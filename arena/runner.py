from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping

from evaluation.packages.loader import SubmissionPackage
from evaluation.runner.models import GameRequest, GameResult
from evaluation.runner.single import run_game

from .models import GameRecord, RatingEvent
from .rating import (
    EloRatingEngine,
    GaussianRatingEngine,
    RatingState,
    apply_checkpoint_status,
)
from .scheduler import Pairing, build_rating_pairs, build_smoke_pairs
from .storage import ArenaStore


GameRunner = Callable[
    [GameRequest, Path, float],
    tuple[GameResult, dict[str, object]],
]


TRACE_RETAIN_LIMIT = 3


@dataclass(frozen=True)
class ArenaRunConfig:
    run_id: str
    phase: str
    games_per_pair: int = 10
    max_steps: int = 1000
    worker_timeout_seconds: float = 30.0
    seed: int = 20260722
    visualize: bool = False
    rating_method: str = "official_gaussian_approx"
    include_demoted: bool = False
    checkpoint_every: int = 10


def _default_game_runner(
    request: GameRequest,
    temp_root: Path,
    timeout_seconds: float,
) -> tuple[GameResult, dict[str, object]]:
    return run_game(request, temp_root=temp_root, timeout_seconds=timeout_seconds)


class ArenaRunner:
    def __init__(
        self,
        store: ArenaStore,
        packages: Mapping[str, SubmissionPackage],
        config: ArenaRunConfig,
        *,
        game_runner: GameRunner = _default_game_runner,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.packages = dict(packages)
        self.config = config
        self.game_runner = game_runner
        self.checkpoint_callback = checkpoint_callback
        self.gaussian_engine = GaussianRatingEngine()
        self.elo_engine = EloRatingEngine()
        self.gaussian_states = {name: RatingState(name) for name in self.packages}
        self.elo_states = {name: RatingState(name) for name in self.packages}
        existing_games = self.store.load_games(self.config.run_id)
        self._known_games = {str(game["game_id"]): game for game in existing_games}
        self.completed_count = len(existing_games)
        checkpoint = self.store.load_latest_checkpoint(self.config.run_id)
        self._last_checkpoint_sequence = int(checkpoint["sequence"]) if checkpoint else None
        self._retained_trace_count = min(
            TRACE_RETAIN_LIMIT,
            sum(
                1
                for path in (self.store.root / "runs" / self.config.run_id / "temp").glob("*.json")
                if path.is_file() and not path.name.endswith((".request.json", ".result.json"))
            ),
        )
        self._restore_states()
        self._restore_checkpoint_status(checkpoint)

    def run_one(self, pairing: Pairing, game_number: int) -> GameRecord:
        if pairing.first not in self.packages or pairing.second not in self.packages:
            raise KeyError(f"unknown Arena package in pairing: {pairing.players}")
        player_a_first = game_number % 2 == 1
        game_id = f"{self.config.run_id}-{self.config.phase}-{pairing.first}-{pairing.second}-{game_number:04d}"
        existing = self._known_games.get(game_id)
        if existing is not None:
            return GameRecord(
                game_id=str(existing["game_id"]),
                run_id=str(existing["run_id"]),
                player_a=str(existing["player_a"]),
                player_b=str(existing["player_b"]),
                player_a_first=bool(existing["player_a_first"]),
                winner=str(existing["winner"]) if existing.get("winner") is not None else None,
                result=str(existing["result"]),
                status=str(existing["status"]),
                error_kind=str(existing["error_kind"]) if existing.get("error_kind") else None,
                error=str(existing["error"]) if existing.get("error") else None,
                steps=int(existing["steps"]),
                payload=dict(existing.get("payload", {})),
            )
        request = GameRequest(
            run_id=self.config.run_id,
            game_id=game_id,
            candidate=self.packages[pairing.first],
            opponent=self.packages[pairing.second],
            candidate_first=player_a_first,
            max_steps=self.config.max_steps,
            visualize=self.config.visualize,
        )
        temp_root = self.store.root / "runs" / self.config.run_id / "temp"
        result, trace = self.game_runner(
            request,
            temp_root,
            self.config.worker_timeout_seconds,
        )
        winner = (
            pairing.first
            if result.finished and result.winner == 0
            else pairing.second
            if result.finished and result.winner == 1
            else None
        )
        finished_draw = result.finished and result.winner is None
        trace_path = result.trace_path
        retain_trace = trace_path.is_file() and self._retained_trace_count < TRACE_RETAIN_LIMIT
        if retain_trace:
            self._retained_trace_count += 1
        record = GameRecord(
            game_id=game_id,
            run_id=self.config.run_id,
            player_a=pairing.first,
            player_b=pairing.second,
            player_a_first=player_a_first,
            winner=winner,
            result="draw" if finished_draw else ("win" if winner else "error"),
            status=result.status,
            error_kind=result.error_kind,
            error=result.error,
            steps=result.steps,
            payload={
                **_trace_metadata(result, trace, retain_trace),
                "candidate_first": player_a_first,
            },
        )
        self.store.record_game(record)
        _cleanup_game_temp_files(result, keep_trace=retain_trace)
        self._known_games[game_id] = record.to_json()
        if result.finished:
            outcome = 0.5 if finished_draw else (1.0 if winner == pairing.first else 0.0)
            self._record_rating(pairing, game_id, outcome)
        self.completed_count += 1
        if self.completed_count % self.config.checkpoint_every == 0:
            self.checkpoint()
        return record

    def run_initial_round(self, deck_ids: tuple[str, ...]) -> None:
        for pairing in build_smoke_pairs(deck_ids):
            for game_number in range(1, self.config.games_per_pair + 1):
                self.run_one(pairing, game_number)
        self.checkpoint()

    def run_continuous_round(self, limit: int) -> None:
        games_to_run = max(0, limit)
        counts = self._pair_counts()
        attempt_counts = self._pair_attempt_counts()
        for sequence in range(games_to_run):
            states = tuple(self.gaussian_states.values())
            if self.config.include_demoted:
                states = tuple(
                    replace(state, status="active") if state.status == "demoted" else state
                    for state in states
                )
            pairs = build_rating_pairs(
                states,
                counts,
                self.config.seed + self.completed_count + sequence,
                1,
            )
            if not pairs:
                break
            pairing = pairs[0]
            pair = tuple(sorted(pairing.players))
            game_number = attempt_counts.get(pair, 0) + 1
            record = self.run_one(pairing, game_number)
            attempt_counts[pair] = game_number
            if record.status == "finished":
                counts[pair] = counts.get(pair, 0) + 1
        self.checkpoint()

    def checkpoint(self) -> None:
        sequence = self.completed_count
        if sequence != self._last_checkpoint_sequence:
            self.gaussian_states = {
                state.name: state
                for state in apply_checkpoint_status(tuple(self.gaussian_states.values()))
            }
            self._last_checkpoint_sequence = sequence
        ratings = {
            name: {
                "mu": round(state.mu, 6),
                "sigma": round(state.sigma, 6),
                "elo": round(self.elo_states[name].elo, 6),
                "games": state.games,
                "status": state.status,
                "below_threshold_checkpoints": state.below_threshold_checkpoints,
            }
            for name, state in self.gaussian_states.items()
        }
        self.store.record_checkpoint(
            self.config.run_id,
            sequence,
            ratings,
            _now(),
        )
        if self.checkpoint_callback is not None:
            self.checkpoint_callback()

    def _record_rating(self, pairing: Pairing, game_id: str, outcome: float) -> None:
        gaussian = self.gaussian_engine.update(
            self.gaussian_states[pairing.first],
            self.gaussian_states[pairing.second],
            outcome,
        )
        elo = self.elo_engine.update(
            self.elo_states[pairing.first],
            self.elo_states[pairing.second],
            outcome,
        )
        self.gaussian_states[pairing.first] = gaussian.after_a
        self.gaussian_states[pairing.second] = gaussian.after_b
        self.elo_states[pairing.first] = elo.after_a
        self.elo_states[pairing.second] = elo.after_b
        outcome_name = "draw" if outcome == 0.5 else ("win" if outcome == 1.0 else "loss")
        for update in (gaussian, elo):
            self.store.record_rating_event(
                RatingEvent(
                    event_id=f"{game_id}:{update.engine}",
                    run_id=self.config.run_id,
                    game_id=game_id,
                    engine=update.engine,
                    revision=update.revision,
                    player_a=pairing.first,
                    player_b=pairing.second,
                    before={
                        "a": _state_json(update.before_a),
                        "b": _state_json(update.before_b),
                    },
                    outcome=outcome_name,
                    expected_a=update.expected_a,
                    delta_a=update.delta_a,
                    delta_b=update.delta_b,
                    after={
                        "a": _state_json(update.after_a),
                        "b": _state_json(update.after_b),
                    },
                    created_at=_now(),
                )
            )

    def _pair_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for game in self._known_games.values():
            if game["player_a"] == game["player_b"] or game.get("status") != "finished":
                continue
            pair = tuple(sorted((str(game["player_a"]), str(game["player_b"]))))
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _pair_attempt_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for game in self._known_games.values():
            if game["player_a"] == game["player_b"]:
                continue
            pair = tuple(sorted((str(game["player_a"]), str(game["player_b"]))))
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _restore_states(self) -> None:
        latest: dict[tuple[str, str], tuple[tuple[str, str], Mapping[str, object]]] = {}
        for event in self.store.load_rating_events(self.config.run_id):
            engine = str(event.get("engine", ""))
            stamp = (str(event.get("created_at", "")), str(event.get("event_id", "")))
            after = event.get("after", {})
            if not isinstance(after, Mapping):
                continue
            for side, player_key in (("a", "player_a"), ("b", "player_b")):
                player = str(event.get(player_key, ""))
                state = after.get(side)
                if player not in self.packages or not isinstance(state, Mapping):
                    continue
                key = (engine, player)
                if key not in latest or stamp >= latest[key][0]:
                    latest[key] = (stamp, state)
        for (engine, player), (_, raw) in latest.items():
            restored = RatingState(
                name=player,
                mu=float(raw.get("mu", 600.0)),
                sigma=float(raw.get("sigma", 200.0)),
                elo=float(raw.get("elo", 600.0)),
                games=int(raw.get("games", 0)),
                status=str(raw.get("status", "active")),
                below_threshold_checkpoints=int(raw.get("below_threshold_checkpoints", 0)),
            )
            if engine == "official_gaussian_approx":
                self.gaussian_states[player] = restored
            elif engine == "elo_compat":
                self.elo_states[player] = restored

    def _restore_checkpoint_status(self, checkpoint: Mapping[str, object] | None) -> None:
        if not checkpoint:
            return
        ratings = checkpoint.get("ratings", {})
        if not isinstance(ratings, Mapping):
            return
        for player, raw in ratings.items():
            if player not in self.gaussian_states or not isinstance(raw, Mapping):
                continue
            current = self.gaussian_states[player]
            if int(raw.get("games", 0)) != current.games:
                continue
            self.gaussian_states[player] = replace(
                current,
                status=str(raw.get("status", current.status)),
                below_threshold_checkpoints=int(
                    raw.get("below_threshold_checkpoints", current.below_threshold_checkpoints)
                ),
            )


def _state_json(state: RatingState) -> dict[str, object]:
    return asdict(state)


def _trace_metadata(
    result: GameResult,
    trace: Mapping[str, object],
    retained: bool,
) -> dict[str, object]:
    raw_trace = trace.get("trace")
    steps = len(raw_trace) if isinstance(raw_trace, list) else 0
    metadata: dict[str, object] = {
        "trace_path": str(result.trace_path),
        "trace_retained": retained,
        "trace_steps": steps,
        "worker_status": result.status,
        "worker_finished": result.finished,
        "worker_winner": result.winner,
    }
    try:
        metadata["trace_size_bytes"] = result.trace_path.stat().st_size
        metadata["trace_sha256"] = _hash_file(result.trace_path)
    except OSError:
        metadata["trace_size_bytes"] = 0
    return metadata


def _cleanup_game_temp_files(result: GameResult, *, keep_trace: bool) -> None:
    trace_path = result.trace_path
    if not keep_trace:
        trace_path.unlink(missing_ok=True)
    for suffix in (".request.json", ".result.json"):
        trace_path.parent.joinpath(f"{result.game_id}{suffix}").unlink(missing_ok=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()
