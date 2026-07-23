"""Find top-ladder exact-deck experts and download their public replays.

The leaderboard and its selected submissions are frozen before any deck
screening.  A submission is considered a match only when the 60-card multiset
from its own player index in a public replay equals the requested deck.
Downloaded replays are stored once by Episode ID even when two matching
experts played each other.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from .download_expert_replays import _download_one, _sha256


SCHEMA_VERSION = "ptcg_top_ladder_exact_deck_replays_v1"
DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"


def canonical_deck_sha256(deck: Iterable[int]) -> str:
    """Hash a deck as an order-independent 60-card multiset."""
    normalized = ",".join(str(card_id) for card_id in sorted(int(card) for card in deck))
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _model_value(model: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    return default


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.astimezone(timezone.utc)


def select_leaderboard_submission(leaderboard_row: Any, submissions: list[Any]) -> tuple[Any, str]:
    """Resolve the submission represented by one leaderboard row.

    The submission timestamp is the primary identity.  Public scores can move
    while the leaderboard and team-submission endpoints are being refreshed.
    """
    if not submissions:
        raise ValueError("team has no submissions")
    leaderboard_date = _model_value(leaderboard_row, "submission_date", "submissionDate")
    exact_dates = [
        row
        for row in submissions
        if _timestamp(_model_value(row, "date_submitted", "dateSubmitted"))
        == _timestamp(leaderboard_date)
    ]
    if len(exact_dates) == 1:
        return exact_dates[0], "submission_date"

    # Kaggle has emitted leaderboard submission timestamps one millisecond away
    # from the team-submission endpoint. Accept only a uniquely nearest row in a
    # narrow window, preserving the date-first identity contract.
    leaderboard_datetime = _utc_datetime(leaderboard_date)
    dated_rows: list[tuple[float, Any]] = []
    if leaderboard_datetime is not None:
        for row in submissions:
            submitted_at = _utc_datetime(
                _model_value(row, "date_submitted", "dateSubmitted")
            )
            if submitted_at is not None:
                dated_rows.append(
                    (abs((submitted_at - leaderboard_datetime).total_seconds()), row)
                )
    dated_rows.sort(key=lambda item: item[0])
    if (
        dated_rows
        and dated_rows[0][0] <= 2.0
        and (len(dated_rows) == 1 or dated_rows[0][0] < dated_rows[1][0])
    ):
        return dated_rows[0][1], "submission_date_nearest_2s"

    leaderboard_score = str(_model_value(leaderboard_row, "score"))
    exact_scores = [
        row
        for row in submissions
        if str(_model_value(row, "public_score", "publicScore")) == leaderboard_score
    ]
    if len(exact_scores) == 1:
        return exact_scores[0], "public_score"
    raise ValueError(
        "cannot uniquely resolve leaderboard submission: "
        f"date={leaderboard_date!r}, score={leaderboard_score!r}"
    )


def _episode_agents(episode: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fallback_index, agent in enumerate(_model_value(episode, "agents", default=[]) or []):
        result.append(
            {
                "index": int(_model_value(agent, "index", default=fallback_index) or 0),
                "submission_id": int(
                    _model_value(agent, "submission_id", "submissionId", default=0) or 0
                ),
                "team_id": int(_model_value(agent, "team_id", "teamId", default=0) or 0),
                "team_name": str(
                    _model_value(agent, "team_name", "teamName", default="") or ""
                ),
                "reward": _model_value(agent, "reward"),
            }
        )
    return result


def episode_player_index(episode: Any, submission_id: int) -> int:
    matches = [
        int(row["index"])
        for row in _episode_agents(episode)
        if int(row["submission_id"]) == submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"submission {submission_id} is not unique in episode "
            f"{_model_value(episode, 'id')}: {matches}"
        )
    return matches[0]


def _decks_from_replay(payload: dict[str, Any]) -> list[list[int]]:
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for row in step:
            if not isinstance(row, dict):
                continue
            for frame in row.get("visualize") or []:
                actions = frame.get("action") if isinstance(frame, dict) else None
                if (
                    isinstance(actions, list)
                    and len(actions) == 2
                    and all(isinstance(deck, list) and len(deck) == 60 for deck in actions)
                ):
                    return [[int(card_id) for card_id in deck] for deck in actions]
    raise ValueError("replay has no public two-player 60-card initial deck frame")


def _leaderboard_snapshot(
    api: KaggleApi,
    *,
    competition: str,
    top: int,
) -> list[dict[str, Any]]:
    leaderboard = api.competition_leaderboard_view(competition, page_size=top) or []
    rows: list[dict[str, Any]] = []
    for rank, leaderboard_row in enumerate(leaderboard[:top], 1):
        team_id = int(_model_value(leaderboard_row, "team_id", "teamId"))
        submissions = list(api.competition_team_submissions(team_id) or [])
        selected, selected_by = select_leaderboard_submission(leaderboard_row, submissions)
        rows.append(
            {
                "rank": rank,
                "team_id": team_id,
                "team_name": str(_model_value(leaderboard_row, "team_name", "teamName")),
                "score": str(_model_value(leaderboard_row, "score")),
                "leaderboard_submission_date": _timestamp(
                    _model_value(leaderboard_row, "submission_date", "submissionDate")
                ),
                "submission_id": int(_model_value(selected, "id")),
                "submission_date": _timestamp(
                    _model_value(selected, "date_submitted", "dateSubmitted")
                ),
                "submission_public_score": str(
                    _model_value(selected, "public_score", "publicScore")
                ),
                "selected_by": selected_by,
            }
        )
    if len(rows) != top:
        raise RuntimeError(f"leaderboard returned {len(rows)} rows, expected {top}")
    return rows


def _eligible_episodes(api: KaggleApi, submission_id: int, limit: int) -> list[Any]:
    episodes = list(api.competition_list_episodes(submission_id) or [])
    eligible: list[Any] = []
    for episode in episodes:
        if "PUBLIC" not in str(_model_value(episode, "type", default="")):
            continue
        if "COMPLETED" not in str(_model_value(episode, "state", default="")):
            continue
        try:
            episode_player_index(episode, submission_id)
        except ValueError:
            continue
        eligible.append(episode)
    eligible.sort(
        key=lambda row: (_timestamp(_model_value(row, "create_time", "createTime")), int(row.id)),
        reverse=True,
    )
    return eligible[:limit]


def _probe_submission(
    api: KaggleApi,
    row: dict[str, Any],
    probe_root: Path,
    target_hash: str,
    *,
    retries: int,
    request_interval: float,
    network_timeout: float,
) -> tuple[dict[str, Any], Any]:
    episodes = _eligible_episodes(api, int(row["submission_id"]), 1)
    if not episodes:
        raise RuntimeError(f"submission {row['submission_id']} has no completed public episode")
    episode = episodes[0]
    episode_id = int(_model_value(episode, "id"))
    _download_one(episode_id, probe_root, retries, request_interval, network_timeout)
    path = probe_root / f"episode-{episode_id}-replay.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    player_index = episode_player_index(episode, int(row["submission_id"]))
    deck = _decks_from_replay(payload)[player_index]
    deck_hash = canonical_deck_sha256(deck)
    result = {
        **row,
        "probe_episode_id": episode_id,
        "probe_player_index": player_index,
        "deck_sha256": deck_hash,
        "exact_deck_match": deck_hash == target_hash,
    }
    return result, episode


def _load_reusable_replays(manifests: Iterable[Path]) -> dict[int, Path]:
    reusable: dict[int, Path] = {}
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        default_root = manifest_path.parent
        for row in payload.get("episodes") or []:
            path = default_root / str(row["file"])
            if path.is_file():
                reusable[int(row["episode_id"])] = path.resolve()
    return reusable


def _episode_metadata(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": int(_model_value(episode, "id")),
        "create_time": _timestamp(_model_value(episode, "create_time", "createTime")),
        "end_time": _timestamp(_model_value(episode, "end_time", "endTime")),
        "state": str(_model_value(episode, "state", default="")),
        "type": str(_model_value(episode, "type", default="")),
        "agents": _episode_agents(episode),
    }


def collect(
    output_root: Path,
    *,
    competition: str,
    deck: list[int],
    top: int = 100,
    max_episodes_per_submission: int = 1000,
    workers: int = 2,
    retries: int = 12,
    request_interval: float = 1.0,
    network_timeout: float = 60.0,
    reuse_manifests: Iterable[Path] = (),
    resume_screening: Path | None = None,
) -> dict[str, Any]:
    if len(deck) != 60:
        raise ValueError("target deck must contain exactly 60 card IDs")
    reuse_manifests = tuple(reuse_manifests)
    output_root.mkdir(parents=True, exist_ok=True)
    episode_root = output_root / "episodes"
    episode_root.mkdir(exist_ok=True)
    probe_root = Path(tempfile.mkdtemp(prefix="top100-deck-probes-", dir=output_root))
    api = KaggleApi()
    api.authenticate()
    target_hash = canonical_deck_sha256(deck)
    if resume_screening is not None:
        frozen = json.loads(resume_screening.read_text(encoding="utf-8"))
        if not frozen.get("complete"):
            raise ValueError(f"screening snapshot is incomplete: {resume_screening}")
        if frozen.get("deck_sha256") != target_hash:
            raise ValueError(f"screening deck hash does not match: {resume_screening}")
        if frozen.get("competition") != competition:
            raise ValueError(f"screening competition does not match: {resume_screening}")
        if int(frozen.get("leaderboard_size", 0)) != top:
            raise ValueError(f"screening leaderboard size does not match: {resume_screening}")
        captured_at = str(frozen["captured_at"])
        leaderboard: list[dict[str, Any]] = []
        screened = list(frozen["leaderboard"])
    else:
        captured_at = datetime.now(timezone.utc).isoformat()
        leaderboard = _leaderboard_snapshot(api, competition=competition, top=top)
        screened = []
    probe_episodes: dict[int, Any] = {}
    screening_path = output_root / "screening.json"
    try:
        for index, row in enumerate(leaderboard, 1):
            try:
                result, episode = _probe_submission(
                    api,
                    row,
                    probe_root,
                    target_hash,
                    retries=retries,
                    request_interval=request_interval,
                    network_timeout=network_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the rest of the snapshot
                result = {
                    **row,
                    "exact_deck_match": None,
                    "screening_error": f"{type(exc).__name__}: {exc}",
                }
                episode = None
            screened.append(result)
            if episode is not None:
                probe_episodes[int(result["probe_episode_id"])] = episode
            screening_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "competition": competition,
                        "captured_at": captured_at,
                        "deck_sha256": target_hash,
                        "leaderboard_size": top,
                        "complete": index == len(leaderboard),
                        "leaderboard": screened,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"screened {index}/{len(leaderboard)}: rank={row['rank']} "
                f"team={row['team_name']!r} exact={result['exact_deck_match']} ",
                flush=True,
            )

        matches = [row for row in screened if row["exact_deck_match"]]
        if not matches:
            raise RuntimeError("no top-ladder submission matched the target deck")

        episode_sources: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        episode_models: dict[int, Any] = dict(probe_episodes)
        for index, row in enumerate(matches, 1):
            submission_id = int(row["submission_id"])
            episodes = _eligible_episodes(api, submission_id, max_episodes_per_submission)
            row["public_episode_count"] = len(episodes)
            for episode in episodes:
                episode_id = int(_model_value(episode, "id"))
                player_index = episode_player_index(episode, submission_id)
                episode_models[episode_id] = episode
                episode_sources[episode_id].append(
                    {
                        "rank": int(row["rank"]),
                        "team_id": int(row["team_id"]),
                        "team_name": str(row["team_name"]),
                        "submission_id": submission_id,
                        "player_index": player_index,
                    }
                )
            print(
                f"indexed {index}/{len(matches)} exact submissions: "
                f"{row['team_name']!r} episodes={len(episodes)}",
                flush=True,
            )

        reusable = _load_reusable_replays(reuse_manifests)
        replay_paths: dict[int, Path] = {}
        missing_ids: list[int] = []
        for episode_id in sorted(episode_sources):
            cached = episode_root / f"episode-{episode_id}-replay.json"
            probe = probe_root / cached.name
            if cached.is_file():
                replay_paths[episode_id] = cached.resolve()
            elif episode_id in reusable:
                replay_paths[episode_id] = reusable[episode_id]
            elif probe.is_file():
                os.replace(probe, cached)
                replay_paths[episode_id] = cached.resolve()
            else:
                missing_ids.append(episode_id)
        missing_ids.sort(
            key=lambda episode_id: _timestamp(
                _model_value(episode_models[episode_id], "create_time", "createTime")
            ),
            reverse=True,
        )

        download_failures: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    _download_one,
                    episode_id,
                    episode_root,
                    retries,
                    request_interval,
                    network_timeout,
                ): episode_id
                for episode_id in missing_ids
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                episode_id = futures[future]
                try:
                    episode_id, _, _ = future.result()
                except Exception as exc:  # noqa: BLE001 - finish and persist the failure set
                    download_failures.append(
                        {
                            "episode_id": episode_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"failed replay {episode_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                replay_paths[episode_id] = (
                    episode_root / f"episode-{episode_id}-replay.json"
                ).resolve()
                if index % 50 == 0 or index == len(futures):
                    print(f"downloaded {index}/{len(futures)} new unique replays", flush=True)
        failure_path = output_root / "download_failures.json"
        if download_failures:
            failure_path.write_text(
                json.dumps(download_failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{len(download_failures)} replay downloads failed; resume after inspecting "
                f"{failure_path}"
            )
        if failure_path.is_file():
            failure_path.unlink()

        episodes_out: list[dict[str, Any]] = []
        for index, episode_id in enumerate(sorted(episode_sources), 1):
            path = replay_paths[episode_id]
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int((payload.get("info") or {}).get("EpisodeId")) != episode_id:
                raise ValueError(f"episode ID mismatch in {path}")
            decks = _decks_from_replay(payload)
            expert_players = sorted(
                episode_sources[episode_id], key=lambda row: (row["player_index"], row["rank"])
            )
            seen_indices: set[int] = set()
            deduplicated_players: list[dict[str, Any]] = []
            for player in expert_players:
                player_index = int(player["player_index"])
                if canonical_deck_sha256(decks[player_index]) != target_hash:
                    raise ValueError(
                        f"exact deck mismatch for submission {player['submission_id']} "
                        f"in episode {episode_id}"
                    )
                if player_index not in seen_indices:
                    deduplicated_players.append(player)
                    seen_indices.add(player_index)
            row = _episode_metadata(episode_models[episode_id])
            row.update(
                {
                    "file": os.path.relpath(path, output_root),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "steps": len(payload.get("steps") or []),
                    "expert_players": deduplicated_players,
                }
            )
            episodes_out.append(row)
            if index % 250 == 0 or index == len(episode_sources):
                print(f"audited {index}/{len(episode_sources)} unique replays", flush=True)

        episodes_out.sort(key=lambda row: (row["create_time"], row["episode_id"]))
        total = len(episodes_out)
        train_end = max(1, int(total * 0.8))
        validation_end = max(train_end + 1, int(total * 0.9)) if total > 1 else total
        for index, row in enumerate(episodes_out):
            row["split"] = (
                "train"
                if index < train_end
                else "validation"
                if index < validation_end
                else "test"
            )

        trajectory_count = sum(len(row["expert_players"]) for row in episodes_out)
        split_counts = collections.Counter(row["split"] for row in episodes_out)
        trajectory_split_counts = collections.Counter()
        for row in episodes_out:
            trajectory_split_counts[row["split"]] += len(row["expert_players"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "competition": competition,
            "captured_at": captured_at,
            "leaderboard_size": top,
            "submission_resolution": (
                "leaderboard submissionDate exact match; unique publicScore fallback"
            ),
            "deck": sorted(deck),
            "deck_counts": dict(sorted(collections.Counter(deck).items())),
            "deck_hash_algorithm": "sha256(comma-separated sorted integer card IDs)",
            "deck_sha256": target_hash,
            "episode_api_limit": 1000,
            "max_episodes_per_submission": max_episodes_per_submission,
            "leaderboard": screened,
            "matching_submissions": matches,
            "matching_submission_count": len(matches),
            "unique_episode_count": total,
            "expert_trajectory_count": trajectory_count,
            "episodes_by_split": dict(sorted(split_counts.items())),
            "trajectories_by_split": dict(sorted(trajectory_split_counts.items())),
            "reuse_manifests": [str(path.resolve()) for path in reuse_manifests],
            "episodes": episodes_out,
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "manifest": str(manifest_path.resolve()),
            "deck_sha256": target_hash,
            "matching_submission_count": len(matches),
            "matching_teams": [row["team_name"] for row in matches],
            "unique_episode_count": total,
            "expert_trajectory_count": trajectory_count,
            "episodes_by_split": dict(sorted(split_counts.items())),
            "downloaded_bytes": sum(
                path.stat().st_size
                for path in replay_paths.values()
                if path.parent == episode_root.resolve()
            ),
        }
    finally:
        for path in probe_root.glob("*"):
            if path.is_file():
                path.unlink()
        probe_root.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--max-episodes-per-submission", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--network-timeout", type=float, default=60.0)
    parser.add_argument("--reuse-manifest", action="append", type=Path, default=[])
    parser.add_argument("--resume-screening", type=Path)
    args = parser.parse_args()
    deck = [
        int(line.strip())
        for line in args.deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = collect(
        args.output,
        competition=args.competition,
        deck=deck,
        top=args.top,
        max_episodes_per_submission=args.max_episodes_per_submission,
        workers=args.workers,
        retries=args.retries,
        request_interval=args.request_interval,
        network_timeout=args.network_timeout,
        reuse_manifests=args.reuse_manifest,
        resume_screening=args.resume_screening,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
