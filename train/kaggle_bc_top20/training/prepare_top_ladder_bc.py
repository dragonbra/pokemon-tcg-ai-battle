"""Freeze top-ladder BC sources and download one audited replay per expert.

This command prepares a source staging area, not a train-ready BC corpus.  Raw
official replays stay under ``data/replays`` while per-expert source records and
60-card decks are mirrored under ``rl_runs/dataset``.  A later training run
must still receive its own globally numbered dataset/checkpoint/run directories.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from .download_expert_replays import _download_one, _sha256
from .download_top100_exact_replays import (
    DEFAULT_COMPETITION,
    _decks_from_replay,
    _eligible_episodes,
    _episode_metadata,
    _leaderboard_snapshot,
    canonical_deck_sha256,
    episode_player_index,
)


SCHEMA_VERSION = "ptcg_top_ladder_bc_sources_v1"
SOURCE_SCHEMA_VERSION = "ptcg_single_expert_bc_source_v1"
DEFAULT_CARD_DATA = Path(__file__).resolve().parents[3] / "data/official/EN_Card_Data.csv"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_slug(team_name: str, team_id: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", team_name.lower()).strip("-")
    return normalized or f"team-{team_id}"


def _agent_for_submission(episode: dict[str, Any], submission_id: int) -> dict[str, Any]:
    matches = [
        agent
        for agent in episode.get("agents") or []
        if int(agent.get("submission_id", 0)) == submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"submission {submission_id} is not unique in episode "
            f"{episode.get('episode_id')}: {matches}"
        )
    return matches[0]


def episode_outcome(episode: dict[str, Any], submission_id: int) -> str:
    own = _agent_for_submission(episode, submission_id).get("reward")
    opponents = [
        agent.get("reward")
        for agent in episode.get("agents") or []
        if int(agent.get("submission_id", 0)) != submission_id
    ]
    if own is None or len(opponents) != 1 or opponents[0] is None:
        return "unknown"
    own_reward = float(own)
    opponent_reward = float(opponents[0])
    if own_reward > opponent_reward:
        return "win"
    if own_reward < opponent_reward:
        return "loss"
    return "draw"


def summarize_outcomes(
    episodes: Iterable[dict[str, Any]], submission_id: int, *, api_limit: int = 1000
) -> dict[str, Any]:
    episode_list = list(episodes)
    counts = collections.Counter(
        episode_outcome(episode, submission_id) for episode in episode_list
    )
    known = counts["win"] + counts["loss"] + counts["draw"]
    return {
        "episodes": len(episode_list),
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "unknown": counts["unknown"],
        "win_rate": counts["win"] / known if known else None,
        "non_loss_rate": (counts["win"] + counts["draw"]) / known if known else None,
        "api_limit": api_limit,
        "window_censored": len(episode_list) >= api_limit,
        "scope": "completed public episodes exposed for the frozen leaderboard submission",
    }


def load_card_catalog(path: Path = DEFAULT_CARD_DATA) -> dict[int, dict[str, str]]:
    catalog: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row.get("Card ID", ""))
            except (TypeError, ValueError):
                continue
            if card_id in catalog:
                continue
            catalog[card_id] = {
                "name": str(row.get("Card Name", "")),
                "stage_or_type": str(
                    row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
                ),
                "previous_stage": str(row.get("Previous stage", "")),
                "hp": str(row.get("HP", "")),
            }
    return catalog


def deck_profile(deck: Iterable[int], catalog: dict[int, dict[str, str]]) -> dict[str, Any]:
    cards = [int(card_id) for card_id in deck]
    if len(cards) != 60:
        raise ValueError(f"deck contains {len(cards)} cards instead of 60")
    counts = collections.Counter(cards)
    entries: list[dict[str, Any]] = []
    pokemon: list[dict[str, Any]] = []
    for card_id, count in sorted(counts.items()):
        metadata = catalog.get(card_id, {})
        row = {
            "card_id": card_id,
            "count": count,
            "name": metadata.get("name", f"Unknown card {card_id}"),
            "stage_or_type": metadata.get("stage_or_type", "unknown"),
        }
        entries.append(row)
        hp = metadata.get("hp", "")
        if hp not in {"", "n/a"}:
            pokemon.append(row)
    return {
        "card_count": len(cards),
        "unique_card_count": len(counts),
        "deck_sha256": canonical_deck_sha256(cards),
        "cards": entries,
        "pokemon": pokemon,
        "pokemon_summary": ", ".join(
            f"{row['name']} x{row['count']}" for row in pokemon
        ),
    }


def _opponent(episode: dict[str, Any], submission_id: int) -> dict[str, Any]:
    matches = [
        agent
        for agent in episode.get("agents") or []
        if int(agent.get("submission_id", 0)) != submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"episode {episode.get('episode_id')} does not have one opponent: {matches}"
        )
    return matches[0]


def summarize_known_deck_matchups(
    episodes: Iterable[dict[str, Any]],
    submission_id: int,
    known_decks: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    episode_list = list(episodes)
    by_hash: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for episode in episode_list:
        opponent = _opponent(episode, submission_id)
        profile = known_decks.get(int(opponent.get("submission_id", 0)))
        if profile is not None:
            by_hash[str(profile["deck_sha256"])].append(episode)
    rows = []
    for deck_hash, matched in sorted(by_hash.items()):
        profile = next(
            profile for profile in known_decks.values() if profile["deck_sha256"] == deck_hash
        )
        rows.append(
            {
                "opponent_deck_sha256": deck_hash,
                "opponent_pokemon_summary": profile["pokemon_summary"],
                **summarize_outcomes(matched, submission_id),
            }
        )
    covered = sum(len(matched) for matched in by_hash.values())
    return {
        "known_episode_count": covered,
        "total_episode_count": len(episode_list),
        "coverage": covered / len(episode_list) if episode_list else 0.0,
        "method": "opponent submission_id matched a frozen Top-N submission with an audited deck",
        "by_opponent_deck": rows,
    }


def collect(
    raw_root: Path,
    dataset_root: Path,
    *,
    competition: str = DEFAULT_COMPETITION,
    top: int = 20,
    workers: int = 2,
    retries: int = 12,
    request_interval: float = 1.0,
    network_timeout: float = 60.0,
    card_data: Path = DEFAULT_CARD_DATA,
) -> dict[str, Any]:
    if top < 1:
        raise ValueError("top must be positive")
    if (raw_root / "manifest.json").exists() or (dataset_root / "campaign.json").exists():
        raise FileExistsError("frozen campaign already exists; choose new output directories")
    raw_root.mkdir(parents=True, exist_ok=True)
    replay_root = raw_root / "episodes"
    metadata_root = raw_root / "metadata"
    replay_root.mkdir(exist_ok=True)
    metadata_root.mkdir(exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    captured_at = datetime.now(timezone.utc).isoformat()
    leaderboard = _leaderboard_snapshot(api, competition=competition, top=top)
    prepared: list[dict[str, Any]] = []
    episode_metadata_by_submission: dict[int, list[dict[str, Any]]] = {}

    for rank, row in enumerate(leaderboard, 1):
        submission_id = int(row["submission_id"])
        models = _eligible_episodes(api, submission_id, 1000)
        metadata = [_episode_metadata(episode) for episode in models]
        episode_metadata_by_submission[submission_id] = metadata
        source_id = f"{rank:04d}-rank-{rank:02d}-{source_slug(row['team_name'], row['team_id'])}"
        metadata_path = metadata_root / f"{source_id}.json"
        episode_window = summarize_outcomes(metadata, submission_id)
        _write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "captured_at": captured_at,
                "competition": competition,
                "leaderboard": row,
                "episode_window": episode_window,
                "episodes": metadata,
            },
        )
        if not models:
            prepared.append(
                {
                    **row,
                    "source_id": source_id,
                    "sample_episode_id": None,
                    "sample_player_index": None,
                    "metadata_file": str(metadata_path.relative_to(raw_root)),
                    "episode_window": episode_window,
                    "source_status": "no_completed_public_episode",
                }
            )
            print(
                f"indexed {rank}/{top}: rank={rank} team={row['team_name']!r} "
                "episodes=0 sample=none",
                flush=True,
            )
            continue

        selected_model = models[0]
        episode_id = int(selected_model.id)
        player_index = episode_player_index(selected_model, submission_id)
        prepared.append(
            {
                **row,
                "source_id": source_id,
                "sample_episode_id": episode_id,
                "sample_player_index": player_index,
                "metadata_file": str(metadata_path.relative_to(raw_root)),
                "episode_window": episode_window,
                "source_status": "sample_available",
            }
        )
        print(
            f"indexed {rank}/{top}: rank={rank} team={row['team_name']!r} "
            f"episodes={len(metadata)} sample={episode_id}",
            flush=True,
        )

    episode_ids = sorted(
        {
            int(row["sample_episode_id"])
            for row in prepared
            if row["sample_episode_id"] is not None
        }
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _download_one,
                episode_id,
                replay_root,
                retries,
                request_interval,
                network_timeout,
            ): episode_id
            for episode_id in episode_ids
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"downloaded/audited {index}/{len(futures)} unique samples", flush=True)

    catalog = load_card_catalog(card_data)
    known_decks: dict[int, dict[str, Any]] = {}
    for row in prepared:
        if row["sample_episode_id"] is None:
            row["deck"] = None
            row["deck_profile"] = None
            row["sample_replay"] = None
            row["sample_replay_sha256"] = None
            row["sample_replay_bytes"] = 0
            continue
        episode_id = int(row["sample_episode_id"])
        replay_path = replay_root / f"episode-{episode_id}-replay.json"
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        if int((payload.get("info") or {}).get("EpisodeId")) != episode_id:
            raise ValueError(f"episode ID mismatch in {replay_path}")
        deck = _decks_from_replay(payload)[int(row["sample_player_index"])]
        profile = deck_profile(deck, catalog)
        row["deck"] = deck
        row["deck_profile"] = profile
        row["sample_replay"] = str(replay_path.relative_to(raw_root))
        row["sample_replay_sha256"] = _sha256(replay_path)
        row["sample_replay_bytes"] = replay_path.stat().st_size
        known_decks[int(row["submission_id"])] = profile

    for row in prepared:
        submission_id = int(row["submission_id"])
        row["known_top_deck_matchups"] = summarize_known_deck_matchups(
            episode_metadata_by_submission[submission_id], submission_id, known_decks
        )
        source_root = dataset_root / str(row["source_id"])
        source_root.mkdir(parents=True, exist_ok=True)
        has_sample = row["sample_episode_id"] is not None
        if has_sample:
            (source_root / "deck.csv").write_text(
                "".join(f"{card_id}\n" for card_id in row["deck"]), encoding="utf-8"
            )
        source_manifest = {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "status": (
                "sample_only_not_train_ready"
                if has_sample
                else "no_completed_public_episode_not_trainable"
            ),
            "source_id": row["source_id"],
            "source_policy": {
                "team_id": row["team_id"],
                "team_name": row["team_name"],
                "submission_id": submission_id,
                "leaderboard_rank": row["rank"],
                "leaderboard_score": row["score"],
            },
            "single_policy_constraint": True,
            "sample": (
                {
                    "episode_id": row["sample_episode_id"],
                    "player_index": row["sample_player_index"],
                    "replay": str((raw_root / row["sample_replay"]).resolve()),
                    "sha256": row["sample_replay_sha256"],
                }
                if has_sample
                else None
            ),
            "deck_file": "deck.csv" if has_sample else None,
            "deck_profile": row["deck_profile"],
            "episode_window": row["episode_window"],
            "metadata_file": str((raw_root / row["metadata_file"]).resolve()),
            "next_step": (
                "download and freeze this submission's full exposed public Episode window, "
                "then assign a global rl_runs experiment ID"
                if has_sample
                else "wait for this submission to expose a completed public Episode"
            ),
        }
        _write_json(source_root / "source_manifest.json", source_manifest)

    campaign = {
        "schema_version": SCHEMA_VERSION,
        "status": "sampled_not_train_ready",
        "competition": competition,
        "captured_at": captured_at,
        "leaderboard_size": top,
        "source_policy_contract": "one frozen team/submission policy per BC dataset",
        "episode_api_limit": 1000,
        "raw_replay_root": str(raw_root.resolve()),
        "dataset_staging_root": str(dataset_root.resolve()),
        "unique_sample_replays": len(episode_ids),
        "players": prepared,
    }
    _write_json(raw_root / "manifest.json", campaign)
    _write_json(dataset_root / "campaign.json", campaign)
    return {
        "manifest": str((raw_root / "manifest.json").resolve()),
        "dataset_campaign": str((dataset_root / "campaign.json").resolve()),
        "players": len(prepared),
        "unique_sample_replays": len(episode_ids),
        "win_rate_windows_censored": sum(
            bool(row["episode_window"]["window_censored"]) for row in prepared
        ),
        "players_without_public_episode": sum(
            row["sample_episode_id"] is None for row in prepared
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--network-timeout", type=float, default=60.0)
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    args = parser.parse_args()
    result = collect(
        args.raw_output,
        args.dataset_output,
        competition=args.competition,
        top=args.top,
        workers=args.workers,
        retries=args.retries,
        request_interval=args.request_interval,
        network_timeout=args.network_timeout,
        card_data=args.card_data,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
