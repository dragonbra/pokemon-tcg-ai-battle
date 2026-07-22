"""Freeze already-downloaded exact-deck replays into an aggregate manifest."""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from .download_expert_replays import _sha256
from .download_top100_exact_replays import (
    SCHEMA_VERSION,
    _decks_from_replay,
    _episode_metadata,
    _model_value,
    canonical_deck_sha256,
    episode_player_index,
)


def _episode_id_from_path(path: Path) -> int:
    parts = path.name.split("-")
    if len(parts) < 3 or parts[0] != "episode":
        raise ValueError(f"not an episode replay filename: {path}")
    return int(parts[1])


def _discover_replays(roots: Iterable[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for root in roots:
        for path in sorted(root.glob("episode-*-replay.json")):
            episode_id = _episode_id_from_path(path)
            previous = result.get(episode_id)
            if previous is not None and previous.resolve() != path.resolve():
                if _sha256(previous) != _sha256(path):
                    raise ValueError(f"conflicting replay copies for episode {episode_id}")
                continue
            result[episode_id] = path.resolve()
    if not result:
        raise ValueError("no downloaded episode replay files found")
    return result


def _source_row(match: dict[str, Any], player_index: int) -> dict[str, Any]:
    return {
        "rank": int(match["rank"]),
        "team_id": int(match["team_id"]),
        "team_name": str(match["team_name"]),
        "submission_id": int(match["submission_id"]),
        "player_index": int(player_index),
    }


def freeze(
    output: Path,
    *,
    screening: Path,
    deck: list[int],
    replay_roots: Iterable[Path],
    source_manifests: Iterable[Path] = (),
) -> dict[str, Any]:
    if len(deck) != 60:
        raise ValueError("target deck must contain exactly 60 card IDs")
    replay_roots = tuple(replay_roots)
    source_manifests = tuple(source_manifests)
    paths = _discover_replays(replay_roots)
    screening_payload = json.loads(screening.read_text(encoding="utf-8"))
    if not screening_payload.get("complete"):
        raise ValueError("screening snapshot is incomplete")
    target_hash = canonical_deck_sha256(deck)
    if screening_payload.get("deck_sha256") != target_hash:
        raise ValueError("screening snapshot uses a different deck hash")
    matches = [
        row
        for row in screening_payload.get("leaderboard") or []
        if row.get("exact_deck_match") is True
    ]
    matches_by_submission = {int(row["submission_id"]): row for row in matches}
    matches_by_name: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in matches:
        matches_by_name[str(row["team_name"])].append(row)

    sources: dict[int, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    metadata: dict[int, dict[str, Any]] = {}
    for manifest_path in source_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        submission_id = int(manifest["submission_id"])
        match = matches_by_submission.get(submission_id)
        if match is None:
            continue
        for episode in manifest.get("episodes") or []:
            episode_id = int(episode["episode_id"])
            if episode_id not in paths:
                continue
            player_index = int(episode["agent_index"])
            sources[episode_id][player_index] = _source_row(match, player_index)
            metadata[episode_id] = {
                "episode_id": episode_id,
                "create_time": str(episode["create_time"]),
                "end_time": str(episode["end_time"]),
                "state": str(episode["state"]),
                "type": str(episode["type"]),
                "agents": [
                    {"index": index, "team_name": name}
                    for index, name in enumerate(episode.get("agents") or [])
                ],
            }

    api = KaggleApi()
    api.authenticate()
    api_error: str | None = None
    for index, match in enumerate(matches, 1):
        if api_error is not None:
            break
        submission_id = int(match["submission_id"])
        try:
            submission_episodes = api.competition_list_episodes(submission_id) or []
        except Exception as exc:  # noqa: BLE001 - offline replay fallback is fail-closed
            api_error = f"{type(exc).__name__}: {exc}"
            print(f"episode metadata API unavailable; using replay fallback: {api_error}")
            break
        for episode in submission_episodes:
            episode_id = int(_model_value(episode, "id"))
            if episode_id not in paths:
                continue
            player_index = episode_player_index(episode, submission_id)
            sources[episode_id][player_index] = _source_row(match, player_index)
            metadata[episode_id] = _episode_metadata(episode)
        print(f"indexed local sources {index}/{len(matches)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    unmatched: list[int] = []
    source_counts: collections.Counter[str] = collections.Counter()
    for index, (episode_id, path) in enumerate(sorted(paths.items()), 1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        replay_id = int((payload.get("info") or {}).get("EpisodeId"))
        if replay_id != episode_id:
            raise ValueError(f"episode ID mismatch in {path}: {replay_id} != {episode_id}")
        decks = _decks_from_replay(payload)
        agents = (payload.get("info") or {}).get("Agents") or []

        # Team-name fallback only repairs episodes that fell outside the API's
        # rolling 1,000-episode window.  Exact deck validation below remains
        # mandatory, so a same-name non-matching player cannot enter the set.
        if not sources.get(episode_id):
            for player_index, agent in enumerate(agents):
                name = str(agent.get("Name", "")) if isinstance(agent, dict) else ""
                name_matches = matches_by_name.get(name) or []
                if (
                    len(name_matches) == 1
                    and canonical_deck_sha256(decks[player_index]) == target_hash
                ):
                    sources[episode_id][player_index] = _source_row(
                        name_matches[0], player_index
                    )

        expert_players = []
        for player_index, source in sorted(sources.get(episode_id, {}).items()):
            if canonical_deck_sha256(decks[player_index]) != target_hash:
                raise ValueError(
                    f"source submission {source['submission_id']} has a deck mismatch "
                    f"in episode {episode_id}"
                )
            expert_players.append(source)
            source_counts[f"{source['submission_id']}:{source['team_name']}"] += 1
        if not expert_players:
            unmatched.append(episode_id)
            continue

        row = metadata.get(episode_id)
        if row is None:
            row = {
                "episode_id": episode_id,
                "create_time": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "end_time": "",
                "state": "COMPLETED",
                "type": "EPISODE_TYPE_PUBLIC",
                "agents": [
                    {"index": i, "team_name": str(agent.get("Name", ""))}
                    for i, agent in enumerate(agents)
                    if isinstance(agent, dict)
                ],
            }
        row = dict(row)
        row.update(
            {
                "file": os.path.relpath(path, output.parent),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "steps": len(payload.get("steps") or []),
                "expert_players": expert_players,
            }
        )
        episodes.append(row)
        if index % 250 == 0 or index == len(paths):
            print(f"audited local replays {index}/{len(paths)}", flush=True)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} downloaded replays have no exact top-100 source; "
            f"first={unmatched[0]}"
        )
    # Episode IDs are monotonic and remain available when Kaggle's metadata
    # endpoint is unavailable.  They provide a deterministic chronological
    # whole-episode split without relying on local file mtimes.
    episodes.sort(key=lambda row: row["episode_id"])
    total = len(episodes)
    train_end = max(1, int(total * 0.8))
    validation_end = max(train_end + 1, int(total * 0.9)) if total > 1 else total
    for index, row in enumerate(episodes):
        row["split"] = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
    split_counts = collections.Counter(row["split"] for row in episodes)
    trajectory_split_counts: collections.Counter[str] = collections.Counter()
    for row in episodes:
        trajectory_split_counts[row["split"]] += len(row["expert_players"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "subset_status": "frozen_downloaded_only",
        "competition": screening_payload["competition"],
        "leaderboard_captured_at": screening_payload["captured_at"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "episode_metadata_api_error": api_error,
        "split_order": "episode_id ascending (monotonic Kaggle Episode ID)",
        "leaderboard_size": int(screening_payload["leaderboard_size"]),
        "deck": sorted(deck),
        "deck_counts": dict(sorted(collections.Counter(deck).items())),
        "deck_hash_algorithm": "sha256(comma-separated sorted integer card IDs)",
        "deck_sha256": target_hash,
        "leaderboard": screening_payload["leaderboard"],
        "matching_submissions": matches,
        "matching_submission_count": len(matches),
        "replay_roots": [str(path.resolve()) for path in replay_roots],
        "source_manifests": [str(path.resolve()) for path in source_manifests],
        "unique_episode_count": total,
        "expert_trajectory_count": sum(len(row["expert_players"]) for row in episodes),
        "episodes_by_split": dict(sorted(split_counts.items())),
        "trajectories_by_split": dict(sorted(trajectory_split_counts.items())),
        "trajectories_by_source": dict(sorted(source_counts.items())),
        "episodes": episodes,
    }
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(output.resolve()),
        "matching_submission_count": len(matches),
        "unique_episode_count": total,
        "expert_trajectory_count": manifest["expert_trajectory_count"],
        "episodes_by_split": manifest["episodes_by_split"],
        "trajectories_by_split": manifest["trajectories_by_split"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--replay-root", action="append", type=Path, required=True)
    parser.add_argument("--source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deck = [
        int(line.strip())
        for line in args.deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = freeze(
        args.output,
        screening=args.screening,
        deck=deck,
        replay_roots=args.replay_root,
        source_manifests=args.source_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
