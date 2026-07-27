"""Build a compact player/deck catalog from an immutable official Episode archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

_ARCHIVE: zipfile.ZipFile | None = None
_MEMBER_RE = re.compile(r"(?:episode-)?(\d+)(?:-replay)?\.json")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deck_hash(cards: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, sorted(cards))).encode("ascii")).hexdigest()


def _model_deck_hash(cards: list[int]) -> str:
    counts = [list(item) for item in sorted(Counter(cards).items())]
    payload = json.dumps(counts, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _outcomes(rewards: list[object]) -> list[str]:
    if not rewards or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in rewards
    ):
        return ["unknown"] * len(rewards)
    maximum = max(float(value) for value in rewards)
    if sum(float(value) == maximum for value in rewards) != 1:
        return ["draw"] * len(rewards)
    return ["win" if float(value) == maximum else "loss" for value in rewards]


def _init_worker(archive_path: str) -> None:
    global _ARCHIVE
    _ARCHIVE = zipfile.ZipFile(archive_path)


def _registration_decks(payload: dict[str, Any]) -> list[list[int]]:
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("replay steps are absent")
    first_member = steps[0][0]
    visualize = first_member.get("visualize") if isinstance(first_member, dict) else None
    if isinstance(visualize, list) and visualize:
        decks = visualize[0].get("action")
    else:
        decks = [member.get("action") for member in steps[1]]
    if not isinstance(decks, list):
        raise ValueError("registration decks are absent")
    result: list[list[int]] = []
    for deck in decks:
        if (
            not isinstance(deck, list)
            or len(deck) != 60
            or any(isinstance(card, bool) or not isinstance(card, int) for card in deck)
        ):
            raise ValueError("registration deck must contain exactly 60 integer IDs")
        result.append(deck)
    return result


def _read_member(member_name: str) -> dict[str, Any]:
    if _ARCHIVE is None:
        raise RuntimeError("archive worker was not initialized")
    raw = _ARCHIVE.read(member_name)
    payload = json.loads(raw)
    info = payload.get("info")
    if not isinstance(info, dict):
        raise ValueError(f"missing info: {member_name}")
    match = _MEMBER_RE.fullmatch(Path(member_name).name)
    if match is None:
        raise ValueError(f"invalid member name: {member_name}")
    declared_id = int(match.group(1))
    episode_id = int(info.get("EpisodeId", -1))
    if episode_id != declared_id:
        raise ValueError(f"Episode ID mismatch: {member_name}")
    teams = info.get("TeamNames")
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    decks = _registration_decks(payload)
    if not (
        isinstance(teams, list)
        and isinstance(rewards, list)
        and isinstance(statuses, list)
        and len(teams) == len(rewards) == len(statuses) == len(decks)
    ):
        raise ValueError(f"player dimensions disagree: {member_name}")
    players = []
    outcomes = _outcomes(rewards)
    for index, deck in enumerate(decks):
        reward = rewards[index]
        players.append(
            {
                "player_index": index,
                "team_name": str(teams[index]),
                "reward": reward,
                "outcome": outcomes[index],
                "deck_sha256": _deck_hash(deck),
                "model_deck_sha256": _model_deck_hash(deck),
                "deck_counts": [list(item) for item in sorted(Counter(deck).items())],
            }
        )
    return {
        "episode_id": episode_id,
        "member": member_name,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "complete": statuses == ["DONE"] * len(statuses),
        "step_count": len(payload.get("steps", [])),
        "players": players,
    }


def build_catalog(archive_path: Path, workers: int) -> dict[str, Any]:
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path) as bundle:
        members = sorted(
            item.filename
            for item in bundle.infolist()
            if not item.is_dir() and _MEMBER_RE.fullmatch(Path(item.filename).name)
        )
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(archive_path),)
    ) as pool:
        episodes = list(pool.map(_read_member, members, chunksize=8))
    return {
        "schema_version": "0015_official_episode_catalog_v1",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "episode_count": len(episodes),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = build_catalog(args.archive, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episode_count": result["episode_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
