"""Freeze auditable 0015 trajectory membership from catalog and targeted overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

TARGET = "dragapult_dusknoir"
PURE = "pure_dragapult"
MARNIE = "marnie_munkidori"
DUSKNOIR_IDS = frozenset({131, 132, 133})


def _source_key(team_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", team_name).encode("ascii", "ignore").decode()
    key = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    if key:
        return key
    full_identity = unicodedata.normalize("NFKC", team_name).casefold()
    suffix = hashlib.sha256(full_identity.encode("utf-8")).hexdigest()[:12]
    return f"unicode_{suffix}"


def _legacy_source_key(team_name: str) -> str:
    """Reproduce the immutable V1 source vocabulary for non-T4 regeneration."""
    key = _source_key(team_name)
    return "non_ascii_source" if key.startswith("unicode_") else key


def _deck_ids(player: dict[str, Any]) -> set[int]:
    return {int(card_id) for card_id, _ in player["deck_counts"]}


def _first_player(payload: dict[str, Any]) -> int:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("replay steps are absent")
    for step in steps:
        if not isinstance(step, list):
            continue
        for member in step:
            if not isinstance(member, dict):
                continue
            frames = member.get("visualize")
            if not isinstance(frames, list):
                continue
            for frame in frames:
                current = (
                    (frame.get("obs") or {}).get("current")
                    if isinstance(frame, dict)
                    else None
                )
                value = current.get("firstPlayer") if isinstance(current, dict) else None
                if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
                    return value
    raise ValueError("replay has no resolved firstPlayer evidence")


def _outcome_from_reward(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unknown"
    return "win" if value > 0 else "loss" if value < 0 else "draw"


def _validation_membership(rows: list[dict[str, Any]]) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        seat = "first" if row["player_index"] == row["first_player"] else "second"
        strata.setdefault((row["outcome"], seat), []).append(row)
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: hashlib.sha256(
                (
                    "0015_target_validation_v1|"
                    f"{key[0]}|{key[1]}|{row['episode_id']}|{row['player_index']}"
                ).encode("ascii")
            ).hexdigest(),
        )
        count = max(1, math.ceil(len(ordered) * 0.2))
        selected.update((row["episode_id"], row["player_index"]) for row in ordered[:count])
    return selected


def _deterministic_t4_cap(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit < 1 or limit > len(rows):
        raise ValueError("T4 cap must be within the available candidate count")
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        strata.setdefault(row["outcome"], []).append(row)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for outcome, values in sorted(strata.items()):
        exact = limit * len(values) / len(rows)
        allocations[outcome] = int(exact)
        remainders.append((exact - int(exact), outcome))
    remaining = limit - sum(allocations.values())
    for _, outcome in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[outcome] += 1
    selected: list[dict[str, Any]] = []
    for outcome, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: hashlib.sha256(
                (
                    "0015_t4_outcome_cap_v1|"
                    f"{outcome}|{row['episode_id']}|{row['player_index']}"
                ).encode("ascii")
            ).hexdigest(),
        )
        selected.extend(ordered[: allocations[outcome]])
    if len(selected) != limit:
        raise RuntimeError("T4 cap allocation did not reach the declared limit")
    return selected


def _read_overlay_rows(overlay_root: Path) -> tuple[list[dict[str, Any]], dict[int, bytes]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[int, bytes] = {}
    builds = {
        "third_ptcg_club": TARGET,
        "lumen_liquidity": PURE,
        "oshbocker": PURE,
    }
    for source_dir in sorted(path for path in overlay_root.iterdir() if path.is_dir()):
        if source_dir.name not in builds:
            continue
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["episodes"]:
            path = source_dir / item["file"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(f"overlay replay hash mismatch: {path}")
            episode_id = int(item["episode_id"])
            payload = json.loads(raw)
            first_player = _first_player(payload)
            rows.append(
                {
                    "episode_id": episode_id,
                    "episode_date": str(item["create_time"])[:10],
                    "player_index": int(item["player_index"]),
                    "first_player": first_player,
                    "team_name": manifest["team_name"],
                    "source_key": _source_key(manifest["team_name"]),
                    "submission_id": int(manifest["submission_id"]),
                    "build": builds[source_dir.name],
                    "deck_sha256": item["deck_sha256"],
                    "outcome": _outcome_from_reward(item["reward"]),
                    "payload_sha256": item["sha256"],
                    "origin": {
                        "kind": "targeted_overlay",
                        "path": str(path),
                        "replay_source": item["replay_source"],
                    },
                }
            )
            payloads[episode_id] = raw
    return rows, payloads


def build_index(
    catalog_path: Path,
    overlay_root: Path,
    archive_path: Path,
    *,
    include_t4: bool = False,
    base_index_path: Path | None = None,
    t4_cap: int | None = None,
) -> dict[str, Any]:
    if include_t4 != (base_index_path is not None):
        raise ValueError("T4 inclusion requires exactly one frozen base index")
    if t4_cap is not None and (not include_t4 or t4_cap < 1):
        raise ValueError("a positive T4 cap requires T4 inclusion")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    overlay_rows, _ = _read_overlay_rows(overlay_root)
    overlay_ids = {row["episode_id"] for row in overlay_rows}
    official_rows: list[dict[str, Any]] = []
    t4_candidates: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as bundle:
        for episode in catalog["episodes"]:
            if not episode["complete"]:
                continue
            raw: bytes | None = None
            for player in episode["players"]:
                ids = _deck_ids(player)
                build = None
                if (
                    player["team_name"] == "LumenLiquidity"
                    and {119, 120, 121, 131, 132, 133}.issubset(ids)
                ):
                    build = TARGET
                elif (
                    player["team_name"] == "flg"
                    and 121 in ids
                    and not DUSKNOIR_IDS.intersection(ids)
                ):
                    build = PURE
                elif {646, 647, 648, 112}.issubset(ids):
                    build = MARNIE
                if build is None:
                    continue
                if episode["episode_id"] in overlay_ids and build != MARNIE:
                    continue
                if player["outcome"] not in {"win", "loss", "draw"}:
                    continue
                if build != MARNIE and raw is None:
                    raw = bundle.read(episode["member"])
                    if hashlib.sha256(raw).hexdigest() != episode["payload_sha256"]:
                        raise ValueError(f"official payload hash mismatch: {episode['member']}")
                first_player = None
                if build != MARNIE:
                    if raw is None:
                        raise RuntimeError("selected official replay was not read")
                    first_player = _first_player(json.loads(raw))
                source_key = (
                    _source_key(player["team_name"])
                    if build != MARNIE or include_t4
                    else _legacy_source_key(player["team_name"])
                )
                row = {
                    "episode_id": int(episode["episode_id"]),
                    "episode_date": "2026-07-26",
                    "player_index": int(player["player_index"]),
                    "first_player": first_player,
                    "team_name": player["team_name"],
                    "source_key": source_key,
                    "submission_id": None,
                    "build": build,
                    "deck_sha256": player["deck_sha256"],
                    "model_deck_sha256": player["model_deck_sha256"],
                    "outcome": player["outcome"],
                    "payload_sha256": episode["payload_sha256"],
                    "origin": {
                        "kind": "official_archive",
                        "archive": str(archive_path),
                        "member": episode["member"],
                    },
                }
                (t4_candidates if build == MARNIE else official_rows).append(row)

    core_rows = official_rows + overlay_rows
    target_teacher = [row for row in core_rows if row["source_key"] == "third_ptcg_club"]
    validation = _validation_membership(target_teacher)
    for row in core_rows:
        row["split"] = (
            "validation"
            if (row["episode_id"], row["player_index"]) in validation
            else "train"
        )
    core_train_count = sum(row["split"] == "train" for row in core_rows)
    resolved_t4_cap = t4_cap if t4_cap is not None else core_train_count
    selected_t4 = (
        _deterministic_t4_cap(t4_candidates, resolved_t4_cap) if include_t4 else []
    )
    if include_t4:
        with zipfile.ZipFile(archive_path) as bundle:
            for row in selected_t4:
                raw = bundle.read(row["origin"]["member"])
                if hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
                    raise ValueError(f"official payload hash mismatch: {row['episode_id']}")
                row["first_player"] = _first_player(json.loads(raw))
    selected_rows = core_rows + selected_t4
    identities: set[tuple[int, int]] = set()
    for row in selected_rows:
        identity = row["episode_id"], row["player_index"]
        if identity in identities:
            raise ValueError(f"duplicate selected trajectory: {identity}")
        identities.add(identity)
    if include_t4:
        if base_index_path is None:
            raise RuntimeError("base index disappeared")
        base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
        source_vocabulary = {
            str(key): int(value) for key, value in base_index["source_vocabulary"].items()
        }
        if sorted(source_vocabulary.values()) != list(range(1, len(source_vocabulary) + 1)):
            raise ValueError("frozen base source vocabulary IDs are not contiguous")
        next_id = max(source_vocabulary.values()) + 1
        for key in sorted(
            {row["source_key"] for row in core_rows + t4_candidates}
            - source_vocabulary.keys()
        ):
            source_vocabulary[key] = next_id
            next_id += 1
        team_keys: dict[str, str] = {}
        key_teams: dict[str, str] = {}
        for row in core_rows + t4_candidates:
            team_name = unicodedata.normalize("NFKC", row["team_name"])
            key = row["source_key"]
            if team_name in team_keys and team_keys[team_name] != key:
                raise ValueError(f"team identity maps to multiple source keys: {team_name}")
            if key in key_teams and key_teams[key] != team_name:
                raise ValueError(f"source key collision: {key}")
            team_keys[team_name] = key
            key_teams[key] = team_name
        for row in selected_t4:
            row["split"] = "train"
    else:
        source_keys = ["third_ptcg_club"] + sorted(
            {row["source_key"] for row in core_rows + t4_candidates} - {"third_ptcg_club"}
        )
        source_vocabulary = {key: index + 1 for index, key in enumerate(source_keys)}
    for row in selected_rows:
        row["source_id"] = source_vocabulary[row["source_key"]]

    counts = Counter(
        (row["build"], row["source_key"], row["split"], row["outcome"])
        for row in selected_rows
    )
    return {
        "schema_version": (
            "0015_trajectory_source_index_v2_t4"
            if include_t4
            else "0015_trajectory_source_index_v1"
        ),
        "catalog": str(catalog_path),
        "official_archive": str(archive_path),
        "overlay_root": str(overlay_root),
        "target_validation": {
            "source_key": "third_ptcg_club",
            "fraction_rule": "ceil(20%) within outcome x first/second strata",
            "hash_rule": "0015_target_validation_v1",
            "trajectory_count": len(validation),
        },
        "source_vocabulary": source_vocabulary,
        "core_trajectories": sorted(
            selected_rows, key=lambda row: (row["episode_id"], row["player_index"])
        ),
        "t4_candidate_trajectories": sorted(
            t4_candidates, key=lambda row: (row["episode_id"], row["player_index"])
        ),
        "audit": {
            "core_unique_trajectories": len(core_rows),
            "core_unique_episodes": len({row["episode_id"] for row in core_rows}),
            "selected_unique_trajectories": len(selected_rows),
            "selected_unique_episodes": len({row["episode_id"] for row in selected_rows}),
            "t4_candidate_trajectories": len(t4_candidates),
            "t4_candidate_episodes": len({row["episode_id"] for row in t4_candidates}),
            "t4_selected_trajectories": len(selected_t4),
            "t4_selected_episodes": len({row["episode_id"] for row in selected_t4}),
            "t4_cap_rule": "0015_t4_outcome_cap_v1" if include_t4 else None,
            "t4_cap": resolved_t4_cap if include_t4 else None,
            "counts": [
                {
                    "build": key[0],
                    "source_key": key[1],
                    "split": key[2],
                    "outcome": key[3],
                    "trajectories": value,
                }
                for key, value in sorted(counts.items())
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-t4", action="store_true")
    parser.add_argument("--base-index", type=Path)
    parser.add_argument("--t4-cap", type=int)
    args = parser.parse_args()
    result = build_index(
        args.catalog,
        args.overlay_root,
        args.official_archive,
        include_t4=args.include_t4,
        base_index_path=args.base_index,
        t4_cap=args.t4_cap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["audit"], ensure_ascii=False))


if __name__ == "__main__":
    main()
