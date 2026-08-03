from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PARTS = [
    ("01_days_0715_0719", "01_days_0715_0719.ipynb", "ptcg-0027-01-days-0715-0719", "PTCG 0027 01 Days 0715-0719", ["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19"]),
    ("02_days_0720_0724", "02_days_0720_0724.ipynb", "ptcg-0027-02-days-0720-0724", "PTCG 0027 02 Days 0720-0724", ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"]),
    ("03_days_0725_0728", "03_days_0725_0728.ipynb", "ptcg-0027-03-days-0725-0728", "PTCG 0027 03 Days 0725-0728", ["2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28"]),
    ("04_days_0729_0801", "04_days_0729_0801.ipynb", "ptcg-0027-04-days-0729-0801", "PTCG 0027 04 Days 0729-0801", ["2026-07-29", "2026-07-30", "2026-07-31", "2026-08-01"]),
]


CODE_TEMPLATE = r'''
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import json
import os
import re
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/{out_dir}")
OUT.mkdir(parents=True, exist_ok=True)

PART_NAME = {part_name!r}
TARGET_DATES = {dates!r}
TARGET_DECK_SHA256 = ""  # Empty keeps all strict positive-reward winners.
SCHEMA_VERSION = "0025_canonical_semantic_decision_v2"
PARTITION_SCHEMA = "0027_daily_semantic_dataset_v1"
RECORDS_PER_SHARD = int(os.environ.get("PTCG_RECORDS_PER_SHARD", "2048"))
MAX_FILES_PER_DATE = int(os.environ.get("PTCG_MAX_FILES_PER_DATE", "0") or "0")
MAX_RECORDS_PER_DATE = int(os.environ.get("PTCG_MAX_RECORDS_PER_DATE", "0") or "0")
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
ACTOR_KEYS = {{
    "global_cat", "global_num", "card_cat", "card_num", "card_parent",
    "resource_cat", "resource_num", "event_cat", "event_num",
    "option_cat", "option_num", "option_state", "option_source", "option_target",
    "option_skill_id", "option_skill_role", "option_skill_parent",
    "option_effect_id", "option_effect_role", "option_effect_parent",
    "min_count", "max_count",
}}
CAT_WIDTHS = {{"global_cat": 11, "card_cat": 5, "resource_cat": 4, "event_cat": 8, "option_cat": 14}}
NUM_WIDTHS = {{"global_num": 17, "card_num": 7, "resource_num": 15, "event_num": 4, "option_num": 16, "option_state": 16}}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_date(path: Path) -> date | None:
    for value in DATE_RE.findall(str(path)):
        try:
            return date.fromisoformat(value)
        except ValueError:
            continue
    return None


def input_files_for_day(day: str) -> list[Path]:
    wanted = date.fromisoformat(day)
    files = [
        path for path in INPUT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {{".json", ".gz", ".zip"}}
        and path_date(path) == wanted
    ]
    files = sorted(files)
    return files[:MAX_FILES_PER_DATE] if MAX_FILES_PER_DATE else files


def payloads(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as bundle:
            for member in bundle.namelist():
                if not member.lower().endswith(".json"):
                    continue
                try:
                    value = json.loads(bundle.read(member))
                except Exception:
                    continue
                if isinstance(value, dict):
                    yield f"{{path}}!{{member}}", value
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, dict):
                            yield f"{{path}}!{{member}}#{{index}}", item
        return

    try:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except Exception:
        return

    if isinstance(value, dict):
        yield str(path), value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                yield f"{{path}}#{{index}}", item


def first(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return default


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def episode_id(payload: Mapping[str, Any], fallback: str) -> str:
    info = payload.get("info")
    if isinstance(info, Mapping):
        value = first(info, "EpisodeId", "episode_id", "episodeId")
        if value is not None:
            return str(value)
    return str(first(payload, "episode_id", "episodeId", "id", default=fallback))


def winner(payload: Mapping[str, Any]) -> int | None:
    rewards = first(payload, "rewards", "reward", "scores", default=[])
    if not isinstance(rewards, list):
        return None
    indexes = [
        index for index, value in enumerate(rewards)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    ]
    return indexes[0] if len(indexes) == 1 else None


def deck_ids(cards: Any) -> list[int]:
    ids: list[int] = []
    for value in as_list(cards):
        if isinstance(value, Mapping):
            value = first(value, "id", "cardId")
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            return []
    return ids


def deck_hash(cards: Any) -> str | None:
    ids = deck_ids(cards)
    if len(ids) != 60:
        return None
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode("ascii")).hexdigest()


def deck_counts(cards: Any) -> list[list[int]]:
    ids = deck_ids(cards)
    if len(ids) != 60:
        return []
    counts = Counter(ids)
    return [[identity, counts[identity]] for identity in sorted(counts)]


def extract_decklists(payload: Mapping[str, Any]) -> dict[int, list[int]]:
    decks: dict[int, list[int]] = {{}}
    for step in as_list(payload.get("steps")):
        if not isinstance(step, list):
            continue
        for player_index, entry in enumerate(step):
            if player_index in decks or not isinstance(entry, Mapping):
                continue
            action = entry.get("action")
            if isinstance(action, list) and len(action) == 60 and all(isinstance(card, int) for card in action):
                decks[player_index] = [int(card) for card in action]
        if len(decks) >= 2:
            break
    return decks


def frames(payload: Mapping[str, Any]):
    traces: list[list[Any]] = []
    singletons: list[Any] = []
    for step in as_list(payload.get("steps")):
        if not isinstance(step, list):
            continue
        for entry in step:
            if not isinstance(entry, Mapping):
                continue
            raw = entry.get("visualize")
            if raw is None:
                raw = entry.get("visual")
            if isinstance(raw, list) and raw:
                traces.append(raw)
            elif isinstance(raw, Mapping):
                singletons.append(raw)
    raw_frames = max(traces, key=len) if traces else singletons
    for index, visual in enumerate(raw_frames):
        if not isinstance(visual, Mapping):
            continue
        observation = visual.get("obs")
        if not isinstance(observation, Mapping) or not isinstance(observation.get("select"), Mapping):
            continue
        current = observation.get("current") if isinstance(observation.get("current"), Mapping) else {{}}
        actor = as_int(current.get("yourIndex"), -1)
        if actor not in (0, 1):
            continue
        yield index, actor, observation, visual.get("selected")


def source_candidates(root: Path) -> list[Path]:
    return sorted({{
        path.parent.parent
        for path in root.rglob("official_public_prototypes_v1.json")
        if path.parent.name == "assets" and (path.parent.parent / "model" / "canonical").is_dir()
    }})


def find_source_root(input_root: Path) -> Path:
    candidates = source_candidates(input_root)
    if len(candidates) == 1:
        return candidates[0]
    extract_root = Path("/kaggle/working/ptcg_0027_0025_source")
    for archive in sorted(input_root.rglob("0025_semantic_foundation_pretraining.zip")):
        with zipfile.ZipFile(archive) as bundle:
            if any(name.endswith("assets/official_public_prototypes_v1.json") for name in bundle.namelist()):
                bundle.extractall(extract_root)
                candidates = source_candidates(extract_root)
                if len(candidates) == 1:
                    return candidates[0]
    raise FileNotFoundError("attach horizen12/ptcg-0027-0025-source or a 0025 source zip dataset")


source_root = find_source_root(INPUT)
sys.path.insert(0, str(source_root.parent))
package = source_root.name
PrototypeIndex = importlib.import_module(f"{{package}}.features.prototypes").PrototypeIndex
compile_canonical_row = importlib.import_module(f"{{package}}.features.canonical.compiler").compile_canonical_row
CausalKnowledge = importlib.import_module(f"{{package}}.knowledge.state").CausalKnowledge
prototypes = PrototypeIndex.load(source_root / "assets" / "official_public_prototypes_v1.json")


def validate_compiled(record: Mapping[str, Any]) -> None:
    actor = record["actor"]
    target = record["target"]
    if set(actor) != ACTOR_KEYS or any(key in actor for key in ("legacy", "action", "source_id", "source_team_name")):
        raise ValueError("actor leakage or key mismatch")
    for key, width in {{**CAT_WIDTHS, **NUM_WIDTHS}}.items():
        values = actor[key]
        if key in ("global_cat", "global_num"):
            if len(values) != width:
                raise ValueError(f"{{key}} width mismatch")
        elif values and len(values[0]) != width:
            raise ValueError(f"{{key}} width mismatch")
    option_count = len(actor["option_cat"])
    action = [int(value) for value in target["ordered_action"]]
    if not option_count or len(set(action)) != len(action) or any(value < 0 or value >= option_count for value in action):
        raise ValueError("illegal ordered action")
    if not int(actor["min_count"]) <= len(action) <= int(actor["max_count"]):
        raise ValueError("min/max contract violated")


class SplitShardWriter:
    def __init__(self, root: Path, split: str):
        self.root = root
        self.split = split
        self.index = 0
        self.records_in_open = 0
        self.open_path: Path | None = None
        self.handle = None
        self.shards: list[dict[str, Any]] = []

    def _open_next(self) -> None:
        self.close_open()
        self.open_path = self.root / self.split / f"shard_{{self.index:05d}}.jsonl.gz"
        self.open_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.open_path, "wt", encoding="utf-8")
        self.records_in_open = 0
        self.index += 1

    def write(self, record: Mapping[str, Any]) -> None:
        if self.handle is None or self.records_in_open >= RECORDS_PER_SHARD:
            self._open_next()
        assert self.handle is not None
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        self.records_in_open += 1

    def close_open(self) -> None:
        if self.handle is None:
            return
        assert self.open_path is not None
        self.handle.close()
        self.shards.append({{
            "path": str(self.open_path.relative_to(self.root)).replace("\\", "/"),
            "records": self.records_in_open,
            "bytes": self.open_path.stat().st_size,
            "sha256": sha256_file(self.open_path),
        }})
        self.handle = None
        self.open_path = None
        self.records_in_open = 0

    def close(self) -> list[dict[str, Any]]:
        self.close_open()
        return self.shards


def process_day(day: str) -> dict[str, Any]:
    day_root = OUT / f"date={{day}}"
    source_files = input_files_for_day(day)
    if not source_files:
        raise FileNotFoundError(f"no official replay json files found for {{day}}")

    writers = {{"train": SplitShardWriter(day_root, "train"), "validation": SplitShardWriter(day_root, "validation")}}
    episode_sets = {{"train": set(), "validation": set()}}
    episode_index: list[dict[str, Any]] = []
    maximum_lengths: dict[str, int] = {{}}
    stats = {{"date": day, "source_files": len(source_files), "episodes": 0, "records": 0, "skipped_non_winner_reward": 0, "skipped_missing_deck": 0, "skipped_deck_filter": 0, "skipped_invalid_action": 0}}

    for source in source_files:
        source_sha = sha256_file(source)
        for payload_name, payload in payloads(source):
            actor_index = winner(payload)
            if actor_index is None:
                stats["skipped_non_winner_reward"] += 1
                continue

            eid = episode_id(payload, payload_name)
            decks = extract_decklists(payload)
            deck = decks.get(actor_index, [])
            dhash = deck_hash(deck)
            if dhash is None:
                stats["skipped_missing_deck"] += 1
                continue
            if TARGET_DECK_SHA256 and dhash != TARGET_DECK_SHA256:
                stats["skipped_deck_filter"] += 1
                continue

            split = "validation" if int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
            knowledge = CausalKnowledge(actor_index, deck_ids(deck))
            decisions = 0
            for frame_index, player, observation, action in frames(payload):
                if MAX_RECORDS_PER_DATE and stats["records"] >= MAX_RECORDS_PER_DATE:
                    break
                if player != actor_index:
                    continue
                select = observation["select"]
                options = as_list(select.get("option"))
                ordered = [as_int(value, -1) for value in as_list(action)]
                minimum = as_int(select.get("minCount"))
                maximum = max(minimum, as_int(select.get("maxCount"), len(options)))
                if not options or not minimum <= len(ordered) <= maximum or len(set(ordered)) != len(ordered) or any(index < 0 or index >= len(options) for index in ordered):
                    stats["skipped_invalid_action"] += 1
                    continue

                raw = {{
                    "schema_version": "0019_universal_winner_decision_v1",
                    "identity": {{"date": day, "episode_id": eid, "player_index": actor_index, "episode_step": frame_index}},
                    "split": split,
                    "deck_manifest": {{"cards": deck_ids(deck), "counts": deck_counts(deck), "sha256": dhash}},
                    "actor_observation": observation,
                    "legal_options": options,
                    "ordered_action": ordered,
                    "action_termination": True,
                    "event_cursor": {{"visual_frame_index": frame_index, "actor_decision_index": decisions, "incoming_log_count": len(as_list(observation.get("logs")))}},
                    "terminal_outcome": "win",
                    "source_payload_sha256": source_sha,
                    "source_id": payload_name,
                    "source_team_name": str((payload.get("info") or {{}}).get("TeamNames", ["", ""])[actor_index]),
                }}
                compiled = compile_canonical_row(raw, knowledge.consume(raw["actor_observation"], raw["event_cursor"]), prototypes)
                for name in ("card_cat", "resource_cat", "event_cat", "option_cat", "option_skill_id", "option_effect_id"):
                    maximum_lengths[name] = max(maximum_lengths.get(name, 0), len(compiled["actor"][name]))
                record = {{"schema_version": SCHEMA_VERSION, "actor": compiled["actor"], "target": compiled["target"], "audit": {{"identity": raw["identity"], "split": split, "source_id": payload_name, "source_payload_sha256": source_sha}}}}
                validate_compiled(record)
                writers[split].write(record)
                stats["records"] += 1
                decisions += 1
                episode_sets[split].add(f"{{day}}::{{eid}}::{{actor_index}}")

            if decisions:
                stats["episodes"] += 1
                episode_index.append({{"episode_id": eid, "winner_player": actor_index, "date": day, "split": split, "decisions": decisions, "deck_sha256": dhash, "source": payload_name}})
            if MAX_RECORDS_PER_DATE and stats["records"] >= MAX_RECORDS_PER_DATE:
                break
        if MAX_RECORDS_PER_DATE and stats["records"] >= MAX_RECORDS_PER_DATE:
            break

    shards = {{split: writer.close() for split, writer in writers.items()}}
    split_counts = {{split: sum(item["records"] for item in items) for split, items in shards.items()}}
    if not split_counts["train"] or not split_counts["validation"]:
        raise RuntimeError(f"empty daily split for {{day}}: {{split_counts}}")
    manifest = {{
        "schema_version": SCHEMA_VERSION,
        "partition_schema": PARTITION_SCHEMA,
        "status": "complete",
        "experiment": "0027_semantic_foundation_pretraining",
        "part": PART_NAME,
        "date": day,
        "winner_only": True,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "source_datasets": [f"kaggle/pokemon-tcg-ai-battle-episodes-{{day}}"],
        "records_per_shard": RECORDS_PER_SHARD,
        "split_counts": split_counts,
        "episode_split_counts": {{key: len(value) for key, value in episode_sets.items()}},
        "episodes": stats["episodes"],
        "records": stats["records"],
        "maximum_lengths": maximum_lengths,
        "shards": shards,
        "actor_forward_excludes": ["legacy", "ordered_action", "source_id", "source_team_name", "source_payload_sha256"],
        "source_identity_actor_visible": False,
        "initialized_from_0025": {{"checkpoint": "best_greedy_exact.pt", "sha256": "adc4eaeca1e62a28bbd762e8e513212c94941044bbc85673f02bbeadc1865aa3", "bytes": 129028675}},
        "stats": stats,
    }}
    write_csv(day_root / "winner_episode_index.csv", episode_index, ["episode_id", "winner_player", "date", "split", "decisions", "deck_sha256", "source"])
    write_json(day_root / "manifest.json", manifest)
    write_json(day_root / "dataset_reference.json", {{"manifest_sha256": sha256_file(day_root / "manifest.json"), "manifest": manifest}})
    return manifest


day_manifests = []
for day in TARGET_DATES:
    manifest = process_day(day)
    row = {{"date": day, "records": manifest["records"], "episodes": manifest["episodes"], "split_counts": manifest["split_counts"], "manifest_path": f"date={{day}}/manifest.json"}}
    day_manifests.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

write_json(OUT / "partition_manifest.json", {{"schema_version": "0027_daily_partition_group_v1", "status": "complete", "part": PART_NAME, "dates": TARGET_DATES, "winner_only": True, "target_deck_sha256": TARGET_DECK_SHA256, "day_manifests": day_manifests}})
print(json.dumps({{"part": PART_NAME, "dates": TARGET_DATES, "output": str(OUT), "day_manifests": day_manifests}}, indent=2, ensure_ascii=False))
'''


def notebook(title: str, description: str, code: str) -> dict:
    return {
        "cells": [
            {"id": "overview", "cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\n", f"{description}\n"]},
            {"id": "build-daily-shards", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.rstrip("\n").splitlines()]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for directory, code_file, slug, title, dates in PARTS:
        target = ROOT / directory
        target.mkdir(parents=True, exist_ok=True)
        code = CODE_TEMPLATE.format(out_dir=f"ptcg_0027_{directory}", part_name=directory, dates=dates)
        ast.parse(code)
        (target / code_file).write_text(
            json.dumps(notebook(title, "Independent day-partition producer for 0027 winner-perspective canonical feature shards.", code), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "id": f"horizen12/{slug}",
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "machine_shape": "None",
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["pokemon-tcg", "0027", "semantic-foundation"],
            "dataset_sources": [*(f"kaggle/pokemon-tcg-ai-battle-episodes-{day}" for day in dates), "horizen12/ptcg-0027-0025-source"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
