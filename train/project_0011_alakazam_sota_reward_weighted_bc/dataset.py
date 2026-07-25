from __future__ import annotations

import gzip
import hashlib
import json
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO

from train.project_0010_alakazam_sota_model.dataset import SPLITS, dataset_paths as base_dataset_paths


DATASET_SCHEMA = "alakazam_sota_reward_weighted_dataset_v1"
REWARD_SCHEMA = "alakazam_offline_reward_metrics_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {split: base / f"{split}.jsonl.gz" for split in SPLITS}


def reward_sidecar_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {split: base / f"reward_sidecar.{split}.jsonl" for split in SPLITS}


def _identity(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["episode_id"]),
        int(record["player_index"]),
        int(record["episode_step"]),
    )


def _iter_json(handle: TextIO):
    for line_number, line in enumerate(handle, 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield value


def build_reward_dataset(
    base_root: Path,
    reward_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    base_paths = base_dataset_paths(base_root)
    sidecar_paths = reward_sidecar_paths(reward_root)
    outputs = dataset_paths(output_root)
    audit_path = output_root / "dataset_audit.json"
    occupied = [path for path in (*outputs.values(), audit_path) if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite 0011 dataset outputs: "
            + ", ".join(map(str, occupied))
        )
    output_root.mkdir(parents=True, exist_ok=True)
    records_by_split = {split: 0 for split in SPLITS}
    episodes_by_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    nonzero_counts: dict[str, int] = {}

    for split in SPLITS:
        if not base_paths[split].is_file() or not sidecar_paths[split].is_file():
            raise FileNotFoundError(
                f"missing paired 0010/reward input for split {split}: "
                f"{base_paths[split]}, {sidecar_paths[split]}"
            )
        with ExitStack() as stack:
            base_handle = stack.enter_context(
                gzip.open(base_paths[split], "rt", encoding="utf-8")
            )
            reward_handle = stack.enter_context(
                sidecar_paths[split].open("r", encoding="utf-8")
            )
            output_handle = stack.enter_context(
                gzip.open(outputs[split], "wt", encoding="utf-8", compresslevel=6)
            )
            pairs = zip_longest(
                _iter_json(base_handle),
                _iter_json(reward_handle),
                fillvalue=None,
            )
            for row_index, (base_row, reward_row) in enumerate(pairs, 1):
                if base_row is None or reward_row is None:
                    raise ValueError(
                        f"base/reward row count mismatch in {split} at row {row_index}"
                    )
                if _identity(base_row) != _identity(reward_row):
                    raise ValueError(
                        f"base/reward identity mismatch in {split} row {row_index}: "
                        f"{_identity(base_row)} != {_identity(reward_row)}"
                    )
                if reward_row.get("reward_schema_version") != REWARD_SCHEMA:
                    raise ValueError(
                        f"unexpected reward schema in {split} row {row_index}: "
                        f"{reward_row.get('reward_schema_version')}"
                    )
                metrics = reward_row.get("reward_metrics")
                if not isinstance(metrics, dict):
                    raise ValueError(f"missing reward metrics in {split} row {row_index}")
                decision = metrics.get("decision") or {}
                for name, value in decision.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        nonzero_counts[name] = nonzero_counts.get(name, 0) + int(value != 0)
                merged = {
                    **base_row,
                    "dataset_schema_version": DATASET_SCHEMA,
                    "reward_schema_version": REWARD_SCHEMA,
                    "reward_metrics": metrics,
                    # The frozen corpus contains only trajectories won by Yushin Ito.
                    # This is intentionally constant and therefore cancels under a
                    # dataset-mean reward baseline.
                    "terminal_outcome": 1.0,
                }
                output_handle.write(
                    json.dumps(merged, ensure_ascii=True, separators=(",", ":")) + "\n"
                )
                records_by_split[split] += 1
                episodes_by_split[split].add(int(base_row["episode_id"]))

    base_audit = base_root / "dataset_audit.json"
    reward_audit = reward_root / "reward_sidecar.jsonl.reward_audit.json"
    audit = {
        "schema_version": DATASET_SCHEMA,
        "reward_schema_version": REWARD_SCHEMA,
        "single_expert_policy": {
            "team_name": "Yushin Ito",
            "selection": "daily winner-only",
        },
        "base_dataset_root": str(base_root.resolve()),
        "base_dataset_audit": str(base_audit.resolve()),
        "base_dataset_audit_sha256": _sha256(base_audit),
        "reward_sidecar_root": str(reward_root.resolve()),
        "reward_sidecar_audit": str(reward_audit.resolve()),
        "reward_sidecar_audit_sha256": _sha256(reward_audit),
        "records_by_split": records_by_split,
        "episodes_by_split": {
            split: len(episodes) for split, episodes in episodes_by_split.items()
        },
        "decision_metric_nonzero_counts": nonzero_counts,
        "terminal_outcome_constant": 1.0,
        "outputs": {
            split: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for split, path in outputs.items()
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def load_dataset_audit(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "dataset_audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("invalid 0011 reward-weighted dataset audit")
    policy = value.get("single_expert_policy") or {}
    if policy.get("team_name") != "Yushin Ito":
        raise ValueError("0011 dataset is not the frozen single-expert policy")
    return value
