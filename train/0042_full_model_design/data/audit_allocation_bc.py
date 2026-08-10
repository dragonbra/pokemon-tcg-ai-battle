"""Produce the immutable coverage report for the Zero-Shot allocation corpus."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from ..league import load_frozen_catalog

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "rl_runs/0042_full_model_design/datasets/allocation_zero_shot_v1/allocation_samples.pt"
MANIFEST = DATASET.with_name("manifest.json")
OUTPUT = ROOT / "experiments/0042_full_model_design/ALLOCATION_BC_DATA_REPORT.json"


def main() -> None:
    source = json.loads(MANIFEST.read_text())
    payload = torch.load(DATASET, map_location="cpu", weights_only=True)
    samples = payload["samples"]
    archetypes = {item.deck_id: item.display_name for item in load_frozen_catalog()}
    label_rows = [{
        "battle_id": item["battle_id"], "seed": item["seed"],
        "opponent_id": item["opponent_id"], "turn": item["turn"],
        "target_ids": item["target_ids"], "counters": item["counters"],
        "split": item["split"],
    } for item in samples]
    label_bytes = json.dumps(label_rows, sort_keys=True, separators=(",", ":")).encode()
    train_allocations = {
        (len(item["counters"]), tuple(item["counters"]))
        for item in samples if item["split"] == "train"
    }
    validation = [item for item in samples if item["split"] == "validation"]
    unseen = sum(
        (len(item["counters"]), tuple(item["counters"])) not in train_allocations
        for item in validation
    )
    report = {
        "schema_version": "0038_allocation_bc_data_audit_v1",
        "teacher_policy_sha256": source["teacher"]["checkpoint_hashes"]["policy_sha256"],
        "teacher_value_sha256": source["teacher"]["checkpoint_hashes"]["value_sha256"],
        "teacher_contract": "Zero-Shot pre-RL legacy sequential; behavior collection only",
        "v5_loaded": False, "ppo_old_logprob_present": False,
        "dataset_sha256": source["dataset_sha256"],
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "official_games": source["valid_games"],
        "unique_battles_with_labels": source["unique_battles"],
        "unique_engine_seeds": source["unique_engine_seeds"],
        "macro_samples": len(samples), "battle_split": source["split"],
        "target_count_n": source["target_count_n"],
        "focal_seat": {"first": source["focal_first"], "second": source["focal_second"]},
        "opponent_archetype": Counter(archetypes.get(item["opponent_id"], "unknown")
                                      for item in samples),
        "immediate_ko_samples": source["immediate_ko_samples"],
        "immediate_prize_samples": source["immediate_prize_samples"],
        "game_stage": source["game_stage"],
        "unseen_validation_allocations": unseen,
        "unseen_validation_allocation_rate": unseen / len(validation),
        "invalid_or_unsupported_chains": source["invalid_or_stale_chains"],
        "known_invalid_reasons": source["invalid_chain_reasons"],
        "note": "Legacy 78 replay labels remain pipeline-only evidence and are not included.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=dict) + "\n")
    print(json.dumps({key: report[key] for key in (
        "dataset_sha256", "label_sha256", "official_games", "macro_samples", "target_count_n"
    )}, indent=2))


if __name__ == "__main__":
    main()
