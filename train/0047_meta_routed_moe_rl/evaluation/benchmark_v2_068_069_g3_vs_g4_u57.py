"""Benchmark V2 comparison for registered Alakazam decks 068/069."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..assets import AssetRegistry
from ..own_archetype import OwnArchetypeVocabulary
from . import benchmark_v2_g3_vs_g4_u57_full67 as full
from . import render_benchmark_v2_g3_vs_g4_u57_full67 as comparison
from . import render_g2_candidate_gate as base


DECK_IDS = ("068", "069")
OWN_ARCHETYPE_ID = 3
REPORT_ROOT = full.ARTIFACT_ROOT / "supplemental_alakazam_068_069"
SLUG = "0044_g3_vs_g4_u57_068_069_alakazam_core16_policy0809_cuda2048_v2"
OUTPUT_ROOT = full.ROOT / "docs/evaluation/combat_mat/benchmark_v2" / SLUG
OUTPUT = OUTPUT_ROOT / "index.html"
MANIFEST = OUTPUT_ROOT / "manifest.json"
DETAIL_ROOT = OUTPUT_ROOT / "reports"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _report(deck_id: str, arm: str) -> Path:
    return REPORT_ROOT / deck_id / arm / "report.json"


def preflight() -> dict[str, Any]:
    full.preflight(require_unused=False)
    registry = AssetRegistry.load(full.PROJECT_ROOT)
    registry.validate_all()
    assets = {row.deck_id: row for row in registry.decks}
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=full.PROJECT_ROOT
    )
    mapping = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    cards = {
        deck_id: tuple(map(int, (full.PROJECT_ROOT / assets[deck_id].deck_path).read_text().splitlines()))
        for deck_id in DECK_IDS
    }
    from collections import Counter as CardCounter
    delta_up = CardCounter(cards["069"]) - CardCounter(cards["068"])
    delta_down = CardCounter(cards["068"]) - CardCounter(cards["069"])
    if (
        any(assets[deck_id].roles != ("evaluation",) for deck_id in DECK_IDS)
        or any(mapping[deck_id] != OWN_ARCHETYPE_ID for deck_id in DECK_IDS)
        or delta_up != CardCounter({1182: 1})
        or delta_down != CardCounter({1097: 1})
    ):
        raise RuntimeError("068/069 asset, parent-delta, or own-archetype contract changed")
    return {
        "schema_version": "0044_068_069_g3_vs_g4_u57_preflight_v1",
        "status": "PASS", "deck_ids": list(DECK_IDS),
        "own_archetype_id": OWN_ARCHETYPE_ID,
        "parent_delta": {"1097_night_stretcher": -1, "1182_bosss_orders": 1},
        "games_per_arm_per_deck": 2048, "arms": list(full.ARMS),
        "source_checkpoint_sha256": {
            "g3": full.G3_CHECKPOINT_SHA256, "g4_u57": full.G4_CHECKPOINT_SHA256,
        },
    }


def _run_arm(deck_id: str, arm: str) -> None:
    report = _report(deck_id, arm)
    if report.is_file():
        return
    output = report.parent
    if output.exists():
        raise FileExistsError(f"incomplete arm must be archived before restart: {output}")
    checkpoint, update = full.CHECKPOINTS[arm]
    subprocess.run([
        sys.executable, "-m",
        "train.0047_meta_routed_moe_rl.evaluation.run_benchmark_v2",
        "--deck-id", deck_id, "--output-root", str(output),
        "--checkpoint", str(checkpoint), "--checkpoint-update", str(update),
    ], cwd=full.ROOT, check=True)


def _read_pair(deck_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reports = tuple(json.loads(_report(deck_id, arm).read_text(encoding="utf-8")) for arm in full.ARMS)
    full.validate_pair(reports[0], reports[1], deck_id=deck_id)
    for report in reports:
        if (
            report.get("focal_own_archetype_id") != OWN_ARCHETYPE_ID
            or report.get("focal_deck_registry_member") is not True
        ):
            raise RuntimeError(f"{deck_id}: registered Alakazam routing evidence mismatch")
    return reports


def _row(deck_id: str, names: dict[str, str], taxonomy: dict[str, Any]) -> dict[str, Any]:
    g3, g4 = _read_pair(deck_id)
    delta = g4["summary"]["win_rate"] - g3["summary"]["win_rate"]
    transitions = Counter(
        (left["outcome"], right["outcome"])
        for left, right in zip(g3["entries"], g4["entries"], strict=True)
    )
    return {
        "deck_id": deck_id, "cohort": "new_alakazam_068_069",
        "display_name": names[deck_id],
        "representative_cards": base._representative_cards(g3["focal_deck_cards"]),
        "exact_deck_sha256": g3["focal_exact_deck_sha256"],
        "g1": g3["summary"], "g2_candidate": g4["summary"],
        "g1_by_opponent": base._by_opponent(g3), "g2_by_opponent": base._by_opponent(g4),
        "by_meta_archetype": base._meta_comparison(g3, g4, taxonomy),
        "g1_candidate_identity": g3["focal_policy_identity_audit"],
        "g2_candidate_identity": g4["focal_policy_identity_audit"],
        "g1_schedule_sha256": g3["schedule"]["schedule_sha256"],
        "g2_schedule_sha256": g4["schedule"]["schedule_sha256"],
        "common_random_schedule_sha256": g3["schedule"]["common_random_schedule_sha256"],
        "opponent_effective_sha256": g3["opponent_policy_identity_audit"]["effective_policy_sha256"],
        "delta": delta,
        "delta_95": base._difference_interval(g3["summary"]["wins"], g4["summary"]["wins"], 2048),
        "classification": "improved" if delta > 0.01 else "regressed" if delta < -0.01 else "flat",
        "paired_transitions": {
            "loss_to_win": transitions[(-1, 1)], "win_to_loss": transitions[(1, -1)],
            "win_to_win": transitions[(1, 1)], "loss_to_loss": transitions[(-1, -1)],
        },
    }


def _relabel(page: str) -> str:
    replacements = {
        "U407 · G2 Candidate Gate": "Alakazam 068/069 · Champion-G3 → G4 U57",
        "0044 U407 G2 Candidate Gate": "0044 068/069 · G3 vs G4 U57",
        "Champion-G1 vs U407 G2 Candidate · Frozen Policy-0809": "Deck068/069 Alakazam · Champion-G3 vs G4 U57 · Policy-0809 Core-16",
        "G1 micro WR": "Champion-G3 micro WR", "G2 candidate micro WR": "G4 U57 micro WR",
        "Champion-G1": "Champion-G3", "U407 G2 candidate": "G4 U57 candidate",
        "U407 candidate": "G4 U57", "U407 source checkpoint": "G4 U57 source checkpoint",
        "U407 effective candidate": "G4 U57 effective candidate",
        "U407 portable checkpoint": "G4 U57 portable checkpoint", "U407 schedule": "G4 U57 schedule",
        "G1 effective candidate": "Champion-G3 effective candidate",
        "G1 portable checkpoint": "Champion-G3 portable checkpoint", "G1 schedule": "Champion-G3 schedule",
        "本报告是 G2 纳入体检，不会自动注册 Champion-G2。": "本报告记录胡地 068/069 扩池前完成的同合同评测；两套 deck 现已进入 live training/opponent pool，但本报告不会自动晋级 Champion-G4。",
        "两臂是同合同、同 exact deck、同完整 Policy-0809 对手池的独立 schedule，不作逐局 paired 推断。": "每个 deck 的两臂使用同一 2,048 个 common-random jobs，可进行 paired 变化定位。",
        "G1/U407 为各自 identity-bound 独立样本。": "Champion-G3/G4 U57 使用相同 common-random jobs。",
        "0044 G2 CANDIDATE · CUDA-2048": "0044 068/069 BENCHMARK V2 · CUDA-2048",
        "G1 vs U407": "Champion-G3 vs G4 U57", "001–010": "Deck068/069",
        "Seeded 030–067": "No second cohort", "Independent-arm delta": "Common-seed delta",
        "independent CUDA-2048 arms": "common-seed CUDA-2048 arms",
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    return page


def publish(*, allow_partial: bool) -> Path:
    completed = [
        deck_id for deck_id in DECK_IDS
        if all(_report(deck_id, arm).is_file() for arm in full.ARMS)
    ]
    if not completed or (not allow_partial and len(completed) != len(DECK_IDS)):
        raise RuntimeError("068/069 paired reports are incomplete")
    registry = AssetRegistry.load(full.PROJECT_ROOT)
    names = {row.deck_id: row.name for row in registry.decks}
    taxonomy = base._meta_taxonomy()
    rows = [_row(deck_id, names, taxonomy) for deck_id in completed]
    overall = comparison._summary(rows)
    empty = comparison._summary([])
    payload = {
        "schema_version": "0044_068_069_g3_vs_g4_u57_benchmark_v2_aggregate_v1",
        "status": "PASS" if len(rows) == 2 else "PARTIAL_EVALUATION_IN_PROGRESS",
        "version": full.VERSION, "published_at": datetime.now(UTC).isoformat(),
        "completed_decks": len(rows), "target_decks": 2, "flat_threshold": 0.01,
        "fixed_deck_ids": list(DECK_IDS), "random_deck_ids": [],
        "execution_order": list(DECK_IDS), "remainder_order_seed": 44_110_067,
        "reporting_meta_taxonomy": taxonomy,
        "summary": {"overall": overall, "fixed_001_010": overall, "seeded_random_030_067": empty},
        "rows": rows, "best_improvement": max(rows, key=lambda row: row["delta"]),
        "worst_regression": min(rows, key=lambda row: row["delta"]),
        "identity": {
            "g3_source_checkpoint_sha256": full.G3_CHECKPOINT_SHA256,
            "g4_u57_source_checkpoint_sha256": full.G4_CHECKPOINT_SHA256,
            "opponent_policy_id": "Policy-0809", "benchmark_id": "Benchmark-V2",
            "games_per_arm_per_deck": 2048, "own_archetype_id": OWN_ARCHETYPE_ID,
        },
    }
    page = _relabel(base.render(payload))
    _atomic_text(OUTPUT, page)
    _atomic_text(MANIFEST, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    for row in rows:
        _atomic_text(DETAIL_ROOT / f"{row['deck_id']}.html", _relabel(base.render_detail(row)))
    return OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2, sort_keys=True))
    if not args.launch:
        return 0
    for deck_id in DECK_IDS:
        for arm in full.ARMS:
            _run_arm(deck_id, arm)
        publish(allow_partial=True)
    print(publish(allow_partial=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
