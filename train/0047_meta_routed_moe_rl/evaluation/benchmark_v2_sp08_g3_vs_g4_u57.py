"""Benchmark V2 comparison for external SP08 Alakazam: G3 versus G4 U57."""

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

from ..assets import canonical_deck_sha256
from . import benchmark_v2_g3_vs_g4_u57_full67 as full
from . import render_g2_candidate_gate as base


DECK_ID = "SP08_MAGA"
DECK_DISPLAY_NAME = "SP08_MAGA - Alakazam / Dudunsparce / Shaymin"
DECK_PATH = full.ROOT / "docs/reports/sp-series/decks/SP08_MAGA/deck.csv"
EXPECTED_DECK_SHA256 = "7b466e7eeb3bc0625f3e780533c8c0f0fb1bc5f59ab328b41f6cdb41153b7b25"
OWN_ARCHETYPE_ID = 3
REPORT_ROOT = full.ARTIFACT_ROOT / "external" / DECK_ID
SLUG = "0044_g3_vs_g4_u57_sp08_alakazam_core16_policy0809_cuda2048_v2"
OUTPUT_ROOT = full.ROOT / "docs/evaluation/combat_mat/benchmark_v2" / SLUG
OUTPUT = OUTPUT_ROOT / "index.html"
MANIFEST = OUTPUT_ROOT / "manifest.json"
DETAIL = OUTPUT_ROOT / "reports" / f"{DECK_ID}.html"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _report(arm: str) -> Path:
    return REPORT_ROOT / arm / "report.json"


def preflight() -> dict[str, Any]:
    cards = tuple(map(int, DECK_PATH.read_text(encoding="utf-8").splitlines()))
    if len(cards) != 60 or canonical_deck_sha256(cards) != EXPECTED_DECK_SHA256:
        raise RuntimeError("SP08 exact-deck identity changed")
    full.preflight(require_unused=False)
    return {
        "schema_version": "0044_sp08_g3_vs_g4_u57_preflight_v1",
        "status": "PASS", "deck_id": DECK_ID,
        "exact_deck_sha256": EXPECTED_DECK_SHA256,
        "own_archetype_id": OWN_ARCHETYPE_ID,
        "own_archetype_assignment": "explicit_user_semantic_alakazam",
        "arms": list(full.ARMS), "games_per_arm": 2048,
    }


def _run_arm(arm: str) -> None:
    checkpoint, update = full.CHECKPOINTS[arm]
    output = REPORT_ROOT / arm
    if output.exists():
        if _report(arm).is_file():
            return
        raise FileExistsError(f"incomplete SP08 arm must be archived first: {output}")
    command = [
        sys.executable, "-m",
        "train.0047_meta_routed_moe_rl.evaluation.run_benchmark_v2",
        "--deck-id", DECK_ID, "--deck-path", str(DECK_PATH),
        "--deck-display-name", DECK_DISPLAY_NAME,
        "--own-archetype-id", str(OWN_ARCHETYPE_ID),
        "--output-root", str(output), "--checkpoint", str(checkpoint),
        "--checkpoint-update", str(update),
    ]
    subprocess.run(command, cwd=full.ROOT, check=True)


def _read_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    reports = tuple(json.loads(_report(arm).read_text(encoding="utf-8")) for arm in full.ARMS)
    full.validate_pair(reports[0], reports[1], deck_id=DECK_ID)
    for report in reports:
        if (
            report.get("focal_exact_deck_sha256") != EXPECTED_DECK_SHA256
            or report.get("focal_own_archetype_id") != OWN_ARCHETYPE_ID
            or report.get("focal_deck_registry_member") is not False
        ):
            raise RuntimeError("SP08 external-deck identity/mapping mismatch")
    return reports


def _relabel(page: str) -> str:
    replacements = {
        "U407 · G2 Candidate Gate": "SP08 Alakazam · Champion-G3 → G4 U57",
        "0044 U407 G2 Candidate Gate": "0044 SP08 · G3 vs G4 U57",
        "Champion-G1 vs U407 G2 Candidate · Frozen Policy-0809": "SP08 Alakazam · Champion-G3 vs G4 U57 · Policy-0809 Core-16",
        "G1 micro WR": "Champion-G3 WR", "G2 candidate micro WR": "G4 U57 WR",
        "Champion-G1": "Champion-G3", "U407 G2 candidate": "G4 U57 candidate",
        "U407 candidate": "G4 U57", "U407 source checkpoint": "G4 U57 source checkpoint",
        "U407 effective candidate": "G4 U57 effective candidate",
        "U407 portable checkpoint": "G4 U57 portable checkpoint",
        "U407 schedule": "G4 U57 schedule",
        "G1 effective candidate": "Champion-G3 effective candidate",
        "G1 portable checkpoint": "Champion-G3 portable checkpoint",
        "G1 schedule": "Champion-G3 schedule",
        "本报告是 G2 纳入体检，不会自动注册 Champion-G2。": "本报告是 SP08 外部 exact deck 诊断，不改变 001–067 资产池，也不会自动晋级 Champion-G4。",
        "两臂是同合同、同 exact deck、同完整 Policy-0809 对手池的独立 schedule，不作逐局 paired 推断。": "两臂使用同一 SP08 exact deck 与同一 2,048 个 common-random jobs，可进行 paired 变化定位。",
        "G1/U407 为各自 identity-bound 独立样本。": "Champion-G3/G4 U57 使用相同 common-random jobs。",
        "0044 G2 CANDIDATE · CUDA-2048": "0044 SP08 BENCHMARK V2 · CUDA-2048",
        "G1 vs U407": "Champion-G3 vs G4 U57",
        "001–010": "SP08 external focal", "Seeded 030–067": "No second cohort",
        "Independent-arm delta": "Common-seed delta",
        "independent CUDA-2048 arms": "common-seed CUDA-2048 arms",
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    return page


def publish() -> Path:
    g3, g4 = _read_pair()
    taxonomy = base._meta_taxonomy()
    delta = g4["summary"]["win_rate"] - g3["summary"]["win_rate"]
    transitions = Counter(
        (left["outcome"], right["outcome"])
        for left, right in zip(g3["entries"], g4["entries"], strict=True)
    )
    row = {
        "deck_id": DECK_ID, "cohort": "external_sp08_alakazam",
        "display_name": DECK_DISPLAY_NAME,
        "representative_cards": base._representative_cards(g3["focal_deck_cards"]),
        "exact_deck_sha256": EXPECTED_DECK_SHA256,
        "g1": g3["summary"], "g2_candidate": g4["summary"],
        "g1_by_opponent": base._by_opponent(g3), "g2_by_opponent": base._by_opponent(g4),
        "by_meta_archetype": base._meta_comparison(g3, g4, taxonomy),
        "g1_candidate_identity": g3["focal_policy_identity_audit"],
        "g2_candidate_identity": g4["focal_policy_identity_audit"],
        "g1_schedule_sha256": g3["schedule"]["schedule_sha256"],
        "g2_schedule_sha256": g4["schedule"]["schedule_sha256"],
        "common_random_schedule_sha256": g3["schedule"]["common_random_schedule_sha256"],
        "opponent_effective_sha256": g3["opponent_policy_identity_audit"]["effective_policy_sha256"],
        "delta": delta, "delta_95": base._difference_interval(g3["summary"]["wins"], g4["summary"]["wins"], 2048),
        "classification": "improved" if delta > 0.01 else "regressed" if delta < -0.01 else "flat",
        "paired_transitions": {
            "loss_to_win": transitions[(-1, 1)], "win_to_loss": transitions[(1, -1)],
            "win_to_win": transitions[(1, 1)], "loss_to_loss": transitions[(-1, -1)],
        },
    }
    summary = {
        "decks": 1, "games_per_arm": 2048,
        "g1_micro_win_rate": g3["summary"]["win_rate"],
        "g2_micro_win_rate": g4["summary"]["win_rate"],
        "g1_wilson_95": g3["summary"]["wilson_95"],
        "g2_wilson_95": g4["summary"]["wilson_95"],
        "micro_delta": delta,
        "micro_delta_95": row["delta_95"], "macro_delta": delta,
        "improved": int(row["classification"] == "improved"),
        "flat": int(row["classification"] == "flat"),
        "regressed": int(row["classification"] == "regressed"),
    }
    payload = {
        "schema_version": "0044_sp08_g3_vs_g4_u57_benchmark_v2_aggregate_v1",
        "status": "PASS", "version": full.VERSION,
        "published_at": datetime.now(UTC).isoformat(), "completed_decks": 1,
        "target_decks": 1, "flat_threshold": 0.01,
        "fixed_deck_ids": [DECK_ID], "random_deck_ids": [],
        "execution_order": [DECK_ID], "remainder_order_seed": 44_110_067,
        "reporting_meta_taxonomy": taxonomy,
        "summary": {"overall": summary, "fixed_001_010": summary, "seeded_random_030_067": base._summary([]) if hasattr(base, '_summary') else summary},
        "rows": [row], "best_improvement": row, "worst_regression": row,
        "identity": {
            "g3_source_checkpoint_sha256": full.G3_CHECKPOINT_SHA256,
            "g4_u57_source_checkpoint_sha256": full.G4_CHECKPOINT_SHA256,
            "opponent_policy_id": "Policy-0809", "benchmark_id": "Benchmark-V2",
            "games_per_arm_per_deck": 2048, "own_archetype_id": OWN_ARCHETYPE_ID,
            "own_archetype_assignment": "explicit_user_semantic_alakazam",
        },
    }
    page = _relabel(base.render(payload)).replace(
        "2 common-seed CUDA-2048 arms · 4,096 terminal games",
        "2 common-seed CUDA-2048 arms · 4,096 terminal games",
    )
    _atomic_text(OUTPUT, page)
    _atomic_text(DETAIL, _relabel(base.render_detail(row)))
    _atomic_text(MANIFEST, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2, sort_keys=True))
    if not args.launch:
        return 0
    for arm in full.ARMS:
        _run_arm(arm)
    print(publish())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
