"""Incrementally publish Champion-G3 versus G4 U57 Full-67 Benchmark V2."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from ..assets import AssetRegistry
from . import benchmark_v2_g3_vs_g4_u57_full67 as full
from . import render_g2_candidate_gate as base


SLUG = "0044_g3_vs_g4_u57_full67_core16_policy0809_cuda2048_v2"
OUTPUT_ROOT = full.ROOT / "docs/evaluation/combat_mat/benchmark_v2" / SLUG
OUTPUT = OUTPUT_ROOT / "index.html"
MANIFEST = OUTPUT_ROOT / "manifest.json"
DETAIL_ROOT = OUTPUT_ROOT / "reports"
REVERSE_LINK = full.ARTIFACT_ROOT / "evaluation.json"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "decks": 0, "games_per_arm": 0, "g1_micro_win_rate": 0.0,
            "g2_micro_win_rate": 0.0, "g1_wilson_95": [0.0, 0.0],
            "g2_wilson_95": [0.0, 0.0], "micro_delta": 0.0,
            "micro_delta_95": [0.0, 0.0], "macro_delta": 0.0,
            "improved": 0, "flat": 0, "regressed": 0,
        }
    games = sum(row["g1"]["games"] for row in rows)
    g3_wins = sum(row["g1"]["wins"] for row in rows)
    g4_wins = sum(row["g2_candidate"]["wins"] for row in rows)
    return {
        "decks": len(rows), "games_per_arm": games,
        "g1_micro_win_rate": g3_wins / games,
        "g2_micro_win_rate": g4_wins / games,
        "g1_wilson_95": base._wilson(g3_wins, games),
        "g2_wilson_95": base._wilson(g4_wins, games),
        "micro_delta": (g4_wins - g3_wins) / games,
        "micro_delta_95": base._difference_interval(g3_wins, g4_wins, games),
        "macro_delta": sum(row["delta"] for row in rows) / len(rows),
        "improved": sum(row["classification"] == "improved" for row in rows),
        "flat": sum(row["classification"] == "flat" for row in rows),
        "regressed": sum(row["classification"] == "regressed" for row in rows),
    }


def aggregate(*, allow_partial: bool) -> dict[str, Any]:
    state = json.loads(full.STATE_PATH.read_text(encoding="utf-8"))
    completed = list(state.get("completed_deck_ids", []))
    if not allow_partial and state.get("status") != "EVALUATION_COMPLETE":
        raise RuntimeError("Full-67 paired Benchmark V2 is incomplete")
    registry = AssetRegistry.load(full.PROJECT_ROOT)
    names = {deck.deck_id: deck.name for deck in registry.decks}
    taxonomy = base._meta_taxonomy()
    rows = []
    for deck_id in full.EXECUTION_DECK_IDS:
        if deck_id not in completed:
            continue
        g3, g4 = full._read_pair(deck_id)
        delta = g4["summary"]["win_rate"] - g3["summary"]["win_rate"]
        transitions = Counter(
            (left["outcome"], right["outcome"])
            for left, right in zip(g3["entries"], g4["entries"], strict=True)
        )
        rows.append({
            "deck_id": deck_id,
            "cohort": "reused_g3_baseline" if deck_id in full.REUSABLE_G3_DECK_IDS else "new_paired_baseline",
            "display_name": names[deck_id],
            "representative_cards": base._representative_cards(g3["focal_deck_cards"]),
            "exact_deck_sha256": g3["focal_exact_deck_sha256"],
            "g1": g3["summary"], "g2_candidate": g4["summary"],
            "g1_by_opponent": base._by_opponent(g3),
            "g2_by_opponent": base._by_opponent(g4),
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
        })
    reused = [row for row in rows if row["deck_id"] in full.REUSABLE_G3_DECK_IDS]
    new = [row for row in rows if row["deck_id"] not in full.REUSABLE_G3_DECK_IDS]
    return {
        "schema_version": "0044_g3_vs_g4_u57_full67_benchmark_v2_aggregate_v1",
        "status": "PASS" if len(rows) == 67 else "PARTIAL_EVALUATION_IN_PROGRESS",
        "version": full.VERSION, "published_at": datetime.now(UTC).isoformat(),
        "completed_decks": len(rows), "target_decks": 67, "flat_threshold": 0.01,
        "fixed_deck_ids": list(full.REUSABLE_G3_DECK_IDS),
        "random_deck_ids": list(full.EXECUTION_DECK_IDS[len(full.REUSABLE_G3_DECK_IDS):]),
        "execution_order": list(full.EXECUTION_DECK_IDS),
        "remainder_order_seed": 44_110_067,
        "reporting_meta_taxonomy": taxonomy,
        "summary": {
            "overall": _summary(rows),
            "fixed_001_010": _summary(reused),
            "seeded_random_030_067": _summary(new),
        },
        "rows": rows,
        "best_improvement": max(rows, key=lambda row: row["delta"]) if rows else None,
        "worst_regression": min(rows, key=lambda row: row["delta"]) if rows else None,
        "identity": {
            "g3_source_checkpoint_sha256": full.G3_CHECKPOINT_SHA256,
            "g4_u57_source_checkpoint_sha256": full.G4_CHECKPOINT_SHA256,
            "opponent_policy_id": "Policy-0809", "benchmark_id": "Benchmark-V2",
            "games_per_arm_per_deck": 2048,
        },
    }


def _relabel(page: str) -> str:
    replacements = {
        "U407 · G2 Candidate Gate": "Champion-G3 → G4 U57 · Full-67 Benchmark V2",
        "0044 U407 G2 Candidate Gate": "0044 G3 vs G4 U57 · Benchmark V2",
        "Champion-G1 vs U407 G2 Candidate · Frozen Policy-0809": "Champion-G3 vs G4 U57 · complete Policy-0809 · Core-16 common random numbers",
        "G1 micro WR": "Champion-G3 micro WR",
        "G2 candidate micro WR": "G4 U57 micro WR",
        "Champion-G1": "Champion-G3",
        "U407 G2 candidate": "G4 U57 candidate",
        "U407 candidate": "G4 U57",
        "U407 source checkpoint": "G4 U57 source checkpoint",
        "U407 effective candidate": "G4 U57 effective candidate",
        "U407 portable checkpoint": "G4 U57 portable checkpoint",
        "U407 schedule": "G4 U57 schedule",
        "G1 effective candidate": "Champion-G3 effective candidate",
        "G1 portable checkpoint": "Champion-G3 portable checkpoint",
        "G1 schedule": "Champion-G3 schedule",
        "本报告是 G2 纳入体检，不会自动注册 Champion-G2。": "本报告比较冻结 Champion-G3 与 G4 U57 candidate，不会自动晋级 Champion-G4。",
        "两臂是同合同、同 exact deck、同完整 Policy-0809 对手池的独立 schedule，不作逐局 paired 推断。": "两臂使用同一 exact deck、同一 2,048 个 opponent/engine/Search/policy/coin jobs，可进行 paired 变化定位。",
        "G1/U407 为各自 identity-bound 独立样本。": "Champion-G3/G4 U57 使用相同 common-random jobs。",
        "两份 identity-bound schedule 在各 exact opponent 的样本数可能不同": "两臂在各 exact opponent 上使用相同样本数与 seeds",
        "0044 G2 CANDIDATE · CUDA-2048": "0044 BENCHMARK V2 · CUDA-2048",
        "G1 vs U407": "Champion-G3 vs G4 U57",
        "001–010": "Existing G3 baseline cohort",
        "Seeded 030–067": "Remaining Full-67",
        "Independent-arm delta": "Common-seed delta",
        "independent CUDA-2048 arms": "common-seed CUDA-2048 arms",
        "430044407": "44110067",
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    return page


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _publish_directory_index() -> None:
    root = OUTPUT_ROOT.parent
    cards = []
    for manifest in sorted(root.glob("*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        schema = payload.get("schema_version", "")
        if schema == "0044_g3_vs_g4_u57_full67_benchmark_v2_aggregate_v1":
            summary = payload["summary"]["overall"]
            label = "Champion-G3 vs G4 U57 · Full-67"
            result = f"{payload['completed_decks']}/67 decks · {summary['micro_delta']*100:+.2f} pp"
        elif schema == "0044_g2_vs_g3_u110_full67_benchmark_v2_aggregate_v1":
            summary = payload["summary"]["overall"]
            label = "Champion-G2 vs G3 U110 · Full-67"
            result = f"{payload['completed_decks']}/67 decks · {summary['micro_delta']*100:+.2f} pp"
        else:
            label = "G2 initialization vs U30 · Deck 007"
            result = f"{payload.get('delta', 0)*100:+.2f} pp"
        cards.append(f'<tr><td><a href="{manifest.parent.name}/index.html">{label}</a></td><td>{result}</td><td><code>{schema}</code></td></tr>')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0044 Benchmark V2</title><style>body{{font:15px/1.6 system-ui;max-width:1100px;margin:auto;padding:36px;background:#edf3f0;color:#172720}}section{{padding:24px;background:#fff;border:1px solid #d4e0da}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #d4e0da;text-align:left}}a{{color:#087f4f;font-weight:700}}code{{overflow-wrap:anywhere}}</style></head><body><h1>0044 · Benchmark V2</h1><p>Core-16 Meta-balanced · complete Policy-0809 · fixed common random numbers · CUDA-2048.</p><section><table><thead><tr><th>Report</th><th>Progress / result</th><th>Schema</th></tr></thead><tbody>{''.join(cards)}</tbody></table></section></body></html>'''
    _atomic_text(root / "index.html", page)


def publish(*, allow_partial: bool) -> Path:
    payload = aggregate(allow_partial=allow_partial)
    if not payload["rows"]:
        raise RuntimeError("no completed paired deck is available")
    page = _relabel(base.render(payload)).replace(
        "40 common-seed CUDA-2048 arms · 81,920 terminal games",
        f"{payload['completed_decks'] * 2} common-seed CUDA-2048 arms · "
        f"{payload['completed_decks'] * 2 * 2048:,} terminal games",
    )
    _atomic_text(OUTPUT, page)
    _atomic_text(MANIFEST, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    for row in payload["rows"]:
        _atomic_text(DETAIL_ROOT / f"{row['deck_id']}.html", _relabel(base.render_detail(row)))
    _atomic_text(REVERSE_LINK, json.dumps({
        "schema_version": "0044_evaluation_reverse_link_v1",
        "version": full.VERSION, "status": payload["status"],
        "authoritative_report": str(OUTPUT.relative_to(full.ROOT)),
    }, indent=2, sort_keys=True) + "\n")
    _publish_directory_index()
    return OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(publish(allow_partial=args.allow_partial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
