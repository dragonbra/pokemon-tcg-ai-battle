"""Incrementally publish Champion-G2 versus G3 U110 Full-67 Benchmark V2."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from ..assets import AssetRegistry
from . import benchmark_v2_g2_vs_g3_u110_full67 as full
from . import render_g2_candidate_gate as base


SLUG = "0044_g2_vs_g3_u110_full67_core16_policy0809_cuda2048_v2"
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
    g2_wins = sum(row["g1"]["wins"] for row in rows)
    g3_wins = sum(row["g2_candidate"]["wins"] for row in rows)
    return {
        "decks": len(rows), "games_per_arm": games,
        "g1_micro_win_rate": g2_wins / games,
        "g2_micro_win_rate": g3_wins / games,
        "g1_wilson_95": base._wilson(g2_wins, games),
        "g2_wilson_95": base._wilson(g3_wins, games),
        "micro_delta": (g3_wins - g2_wins) / games,
        "micro_delta_95": base._difference_interval(g2_wins, g3_wins, games),
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
    for deck_id in sorted(completed):
        reports = {
            arm: json.loads((full.REPORT_ROOT / deck_id / arm / "report.json").read_text(encoding="utf-8"))
            for arm in full.ARMS
        }
        g2, g3 = reports["g2"], reports["g3_u110"]
        full.validate_pair(g2, g3, deck_id=deck_id)
        delta = g3["summary"]["win_rate"] - g2["summary"]["win_rate"]
        transitions = Counter(
            (left["outcome"], right["outcome"])
            for left, right in zip(g2["entries"], g3["entries"], strict=True)
        )
        rows.append({
            "deck_id": deck_id,
            "cohort": "fixed_001_010" if deck_id in full.PRIORITY_DECK_IDS else "seeded_random_030_067",
            "display_name": names[deck_id],
            "representative_cards": base._representative_cards(g2["focal_deck_cards"]),
            "exact_deck_sha256": g2["focal_exact_deck_sha256"],
            "g1": g2["summary"], "g2_candidate": g3["summary"],
            "g1_by_opponent": base._by_opponent(g2),
            "g2_by_opponent": base._by_opponent(g3),
            "by_meta_archetype": base._meta_comparison(g2, g3, taxonomy),
            "g1_candidate_identity": g2["focal_policy_identity_audit"],
            "g2_candidate_identity": g3["focal_policy_identity_audit"],
            "g1_schedule_sha256": g2["schedule"]["schedule_sha256"],
            "g2_schedule_sha256": g3["schedule"]["schedule_sha256"],
            "common_random_schedule_sha256": g2["schedule"]["common_random_schedule_sha256"],
            "opponent_effective_sha256": g2["opponent_policy_identity_audit"]["effective_policy_sha256"],
            "delta": delta,
            "delta_95": base._difference_interval(g2["summary"]["wins"], g3["summary"]["wins"], 2048),
            "classification": "improved" if delta > 0.01 else "regressed" if delta < -0.01 else "flat",
            "paired_transitions": {
                "loss_to_win": transitions[(-1, 1)], "win_to_loss": transitions[(1, -1)],
                "win_to_win": transitions[(1, 1)], "loss_to_loss": transitions[(-1, -1)],
            },
        })
    priority = [row for row in rows if row["deck_id"] in full.PRIORITY_DECK_IDS]
    remainder = [row for row in rows if row["deck_id"] not in full.PRIORITY_DECK_IDS]
    return {
        "schema_version": "0044_g2_vs_g3_u110_full67_benchmark_v2_aggregate_v1",
        "status": "PASS" if len(rows) == 67 else "PARTIAL_EVALUATION_IN_PROGRESS",
        "version": full.VERSION, "published_at": datetime.now(UTC).isoformat(),
        "completed_decks": len(rows), "target_decks": 67, "flat_threshold": 0.01,
        "fixed_deck_ids": list(full.PRIORITY_DECK_IDS),
        "random_deck_ids": list(full.EXECUTION_DECK_IDS[3:]),
        "execution_order": list(full.EXECUTION_DECK_IDS),
        "remainder_order_seed": full.REMAINDER_ORDER_SEED,
        "reporting_meta_taxonomy": taxonomy,
        "summary": {"overall": _summary(rows), "fixed_001_010": _summary(priority), "seeded_random_030_067": _summary(remainder)},
        "rows": rows,
        "best_improvement": max(rows, key=lambda row: row["delta"]) if rows else None,
        "worst_regression": min(rows, key=lambda row: row["delta"]) if rows else None,
        "identity": {
            "g2_source_checkpoint_sha256": full.G2_CHECKPOINT_SHA256,
            "g3_u110_source_checkpoint_sha256": full.G3_CHECKPOINT_SHA256,
            "opponent_policy_id": "Policy-0809",
            "benchmark_id": "Benchmark-V2",
            "games_per_arm_per_deck": 2048,
        },
    }


def _relabel(page: str) -> str:
    replacements = {
        "U407 · G2 Candidate Gate": "Champion-G2 → G3 U110 · Full-67 Benchmark V2",
        "0044 U407 G2 Candidate Gate": "0044 G2 vs G3 U110 · Benchmark V2",
        "Champion-G1 vs U407 G2 Candidate · Frozen Policy-0809": "Champion-G2 vs G3 U110 · complete Policy-0809 · Core-16 common random numbers",
        "G1 micro WR": "Champion-G2 micro WR",
        "G2 candidate micro WR": "G3 U110 micro WR",
        "Champion-G1": "Champion-G2",
        "U407 G2 candidate": "G3 U110 candidate",
        "U407 candidate": "G3 U110",
        "U407 source checkpoint": "G3 U110 source checkpoint",
        "U407 effective candidate": "G3 U110 effective candidate",
        "U407 portable checkpoint": "G3 U110 portable checkpoint",
        "U407 schedule": "G3 U110 schedule",
        "G1 effective candidate": "Champion-G2 effective candidate",
        "G1 portable checkpoint": "Champion-G2 portable checkpoint",
        "G1 schedule": "Champion-G2 schedule",
        "本报告是 G2 纳入体检，不会自动注册 Champion-G2。": "本报告比较冻结 Champion-G2 与 G3 U110，不会自动晋级 Champion-G3。",
        "两臂是同合同、同 exact deck、同完整 Policy-0809 对手池的独立 schedule，不作逐局 paired 推断。": "两臂使用同一 exact deck、同一 2,048 个 opponent/engine/Search/policy/coin jobs，可进行 paired 变化定位。",
        "G1/U407 为各自 identity-bound 独立样本。": "Champion-G2/G3 U110 使用相同 common-random jobs。",
        "两份 identity-bound schedule 在各 exact opponent 的样本数可能不同": "两臂在各 exact opponent 上使用相同样本数与 seeds",
        "0044 G2 CANDIDATE · CUDA-2048": "0044 BENCHMARK V2 · CUDA-2048",
        "G1 vs U407": "Champion-G2 vs G3 U110",
        "001–010": "Priority 007/003/002",
        "Seeded 030–067": "Seeded remainder",
        "Independent-arm delta": "Common-seed delta",
        "independent CUDA-2048 arms": "common-seed CUDA-2048 arms",
        "430044407": str(full.REMAINDER_ORDER_SEED),
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
        if schema == "0044_g2_vs_g3_u110_full67_benchmark_v2_aggregate_v1":
            s = payload["summary"]["overall"]
            label = "Champion-G2 vs G3 U110 · Full-67"
            result = f"{payload['completed_decks']}/67 decks · {s['micro_delta']*100:+.2f} pp"
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
