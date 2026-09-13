"""Aggregate and render the U407 G2-candidate versus Champion-G1 gate."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import html
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from evaluation.cards import card_image_url, load_card_catalog

from ..assets import AssetRegistry, sha256_file
from .g2_candidate_gate import (
    ARMS,
    ARTIFACT_ROOT,
    EVALUATION_DECK_IDS,
    FIXED_DECK_IDS,
    RANDOM_DECK_IDS,
    REPORT_ROOT,
    ROOT,
    STATE_PATH,
    VERSION,
    _wilson,
    validate_arm_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SLUG = (
    "u407_g2_candidate_vs_champion_g1_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1"
)
OUTPUT_ROOT = ROOT / "docs/evaluation/combat_mat/policy_0809" / REPORT_SLUG
OUTPUT = OUTPUT_ROOT / "index.html"
MANIFEST = OUTPUT_ROOT / "manifest.json"
DETAIL_ROOT = OUTPUT_ROOT / "reports"
REVERSE_LINK = ARTIFACT_ROOT / "evaluation.json"
FLAT_THRESHOLD = 0.01
META_TAXONOMY_PATH = PROJECT_ROOT / "assets/taxonomy/own_archetypes_v2.json"
META_MAPPING_PATH = PROJECT_ROOT / "assets/taxonomy/deck_own_archetype_mapping_v2.json"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _interval(values: list[float]) -> str:
    return f"{values[0] * 100:.2f}–{values[1] * 100:.2f}%"


def _difference_interval(left_wins: int, right_wins: int, games: int) -> list[float]:
    """Unpooled normal 95% CI for two independent binomial proportions."""
    left = left_wins / games
    right = right_wins / games
    half = 1.959963984540054 * math.sqrt(
        left * (1 - left) / games + right * (1 - right) / games
    )
    return [right - left - half, right - left + half]


def _read_reports(*, allow_partial: bool = False) -> list[dict[str, Any]]:
    rows = []
    for deck_id in EVALUATION_DECK_IDS:
        paths = {arm: REPORT_ROOT / deck_id / arm / "report.json" for arm in ARMS}
        if not all(path.is_file() for path in paths.values()):
            if allow_partial:
                continue
            raise RuntimeError(f"{deck_id}: both arm reports are required")
        arms = {
            arm: json.loads(path.read_text()) for arm, path in paths.items()
        }
        rows.append({"deck_id": deck_id, "arms": arms})
    return rows


def _representative_cards(cards: list[int]) -> list[dict[str, Any]]:
    catalog = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    counts = Counter(cards)
    pokemon = []
    for card_id, count in counts.items():
        card = catalog.get(card_id)
        if not card or "Pokémon" not in card["stage_or_type"]:
            continue
        pokemon.append({
            "card_id": card_id, "count": count, "name": card["name"],
            "image_url": card_image_url(card["expansion"], card["collection_number"]),
            "stage_or_type": card["stage_or_type"],
        })
    pokemon.sort(key=lambda row: (
        -("Stage 2" in row["stage_or_type"]), -("Stage 1" in row["stage_or_type"]),
        -(" ex" in row["name"]), -row["count"], row["card_id"],
    ))
    return pokemon[:2]


def _deck_representative_art() -> dict[str, list[dict[str, Any]]]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    expected = {f"{value:03d}" for value in range(1, 68)}
    result = {}
    for deck in registry.decks:
        if deck.deck_id not in expected:
            continue
        cards = list(map(int, (PROJECT_ROOT / deck.deck_path).read_text().splitlines()))
        representatives = _representative_cards(cards)
        if not representatives or any(not row["image_url"] for row in representatives):
            raise RuntimeError(f"deck {deck.deck_id} has no complete representative art")
        result[deck.deck_id] = representatives
    if set(result) != expected:
        raise RuntimeError("representative art does not cover exact 001-067")
    return result


def _by_opponent(report: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, dict[str, int | float]] = {}
    for entry in report["entries"]:
        row = grouped.setdefault(
            str(entry["opponent_id"]), {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        )
        row["games"] += 1
        if entry["outcome"] == 1:
            row["wins"] += 1
        elif entry["outcome"] == -1:
            row["losses"] += 1
        else:
            row["draws"] += 1
    for row in grouped.values():
        row["win_rate"] = row["wins"] / row["games"]
    return grouped


def _meta_taxonomy() -> dict[str, Any]:
    taxonomy = json.loads(META_TAXONOMY_PATH.read_text(encoding="utf-8"))
    mapping = json.loads(META_MAPPING_PATH.read_text(encoding="utf-8"))
    reporting_deck_ids = {f"{value:03d}" for value in range(1, 68)}
    classes = sorted(taxonomy["classes"], key=lambda row: int(row["archetype_id"]))
    if [int(row["archetype_id"]) for row in classes] != list(range(29)):
        raise RuntimeError("reporting taxonomy is not exact 29 classes")
    deck_to_class: dict[str, int] = {}
    for item in mapping["decks"]:
        deck_id = str(item["deck_id"])
        if deck_id not in reporting_deck_ids:
            continue
        archetype_id = int(item["archetype_id"])
        if deck_id in deck_to_class:
            raise RuntimeError(f"duplicate reporting taxonomy deck {deck_id}")
        deck_to_class[deck_id] = archetype_id
    expected = reporting_deck_ids
    if set(deck_to_class) != expected:
        raise RuntimeError("reporting taxonomy does not cover exact 001-067")
    by_class = {int(row["archetype_id"]): row for row in classes}
    discrepancies = []
    for archetype_id, row in by_class.items():
        declared = sorted(
            deck_id for deck_id in map(str, row["deck_ids"])
            if deck_id in reporting_deck_ids
        )
        mapped = sorted(deck for deck, value in deck_to_class.items() if value == archetype_id)
        if declared != mapped:
            discrepancies.append({
                "archetype_id": archetype_id,
                "registry_only_deck_ids": sorted(set(declared) - set(mapped)),
                "mapping_only_deck_ids": sorted(set(mapped) - set(declared)),
            })
        row["deck_ids"] = mapped
    return {
        "taxonomy_version": taxonomy["taxonomy_version"],
        "taxonomy_sha256": sha256_file(META_TAXONOMY_PATH),
        "mapping_sha256": sha256_file(META_MAPPING_PATH),
        "classes": classes, "deck_to_class": deck_to_class,
        "membership_authority": "deck_own_archetype_mapping_v2",
        "registry_membership_discrepancies": discrepancies,
        "reporting_only": True, "model_opponent_meta_head_unchanged": True,
        "model_opponent_meta_classes": 15,
    }


def _by_meta_archetype(
    report: dict[str, Any], taxonomy: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {
        int(row["archetype_id"]): {
            "archetype_id": int(row["archetype_id"]),
            "name": str(row["name"]), "display_name": str(row["display_name"]),
            "definition": str(row["definition"]),
            "parent_archetype": row.get("parent_archetype"),
            "deck_ids": sorted(map(str, row["deck_ids"])),
            "observed_deck_ids": set(), "games": 0, "wins": 0,
            "losses": 0, "draws": 0,
        }
        for row in taxonomy["classes"]
    }
    for entry in report["entries"]:
        opponent_id = str(entry["opponent_id"])
        if opponent_id not in taxonomy["deck_to_class"]:
            raise RuntimeError(f"unknown opponent deck in reporting taxonomy: {opponent_id}")
        row = by_id[int(taxonomy["deck_to_class"][opponent_id])]
        row["observed_deck_ids"].add(opponent_id)
        row["games"] += 1
        if entry["outcome"] == 1:
            row["wins"] += 1
        elif entry["outcome"] == -1:
            row["losses"] += 1
        elif entry["outcome"] == 0:
            row["draws"] += 1
        else:
            raise RuntimeError("invalid outcome in reporting taxonomy aggregation")
    result = []
    for archetype_id in range(29):
        row = by_id[archetype_id]
        row["observed_deck_ids"] = sorted(row["observed_deck_ids"])
        row["win_rate"] = row["wins"] / row["games"] if row["games"] else None
        result.append(row)
    if sum(row["games"] for row in result) != len(report["entries"]):
        raise RuntimeError("reporting taxonomy aggregation lost games")
    return result


def _meta_comparison(
    g1: dict[str, Any], g2: dict[str, Any], taxonomy: dict[str, Any]
) -> list[dict[str, Any]]:
    left = _by_meta_archetype(g1, taxonomy)
    right = _by_meta_archetype(g2, taxonomy)
    result = []
    for g1_row, g2_row in zip(left, right, strict=True):
        if g1_row["archetype_id"] != g2_row["archetype_id"]:
            raise RuntimeError("Meta Archetype aggregation order mismatch")
        result.append({
            "archetype_id": g1_row["archetype_id"], "name": g1_row["name"],
            "display_name": g1_row["display_name"],
            "definition": g1_row["definition"],
            "parent_archetype": g1_row["parent_archetype"],
            "deck_ids": g1_row["deck_ids"],
            "observed_deck_ids": sorted(
                set(g1_row["observed_deck_ids"]) | set(g2_row["observed_deck_ids"])
            ),
            "g1": {key: g1_row[key] for key in ("games", "wins", "losses", "draws", "win_rate")},
            "g2_candidate": {
                key: g2_row[key] for key in ("games", "wins", "losses", "draws", "win_rate")
            },
            "delta": (
                g2_row["win_rate"] - g1_row["win_rate"]
                if g1_row["win_rate"] is not None and g2_row["win_rate"] is not None
                else None
            ),
        })
    return result


def aggregate(*, allow_partial: bool = False) -> dict[str, Any]:
    state = json.loads(STATE_PATH.read_text())
    if not allow_partial and state.get("status") != "EVALUATION_COMPLETE_REPORT_PENDING":
        raise RuntimeError("all forty CUDA-2048 arms must complete before rendering")
    registry = AssetRegistry.load(PROJECT_ROOT)
    taxonomy = _meta_taxonomy()
    names = {row.deck_id: row.name for row in registry.decks}
    rows = []
    for item in _read_reports(allow_partial=allow_partial):
        deck_id = item["deck_id"]
        g1 = item["arms"]["g1"]
        g2 = item["arms"]["g2_candidate"]
        # Re-run the controller's fail-closed gate at publication time.
        validate_arm_report(g1, arm="g1", deck_id=deck_id)
        validate_arm_report(g2, arm="g2_candidate", deck_id=deck_id)
        if (
            g1["focal_exact_deck_sha256"] != g2["focal_exact_deck_sha256"]
            or g1["opponent_policy_identity_audit"]["effective_policy_sha256"]
            != g2["opponent_policy_identity_audit"]["effective_policy_sha256"]
        ):
            raise RuntimeError(f"{deck_id}: comparison identity mismatch")
        g1_wr = float(g1["summary"]["win_rate"])
        g2_wr = float(g2["summary"]["win_rate"])
        delta = g2_wr - g1_wr
        classification = (
            "improved" if delta > FLAT_THRESHOLD
            else "regressed" if delta < -FLAT_THRESHOLD
            else "flat"
        )
        rows.append({
            "deck_id": deck_id,
            "cohort": "fixed_001_010" if deck_id in FIXED_DECK_IDS else "seeded_random_030_067",
            "display_name": names[deck_id],
            "representative_cards": _representative_cards(g1["focal_deck_cards"]),
            "exact_deck_sha256": g1["focal_exact_deck_sha256"],
            "g1": g1["summary"], "g2_candidate": g2["summary"],
            "g1_by_opponent": _by_opponent(g1),
            "g2_by_opponent": _by_opponent(g2),
            "by_meta_archetype": _meta_comparison(g1, g2, taxonomy),
            "g1_candidate_identity": g1["candidate_deployment_identity_audit"],
            "g2_candidate_identity": g2["candidate_deployment_identity_audit"],
            "g1_schedule_sha256": g1["schedule"]["schedule_sha256"],
            "g2_schedule_sha256": g2["schedule"]["schedule_sha256"],
            "opponent_effective_sha256": g1["opponent_policy_identity_audit"]["effective_policy_sha256"],
            "delta": delta,
            "delta_95": _difference_interval(
                int(g1["summary"]["wins"]), int(g2["summary"]["wins"]), 2048
            ),
            "classification": classification,
        })

    def cohort_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        games = sum(row["g1"]["games"] for row in selected)
        g1_wins = sum(row["g1"]["wins"] for row in selected)
        g2_wins = sum(row["g2_candidate"]["wins"] for row in selected)
        return {
            "decks": len(selected), "games_per_arm": games,
            "g1_micro_win_rate": g1_wins / games,
            "g2_micro_win_rate": g2_wins / games,
            "g1_wilson_95": _wilson(g1_wins, games),
            "g2_wilson_95": _wilson(g2_wins, games),
            "micro_delta_95": _difference_interval(g1_wins, g2_wins, games),
            "micro_delta": (g2_wins - g1_wins) / games,
            "macro_delta": sum(row["delta"] for row in selected) / len(selected),
            "improved": sum(row["classification"] == "improved" for row in selected),
            "flat": sum(row["classification"] == "flat" for row in selected),
            "regressed": sum(row["classification"] == "regressed" for row in selected),
        }
    if not rows:
        raise RuntimeError("no complete G1/U407 deck pair is available for publication")
    summary = {
        "overall": cohort_summary(rows),
        "fixed_001_010": cohort_summary([row for row in rows if row["deck_id"] in FIXED_DECK_IDS]),
        "seeded_random_030_067": cohort_summary([row for row in rows if row["deck_id"] in RANDOM_DECK_IDS]),
    }
    return {
        "schema_version": "0044_u407_g2_candidate_gate_aggregate_v1",
        "status": (
            "HUMAN_DECISION_REQUIRED" if len(rows) == len(EVALUATION_DECK_IDS)
            else "PARTIAL_EVALUATION_IN_PROGRESS"
        ),
        "published_at": datetime.now(UTC).isoformat(),
        "completed_decks": len(rows), "target_decks": len(EVALUATION_DECK_IDS),
        "version": VERSION, "flat_threshold": FLAT_THRESHOLD,
        "fixed_deck_ids": list(FIXED_DECK_IDS),
        "random_deck_ids": list(RANDOM_DECK_IDS),
        "reporting_meta_taxonomy": taxonomy,
        "summary": summary, "rows": rows,
        "best_improvement": max(rows, key=lambda row: row["delta"]),
        "worst_regression": min(rows, key=lambda row: row["delta"]),
    }


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]["overall"]
    def art(row: dict[str, Any]) -> str:
        return '<span class="art">' + "".join(
            f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
            f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
            for card in row["representative_cards"] if card["image_url"]
        ) + "</span>"
    rows = "".join(
        f"<tr data-search='{row['deck_id']} {html.escape(row['display_name'], quote=True).lower()}'><td class='number'>{row['deck_id']}</td><td><span class='deck-name'>{art(row)}<span><a href='reports/{row['deck_id']}.html'>{html.escape(row['display_name'])}</a></span></span></td>"
        f"<td>{'001–010' if row['cohort']=='fixed_001_010' else 'Seeded 030–067'}</td>"
        f"<td>{_pct(row['g1']['win_rate'])}<small>{row['g1']['wins']}-{row['g1']['losses']}-{row['g1']['draws']}<br>95% {_interval(row['g1']['wilson_95'])}<br>先/后 {_pct(row['g1']['focal_first_win_rate'])} / {_pct(row['g1']['focal_second_win_rate'])}</small></td>"
        f"<td>{_pct(row['g2_candidate']['win_rate'])}<small>{row['g2_candidate']['wins']}-{row['g2_candidate']['losses']}-{row['g2_candidate']['draws']}<br>95% {_interval(row['g2_candidate']['wilson_95'])}<br>先/后 {_pct(row['g2_candidate']['focal_first_win_rate'])} / {_pct(row['g2_candidate']['focal_second_win_rate'])}</small></td>"
        f"<td class='{row['classification']}'>{_pp(row['delta'])}<small>独立样本 95% CI {_pp(row['delta_95'][0])} 至 {_pp(row['delta_95'][1])}</small></td>"
        f"<td><code>{row['exact_deck_sha256'][:12]}…</code></td></tr>"
        for row in payload["rows"]
    )
    cohort_rows = "".join(
        f"<tr><td>{label}</td><td>{row['decks']}</td><td>{_pct(row['g1_micro_win_rate'])}<small>95% {_interval(row['g1_wilson_95'])}</small></td>"
        f"<td>{_pct(row['g2_micro_win_rate'])}<small>95% {_interval(row['g2_wilson_95'])}</small></td><td>{_pp(row['micro_delta'])}<small>95% {_pp(row['micro_delta_95'][0])} 至 {_pp(row['micro_delta_95'][1])}</small></td>"
        f"<td>{row['improved']} / {row['flat']} / {row['regressed']}</td></tr>"
        for label, row in (
            ("Overall", payload["summary"]["overall"]),
            ("001–010", payload["summary"]["fixed_001_010"]),
            ("Seeded 030–067", payload["summary"]["seeded_random_030_067"]),
        )
    )
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d;--red:#a54343;--amber:#986a12}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}header{padding:28px max(20px,calc((100vw - 1500px)/2));background:#18382d;color:#fff}header h1{margin:0;font-size:30px}header p{max-width:1100px;margin:7px 0 0;color:#cfe1da}main{max-width:1500px;margin:auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--paper);margin-bottom:18px}.metric{padding:15px 18px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font-size:23px}.metric small,small,.muted{display:block;color:var(--muted)}section{padding:20px;margin-top:18px;border:1px solid var(--line);background:#fff}h2{margin:0 0 8px}.tools{display:flex;margin:18px 0}.tools input{width:min(420px,100%);padding:9px 11px;border:1px solid #b9c9c1;border-radius:4px;background:#fff}.table{overflow:auto}table{width:100%;min-width:1180px;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:middle;white-space:nowrap}th{background:#e9f0ec;color:#486158;font-size:12px}tr:hover td{background:#f8fbf9}a{color:var(--green);font-weight:700;text-decoration:none}code{overflow-wrap:anywhere}.number{font-size:16px;font-weight:850;color:var(--green)}.improved{color:var(--green);font-weight:700}.regressed{color:var(--red);font-weight:700}.flat{color:var(--amber);font-weight:700}.audit{display:grid;grid-template-columns:230px 1fr;gap:7px 14px}.deck-name{display:flex;align-items:center;gap:10px;min-width:240px}.art{display:flex;width:68px;flex:0 0 68px}.art img{width:38px;height:53px;object-fit:cover;border-radius:3px;border:1px solid #c9d5cf;margin-right:-8px;background:#e4ebe7}@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.audit{grid-template-columns:1fr}table{font-size:12px}.art{display:none}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0044 U407 G2 Candidate Gate</title><style>{style}</style></head><body><main>
<header><p>0044 CHAMPION LEAGUE RL · PROMOTE DIAGNOSTIC</p><h1>U407 · G2 Candidate Gate</h1><p>Champion-G1 vs U407 G2 Candidate · Frozen Policy-0809 · 已完成 {payload['completed_decks']}/{payload['target_decks']} focal decks × 2 arms × CUDA-2048</p></header>
<div class="grid"><div class="metric"><small>G1 micro WR</small><b>{_pct(s['g1_micro_win_rate'])}</b></div><div class="metric"><small>G2 candidate micro WR</small><b>{_pct(s['g2_micro_win_rate'])}</b></div><div class="metric"><small>Overall delta</small><b>{_pp(s['micro_delta'])}</b></div><div class="metric"><small>Improved / Flat / Regressed</small><b>{s['improved']} / {s['flat']} / {s['regressed']}</b></div></div>
<section><h2>结论边界</h2><p><b>状态：{payload['status']}。</b>{'这是增量快照，尚不可作最终晋级判断；' if payload['status']=='PARTIAL_EVALUATION_IN_PROGRESS' else ''}本报告是 G2 纳入体检，不会自动注册 Champion-G2。Flat 的描述阈值固定为 ±{payload['flat_threshold']*100:.1f} pp；这不是统计等价检验。G1 与 U407 均满足同一 FrozenMeta2048 合同，但 schedule 按各自 deployment identity 独立派生，故区间按两份独立二项样本计算，不作逐局 paired 推断。</p></section>
<section><h2>Cohort summary</h2><table><thead><tr><th>Cohort</th><th>Decks</th><th>G1</th><th>G2 candidate</th><th>Delta</th><th>I / F / R</th></tr></thead><tbody>{cohort_rows}</tbody></table></section>
<div class="tools"><input id="search" type="search" placeholder="筛选编号或牌型"></div><section class="table"><h2>Per-deck CUDA-2048 comparison</h2><table id="results"><thead><tr><th>Deck</th><th>Name / report</th><th>Cohort</th><th>Champion-G1</th><th>U407 candidate</th><th>Delta</th><th>Exact deck</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Identity / evidence audit</h2><div class="audit"><span>Candidate contract</span><code>kaggle_fp16_storage_fp32_runtime_v1</code><span>Opponent</span><code>complete immutable Policy-0809 · {payload['rows'][0]['opponent_effective_sha256']}</code><span>Evaluation</span><code>40 independent CUDA-2048 arms · 81,920 terminal games</code><span>Sample seed</span><code>430044407</code><span>Random deck IDs</span><code>{', '.join(payload['random_deck_ids'])}</code><span>Selection mode</span><code>greedy</code><span>CUDA routing</span><code>CUDA Engine 2.0 · device-resident features · zero focal/opponent parameter-storage alias</code><span>Promotion</span><b>Only human PROMOTE / HOLD / REJECT</b></div></section>
<script id="report-data" type="application/json">{embedded}</script><script>const q=document.getElementById('search');q.addEventListener('input',()=>{{const v=q.value.trim().toLowerCase();document.querySelectorAll('#results tbody tr').forEach(r=>r.hidden=!r.dataset.search.includes(v))}});</script></main></body></html>'''


def render_detail(row: dict[str, Any]) -> str:
    g1, g2 = row["g1"], row["g2_candidate"]
    opponent_ids = sorted(set(row["g1_by_opponent"]) | set(row["g2_by_opponent"]))
    def matchup_cell(arm: str, opponent_id: str) -> str:
        value = row[arm].get(opponent_id)
        if value is None:
            return "—"
        return (
            f"{_pct(float(value['win_rate']))}<small>"
            f"{value['wins']}-{value['losses']}-{value['draws']}</small>"
        )
    matchup_rows = "".join(
        f"<tr><td>{opponent_id}</td>"
        f"<td>{matchup_cell('g1_by_opponent', opponent_id)}</td>"
        f"<td>{matchup_cell('g2_by_opponent', opponent_id)}</td>"
        f"<td>{_pp(float(row['g2_by_opponent'][opponent_id]['win_rate']) - float(row['g1_by_opponent'][opponent_id]['win_rate'])) if opponent_id in row['g1_by_opponent'] and opponent_id in row['g2_by_opponent'] else '—'}</td></tr>"
        for opponent_id in opponent_ids
    )
    def meta_result(value: dict[str, Any]) -> str:
        if not value["games"]:
            return '<span class="no-sample">无样本</span>'
        return (
            f"<b>{_pct(float(value['win_rate']))}</b>"
            f"<small>{value['wins']}-{value['losses']}-{value['draws']} · n={value['games']}</small>"
        )
    def meta_delta(value: float | None) -> str:
        return _pp(float(value)) if value is not None else '<span class="no-sample">无样本</span>'
    deck_art = _deck_representative_art()
    def observed_decks(meta: dict[str, Any]) -> str:
        if not meta["observed_deck_ids"]:
            return '<span class="no-sample">无</span>'
        return '<span class="observed-decks">' + "".join(
            '<span class="opponent-deck-chip">'
            + '<span class="opponent-art">'
            + "".join(
                f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
                f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy" '
                f'title="{html.escape(str(card["name"]), quote=True)}">'
                for card in deck_art[deck_id]
            )
            + f'</span><b>{deck_id}</b></span>'
            for deck_id in meta["observed_deck_ids"]
        ) + "</span>"
    meta_rows = "".join(
        f"<tr data-meta='{meta['archetype_id']} {html.escape(meta['display_name'], quote=True).lower()} {' '.join(meta['deck_ids'])}'>"
        f"<td><span class='meta-id'>{meta['archetype_id']:02d}</span> <b>{html.escape(meta['display_name'])}</b>"
        f"<small>{html.escape(meta['definition'])}</small>"
        f"<small>Parent: {html.escape(str(meta['parent_archetype'] or '—'))}</small></td>"
        f"<td><b>{', '.join(meta['deck_ids']) or '—'}</b><small>本次实际覆盖：</small>{observed_decks(meta)}</td>"
        f"<td>{meta_result(meta['g1'])}</td><td>{meta_result(meta['g2_candidate'])}</td>"
        f"<td>{meta_delta(meta['delta'])}</td></tr>"
        for meta in row["by_meta_archetype"]
    )
    audit_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for label, value in (
            ("Exact deck SHA-256", row["exact_deck_sha256"]),
            ("G1 effective candidate", row["g1_candidate_identity"]["effective_candidate_sha256"]),
            ("G1 portable checkpoint", row["g1_candidate_identity"]["portable_checkpoint_sha256"]),
            ("G1 schedule", row["g1_schedule_sha256"]),
            ("U407 source checkpoint", row["g2_candidate_identity"]["source_checkpoint_sha256"]),
            ("U407 effective candidate", row["g2_candidate_identity"]["effective_candidate_sha256"]),
            ("U407 portable checkpoint", row["g2_candidate_identity"]["portable_checkpoint_sha256"]),
            ("U407 schedule", row["g2_schedule_sha256"]),
            ("Policy-0809 effective", row["opponent_effective_sha256"]),
        )
    )
    cards = "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
        for card in row["representative_cards"] if card["image_url"]
    )
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#f3f7f5;--surface:#fff;--soft:#f7faf8;--ink:#172b25;--muted:#60736c;--line:#dce7e2;--brand:#217a58;--dark:#14563d}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,"PingFang SC",sans-serif}main{max-width:1240px;margin:auto;padding:36px 28px 64px}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:22px;padding:30px 32px;border-radius:8px;color:#fff;background:var(--dark)}.hero h1{margin:0;font-size:32px}.hero p{margin:4px 0;color:#c8eadb}.hero-art{display:flex}.hero-art img{width:82px;height:114px;object-fit:cover;border-radius:6px;border:1px solid #ffffff55;margin-left:-14px}.back{color:#d8eee5;text-decoration:none;font-weight:700}section{margin:18px 0;padding:22px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}h2{margin:0 0 10px}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{padding:16px;border:1px solid var(--line);border-radius:10px}.card small{display:block;color:var(--muted)}.card b{display:block;font-size:24px}.delta{color:var(--brand)}table{width:100%;border-collapse:collapse}th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--soft);color:#496159}td small{display:block;color:var(--muted)}code{overflow-wrap:anywhere}.boundary{border-left:4px solid var(--brand);color:var(--muted)}.meta-id{display:inline-flex;min-width:28px;justify-content:center;padding:1px 5px;border-radius:4px;background:#e4f3ec;color:var(--dark);font-weight:800}.no-sample{color:var(--muted);font-style:italic}.meta-note{color:var(--muted)}.observed-decks{display:flex;flex-wrap:wrap;gap:7px;margin-top:5px}.opponent-deck-chip{display:inline-flex;align-items:center;gap:6px;padding:3px 7px 3px 3px;border:1px solid #d6e2dc;border-radius:6px;background:#f7faf8}.opponent-art{display:flex;width:36px}.opponent-art img{width:25px;height:35px;object-fit:cover;border:1px solid #c9d5cf;border-radius:3px;margin-right:-12px;background:#e4ebe7}.opponent-deck-chip b{font-variant-numeric:tabular-nums}@media(max-width:700px){.summary{grid-template-columns:1fr}.hero-art{display:none}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{row['deck_id']} · G1 vs U407</title><style>{style}</style></head><body><main><div class="hero"><div><a class="back" href="../index.html">← 返回总览</a><p>0044 G2 CANDIDATE · CUDA-2048</p><h1>{row['deck_id']} · {html.escape(row['display_name'])}</h1></div><div class="hero-art">{cards}</div></div>
<section class="summary"><div class="card"><small>Champion-G1</small><b>{_pct(g1['win_rate'])}</b><span>{g1['wins']}-{g1['losses']}-{g1['draws']} · 95% {_interval(g1['wilson_95'])}</span><small>实际先/后 {_pct(g1['focal_first_win_rate'])} / {_pct(g1['focal_second_win_rate'])}</small></div><div class="card"><small>U407 G2 candidate</small><b>{_pct(g2['win_rate'])}</b><span>{g2['wins']}-{g2['losses']}-{g2['draws']} · 95% {_interval(g2['wilson_95'])}</span><small>实际先/后 {_pct(g2['focal_first_win_rate'])} / {_pct(g2['focal_second_win_rate'])}</small></div><div class="card"><small>Independent-arm delta</small><b class="delta">{_pp(row['delta'])}</b><span>95% CI {_pp(row['delta_95'][0])} 至 {_pp(row['delta_95'][1])}</span><small>{row['classification']} · ±1 pp descriptive threshold</small></div></section>
<section class="boundary"><b>HUMAN_DECISION_REQUIRED</b> · 本页不自动产生 Champion-G2。两臂是同合同、同 exact deck、同完整 Policy-0809 对手池的独立 schedule，不作逐局 paired 推断。</section>
<section id="meta-archetypes"><h2>Against Meta Archetype · 29-class reporting view</h2><p class="meta-note">按 <code>own_archetypes_v2</code> 的新分化类别离线聚合真实 opponent deck；它只用于分析，不改变模型的 15-way opponent Meta head。分类全集列出 001–067 的成员，本次实际覆盖列出 Frozen Policy-0809 schedule 真正出现过的成员。G1/U407 为各自 identity-bound 独立样本。</p><table><thead><tr><th>Meta Archetype</th><th>类内 deck IDs / 实际覆盖</th><th>Champion-G1</th><th>U407 candidate</th><th>Delta</th></tr></thead><tbody>{meta_rows}</tbody></table></section>
<section><h2>By opponent exact deck</h2><p>两份 identity-bound schedule 在各 exact opponent 的样本数可能不同，因此这里只作 matchup 定位，不作 paired 显著性结论。</p><table><thead><tr><th>Opponent deck</th><th>Champion-G1</th><th>U407 candidate</th><th>Delta</th></tr></thead><tbody>{matchup_rows}</tbody></table></section>
<section><h2>Deployment / routing identity</h2><table>{audit_rows}</table></section>
<script id="report-data" type="application/json">{payload}</script></main></body></html>'''


def publish(*, allow_partial: bool = False) -> Path:
    payload = aggregate(allow_partial=allow_partial)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".html.tmp")
    temporary.write_text(render(payload), encoding="utf-8")
    temporary.replace(OUTPUT)
    manifest_temporary = MANIFEST.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temporary, MANIFEST)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    for row in payload["rows"]:
        detail = DETAIL_ROOT / f"{row['deck_id']}.html"
        temporary = detail.with_suffix(".html.tmp")
        temporary.write_text(render_detail(row), encoding="utf-8")
        temporary.replace(detail)
    reverse_temporary = REVERSE_LINK.with_suffix(".json.tmp")
    reverse_temporary.write_text(json.dumps({
        "schema_version": "0044_evaluation_reverse_link_v1",
        "version": VERSION, "status": payload["status"],
        "authoritative_report": str(OUTPUT.relative_to(ROOT)),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(reverse_temporary, REVERSE_LINK)
    return OUTPUT


def watch(interval_seconds: float = 10.0) -> None:
    last_complete: tuple[str, ...] = ()
    while True:
        complete = tuple(
            deck_id for deck_id in EVALUATION_DECK_IDS
            if all((REPORT_ROOT / deck_id / arm / "report.json").is_file() for arm in ARMS)
        )
        if complete and complete != last_complete:
            publish(allow_partial=True)
            print(f"published {len(complete)}/{len(EVALUATION_DECK_IDS)} decks", flush=True)
            last_complete = complete
        state = json.loads(STATE_PATH.read_text())
        if state.get("status") == "EVALUATION_COMPLETE_REPORT_PENDING":
            publish()
            return
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.watch:
        watch(args.interval_seconds)
        return 0
    print(publish(allow_partial=args.allow_partial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
