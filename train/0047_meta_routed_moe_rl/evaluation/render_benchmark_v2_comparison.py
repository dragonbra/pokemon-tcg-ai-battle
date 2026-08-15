"""Publish the Core-16 Benchmark V2 U0-versus-U30 comparison."""

from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .render_g2_candidate_gate import (
    ROOT, _deck_representative_art, _meta_taxonomy, _pct,
    _representative_cards,
)
from .run_benchmark_v2 import validate_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/evaluation/combat_mat/benchmark_v2"
SLUG = "0044_u0_g2_init_vs_u30_deck_007_core16_policy0809_cuda2048_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    return {
        "games": games, "wins": wins, "losses": losses,
        "draws": games - wins - losses,
        "win_rate": wins / games if games else None,
    }


def aggregate(u0: dict[str, Any], u30: dict[str, Any]) -> dict[str, Any]:
    validate_report(u0)
    validate_report(u30)
    if u0["focal_checkpoint_update"] != 0 or u30["focal_checkpoint_update"] != 30:
        raise RuntimeError("comparison must bind exact U0 and U30 checkpoints")
    if u0["focal_deck_id"] != "007" or u30["focal_deck_id"] != "007":
        raise RuntimeError("comparison must bind focal exact deck 007")
    common_hash = u0["schedule"]["common_random_schedule_sha256"]
    if common_hash != u30["schedule"]["common_random_schedule_sha256"]:
        raise RuntimeError("U0 and U30 do not share the common-random schedule")
    job_fields = (
        "game_id", "opponent_id", "opponent_meta_archetype_id", "engine_seed",
        "search_seed", "policy_seed", "coin_winner_seed",
    )
    for left, right in zip(u0["entries"], u30["entries"], strict=True):
        if tuple(left[key] for key in job_fields) != tuple(right[key] for key in job_fields):
            raise RuntimeError("paired Benchmark V2 job identity mismatch")

    taxonomy = _meta_taxonomy()
    art = _deck_representative_art()
    classes = {int(row["archetype_id"]): row for row in taxonomy["classes"]}
    selected = list(map(int, u0["schedule"]["selected_class_ids"]))
    by_meta = []
    for class_id in selected:
        left = [row for row in u0["entries"] if row["opponent_meta_archetype_id"] == class_id]
        right = [row for row in u30["entries"] if row["opponent_meta_archetype_id"] == class_id]
        lstat, rstat = _aggregate(left), _aggregate(right)
        row = classes[class_id]
        by_meta.append({
            "archetype_id": class_id, "display_name": row["display_name"],
            "definition": row["definition"], "deck_ids": list(map(str, row["deck_ids"])),
            "representative_decks": [
                {"deck_id": deck_id, "representative_cards": art[deck_id]}
                for deck_id in map(str, row["deck_ids"])
            ],
            "u0": lstat, "u30": rstat,
            "delta": rstat["win_rate"] - lstat["win_rate"],
            "net_wins": rstat["wins"] - lstat["wins"],
        })

    observed_decks = sorted({str(row["opponent_id"]) for row in u0["entries"]})
    by_deck = []
    for deck_id in observed_decks:
        left = [row for row in u0["entries"] if str(row["opponent_id"]) == deck_id]
        right = [row for row in u30["entries"] if str(row["opponent_id"]) == deck_id]
        lstat, rstat = _aggregate(left), _aggregate(right)
        by_deck.append({
            "deck_id": deck_id, "representative_cards": art[deck_id],
            "u0": lstat, "u30": rstat,
            "delta": rstat["win_rate"] - lstat["win_rate"],
            "net_wins": rstat["wins"] - lstat["wins"],
        })

    transitions = Counter(
        (left["outcome"], right["outcome"])
        for left, right in zip(u0["entries"], u30["entries"], strict=True)
    )
    return {
        "schema_version": "0044_benchmark_v2_u0_u30_comparison_v1",
        "status": "PASS",
        "contract_id": u0["schedule"]["contract_id"],
        "common_random_schedule_sha256": common_hash,
        "focal_deck_id": "007",
        "focal_deck_display_name": u0["focal_deck_display_name"],
        "focal_representative_cards": _representative_cards(u0["focal_deck_cards"]),
        "opponent_policy_id": "Policy-0809",
        "u0": {
            "label": "0044 U0 (Champion-G2 initialization)",
            "summary": u0["summary"],
            "source_checkpoint_sha256": u0["focal_policy_identity_audit"]["source_checkpoint_sha256"],
            "deployment_effective_sha256": u0["focal_deployment_effective_sha256"],
            "schedule_sha256": u0["schedule"]["schedule_sha256"],
        },
        "u30": {
            "label": "0044 U30",
            "summary": u30["summary"],
            "source_checkpoint_sha256": u30["focal_policy_identity_audit"]["source_checkpoint_sha256"],
            "deployment_effective_sha256": u30["focal_deployment_effective_sha256"],
            "schedule_sha256": u30["schedule"]["schedule_sha256"],
        },
        "delta": u30["summary"]["win_rate"] - u0["summary"]["win_rate"],
        "net_wins": u30["summary"]["wins"] - u0["summary"]["wins"],
        "paired_transitions": {
            "loss_to_win": transitions[(-1, 1)], "win_to_loss": transitions[(1, -1)],
            "win_to_win": transitions[(1, 1)], "loss_to_loss": transitions[(-1, -1)],
        },
        "by_meta_archetype": by_meta,
        "by_opponent_deck": by_deck,
        "audit": {
            "same_common_random_schedule": True,
            "same_2048_job_identities": True,
            "selected_meta_ids": selected,
            "games_per_meta": 128,
            "u0_status": u0["status"], "u30_status": u30["status"],
            "u0_routing_failures": u0["collector_metrics"]["rollout/lane_routing_audit_failures"],
            "u30_routing_failures": u30["collector_metrics"]["rollout/lane_routing_audit_failures"],
            "u0_feature_d2h_bytes": u0["collector_metrics"]["rollout/cuda_feature_d2h_bytes"],
            "u30_feature_d2h_bytes": u30["collector_metrics"]["rollout/cuda_feature_d2h_bytes"],
        },
    }


def _art(cards: list[dict[str, Any]], css: str = "art") -> str:
    return f'<span class="{css}">' + "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy" '
        f'title="{html.escape(str(card["name"]), quote=True)}">'
        for card in cards
    ) + "</span>"


def _pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _comparison_cells(row: dict[str, Any]) -> str:
    css = "up" if row["delta"] > 0 else "down" if row["delta"] < 0 else "flat"
    return (
        f'<td>{_pct(row["u0"]["win_rate"])}<small>{row["u0"]["wins"]}-'
        f'{row["u0"]["losses"]}-{row["u0"]["draws"]} · n={row["u0"]["games"]}</small></td>'
        f'<td>{_pct(row["u30"]["win_rate"])}<small>{row["u30"]["wins"]}-'
        f'{row["u30"]["losses"]}-{row["u30"]["draws"]} · n={row["u30"]["games"]}</small></td>'
        f'<td class="{css}"><b>{_pp(row["delta"])}</b><small>{row["net_wins"]:+d} wins</small></td>'
    )


def render(payload: dict[str, Any]) -> str:
    u0, u30 = payload["u0"], payload["u30"]
    meta_rows = "".join(
        f'<tr><td><span class="id">{row["archetype_id"]:02d}</span> '
        f'<b>{html.escape(row["display_name"])}</b><small>{html.escape(row["definition"])}</small></td>'
        f'<td><div class="chips">' + "".join(
            f'<span class="chip">{_art(deck["representative_cards"], "thumbs")}<b>{deck["deck_id"]}</b></span>'
            for deck in row["representative_decks"]
        ) + f'</div><small>{", ".join(row["deck_ids"])}</small></td>'
        + _comparison_cells(row) + "</tr>"
        for row in payload["by_meta_archetype"]
    )
    deck_rows = "".join(
        f'<tr><td><span class="id">{row["deck_id"]}</span></td>'
        f'<td>{_art(row["representative_cards"])}</td>' + _comparison_cells(row) + "</tr>"
        for row in sorted(payload["by_opponent_deck"], key=lambda row: row["delta"], reverse=True)
    )
    audit_rows = "".join(
        f'<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(str(value))}</code></td></tr>'
        for key, value in {
            "contract_id": payload["contract_id"],
            "common_random_schedule_sha256": payload["common_random_schedule_sha256"],
            "U0 source checkpoint": u0["source_checkpoint_sha256"],
            "U0 deployment effective": u0["deployment_effective_sha256"],
            "U30 source checkpoint": u30["source_checkpoint_sha256"],
            "U30 deployment effective": u30["deployment_effective_sha256"],
            **payload["audit"],
        }.items()
    )
    transitions = payload["paired_transitions"]
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#edf3f0;--paper:#fff;--ink:#172720;--muted:#65766f;--line:#d4e0da;--brand:#17694a;--up:#087f4f;--down:#b13b3b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"PingFang SC",sans-serif}main{max-width:1440px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;align-items:end;gap:20px;padding:30px;border-radius:12px;background:#153e30;color:#fff}.hero h1{margin:4px 0;font-size:31px}.hero p{margin:0;color:#cfe3da}.hero .art img{width:82px;height:114px}.summary{display:grid;grid-template-columns:repeat(4,1fr);margin:18px 0;background:#fff;border:1px solid var(--line)}.metric{padding:17px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font-size:24px}.metric small,small{display:block;color:var(--muted)}section{margin-top:18px;padding:20px;background:#fff;border:1px solid var(--line);overflow:auto}h2{margin:0 0 9px}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{background:#edf3f0;color:#50645c}.up{color:var(--up)}.down{color:var(--down)}.flat{color:#68756f}.id{display:inline-block;padding:2px 6px;border-radius:4px;background:#e0f0e8;color:var(--brand);font-weight:850}.art,.thumbs{display:inline-flex;vertical-align:middle;width:60px}.art img{width:36px;height:50px}.thumbs{width:33px}.thumbs img{width:24px;height:34px;margin-right:-12px}.art img,.thumbs img{object-fit:cover;border:1px solid #c8d5cf;border-radius:3px;margin-right:-8px;background:#e5ece8}.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 7px 3px 3px;border:1px solid var(--line);border-radius:6px;background:#f7faf8}code{overflow-wrap:anywhere}.note{padding:12px 14px;border-left:4px solid #d89a26;background:#fff8e8}a{color:var(--brand);font-weight:700}@media(max-width:800px){.summary{grid-template-columns:1fr 1fr}.hero>.art{display:none}}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0044 Benchmark V2 · U0 vs U30</title><style>{style}</style></head><body><main>
<div class="hero"><div><p>0044 · BENCHMARK V2 · CORE-16 · COMMON RANDOM NUMBERS</p><h1>U0 → U30 · Deck 007</h1><p>Focal: Dragapult ex · Opponent: complete immutable Policy-0809 · CUDA-2048 per checkpoint</p></div>{_art(payload['focal_representative_cards'])}</div>
<div class="summary"><div class="metric"><small>U0 · G2 initialization</small><b>{_pct(u0['summary']['win_rate'])}</b><span>{u0['summary']['wins']}-{u0['summary']['losses']}-{u0['summary']['draws']}</span></div><div class="metric"><small>U30</small><b>{_pct(u30['summary']['win_rate'])}</b><span>{u30['summary']['wins']}-{u30['summary']['losses']}-{u30['summary']['draws']}</span></div><div class="metric"><small>Net change</small><b class="up">{_pp(payload['delta'])}</b><span>{payload['net_wins']:+d} wins</span></div><div class="metric"><small>Paired flips</small><b>{transitions['loss_to_win']} ↗ / {transitions['win_to_loss']} ↘</b><span>same 2,048 seeded jobs</span></div></div>
<section><h2>结论与证据边界</h2><p>这是严格同合同的成对比较：两端均先形成完整 effective candidate，按 FP16 storage → FP32 runtime 部署，使用相同 2,048 个 exact-deck、Meta、engine、Search、policy 与 coin-winner seeds。U0 表示 0044 从 Champion-G2 初始化后的起点；它包含新增 LoRA 参数结构，但零残差初始化保证起点策略 forward 与 G2 对齐。</p><p class="note">净胜率上升不等于所有 matchup 同步上涨。请以以下 Meta 与 exact-deck 明细定位能力迁移。</p><p><a href="u0_report.json">U0 原始 PASS report</a> · <a href="u30_report.json">U30 原始 PASS report</a> · <a href="manifest.json">聚合审计 manifest</a></p></section>
<section><h2>By Meta Archetype</h2><table><thead><tr><th>Meta</th><th>成员 deck / 卡图</th><th>U0</th><th>U30</th><th>变化</th></tr></thead><tbody>{meta_rows}</tbody></table></section>
<section><h2>By opponent exact deck</h2><table><thead><tr><th>Deck</th><th>代表卡图</th><th>U0</th><th>U30</th><th>变化</th></tr></thead><tbody>{deck_rows}</tbody></table></section>
<section><h2>Identity / routing audit</h2><table>{audit_rows}</table></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def _write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def publish(*, u0_report: Path, u30_report: Path, output_root: Path = OUTPUT_ROOT) -> Path:
    target = output_root / SLUG
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    u0 = json.loads(u0_report.read_text(encoding="utf-8"))
    u30 = json.loads(u30_report.read_text(encoding="utf-8"))
    payload = aggregate(u0, u30)
    payload["source_reports"] = {
        "u0_sha256": _sha256(u0_report), "u30_sha256": _sha256(u30_report),
        "u0_archived_path": "u0_report.json", "u30_archived_path": "u30_report.json",
    }
    shutil.copyfile(u0_report, target / "u0_report.json")
    shutil.copyfile(u30_report, target / "u30_report.json")
    _write(target / "manifest.json", json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _write(target / "index.html", render(payload))
    publish_index(output_root=output_root)
    return target / "index.html"


def publish_index(*, output_root: Path = OUTPUT_ROOT) -> Path:
    rows = []
    for manifest in sorted(output_root.glob("*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows.append(
            f'<tr><td>{payload["focal_deck_id"]}</td><td>{html.escape(payload["u0"]["label"])}</td>'
            f'<td>{_pct(payload["u0"]["summary"]["win_rate"])}</td>'
            f'<td>{_pct(payload["u30"]["summary"]["win_rate"])}</td>'
            f'<td class="{"up" if payload["delta"] >= 0 else "down"}">{_pp(payload["delta"])}</td>'
            f'<td><a href="{manifest.parent.name}/index.html">打开报告</a></td></tr>'
        )
    output_root.mkdir(parents=True, exist_ok=True)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0044 Benchmark V2</title><style>body{{font:15px/1.6 system-ui;max-width:1100px;margin:auto;padding:36px;background:#edf3f0;color:#172720}}section{{padding:24px;background:#fff;border:1px solid #d4e0da}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #d4e0da;text-align:left}}a,.up{{color:#087f4f;font-weight:700}}.down{{color:#b13b3b}}</style></head><body><h1>0044 · Benchmark V2</h1><p>Core-16 Meta-balanced · complete Policy-0809 opponent · fixed common random numbers · CUDA-2048.</p><section><table><thead><tr><th>Focal deck</th><th>Baseline</th><th>U0</th><th>U30</th><th>Delta</th><th>Report</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section></body></html>'''
    _write(output_root / "index.html", page)
    return output_root / "index.html"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u0-report", required=True, type=Path)
    parser.add_argument("--u30-report", required=True, type=Path)
    args = parser.parse_args()
    print(publish(u0_report=args.u0_report.resolve(), u30_report=args.u30_report.resolve()))
