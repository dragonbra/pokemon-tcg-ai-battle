"""Render the deck-069 V18 late-checkpoint and finite-EMA selection report."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .render_g2_candidate_gate import ROOT, _deck_representative_art, _meta_taxonomy
from .run_benchmark_v2 import validate_report


OUTPUT_ROOT = ROOT / "docs/evaluation/combat_mat/benchmark_v2"
SLUG = "0044_v18_deck069_u14_u17_ema_checkpoint_selection_core16_policy0809_cuda2048_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(entries)
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": games - wins - losses,
        "win_rate": wins / games if games else None,
    }


def aggregate(
    *, baseline_manifest: dict[str, Any], reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = next(
        row for row in baseline_manifest["rows"] if row["deck_id"] == "069"
    )
    expected_labels = ("U14", "U15", "U16", "U17", "EMA U14–U17")
    if tuple(reports) != expected_labels:
        raise RuntimeError(f"expected ordered candidates {expected_labels}")
    first = reports[expected_labels[0]]
    common_hash = first["schedule"]["common_random_schedule_sha256"]
    job_fields = (
        "game_id", "opponent_id", "opponent_meta_archetype_id", "engine_seed",
        "search_seed", "policy_seed", "coin_winner_seed",
    )
    for label, report in reports.items():
        validate_report(report)
        if report["focal_deck_id"] != "069" or len(report["entries"]) != 2048:
            raise RuntimeError(f"{label} is not a deck-069 CUDA-2048 report")
        if report["schedule"]["common_random_schedule_sha256"] != common_hash:
            raise RuntimeError(f"{label} uses a different common-random schedule")
        for left, right in zip(first["entries"], report["entries"], strict=True):
            if tuple(left[key] for key in job_fields) != tuple(
                right[key] for key in job_fields
            ):
                raise RuntimeError(f"{label} paired job identity mismatch")

    candidates = []
    for label, report in reports.items():
        candidates.append({
            "label": label,
            "summary": report["summary"],
            "checkpoint_update": report["focal_checkpoint_update"],
            "source_checkpoint_sha256": report["focal_policy_identity_audit"][
                "source_checkpoint_sha256"
            ],
            "deployment_effective_sha256": report[
                "focal_deployment_effective_sha256"
            ],
            "schedule_sha256": report["schedule"]["schedule_sha256"],
            "derived": label.startswith("EMA"),
        })
    best = max(candidates, key=lambda row: row["summary"]["win_rate"])

    taxonomy = _meta_taxonomy()
    classes = {int(row["archetype_id"]): row for row in taxonomy["classes"]}
    selected = list(map(int, first["schedule"]["selected_class_ids"]))
    baseline_meta = {
        int(row["archetype_id"]): row for row in baseline["by_meta_archetype"]
    }
    by_meta = []
    for class_id in selected:
        row = classes[class_id]
        values = {}
        for label, report in reports.items():
            values[label] = _stats([
                entry for entry in report["entries"]
                if int(entry["opponent_meta_archetype_id"]) == class_id
            ])
        by_meta.append({
            "archetype_id": class_id,
            "display_name": row["display_name"],
            "deck_ids": list(map(str, row["deck_ids"])),
            "champion_g3": baseline_meta[class_id]["g1"],
            "v18_u0": baseline_meta[class_id]["g2_candidate"],
            "candidates": values,
        })

    baseline_g3_decks = baseline["g1_by_opponent"]
    baseline_u0_decks = baseline["g2_by_opponent"]
    observed_decks = sorted({str(row["opponent_id"]) for row in first["entries"]})
    by_deck = []
    for deck_id in observed_decks:
        values = {}
        for label, report in reports.items():
            values[label] = _stats([
                entry for entry in report["entries"]
                if str(entry["opponent_id"]) == deck_id
            ])
        by_deck.append({
            "deck_id": deck_id,
            "champion_g3": baseline_g3_decks[deck_id],
            "v18_u0": baseline_u0_decks[deck_id],
            "candidates": values,
        })

    return {
        "schema_version": "0044_v18_late_checkpoint_selection_v1",
        "status": "PASS",
        "focal_deck_id": "069",
        "opponent_policy_id": "Policy-0809",
        "contract_id": first["schedule"]["contract_id"],
        "common_random_schedule_sha256": common_hash,
        "baseline": {
            "champion_g3": baseline["g1"],
            "v18_u0": baseline["g2_candidate"],
        },
        "candidates": candidates,
        "best_observed": {
            "label": best["label"],
            "win_rate": best["summary"]["win_rate"],
            "selection_rule": "highest raw win rate on the common 2048-game schedule",
            "promotion_decision": "none",
        },
        "ema_contract": {
            "source_updates": [14, 15, 16, 17],
            "decay": 0.5,
            "normalized_oldest_to_newest_weights": [1 / 15, 2 / 15, 4 / 15, 8 / 15],
            "not_a_true_optimizer_update": True,
        },
        "by_meta_archetype": by_meta,
        "by_opponent_deck": by_deck,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _cell(stat: dict[str, Any], *, best: bool = False) -> str:
    klass = ' class="best"' if best else ""
    return (
        f"<td{klass}><b>{_pct(stat['win_rate'])}</b>"
        f"<small>{stat['wins']}-{stat['losses']}-{stat['draws']} · n={stat['games']}</small></td>"
    )


def render(payload: dict[str, Any]) -> str:
    labels = [row["label"] for row in payload["candidates"]]
    best_label = payload["best_observed"]["label"]
    headers = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    summary_cells = "".join(
        _cell(row["summary"], best=row["label"] == best_label)
        for row in payload["candidates"]
    )
    meta_rows = "".join(
        "<tr><td><b>"
        f"{row['archetype_id']:02d} · {html.escape(row['display_name'])}</b>"
        f"<small>{', '.join(row['deck_ids'])}</small></td>"
        + _cell(row["champion_g3"])
        + _cell(row["v18_u0"])
        + "".join(_cell(row["candidates"][label]) for label in labels)
        + "</tr>"
        for row in payload["by_meta_archetype"]
    )
    art = _deck_representative_art()
    deck_rows = "".join(
        "<tr><td><b>"
        f"{row['deck_id']}</b><span class=\"art\">"
        + "".join(
            f'<img src="{html.escape(card["image_url"], quote=True)}" '
            f'alt="{html.escape(card["name"], quote=True)}" loading="lazy">'
            for card in art[row["deck_id"]]
        )
        + "</span></td>"
        + _cell(row["champion_g3"])
        + _cell(row["v18_u0"])
        + "".join(_cell(row["candidates"][label]) for label in labels)
        + "</tr>"
        for row in payload["by_opponent_deck"]
    )
    summary = payload["baseline"]
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    style = """
:root{--bg:#edf3f0;--paper:#fff;--ink:#172720;--muted:#65766f;--line:#d4e0da;--brand:#17694a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}main{max-width:1550px;margin:auto;padding:28px}.hero{padding:28px;border-radius:12px;background:#153e30;color:#fff}.hero h1{margin:4px 0}.hero p{margin:0;color:#cfe3da}section{margin-top:18px;padding:20px;background:#fff;border:1px solid var(--line);overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left}th{position:sticky;top:0;background:#edf3f0}.best{background:#dff5e8;color:#086b42}small{display:block;color:var(--muted)}.art{display:inline-flex;margin-left:8px;vertical-align:middle}.art img{width:28px;height:39px;object-fit:cover;border-radius:3px;margin-right:-5px;border:1px solid #c8d5cf}.note{padding:12px;border-left:4px solid #d89a26;background:#fff8e8}a{color:var(--brand);font-weight:700}code{overflow-wrap:anywhere}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0044 V18 late checkpoint selection</title><style>{style}</style></head><body><main>
<div class="hero"><p>0044 · BENCHMARK V2 · POLICY-0809 · CUDA-2048 · DECK 069</p><h1>V18 U14–U17 + EMA checkpoint 体检</h1><p>所有候选共享同一组 2,048 个 Meta / exact-deck / engine / search / policy / coin seeds。</p></div>
<section><h2>总体结果</h2><table><thead><tr><th>Champion-G3</th><th>V18 U0</th>{headers}</tr></thead><tbody><tr>{_cell(summary['champion_g3'])}{_cell(summary['v18_u0'])}{summary_cells}</tr></tbody></table><p class="note">当前 raw 最优：<b>{html.escape(best_label)}</b>（{_pct(payload['best_observed']['win_rate'])}）。这只是同合同 checkpoint 选择，不构成 Champion promote。EMA 是 U14–U17 的有限归一化参数平均，不是一个真实 optimizer update。</p><p><a href="manifest.json">审计 manifest</a> · <code>{payload['common_random_schedule_sha256']}</code></p></section>
<section><h2>按 Meta Archetype</h2><table><thead><tr><th>Meta / 成员 decks</th><th>Champion-G3</th><th>V18 U0</th>{headers}</tr></thead><tbody>{meta_rows}</tbody></table></section>
<section><h2>按对手 exact deck</h2><table><thead><tr><th>Deck / 卡图</th><th>Champion-G3</th><th>V18 U0</th>{headers}</tr></thead><tbody>{deck_rows}</tbody></table></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def publish(
    *, baseline_manifest: Path, report_paths: dict[str, Path],
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    target = output_root / SLUG
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    reports = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in report_paths.items()
    }
    payload = aggregate(
        baseline_manifest=json.loads(baseline_manifest.read_text(encoding="utf-8")),
        reports=reports,
    )
    payload["source_reports"] = {}
    for label, path in report_paths.items():
        archived = f"{label.lower().replace(' ', '_').replace('–', '_')}_report.json"
        shutil.copyfile(path, target / archived)
        payload["source_reports"][label] = {
            "sha256": _sha256(path), "archived_path": archived,
        }
    _write(target / "manifest.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _write(target / "index.html", render(payload))
    return target / "index.html"

