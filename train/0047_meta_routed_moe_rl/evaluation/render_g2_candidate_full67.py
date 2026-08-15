"""Publish the completed reverse-order Full-67 G1 versus U407 assessment."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from ..assets import AssetRegistry
from . import g2_candidate_full67 as full
from . import g2_candidate_gate as gate
from . import render_g2_candidate_gate as base


REPORT_SLUG = base.REPORT_SLUG
OUTPUT_ROOT = base.OUTPUT_ROOT
OUTPUT = OUTPUT_ROOT / "index.html"
MANIFEST = OUTPUT_ROOT / "manifest.json"
DETAIL_ROOT = OUTPUT_ROOT / "reports"
REVERSE_LINK = full.ARTIFACT_ROOT / "evaluation.json"


def aggregate() -> dict[str, Any]:
    state = json.loads(full.STATE_PATH.read_text(encoding="utf-8"))
    if state.get("status") != "EVALUATION_COMPLETE_REPORT_PENDING":
        raise RuntimeError("Full-67 evaluation is incomplete")
    registry = AssetRegistry.load(gate.PROJECT_ROOT)
    taxonomy = base._meta_taxonomy()
    names = {deck.deck_id: deck.name for deck in registry.decks}
    rows = []
    for deck_id in (f"{value:03d}" for value in range(1, 68)):
        reports = {}
        for arm in gate.ARMS:
            path = full.source_report(deck_id, arm)
            if path is None:
                raise RuntimeError(f"{deck_id}:{arm} report missing")
            report = json.loads(path.read_text(encoding="utf-8"))
            gate.validate_arm_report(report, arm=arm, deck_id=deck_id)
            reports[arm] = report
        g1, g2 = reports["g1"], reports["g2_candidate"]
        if (
            g1["focal_exact_deck_sha256"] != g2["focal_exact_deck_sha256"]
            or g1["opponent_policy_identity_audit"]["effective_policy_sha256"]
            != g2["opponent_policy_identity_audit"]["effective_policy_sha256"]
        ):
            raise RuntimeError(f"{deck_id}: Full-67 identity mismatch")
        delta = g2["summary"]["win_rate"] - g1["summary"]["win_rate"]
        rows.append({
            "deck_id": deck_id, "cohort": "full_001_067",
            "display_name": names[deck_id],
            "representative_cards": base._representative_cards(g1["focal_deck_cards"]),
            "exact_deck_sha256": g1["focal_exact_deck_sha256"],
            "g1": g1["summary"], "g2_candidate": g2["summary"],
            "g1_by_opponent": base._by_opponent(g1),
            "g2_by_opponent": base._by_opponent(g2),
            "by_meta_archetype": base._meta_comparison(g1, g2, taxonomy),
            "g1_candidate_identity": g1["candidate_deployment_identity_audit"],
            "g2_candidate_identity": g2["candidate_deployment_identity_audit"],
            "g1_schedule_sha256": g1["schedule"]["schedule_sha256"],
            "g2_schedule_sha256": g2["schedule"]["schedule_sha256"],
            "opponent_effective_sha256": g1["opponent_policy_identity_audit"]["effective_policy_sha256"],
            "delta": delta,
            "delta_95": base._difference_interval(
                g1["summary"]["wins"], g2["summary"]["wins"], 2048
            ),
            "classification": (
                "improved" if delta > base.FLAT_THRESHOLD
                else "regressed" if delta < -base.FLAT_THRESHOLD else "flat"
            ),
        })
    games = sum(row["g1"]["games"] for row in rows)
    g1_wins = sum(row["g1"]["wins"] for row in rows)
    g2_wins = sum(row["g2_candidate"]["wins"] for row in rows)
    summary = {
        "decks": 67, "games_per_arm": games,
        "g1_micro_win_rate": g1_wins / games,
        "g2_micro_win_rate": g2_wins / games,
        "g1_wilson_95": gate._wilson(g1_wins, games),
        "g2_wilson_95": gate._wilson(g2_wins, games),
        "micro_delta": (g2_wins - g1_wins) / games,
        "micro_delta_95": base._difference_interval(g1_wins, g2_wins, games),
        "macro_delta": sum(row["delta"] for row in rows) / 67,
        "improved": sum(row["classification"] == "improved" for row in rows),
        "flat": sum(row["classification"] == "flat" for row in rows),
        "regressed": sum(row["classification"] == "regressed" for row in rows),
    }
    return {
        "schema_version": "0044_u407_g2_candidate_full67_aggregate_v1",
        "status": "HUMAN_DECISION_REQUIRED", "version": full.VERSION,
        "published_at": datetime.now(UTC).isoformat(),
        "completed_decks": 67, "target_decks": 67,
        "flat_threshold": base.FLAT_THRESHOLD,
        "reporting_meta_taxonomy": taxonomy,
        "fixed_deck_ids": [row["deck_id"] for row in rows],
        "random_deck_ids": [], "summary": {
            "overall": summary, "fixed_001_010": summary,
            "seeded_random_030_067": summary,
        },
        "rows": rows, "best_improvement": max(rows, key=lambda row: row["delta"]),
        "worst_regression": min(rows, key=lambda row: row["delta"]),
    }


def render(payload: dict[str, Any]) -> str:
    page = base.render(payload)
    page = page.replace("U407 · G2 Candidate Gate", "U407 · G2 Candidate Full-67")
    page = page.replace("已完成 67/67 focal decks", "Full 001–067 · 67/67 focal decks")
    page = page.replace("Seeded 030–067", "Full 001–067")
    page = page.replace(
        "<section><h2>Cohort summary</h2>",
        "<section><h2>Full-67 summary</h2>",
    )
    return page


def publish() -> Path:
    payload = aggregate()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".html.tmp")
    temporary.write_text(render(payload), encoding="utf-8")
    os.replace(temporary, OUTPUT)
    manifest_tmp = MANIFEST.with_suffix(".json.tmp")
    manifest_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, MANIFEST)
    for row in payload["rows"]:
        target = DETAIL_ROOT / f"{row['deck_id']}.html"
        temporary = target.with_suffix(".html.tmp")
        temporary.write_text(base.render_detail(row), encoding="utf-8")
        os.replace(temporary, target)
    reverse_tmp = REVERSE_LINK.with_suffix(".json.tmp")
    reverse_tmp.write_text(json.dumps({
        "schema_version": "0044_evaluation_reverse_link_v1",
        "version": full.VERSION, "status": payload["status"],
        "authoritative_report": str(OUTPUT.relative_to(gate.ROOT)),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(reverse_tmp, REVERSE_LINK)
    return OUTPUT


if __name__ == "__main__":
    print(publish())
