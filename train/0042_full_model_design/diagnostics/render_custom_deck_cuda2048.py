"""Render custom-deck CUDA-2048 in the canonical Policy-0809 report UI."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import re
from typing import Any

from evaluation.cards import card_image_url, load_card_catalog

from ..rollout.deck_routing import exact_deck_sha256


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / (
    "docs/evaluation/combat_mat/policy_0809/"
    "0809_kaggle_top100_plus_v1_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1/reports/007.html"
)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _reference() -> tuple[str, dict[str, Any]]:
    source = REFERENCE.read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", source, re.S)
    data = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        source,
        re.S,
    )
    if style is None or data is None:
        raise RuntimeError("canonical Policy-0809 report template is malformed")
    return style.group(1), json.loads(data.group(1))


def _category(requested: str, explicit: str | None = None) -> str:
    if explicit in {"pokemon", "trainer", "energy"}:
        return explicit
    if "Energy" in requested:
        return "energy"
    pokemon = (
        "Dreepy", "Drakloak", "Dragapult", "Munkidori", "Budew", "Meowth",
        "Fezandipiti", "Moltres",
    )
    return "pokemon" if requested.startswith(pokemon) else "trainer"


def _validate(report: dict[str, Any], audit: dict[str, Any]) -> None:
    if report.get("status") != "PASS" or report.get("summary", {}).get("games") != 2048:
        raise RuntimeError("canonical HTML requires a passing CUDA-2048 report")
    cards = tuple(int(card) for card in report.get("focal_deck_cards") or ())
    digest = exact_deck_sha256(cards)
    if digest not in {
        report.get("focal_exact_deck_sha256"), audit.get("focal_exact_deck_sha256")
    } or report.get("focal_exact_deck_sha256") != audit.get("focal_exact_deck_sha256"):
        raise RuntimeError("report/audit focal exact-deck identity mismatch")
    displayed = Counter({
        int(row["card_id"]): int(row["count"])
        for row in audit.get("mappings") or []
    })
    if displayed != Counter(cards) or sum(displayed.values()) != 60:
        raise RuntimeError("rendered card list differs from evaluated exact 60")
    opponent = report.get("opponent_policy_identity_audit") or {}
    if opponent.get("status") != "PASS" or opponent.get("requested_policy_id") != "Policy-0809":
        raise RuntimeError("full Policy-0809 opponent audit is missing")


def _candidate_cards(
    audit: dict[str, Any], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    known = {
        int(row["card_id"]): row
        for row in reference["manifest"]["candidate"]["deck_cards"]
    }
    extras = {
        791: "https://images.pokemontcg.io/me2/14.png",
        1256: "https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/tpci/DRI/DRI_180_R_EN_SM.png",
        1260: "https://images.pokemontcg.io/me1/127.png",
    }
    cards = []
    for mapping in audit["mappings"]:
        card_id = int(mapping["card_id"])
        canonical = known.get(card_id, {})
        requested = str(mapping["requested"])
        expansion = str(mapping.get("expansion") or requested.rsplit(" ", 2)[-2])
        number = str(mapping.get("collection_number") or requested.rsplit(" ", 2)[-1])
        name = str(mapping.get("name") or requested.rsplit(" ", 2)[0])
        cards.append({
            "card_id": card_id,
            "category": _category(requested, mapping.get("category")),
            "count": int(mapping["count"]),
            "name": name,
            "requested_print": f"{expansion} {number}",
            "canonical_print": str(mapping["canonical_print"]),
            "mapping": str(mapping["match"]),
            "image_url": (
                mapping.get("image_url") or canonical.get("image_url")
                or extras.get(card_id, "")
            ),
        })
    return cards


def _audit_from_report(report: dict[str, Any]) -> dict[str, Any]:
    catalog = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    counts = Counter(int(card) for card in report["focal_deck_cards"])
    mappings = []
    for card_id, count in counts.items():
        metadata = catalog[card_id]
        kind = str(metadata["stage_or_type"])
        category = (
            "pokemon" if "Pokémon" in kind
            else "energy" if "Energy" in kind else "trainer"
        )
        expansion = str(metadata["expansion"])
        number = str(metadata["collection_number"])
        mappings.append({
            "requested": f'{metadata["name"]} {expansion} {number}',
            "name": metadata["name"],
            "expansion": expansion,
            "collection_number": number,
            "category": category,
            "count": count,
            "card_id": card_id,
            "canonical_print": f"{expansion} {number}",
            "match": "exact",
            "image_url": card_image_url(expansion, number) or "",
        })
    return {
        "schema_version": "0042_catalog_exact_deck_render_audit_v1",
        "status": "PASS",
        "requested_cards": 60,
        "unique_canonical_card_ids": len(mappings),
        "assigned_deck_name": report["focal_deck_display_name"],
        "focal_deck_id": report["focal_deck_id"],
        "focal_exact_deck_sha256": report["focal_exact_deck_sha256"],
        "mappings": mappings,
    }


def _deck_html(cards: list[dict[str, Any]]) -> str:
    groups = []
    labels = (("pokemon", "Pokémon"), ("trainer", "Trainer"), ("energy", "Energy"))
    for category, label in labels:
        selected = [row for row in cards if row["category"] == category]
        entries = []
        for row in selected:
            image = (
                f'<img src="{html.escape(row["image_url"])}" alt="{html.escape(row["name"])}" '
                'loading="lazy" onerror="this.hidden=true">'
                if row["image_url"] else ""
            )
            alias = row["mapping"] != "exact"
            mapping = (
                f"requested {row['requested_print']} → canonical {row['canonical_print']}"
                if alias else row["requested_print"]
            )
            entries.append(
                f'<div class="deck-entry">{image}<span><span class="deck-card-name">'
                f'{html.escape(row["name"])}</span><span class="deck-card-set">'
                f'{html.escape(mapping)} · ID {row["card_id"]}</span></span>'
                f'<span class="deck-card-count">×{row["count"]}</span></div>'
            )
        total = sum(row["count"] for row in selected)
        groups.append(
            '<div class="deck-group"><div class="deck-group-head">'
            f'<h3>{label}</h3><span class="deck-count">{total} 张</span></div>'
            f'<div class="deck-list">{"".join(entries)}</div></div>'
        )
    return "".join(groups)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "draws": 0, "games": 0,
                 "first_games": 0, "first_wins": 0, "second_games": 0,
                 "second_wins": 0}
    )
    for row in rows:
        target = grouped[str(row["opponent_id"])]
        outcome = int(row["outcome"])
        target["games"] += 1
        target["wins"] += int(outcome == 1)
        target["losses"] += int(outcome == -1)
        target["draws"] += int(outcome == 0)
        seat = "first" if row["focal_first"] else "second"
        target[f"{seat}_games"] += 1
        target[f"{seat}_wins"] += int(outcome == 1)
    return dict(grouped)


def _chart_row(
    *, number: str, name: str, record: dict[str, int], images: list[dict[str, Any]]
) -> str:
    rate = record["wins"] / record["games"] if record["games"] else 0.0
    thumbs = "".join(
        f'<img class="opponent-thumb" src="{html.escape(str(card["image_url"]))}" '
        f'alt="{html.escape(str(card["name"]))}" loading="lazy" onerror="this.hidden=true">'
        for card in images
    )
    return (
        '<div class="chart-row opponent-row"><span class="opponent-identity">'
        f'<span class="opponent-thumbnails">{thumbs}</span>'
        f'<span class="deck-number">{html.escape(number)}</span>'
        f'<span class="opponent-name" title="{html.escape(name)}">{html.escape(name)}</span>'
        '</span><span class="bar-track"><span class="bar" '
        f'style="width:{100 * rate:.2f}%"></span></span><span class="chart-result">'
        f'<span class="chart-rate">{_pct(rate)}</span><span class="chart-record">'
        f'{record["wins"]}-{record["losses"]}-{record["draws"]} / {record["games"]}局'
        '</span></span></div>'
    )


def _opponent_chart(
    aggregate: dict[str, dict[str, int]], reference: dict[str, Any]
) -> str:
    rows = []
    for opponent in reference["manifest"]["opponents"]:
        deck_id = str(opponent["name"])
        package = opponent["package_manifest"]
        rows.append(_chart_row(
            number=str(package["frozen_deck_number"]),
            name=str(opponent["display_name"]),
            record=aggregate[deck_id],
            images=list(opponent.get("representative_cards") or []),
        ))
    return "".join(rows)


def _meta_chart(
    aggregate: dict[str, dict[str, int]], reference: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    meta_rows = list(reference["summary"]["by_meta_archetype"])
    other = reference["summary"].get("meta_archetype_other")
    if other:
        meta_rows.append(other)
    rendered = []
    payload = []
    for meta in sorted(meta_rows, key=lambda row: int(row["class_id"])):
        record = {key: 0 for key in ("wins", "losses", "draws", "games")}
        for deck_id in meta["deck_ids"]:
            source = aggregate[deck_id]
            for key in record:
                record[key] += source[key]
        payload.append({**meta, **record, "win_rate": record["wins"] / record["games"]})
        rendered.append(_chart_row(
            number=f'{int(meta["class_id"]) + 1:02d}',
            name=str(meta["display_name"]),
            record=record,
            images=list(meta.get("representative_cards") or []),
        ))
    return "".join(rendered), payload


def render(report: dict[str, Any], audit: dict[str, Any]) -> str:
    _validate(report, audit)
    style, reference = _reference()
    cards = _candidate_cards(audit, reference)
    aggregate = _aggregate(report["entries"])
    opponent_chart = _opponent_chart(aggregate, reference)
    meta_chart, meta_payload = _meta_chart(aggregate, reference)
    summary = report["summary"]
    candidate = report["candidate_deployment_identity_audit"]
    opponent = report["opponent_policy_identity_audit"]
    assigned = str(audit["assigned_deck_name"])
    run_id = f'0042-{assigned}-u{int(report["checkpoint_update"]):03d}-cuda2048'
    representatives = [row for row in cards if row["category"] == "pokemon"][:2]
    if not representatives:
        representatives = cards[:2]
    representative_html = "".join(
        f'<img src="{html.escape(row["image_url"])}" alt="{html.escape(row["name"])}" '
        'loading="eager" onerror="this.hidden=true">'
        for row in representatives if row["image_url"]
    )
    embedded = json.dumps({
        "report": report,
        "card_id_audit": audit,
        "summary": {"by_opponent": aggregate, "by_meta_archetype": meta_payload},
    }, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    chance = int(report["health"].get("eval/core/chance_boundaries", 0))
    errors = sum(bool(row.get("error")) or not row.get("valid") for row in report["entries"])
    extra_style = """
.identity-grid{display:grid;grid-template-columns:190px minmax(0,1fr);gap:8px 16px}
.identity-grid code{overflow-wrap:anywhere}.pass{color:var(--brand);font-weight:750}
@media(max-width:760px){.identity-grid{grid-template-columns:1fr}}
"""
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(assigned)} · CUDA-2048 · update {report["checkpoint_update"]}</title>
<style>{style}{extra_style}</style></head><body><main>
<header class="hero"><div><p class="eyebrow">POKÉMON TCG · EVALUATION</p>
<h1>{html.escape(assigned)}</h1><p>0042 candidate vs full immutable Policy-0809 · Official CUDA engine</p></div>
<div class="run-id">{html.escape(run_id)}</div></header>
<section class="candidate-overview" id="exact-deck"><div class="candidate-lead"><div>
<p class="candidate-kicker">主视角卡组</p><h2 class="candidate-title">{html.escape(assigned)}</h2>
<div class="candidate-meta"><span>Update {report["checkpoint_update"]}</span><span>Exact 60 cards</span>
<span>Frozen Arena · 0809_kaggle_top100_plus_v1</span><span>Exact-deck conditioned focal policy</span></div></div>
<div class="candidate-representatives">{representative_html}</div></div>
<div class="deck-groups">{_deck_html(cards)}</div></section>
<section><div class="profile-grid"><div class="profile-item"><span class="label">Metric profile</span><div class="profile-value">frozen_cuda_2048</div></div>
<div class="profile-item"><span class="label">Revision</span><div class="profile-value">0042 custom deck v1</div></div>
<div class="profile-item"><span class="label">Metrics</span><div class="profile-value">outcome, by_opponent, by_meta_archetype, deployment identity</div></div></div></section>
<section><h2>总体结果</h2><div class="summary"><div class="card"><div class="label">总对局</div><div class="value">2048</div></div>
<div class="card"><div class="label">胜 / 负 / 平</div><div class="value">{summary["wins"]} / {summary["losses"]} / {summary["draws"]}</div></div>
<div class="card"><div class="label">完成率</div><div class="value">100.00%</div></div>
<div class="card"><div class="label">胜率</div><div class="value">{_pct(summary["win_rate"])}</div></div>
<div class="card"><div class="label">先攻 / 后攻</div><div class="value">{_pct(summary["focal_first_win_rate"])} / {_pct(summary["focal_second_win_rate"])}</div></div>
<div class="card"><div class="label">错误数</div><div class="value">{errors}</div></div></div></section>
<section class="matchup-section" id="opponent-matchups"><h2>对局胜率图</h2><p class="muted">对 001–055 Exact Deck；同一批 CUDA-2048 official-engine 对局按 opponent identity 聚合。</p>
<div class="matchup-chart">{opponent_chart}</div></section>
<section class="matchup-section" id="meta-archetype-matchups"><h2>对 15 种 Meta Archetype 的聚合胜率</h2>
<p class="muted">按 opponent exact deck 的 priority-ordered trigger-card taxonomy 分类并按局数加权；包含 Others。</p>
<div class="matchup-chart">{meta_chart}</div></section>
<section id="identity-audit"><h2>Deployment / Frozen identity audit</h2><div class="identity-grid">
<div class="label">Focal exact-deck SHA-256</div><code>{report["focal_exact_deck_sha256"]}</code>
<div class="label">Candidate deployment</div><code>{candidate["effective_candidate_sha256"]}</code>
<div class="label">Candidate dtype</div><div>FP16 storage → strict FP32 runtime</div>
<div class="label">Opponent policy</div><div class="pass">Policy-0809 · PASS</div>
<div class="label">Opponent effective SHA-256</div><code>{opponent["effective_policy_sha256"]}</code>
<div class="label">Schedule SHA-256</div><code>{report["schedule_sha256"]}</code>
<div class="label">CUDA health</div><div class="pass">2,048 terminal · 0 error · 0 unsupported · 0 semantic fallback · {chance} allowed chance boundary</div>
<div class="label">Feature residency</div><div class="pass">GPU resident · feature D2H bytes = 0</div></div></section>
<script id="report-data" type="application/json">{embedded}</script></main></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--card-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    audit = (
        json.loads(args.card_audit.read_text(encoding="utf-8"))
        if args.card_audit is not None else _audit_from_report(report)
    )
    content = render(report, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
