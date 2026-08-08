"""Render one 0037 seeded-512 checkpoint in the Frozen-0806 report style."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import html
import json
from pathlib import Path
from typing import Any

from evaluation.cards import card_image_url, load_card_catalog
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog


ROOT = Path(__file__).resolve().parents[2]
FOCAL_DECK_ID = "dragapult_ex_07bedfffbfad"
FOCAL_DECK = (
    ROOT / "train/0037_dragapult_value_initialized_rl/league/decks"
    / FOCAL_DECK_ID / "deck.csv"
)


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _result_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    games = payload.get("games") or []
    if (
        payload.get("official_engine") is not True
        or payload.get("mode") != "greedy"
        or len(games) != 512
        or any(not game.get("valid") or game.get("error") for game in games)
    ):
        raise ValueError(f"not a complete official seeded-512 result: {path}")
    return payload


def _catalog_rows() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    rows: dict[str, dict[str, Any]] = {}
    for opponent in catalog.opponents:
        manifest = opponent.package_manifest or {}
        rows[opponent.name] = {
            "number": str(manifest["frozen_deck_number"]),
            "display_name": opponent.display_name,
            "representative_cards": list(opponent.representative_cards),
        }
    if len(rows) != 55 or {row["number"] for row in rows.values()} != {
        f"{number:03d}" for number in range(1, 56)
    }:
        raise RuntimeError("Frozen-0806 presentation numbering is not exactly 001-055")
    focal = rows.get(FOCAL_DECK_ID)
    if focal is None or focal["number"] != "007":
        raise RuntimeError("Frozen-0806 focal Dragapult deck is not deck 007")
    return rows, catalog


def _deck_cards() -> list[dict[str, Any]]:
    catalog = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    counts = Counter(int(line) for line in FOCAL_DECK.read_text().splitlines())
    if sum(counts.values()) != 60:
        raise ValueError("0037 focal deck is not exact 60")
    cards = []
    for card_id, count in counts.items():
        metadata = catalog[card_id]
        kind = str(metadata["stage_or_type"])
        category = "pokemon" if "Pokémon" in kind else "energy" if "Energy" in kind else "trainer"
        cards.append({
            "card_id": card_id,
            "name": metadata["name"],
            "expansion": metadata["expansion"],
            "collection_number": metadata["collection_number"],
            "stage_or_type": kind,
            "category": category,
            "count": count,
            "image_url": card_image_url(
                metadata["expansion"], metadata["collection_number"]
            ),
        })
    category_order = {"pokemon": 0, "trainer": 1, "energy": 2}
    return sorted(cards, key=lambda card: (category_order[card["category"]], card["name"], card["card_id"]))


def _card_groups(cards: list[dict[str, Any]]) -> str:
    groups = []
    for category, label in (("pokemon", "Pokémon"), ("trainer", "Trainer"), ("energy", "Energy")):
        entries = [card for card in cards if card["category"] == category]
        total = sum(card["count"] for card in entries)
        body = "".join(
            '<div class="deck-entry">'
            f'<img src="{_text(card["image_url"])}" alt="{_text(card["name"])}" loading="lazy" onerror="this.hidden=true">'
            f'<span><b>{_text(card["name"])}</b><small>{_text(card["expansion"])} {_text(card["collection_number"])} · {_text(card["stage_or_type"])}</small></span>'
            f'<strong>×{card["count"]}</strong></div>'
            for card in entries
        )
        groups.append(
            f'<div class="deck-group"><h3>{label}<span>{total} 张</span></h3>{body}</div>'
        )
    return "".join(groups)


def render(result: dict[str, Any], *, label: str, output: Path) -> dict[str, Any]:
    catalog, runtime = _catalog_rows()
    cards = _deck_cards()
    by_opponent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in result["games"]:
        if game["opponent_id"] not in catalog:
            raise ValueError(f"result contains an unknown Frozen-0806 deck: {game['opponent_id']}")
        by_opponent[game["opponent_id"]].append(game)
    if set(by_opponent) != set(catalog):
        raise ValueError("result does not cover all 55 Frozen-0806 decks")

    matchups = []
    for deck_id, games in by_opponent.items():
        identity = catalog[deck_id]
        first = [game for game in games if game["focal_first"]]
        second = [game for game in games if not game["focal_first"]]
        wins = sum(game["reward"] == 1.0 for game in games)
        losses = sum(game["reward"] == -1.0 for game in games)
        draws = len(games) - wins - losses
        matchups.append({
            **identity,
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(games),
            "first_games": len(first),
            "first_wins": sum(game["reward"] == 1.0 for game in first),
            "second_games": len(second),
            "second_wins": sum(game["reward"] == 1.0 for game in second),
        })
    matchups.sort(key=lambda row: int(row["number"]))
    if sum(row["games"] for row in matchups) != 512:
        raise RuntimeError("matchup aggregation changed the game count")

    metrics = result["metrics"]
    wins = int(metrics["eval/wins"])
    losses = int(metrics["eval/losses"])
    draws = int(metrics["eval/draws"])
    first_wins = round(float(metrics["eval/first_win_rate"]) * 256)
    second_wins = round(float(metrics["eval/second_win_rate"]) * 256)
    checkpoint_update = int(result["checkpoint_identity"]["checkpoint_update"])
    run_id = f"run-20260808-v5-checkpoint-{checkpoint_update:06d}-seeded512"
    created_at = datetime.now(UTC).isoformat()

    matchup_rows = "".join(
        "<tr>"
        f'<td class="number">{row["number"]}</td>'
        '<td><div class="opponent">'
        + '<span class="art">' + "".join(
            f'<img src="{_text(card["image_url"])}" alt="{_text(card["name"])}" loading="lazy" onerror="this.hidden=true">'
            for card in row["representative_cards"]
        ) + '</span>'
        f'<strong>{_text(row["display_name"])}</strong></div></td>'
        f'<td>{row["games"]}</td><td>{row["wins"]}</td><td>{row["losses"]}</td><td>{row["draws"]}</td>'
        f'<td><b>{_percent(row["win_rate"])}</b><span class="bar"><i style="width:{100*row["win_rate"]:.2f}%"></i></span></td>'
        f'<td>{row["first_wins"]}/{row["first_games"]}</td><td>{row["second_wins"]}/{row["second_games"]}</td>'
        "</tr>"
        for row in matchups
    )
    representative = next(row for row in catalog.values() if row["number"] == "007")
    representative_images = "".join(
        f'<img src="{_text(card["image_url"])}" alt="{_text(card["name"])}" onerror="this.hidden=true">'
        for card in representative["representative_cards"]
    )
    embedded = json.dumps({
        "manifest": {
            "run_id": run_id,
            "candidate": {"name": f"V5 checkpoint {checkpoint_update}", "display_name": f"007 · Dragapult ex · {label}"},
            "finished_at": created_at,
        },
        "summary": {
            "total_games": 512,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "errors": 0,
            "unfinished": 0,
            "win_rate": wins / 512,
            "completion_rate": 1.0,
        },
    }, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>007 · Dragapult ex · {_text(label)}</title><style>
:root{{--bg:#f3f7f5;--paper:#fff;--soft:#f7faf8;--ink:#172b25;--muted:#60736c;--line:#dce7e2;--green:#217a58;--dark:#14563d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"PingFang SC",sans-serif}}main{{max-width:1440px;margin:auto;padding:30px 24px 60px}}
header{{display:flex;justify-content:space-between;align-items:end;gap:24px;padding:26px 30px;background:var(--dark);color:#fff;border-radius:8px}}header p{{margin:4px 0 0;color:#c8eadb}}h1{{margin:0;font-size:30px}}code{{font-size:11px}}section{{margin-top:18px;padding:22px;background:var(--paper);border:1px solid var(--line);border-radius:8px}}
.candidate{{display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center}}.candidate h2{{margin:0;font-size:25px}}.candidate p{{color:var(--muted)}}.representatives img{{width:105px;border:1px solid var(--line);border-radius:6px}}
.deck-groups{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;margin-top:20px}}.deck-group h3{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px}}.deck-group h3 span{{color:var(--muted);font-size:12px}}
.deck-entry{{display:grid;grid-template-columns:40px 1fr auto;gap:9px;align-items:center;min-height:56px;border-bottom:1px solid #edf2ef;padding:4px 0}}.deck-entry img{{width:38px;height:52px;object-fit:cover;border-radius:3px}}small{{display:block;color:var(--muted);font-size:10px}}
.stats{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.stat{{padding:13px 15px;background:var(--soft);border-radius:7px}}.stat b{{display:block;font-size:22px}}.stat span{{color:var(--muted);font-size:12px}}
.tools{{margin:18px 0 10px}}input{{width:min(440px,100%);padding:9px 11px;border:1px solid #b9c9c1;border-radius:5px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:7px}}
table{{width:100%;min-width:1050px;border-collapse:collapse;background:#fff}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#e9f0ec;color:#486158;font-size:12px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f8fbf9}}
.number{{font-size:16px;font-weight:850;color:var(--green)}}.opponent{{display:flex;align-items:center;gap:10px}}.art{{display:flex;width:66px}}.art img{{width:37px;height:51px;margin-right:-8px;object-fit:cover;border:1px solid #c9d5cf;border-radius:3px}}.bar{{display:block;width:90px;height:6px;margin-top:4px;background:#e4ece8;border-radius:9px}}.bar i{{display:block;height:100%;background:var(--green);border-radius:9px}}
@media(max-width:800px){{.deck-groups{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr 1fr}}header{{align-items:start;flex-direction:column}}}}
</style></head><body><main>
<header><div><p>POKÉMON TCG · 0037 FROZEN-0806</p><h1>007 · Dragapult ex · {_text(label)}</h1><p>Official engine · Greedy · 固定 512 局 · 256 个 seed pair 交换先后手</p></div><code>{run_id}</code></header>
<section><div class="candidate"><div><h2>我的卡组：007 · Dragapult ex</h2><p>Exact 60 cards · Frozen-0806 编号与 Policy-0806 报告完全一致</p></div><div class="representatives">{representative_images}</div></div><div class="deck-groups">{_card_groups(cards)}</div></section>
<section><div class="stats"><div class="stat"><b>512</b><span>实际对局</span></div><div class="stat"><b>{wins}</b><span>胜场</span></div><div class="stat"><b>{losses}</b><span>负场</span></div><div class="stat"><b>{_percent(wins/512)}</b><span>总胜率</span></div><div class="stat"><b>{first_wins}/256</b><span>先手胜场</span></div><div class="stat"><b>{second_wins}/256</b><span>后手胜场</span></div></div>
<div class="tools"><input id="search" type="search" placeholder="筛选 001–055 编号或牌型"></div><div class="table"><table><thead><tr><th>编号</th><th>Frozen-0806 卡组</th><th>对局</th><th>胜</th><th>负</th><th>平</th><th>胜率</th><th>先手胜/局</th><th>后手胜/局</th></tr></thead><tbody>{matchup_rows}</tbody></table></div></section>
<script id="report-data" type="application/json">{embedded}</script><script>const q=document.querySelector('#search');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();for(const r of document.querySelector('tbody').rows)r.hidden=!r.innerText.toLowerCase().includes(s)}});</script>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return {"label": label, "update": checkpoint_update, "wins": wins, "losses": losses, "matchups": matchups, "run_id": run_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = render(_result_payload(args.result.resolve()), label=args.label, output=args.output.resolve())
    print(json.dumps({key: summary[key] for key in ("label", "update", "wins", "losses", "run_id")}, indent=2))


if __name__ == "__main__":
    main()
