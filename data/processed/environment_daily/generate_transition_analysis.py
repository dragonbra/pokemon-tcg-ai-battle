"""Render an evidence-bounded Top 100 environment transition analysis."""

from __future__ import annotations

import argparse
from collections import Counter
from html.parser import HTMLParser
import html
import json
from pathlib import Path
import re

from .generate_live_snapshot import (
    DAILY_REPORT_ROOT,
    REPOSITORY_ROOT,
    _baseline_css,
    _card_catalog,
    _card_group,
    _card_thumb,
)


TRANSITION_REPORT = DAILY_REPORT_ROOT.parent / "environment-transition.html"
INDEX_PATH = DAILY_REPORT_ROOT.parent / "index.html"
ROSTER_DATES = ("2026-07-26", "2026-07-27", "2026-07-28")
STRICT_DATES = ("2026-07-27", "2026-07-28")


class _RosterParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, object]] = []
        self._row: dict[str, object] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "tr" and "data-player-row" in values:
            self._row = {"archetype": values.get("data-archetype") or "", "cells": []}
        elif self._row is not None and tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if self._row is not None and tag in {"td", "th"} and self._cell is not None:
            cells = self._row["cells"]
            assert isinstance(cells, list)
            cells.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _rank_value(value: object) -> int:
    match = re.search(r"#(\d+)", str(value))
    if not match:
        raise ValueError(f"missing rank: {value!r}")
    return int(match.group(1))


def _player_name(value: object) -> str:
    match = re.match(r"(.+?)Team \d+", str(value))
    if not match:
        raise ValueError(f"missing team name: {value!r}")
    return match.group(1).strip()


def _complete_snapshot(root: Path, date: str) -> dict[str, object]:
    candidates = []
    for path in (root / ".tmp/environment_daily" / date).glob("run-*/snapshot.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            candidates.append((str(payload.get("captured_at_utc") or ""), path, payload))
    if not candidates:
        raise FileNotFoundError(f"no complete environment snapshot for {date}")
    _, path, payload = max(candidates)
    payload["_path"] = str(path)
    return payload


def _legacy_roster(report: Path) -> list[dict[str, object]]:
    parser = _RosterParser()
    parser.feed(report.read_text(encoding="utf-8"))
    if len(parser.rows) != 100:
        raise ValueError(f"{report} must provide 100 player rows, found {len(parser.rows)}")
    result = []
    for row in parser.rows:
        cells = row["cells"]
        assert isinstance(cells, list)
        if len(cells) < 7:
            raise ValueError(f"{report} has a malformed player row")
        games = int(str(cells[4]).replace(",", ""))
        rate = re.search(r"(\d+(?:\.\d+)?)%", str(cells[6]))
        result.append(
            {
                "rank": _rank_value(cells[0]),
                "team_name": _player_name(cells[1]),
                "archetype": str(row["archetype"]),
                "score": float(cells[3]),
                "valid_games": games,
                "win_rate": float(rate.group(1)) / 100 if rate else None,
                "deck_sha256": None,
                "deck": [],
            }
        )
    return sorted(result, key=lambda player: int(player["rank"]))


def _audited_roster(snapshot: dict[str, object]) -> list[dict[str, object]]:
    players = snapshot.get("players")
    if not isinstance(players, dict) or len(players) != 100:
        raise ValueError("complete snapshot must contain 100 player records")
    result = []
    for rank in range(1, 101):
        raw = players.get(str(rank))
        if not isinstance(raw, dict) or len(raw.get("deck") or []) != 60:
            raise ValueError(f"snapshot rank {rank} has no exact 60-card deck")
        result.append(
            {
                "rank": rank,
                "team_name": str(raw["team_name"]),
                "archetype": str(raw["archetype"]),
                "score": float(raw["score"]),
                "valid_games": int(raw["valid_games"]),
                "win_rate": float(raw["win_rate"]),
                "deck_sha256": str(raw["deck_sha256"]),
                "deck": [int(card_id) for card_id in raw["deck"]],
            }
        )
    return result


def _distribution(players: list[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(player["archetype"]) for player in players).items()))


def _pool(players: list[dict[str, object]]) -> Counter[int]:
    result: Counter[int] = Counter()
    for player in players:
        result.update(set(int(card_id) for card_id in player["deck"]))
    return result


def _representative_ids(
    archetype: str, strict_rosters: list[list[dict[str, object]]], catalog: dict[int, dict[str, str]]
) -> list[int]:
    counts: Counter[int] = Counter()
    for roster in strict_rosters:
        for player in roster:
            if player["archetype"] == archetype:
                counts.update(set(int(card_id) for card_id in player["deck"]))
    choices = []
    for card_id, presence in counts.items():
        row = catalog.get(card_id, {})
        if _card_group(row) != "Pokémon":
            continue
        stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        priority = 3 if "Stage 2" in stage else 2 if "Stage 1" in stage else 1
        if " ex" in row.get("Card Name", "") or row.get("Card Name", "").startswith("Mega "):
            priority += 2
        choices.append((priority, presence, card_id))
    return [card_id for _, _, card_id in sorted(choices, reverse=True)[:2]]


def build_transition_data(root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    report_root = root / "docs/environment-daily_kaggle_top100/daily"
    legacy = _legacy_roster(report_root / "2026-07-26.html")
    snapshot_27 = _complete_snapshot(root, STRICT_DATES[0])
    snapshot_28 = _complete_snapshot(root, STRICT_DATES[1])
    roster_27 = _audited_roster(snapshot_27)
    roster_28 = _audited_roster(snapshot_28)
    rosters = [legacy, roster_27, roster_28]
    by_day = [{str(player["team_name"]): player for player in roster} for roster in rosters]
    shared_26_27 = set(by_day[0]) & set(by_day[1])
    shared_27_28 = set(by_day[1]) & set(by_day[2])
    all_three = shared_26_27 & set(by_day[2])
    deck_changes = []
    aggregate_added: Counter[int] = Counter()
    aggregate_removed: Counter[int] = Counter()
    for name in shared_27_28:
        old, new = by_day[1][name], by_day[2][name]
        if old["deck_sha256"] == new["deck_sha256"]:
            continue
        old_counts, new_counts = Counter(old["deck"]), Counter(new["deck"])
        added, removed = new_counts - old_counts, old_counts - new_counts
        aggregate_added.update(added)
        aggregate_removed.update(removed)
        deck_changes.append(
            {
                "team_name": name,
                "old_rank": old["rank"],
                "new_rank": new["rank"],
                "old_archetype": old["archetype"],
                "new_archetype": new["archetype"],
                "added": dict(added),
                "removed": dict(removed),
            }
        )
    deck_changes.sort(key=lambda row: (abs(int(row["old_rank"]) - int(row["new_rank"])), row["team_name"]), reverse=True)
    movers = [
        {
            "team_name": name,
            "old_rank": old["rank"],
            "new_rank": new["rank"],
            "rank_delta": int(old["rank"]) - int(new["rank"]),
            "archetype": new["archetype"],
            "deck_changed": old["deck_sha256"] != new["deck_sha256"],
        }
        for name in shared_27_28
        for old, new in [(by_day[1][name], by_day[2][name])]
    ]
    movers.sort(key=lambda row: (abs(int(row["rank_delta"])), row["team_name"]), reverse=True)
    pool_27, pool_28 = _pool(roster_27), _pool(roster_28)
    pool_delta = [
        {
            "card_id": card_id,
            "old_presence": pool_27[card_id],
            "new_presence": pool_28[card_id],
            "delta": pool_28[card_id] - pool_27[card_id],
        }
        for card_id in set(pool_27) | set(pool_28)
        if pool_27[card_id] != pool_28[card_id]
    ]
    pool_delta.sort(key=lambda row: (abs(int(row["delta"])), int(row["new_presence"])), reverse=True)
    audits = {}
    for date, snapshot in zip(STRICT_DATES, (snapshot_27, snapshot_28), strict=True):
        audit = snapshot.get("identity_audit")
        if not isinstance(audit, dict) or audit.get("audited_players") != 100:
            raise ValueError(f"{date} has no complete identity audit")
        stabilization = audit.get("stabilization") or snapshot.get("stabilization") or {}
        stable_sweeps = stabilization.get("consecutive_stable_sweeps")
        if stable_sweeps != 2:
            raise ValueError(f"{date} did not reach two stable Meta sweeps")
        audits[date] = {
            "audited_players": audit["audited_players"],
            "bounded_player_views": sum(
                int(player.get("valid_games") or 0)
                for player in (snapshot.get("players") or {}).values()
            ),
            "consecutive_stable_sweeps": stable_sweeps,
            "captured_at_utc": snapshot["captured_at_utc"],
            "source": str(snapshot["_path"]),
        }
    snapshots = [
        {
            "date": date,
            "players": len(roster),
            "archetypes": len(_distribution(roster)),
            "average_score": round(sum(float(player["score"]) for player in roster) / len(roster), 1),
            "strict": date in STRICT_DATES,
        }
        for date, roster in zip(ROSTER_DATES, rosters, strict=True)
    ]
    return {
        "snapshots": snapshots,
        "strict_comparison_dates": list(STRICT_DATES),
        "distributions": dict(zip(ROSTER_DATES, [_distribution(roster) for roster in rosters], strict=True)),
        "strict_rosters": {STRICT_DATES[0]: roster_27, STRICT_DATES[1]: roster_28},
        "continuity": {
            "shared_26_27": len(shared_26_27),
            "shared_27_28": len(shared_27_28),
            "all_three": len(all_three),
            "entered_28": sorted(set(by_day[2]) - set(by_day[1])),
            "exited_28": sorted(set(by_day[1]) - set(by_day[2])),
            "deck_changes": len(deck_changes),
        },
        "movers": movers,
        "deck_changes": deck_changes,
        "card_pool_delta": pool_delta,
        "aggregate_added": dict(aggregate_added),
        "aggregate_removed": dict(aggregate_removed),
        "audit": audits,
    }


def _archetype_visual(name: str, representatives: dict[str, list[int]], catalog: dict[int, dict[str, str]]) -> str:
    thumbs = "".join(_card_thumb(card_id, catalog, compact=True) for card_id in representatives.get(name, []))
    return (
        f'<span class="archetype-visual" title="{html.escape(name)}"><span class="archetype-thumbs">'
        f"{thumbs}</span><span class=\"archetype-label\">{html.escape(name)}</span></span>"
    )


def _index_transition_link() -> None:
    text = INDEX_PATH.read_text(encoding="utf-8")
    if "environment-transition.html" in text:
        return
    article = '''    <article class="report transition-report">
      <time datetime="2026-07-28">截至 2026-07-28</time>
      <div>
        <h3>Top 100 跨日环境变迁分析 · 0726–0728</h3>
        <p>聚合已发布日报，审计构筑、排名、卡池与选手流动；0727–0728 保持同一冻结合同。</p>
      </div>
      <a class="button" href="environment-transition.html">查看分析</a>
    </article>\n'''
    marker = '    <article class="report">'
    if marker not in text:
        raise ValueError(f"report article marker missing: {INDEX_PATH}")
    INDEX_PATH.write_text(text.replace(marker, article + marker, 1), encoding="utf-8")


def render_transition_report(data: dict[str, object], output: Path) -> None:
    catalog = _card_catalog()
    strict_rosters = data["strict_rosters"]
    assert isinstance(strict_rosters, dict)
    roster_27 = strict_rosters[STRICT_DATES[0]]
    roster_28 = strict_rosters[STRICT_DATES[1]]
    assert isinstance(roster_27, list) and isinstance(roster_28, list)
    distributions = data["distributions"]
    assert isinstance(distributions, dict)
    archetypes = sorted(set().union(*(set(value) for value in distributions.values())))
    representatives = {
        archetype: _representative_ids(archetype, [roster_27, roster_28], catalog)
        for archetype in archetypes
    }
    max_count = max(max(values.values()) for values in distributions.values())
    top_archetypes = sorted(
        archetypes,
        key=lambda name: max(int(distributions[date].get(name, 0)) for date in ROSTER_DATES),
        reverse=True,
    )[:14]
    archetype_rows = "".join(
        f'<tr><td>{_archetype_visual(name, representatives, catalog)}</td>'
        + "".join(
            f'<td><div class="day-count"><b>{int(distributions[date].get(name, 0))}</b>'
            f'<i style="--width:{100 * int(distributions[date].get(name, 0)) / max_count:.1f}%"></i></div></td>'
            for date in ROSTER_DATES
        )
        + f'<td><b class="delta {"up" if int(distributions[STRICT_DATES[1]].get(name, 0)) - int(distributions[STRICT_DATES[0]].get(name, 0)) > 0 else "down" if int(distributions[STRICT_DATES[1]].get(name, 0)) - int(distributions[STRICT_DATES[0]].get(name, 0)) < 0 else "flat"}">{int(distributions[STRICT_DATES[1]].get(name, 0)) - int(distributions[STRICT_DATES[0]].get(name, 0)):+d}</b></td></tr>'
        for name in top_archetypes
    )
    mover_rows = "".join(
        f'<tr data-mover data-search="{html.escape(str(row["team_name"]).lower())} {html.escape(str(row["archetype"]).lower())}">'
        f'<td><b>{html.escape(str(row["team_name"]))}</b></td><td>{_archetype_visual(str(row["archetype"]), representatives, catalog)}</td>'
        f'<td>#{row["old_rank"]}</td><td>#{row["new_rank"]}</td><td><b class="delta {"up" if int(row["rank_delta"]) > 0 else "down" if int(row["rank_delta"]) < 0 else "flat"}">{int(row["rank_delta"]):+d}</b></td>'
        f'<td>{"已换 exact deck" if row["deck_changed"] else "deck hash 不变"}</td></tr>'
        for row in data["movers"][:40]
    )
    changed_rows = "".join(
        f'<tr><td><b>{html.escape(str(row["team_name"]))}</b></td><td>#{row["old_rank"]} → #{row["new_rank"]}</td>'
        f'<td>{_archetype_visual(str(row["old_archetype"]), representatives, catalog)}</td><td>{_archetype_visual(str(row["new_archetype"]), representatives, catalog)}</td>'
        f'<td>{sum(int(value) for value in row["added"].values())} 张加入 / {sum(int(value) for value in row["removed"].values())} 张移除</td></tr>'
        for row in data["deck_changes"][:30]
    ) or '<tr><td colspan="5" class="muted">共同上榜选手没有 exact deck hash 变化。</td></tr>'
    def pool_row(row: dict[str, object]) -> str:
        card_id = int(row["card_id"])
        card_name = catalog.get(card_id, {}).get("Card Name", f"Card ID {card_id}")
        direction = "up" if int(row["delta"]) > 0 else "down"
        return (
            f'<tr><td><span class="pool-card">{_card_thumb(card_id, catalog, compact=False)}'
            f'<span><b>{html.escape(card_name)}</b><small>ID {card_id}</small></span></span></td>'
            f'<td>{row["old_presence"]}</td><td>{row["new_presence"]}</td>'
            f'<td><b class="delta {direction}">{int(row["delta"]):+d}</b></td></tr>'
        )

    pool_rows = "".join(pool_row(row) for row in data["card_pool_delta"][:36])
    continuity = data["continuity"]
    assert isinstance(continuity, dict)
    audits = data["audit"]
    assert isinstance(audits, dict)
    css = _baseline_css(output) + """
    .transition-hero{background:linear-gradient(135deg,#102b24,#166d55);position:relative;overflow:hidden}
    .transition-hero:after{content:"";position:absolute;right:-8%;bottom:-40%;width:52%;aspect-ratio:1;border:1px solid #a8e9cd55;border-radius:50%;box-shadow:0 0 0 45px #a8e9cd12,0 0 0 90px #a8e9cd0d}.transition-hero>*{position:relative;z-index:1}
    .transition-metrics{grid-template-columns:repeat(5,1fr)}.transition-metrics .metric{background:#ffffff12}.transition-rail{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.day-node{position:relative;padding:15px;border:1px solid var(--line);border-radius:12px;background:#f8fbf9}.day-node.strict{border-top:5px solid var(--green)}.day-node.legacy{border-top:5px solid var(--amber)}.day-node b,.day-node small{display:block}.day-node small{color:var(--muted)}
    .transition-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}.finding-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.finding-grid article{padding:14px;border-left:5px solid var(--green);border-radius:10px;background:#edf6f1}.finding-grid p{margin:5px 0 0;color:#405149}.audit-note{padding:15px 17px;border:1px solid #e5bb76;border-radius:12px;background:#fff8e9;color:#5e4218}.audit-note b{color:#7b4608}
    .transition-table{min-width:900px}.transition-table td:first-child{min-width:260px}.day-count{position:relative;min-width:120px;height:26px;padding:3px 8px;overflow:hidden;border-radius:5px;background:#f1f5f2}.day-count b{position:relative;z-index:1}.day-count i{position:absolute;inset:0 auto 0 0;width:var(--width);background:#9fd0ba}.delta{font-variant-numeric:tabular-nums}.delta.up{color:#087653}.delta.down{color:#aa413a}.delta.flat{color:#68746e}.toolbar input{min-width:250px}.table-scroll{max-height:680px}.pool-card{display:flex;align-items:center;gap:8px;min-width:190px}.pool-card>.card-thumb{flex-basis:28px;width:28px!important;height:39px!important;min-width:28px;max-width:28px!important;min-height:39px;max-height:39px!important}.pool-card b,.pool-card small{display:block}.pool-card small{color:var(--muted)}
    .card-preview{position:fixed;z-index:9999;display:none;width:250px;padding:8px;border:1px solid #bdcbc4;border-radius:10px;background:#fff;box-shadow:0 18px 55px rgba(8,26,19,.32);pointer-events:none}.card-preview.visible{display:block}.card-preview img{display:block;width:234px;height:328px;object-fit:contain;background:#eef2f0}.card-preview b{display:block;margin-top:6px;overflow:hidden;text-align:center;text-overflow:ellipsis;white-space:nowrap}.card-thumb:focus{outline:2px solid var(--green);outline-offset:2px}
    @media(max-width:900px){.transition-grid{grid-template-columns:1fr}.transition-metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:640px){.transition-rail,.finding-grid{grid-template-columns:1fr}.transition-metrics{grid-template-columns:repeat(2,1fr)}}
    """
    snapshots = data["snapshots"]
    assert isinstance(snapshots, list)
    rail = "".join(
        f'<article class="day-node {"strict" if snapshot["strict"] else "legacy"}"><b>{snapshot["date"]}</b><small>{snapshot["players"]} 名 Top 100 · {snapshot["archetypes"]} 个牌型</small><small>{"同合同完整审计" if snapshot["strict"] else "历史 roster 锚点"}</small></article>'
        for snapshot in snapshots
    )
    audit_rows = "".join(
        f'<li><a href="daily/{date}.html">{date} 日报</a>：{audit["audited_players"]}/100 身份链，{audit["bounded_player_views"]:,} 个截止点前玩家视角，稳定 sweep {audit["consecutive_stable_sweeps"]}。</li>'
        for date, audit in audits.items()
    )
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top 100 跨日环境变迁分析 · 0726–0728</title><style>{css}</style></head><body><main class="page">
<section class="hero transition-hero"><p class="eyebrow">POKEMON TCG AI BATTLE · ENVIRONMENT TRANSITION</p><h1>Top 100 环境变迁</h1><p class="lead">把已发布的 0726–0728 环境日报放到一张时间轴上。构筑与排名的严格差分只使用具有相同冻结合同的 0727 与 0728；0726 保留为 roster 历史锚点。</p><div class="metrics transition-metrics"><div class="metric"><b>3</b><span>已发布快照</span></div><div class="metric"><b>{continuity["all_three"]}</b><span>连续三日同榜选手</span></div><div class="metric"><b>{continuity["shared_27_28"]}</b><span>0727 → 0728 共同上榜</span></div><div class="metric"><b>{continuity["deck_changes"]}</b><span>共同上榜者换 exact deck</span></div><div class="metric"><b>{len(data["card_pool_delta"])}</b><span>卡池覆盖变化卡牌</span></div></div></section>
<nav class="section-nav" aria-label="跨日环境分析导航"><a href="#transition-overview">时间线</a><a href="#archetype-migration">牌型迁移</a><a href="#rank-movement">排名流动</a><a href="#deck-transition">构筑变动</a><a href="#card-pool-delta">卡池变化</a><a href="#evidence-boundary">证据边界</a></nav>
<section class="panel" id="transition-overview"><div class="heading"><div><p class="eyebrow">SNAPSHOT TIMELINE</p><h2>三日观察窗口</h2></div><p>时间线按日报冻结顺序排列；只有绿色节点之间允许 exact deck、卡池和 Meta 口径的严格对比。</p></div><div class="transition-rail">{rail}</div><div class="transition-grid"><div class="finding-grid"><article><b>连续性</b><p>0726→0727 同榜 {continuity["shared_26_27"]} 人；0727→0728 同榜 {continuity["shared_27_28"]} 人。</p></article><article><b>榜单换血</b><p>0728 新进 {len(continuity["entered_28"])} 人，退出 {len(continuity["exited_28"])} 人。</p></article><article><b>构筑更新</b><p>{continuity["deck_changes"]} 位共同上榜选手更换 exact deck hash，详见构筑变动表。</p></article><article><b>观测口径</b><p>0727–0728 的两份冻结均通过 100/100 身份链和两轮稳定 sweep。</p></article></div><div class="audit-note"><b>怎样阅读：</b>这张页面描述的是 Top 100 的环境组成与短期流动，不将离线构筑频率或滚动 Meta 胜率表述为策略强度因果证据。</div></div></section>
<section class="panel" id="archetype-migration"><div class="heading"><div><p class="eyebrow">ARCHETYPE MIGRATION</p><h2>牌型席位的三日迁移</h2></div><p>展示三日任一日进入前列的牌型；末列仅对 0727→0728 作差，避免混用 0726 的不同报告合同。</p></div><div class="table-scroll"><table class="transition-table"><thead><tr><th>牌型</th><th>0726</th><th>0727</th><th>0728</th><th>严格差分</th></tr></thead><tbody>{archetype_rows}</tbody></table></div></section>
<section class="panel" id="rank-movement"><div class="heading"><div><p class="eyebrow">ROSTER CONTINUITY</p><h2>0727 → 0728 排名流动</h2></div><p>只列共同上榜选手；正值代表排名上升。可按选手或牌型筛选，不将排名流动解释为单一 deck 或卡牌的因果结果。</p></div><div class="toolbar"><input id="mover-search" type="search" placeholder="筛选选手或牌型"><span id="mover-count" class="count-visible">显示 40</span></div><div class="table-scroll"><table class="transition-table"><thead><tr><th>选手</th><th>0728 牌型</th><th>0727</th><th>0728</th><th>名次变化</th><th>构筑</th></tr></thead><tbody>{mover_rows}</tbody></table></div></section>
<section class="panel" id="deck-transition"><div class="heading"><div><p class="eyebrow">EXACT DECK DELTA</p><h2>共同上榜者的构筑更新</h2></div><p>只比较每名选手在两个 complete 快照中的 exact 60-card deck；“加入/移除”是卡位数量，并非胜率归因。</p></div><div class="table-scroll"><table class="transition-table"><thead><tr><th>选手</th><th>排名</th><th>0727 牌型</th><th>0728 牌型</th><th>卡位变更</th></tr></thead><tbody>{changed_rows}</tbody></table></div></section>
<section class="panel" id="card-pool-delta"><div class="heading"><div><p class="eyebrow">CARD POOL DELTA</p><h2>100 份 exact deck 的覆盖变化</h2></div><p>每行统计包含该 Card ID 的 Top 100 构筑数。正负值仅是 0727 与 0728 的覆盖差，不等同于单卡强度。</p></div><div class="table-scroll"><table class="transition-table"><thead><tr><th>卡牌</th><th>0727 使用构筑</th><th>0728 使用构筑</th><th>变化</th></tr></thead><tbody>{pool_rows}</tbody></table></div></section>
<section class="panel provenance" id="evidence-boundary"><div class="heading"><div><p class="eyebrow">EVIDENCE BOUNDARY</p><h2>来源、合同与不可比较部分</h2></div></div><ul><li>来源日报：<a href="daily/2026-07-26.html">0726</a>、<a href="daily/2026-07-27.html">0727</a>、<a href="daily/2026-07-28.html">0728</a>。</li>{audit_rows}<li>0726 用于 Top 100 roster、排名与牌型的历史观察；它不进入 0727–0728 的 Meta 胜率、exact deck 或卡池差分。</li><li>0727–0728 的 exact deck、Card ID 覆盖和选手身份来自各自冻结点以前最新的 PUBLIC + COMPLETED Episode 与代表 replay；两次稳定扫描均零新增。</li><li>本页不重新抓取 Kaggle，不改变任一历史日报或其冻结边界。</li></ul></section>
</main><script type="application/json" id="transition-data">{payload}</script><script>const input=document.getElementById('mover-search'),count=document.getElementById('mover-count');function filter(){{const q=input.value.trim().toLowerCase();let n=0;document.querySelectorAll('[data-mover]').forEach(row=>{{const show=!q||row.dataset.search.includes(q);row.hidden=!show;if(show)n++;}});count.textContent=`显示 ${{n}}`;}}input.addEventListener('input',filter);filter();const preview=document.createElement('div');preview.className='card-preview';preview.innerHTML='<img alt="卡牌大图预览" width="234" height="328"><b></b>';document.body.appendChild(preview);const image=preview.querySelector('img'),name=preview.querySelector('b');let active=null;function position(e){{const w=250,h=370,m=14;let l=e.clientX+18,t=e.clientY+18;if(l+w+m>innerWidth)l=e.clientX-w-18;if(t+h+m>innerHeight)t=innerHeight-h-m;preview.style.left=`${{Math.max(m,l)}}px`;preview.style.top=`${{Math.max(m,t)}}px`;}}function show(thumb,e){{const img=thumb.querySelector('img');if(!img||img.hidden)return;active=thumb;image.src=thumb.dataset.cardPreviewUrl||img.src;name.textContent=thumb.dataset.cardName||img.alt;preview.classList.add('visible');position(e);}}function hide(){{active=null;preview.classList.remove('visible');}}document.addEventListener('pointerover',e=>{{const thumb=e.target.closest&&e.target.closest('.card-thumb');if(thumb&&thumb!==active)show(thumb,e);}});document.addEventListener('pointermove',e=>{{if(active)position(e);}});document.addEventListener('pointerout',e=>{{if(active&&!e.relatedTarget?.closest?.('.card-thumb'))hide();}});</script></body></html>''',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the audited 0726–0728 environment transition page.")
    parser.add_argument("--output", type=Path, default=TRANSITION_REPORT)
    args = parser.parse_args()
    data = build_transition_data(REPOSITORY_ROOT)
    render_transition_report(data, args.output)
    if args.output.resolve() == TRANSITION_REPORT.resolve():
        _index_transition_link()


if __name__ == "__main__":
    main()
