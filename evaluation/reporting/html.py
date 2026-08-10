from __future__ import annotations

import html as html_module
import json
from collections.abc import Mapping

from .models import (
    ReportData,
    as_mapping,
    control_differences,
    display_value,
    json_ready,
    ordered_mapping,
    percentage,
)


REPORT_STYLES = """
:root{
  color-scheme:light;
  --bg:#f3f7f5;
  --surface:#ffffff;
  --surface-soft:#f7faf8;
  --ink:#172b25;
  --muted:#60736c;
  --line:#dce7e2;
  --brand:#217a58;
  --brand-dark:#14563d;
  --brand-soft:#e4f3ec;
  --shadow:0 12px 32px rgba(26,71,55,.08);
}
*{box-sizing:border-box}
body{
  margin:0;
  background:var(--bg);
  color:var(--ink);
  font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;
}
main{max-width:1240px;margin:0 auto;padding:36px 28px 64px}
.hero{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:24px;
  margin-bottom:22px;
  padding:30px 32px;
  border-radius:8px;
  color:#fff;
  background:var(--brand-dark);
}
.eyebrow{margin:0 0 6px;color:#c8eadb;font-size:12px;font-weight:700;letter-spacing:.12em}
.back-link{display:inline-block;margin-bottom:8px;color:#d8eee5;font-size:12px;font-weight:700;text-decoration:none}.back-link:hover{text-decoration:underline}
h1{margin:0;font-size:32px;line-height:1.2;letter-spacing:0}
h2{margin:0 0 6px;font-size:20px;line-height:1.35;letter-spacing:0}
h3{margin:0 0 10px}
.run-id{
  max-width:48%;
  padding:8px 12px;
  border:1px solid rgba(255,255,255,.22);
  border-radius:6px;
  background:rgba(255,255,255,.1);
  color:#eaf7f1;
  font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;
  overflow-wrap:anywhere;
}
section{
  margin:18px 0;
  padding:22px;
  overflow-x:auto;
  border:1px solid var(--line);
  border-radius:8px;
  background:rgba(255,255,255,.96);
}
.profile-grid{display:grid;grid-template-columns:1fr 110px 2fr;gap:12px}
.profile-item{padding:12px 14px;border-radius:10px;background:var(--surface-soft)}
.profile-item .label{display:block;margin-bottom:3px}
.profile-value{font-weight:650;overflow-wrap:anywhere}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{
  min-height:92px;
  padding:15px 16px;
  border:1px solid var(--line);
  border-radius:11px;
  background:#fff;
}
.label{color:var(--muted);font-size:12px;font-weight:600;letter-spacing:.02em}
.value{margin-top:5px;font-size:23px;font-weight:750;letter-spacing:-.02em}
.matchup-section{padding:18px 20px}
.matchup-chart{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:2px 20px;
  margin-top:8px;
  font-size:12px;
}
.chart-row{
  display:grid;
  grid-template-columns:minmax(210px,270px) 1fr 150px;
  gap:8px;
  align-items:center;
  padding:3px 5px;
  border-radius:6px;
}
.chart-row:hover{background:var(--surface-soft)}
.opponent-identity{display:flex;align-items:center;min-width:0;gap:7px}
.opponent-link{color:inherit;text-decoration:none}.opponent-link:hover .opponent-name{color:var(--brand);text-decoration:underline}
.deck-number{display:inline-flex;align-items:center;justify-content:center;min-width:34px;padding:2px 5px;border:1px solid #b8cec4;border-radius:4px;background:#edf6f1;color:var(--brand-dark);font-weight:800;font-variant-numeric:tabular-nums}
.opponent-thumbnails{display:flex;flex:0 0 auto;padding-left:3px}
.opponent-thumb{
  width:29px;
  height:38px;
  margin-left:-3px;
  object-fit:cover;
  border:1px solid rgba(23,43,37,.2);
  border-radius:4px;
  background:#e6eee9;
  box-shadow:0 2px 5px rgba(23,43,37,.12);
}
.opponent-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chart-result{text-align:right;font-variant-numeric:tabular-nums}.chart-rate{font-weight:800}.chart-record{margin-left:5px;color:var(--muted);font-size:11px;white-space:nowrap}
.bar-track{height:7px;overflow:hidden;border-radius:999px;background:#e4ece8}
.bar{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#2a8d65,#54b184)}
table{width:100%;min-width:820px;margin-top:14px;border-collapse:separate;border-spacing:0}
th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{
  background:var(--surface-soft);
  color:#496159;
  font-size:12px;
  font-weight:700;
  letter-spacing:.02em;
  white-space:nowrap;
}
th:first-child{border-radius:9px 0 0 9px}
th:last-child{border-radius:0 9px 9px 0}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:#fbfdfc}
td:first-child{font-weight:600}
.muted{margin:0;color:var(--muted)}
.metric-id{
  color:#286d54;
  background:transparent;
  font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;
}
.tag{
  display:inline-flex;
  align-items:center;
  padding:3px 8px;
  border-radius:999px;
  color:#315f50;
  background:var(--brand-soft);
  font-size:12px;
  white-space:nowrap;
}
.metric-value{color:var(--brand-dark);font-size:15px}
.semantic-detail{margin-top:5px;color:#354b43}
.tracking-target{margin-top:5px;color:var(--muted);font-size:12px}
.length-distribution-cell{padding:18px 12px 8px}
.length-distribution-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.length-chart{min-width:0;padding:16px;border:1px solid var(--line);border-radius:12px;background:var(--surface-soft)}
.length-chart-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:10px}
.length-chart-summary{color:var(--muted);font-size:12px;text-align:right}
.length-bars{display:flex;align-items:flex-end;gap:6px;min-height:205px;padding:14px 8px 0;overflow-x:auto;border-bottom:1px solid #b9c9c1}
.length-bin{display:grid;grid-template-rows:20px 150px 24px;flex:1 0 34px;min-width:34px;align-items:end;text-align:center}
.length-count{align-self:center;color:#41564f;font-size:11px;font-variant-numeric:tabular-nums}
.length-bar-space{display:flex;height:150px;align-items:flex-end;justify-content:center}
.length-bar{display:flex;width:25px;min-height:2px;overflow:hidden;flex-direction:column-reverse;border-radius:5px 5px 0 0;box-shadow:0 2px 5px rgba(23,43,37,.12)}
.length-chart.win .candidate-first{background:#187a55}.length-chart.win .candidate-second{background:#65bd96}
.length-chart.loss .candidate-first{background:#b84b4b}.length-chart.loss .candidate-second{background:#e69b8f}
.length-round{align-self:center;color:var(--muted);font-size:11px;white-space:nowrap}
.length-legend{display:flex;gap:14px;margin-top:10px;color:var(--muted);font-size:11px}
.length-legend span{display:inline-flex;align-items:center;gap:5px}
.length-legend i{width:9px;height:9px;border-radius:2px;background:#526e64}
.length-legend .second i{opacity:.48}
.candidate-overview{padding:0;overflow:hidden}
.candidate-lead{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:28px;padding:24px;border-bottom:1px solid var(--line);background:#fff}
.candidate-kicker{margin:0 0 5px;color:var(--brand);font-size:12px;font-weight:750;text-transform:uppercase}
.candidate-title{font-size:26px}
.candidate-meta{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:12px;color:var(--muted);font-size:12px}
.candidate-representatives{display:flex;align-items:center;gap:10px}
.candidate-representatives img{width:112px;aspect-ratio:2.5/3.5;object-fit:cover;border:1px solid #cbd9d2;border-radius:6px;background:#e7eeea}
.deck-groups{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;padding:18px 24px 24px}
.deck-group{min-width:0;padding:0 20px;border-left:1px solid var(--line)}
.deck-group:first-child{padding-left:0;border-left:0}.deck-group:last-child{padding-right:0}
.deck-group-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:10px}
.deck-group-head h3{margin:0;font-size:15px}.deck-count{color:var(--muted);font-size:12px}
.deck-list{display:grid;gap:3px}
.deck-entry{display:grid;grid-template-columns:42px minmax(0,1fr) auto;align-items:center;gap:8px;min-height:48px;padding:4px 0;border-top:1px solid #edf2ef}
.deck-entry:first-child{border-top:0}
.deck-entry img{width:38px;height:52px;object-fit:cover;border:1px solid #d2ddd7;border-radius:3px;background:#e7eeea}
.deck-card-name{display:block;font-size:12px;font-weight:650;line-height:1.25;overflow-wrap:anywhere}
.deck-card-set{display:block;color:var(--muted);font-size:10px}
.deck-card-count{font-size:14px;font-weight:750;font-variant-numeric:tabular-nums}
.league-contract{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(240px,.7fr);gap:18px}
.quality-callout{padding:16px;border-left:4px solid var(--brand);background:var(--surface-soft)}
.quality-callout p{margin:6px 0 0}.reward-warning{color:#7b4a24}
.quality-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:16px;border-top:1px solid var(--line);border-left:1px solid var(--line)}
.quality-item{min-height:88px;padding:12px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);background:#fff}
.quality-item.focus{box-shadow:inset 0 3px 0 var(--brand)}
.quality-value{margin-top:5px;font-size:18px;font-weight:750;font-variant-numeric:tabular-nums}
.quality-detail{margin-top:3px;color:var(--muted);font-size:11px}
.evidence-note{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
@media (max-width:760px){
  main{padding:18px 12px 40px}
  .hero{align-items:flex-start;flex-direction:column;padding:24px 20px;border-radius:8px}
  .run-id{max-width:100%}
  section{padding:17px 14px;border-radius:8px}
  .profile-grid{grid-template-columns:1fr}
  .matchup-chart{grid-template-columns:1fr}
  .chart-row{grid-template-columns:minmax(180px,1fr) 72px 128px;gap:7px;padding:4px 0}
  .summary{grid-template-columns:repeat(2,minmax(0,1fr))}
  .length-distribution-grid{grid-template-columns:1fr}
  .candidate-lead{grid-template-columns:1fr;padding:18px}
  .candidate-representatives img{width:88px}
  .deck-groups{grid-template-columns:1fr;padding:10px 18px 18px}
  .deck-group,.deck-group:first-child,.deck-group:last-child{padding:14px 0;border-left:0;border-top:1px solid var(--line)}
  .deck-group:first-child{border-top:0}
  .league-contract{grid-template-columns:1fr}
  .quality-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
""".strip()


def render_html(data: ReportData) -> str:
    """渲染可直接用浏览器打开的单文件 HTML 报告。"""
    manifest = as_mapping(data.manifest)
    summary = as_mapping(data.summary)
    run_id = _text(manifest.get("run_id", "评测运行"))
    embedded_summary = dict(data.summary)
    if "control" in embedded_summary:
        embedded_summary["control"] = {
            "differences": dict(control_differences(summary))
        }
    document_data = json.dumps(
        json_ready(
            {
                "manifest": data.manifest,
                "summary": embedded_summary,
                "games": data.games,
                "metrics": data.metrics,
                "cases": data.cases,
                "metric_profile": data.metric_profile,
                "presentations": {
                    metric_id: {
                        "metric_id": presentation.metric_id,
                        "title": presentation.title,
                        "markdown": presentation.markdown,
                        "html": presentation.html,
                    }
                    for metric_id, presentation in data.presentations.items()
                },
                "presentation_errors": data.presentation_errors,
            }
        ),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        default=str,
    ).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    semantic_metrics = _semantic_metrics_html(data.metric_profile, data.metrics)
    visible_metrics = semantic_metrics or _metrics_html(data.metrics)
    candidate = as_mapping(manifest.get("candidate"))
    candidate_title = candidate.get("display_name") or candidate.get("name") or "评测报告"
    package_manifest = as_mapping(candidate.get("package_manifest"))
    deck_number = package_manifest.get("frozen_deck_number")
    numbered_title = f"{deck_number} · {candidate_title}" if deck_number else candidate_title
    back_link = (
        '<a class="back-link" href="../index.html">返回 Combat Mat 总览</a>'
        if deck_number
        else ""
    )
    update = package_manifest.get("update")
    candidate_policy = package_manifest.get("frozen_policy_label")
    opponent_policy = as_mapping(manifest.get("opponent_pool")).get("policy_label")
    hero_label = (
        f"{candidate_policy} vs {opponent_policy} · Official-engine evaluation"
        if candidate_policy and opponent_policy
        else f"League update {update}" if update is not None else "Official-engine evaluation"
    )
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_text(numbered_title)} - {run_id}</title>",
            f"<style>{REPORT_STYLES}</style>",
            "</head>",
            "<body><main>",
            f'<header class="hero"><div>{back_link}'
            '<p class="eyebrow">POKÉMON TCG · EVALUATION</p>'
            f'<h1>{_text(numbered_title)}</h1><p>{_text(hero_label)}</p></div>'
            f'<div class="run-id">{run_id}</div></header>',
            _candidate_overview_html(manifest, data.metric_profile),
            _profile_html(data.metric_profile),
            _summary_html(summary),
            _matchup_html(summary, manifest),
            _meta_matchup_html(summary),
            _league_quality_html(data.metric_profile, data.metrics),
            visible_metrics,
            f'<script id="report-data" type="application/json">{document_data}</script>',
            "</main></body></html>",
        )
    )


def _candidate_overview_html(
    manifest: Mapping[str, object], profile: Mapping[str, object]
) -> str:
    candidate = as_mapping(manifest.get("candidate"))
    cards = candidate.get("deck_cards")
    if not isinstance(cards, list | tuple):
        return ""
    package_manifest = as_mapping(candidate.get("package_manifest"))
    title = candidate.get("display_name") or candidate.get("name") or "Candidate"
    deck_number = package_manifest.get("frozen_deck_number")
    numbered_title = f"{deck_number} · {title}" if deck_number else title
    update = package_manifest.get("update", "-")
    representatives = "".join(
        f'<img src="{_text(as_mapping(card).get("image_url"))}" '
        f'alt="{_text(as_mapping(card).get("name", "代表宝可梦"))}" loading="eager" '
        'onerror="this.hidden=true">'
        for card in candidate.get("representative_cards", ())
        if as_mapping(card).get("image_url")
    )
    labels = (("pokemon", "Pokémon"), ("trainer", "Trainer"), ("energy", "Energy"))
    groups = []
    for category, label in labels:
        entries = [as_mapping(card) for card in cards if as_mapping(card).get("category") == category]
        total = sum(int(card.get("count", 0)) for card in entries)
        rows = "".join(_deck_entry_html(card) for card in entries)
        groups.append(
            '<div class="deck-group"><div class="deck-group-head">'
            f'<h3>{label}</h3><span class="deck-count">{total} 张</span></div>'
            f'<div class="deck-list">{rows}</div></div>'
        )
    return (
        '<section class="candidate-overview" id="exact-deck">'
        '<div class="candidate-lead"><div><p class="candidate-kicker">主视角卡组</p>'
        f'<h2 class="candidate-title">{_text(numbered_title)}</h2>'
        '<div class="candidate-meta">'
        f'<span>Update {_text(update)}</span><span>Exact {_text(candidate.get("deck_total", "-"))} cards</span>'
        f'<span>Frozen Arena · {_text(manifest.get("opponent_pool", {}).get("pool_id", "-"))}</span>'
        '</div></div>'
        f'<div class="candidate-representatives">{representatives}</div></div>'
        f'<div class="deck-groups">{"".join(groups)}</div></section>'
    )


def _deck_entry_html(card: Mapping[str, object]) -> str:
    image_url = card.get("image_url")
    image = (
        f'<img src="{_text(image_url)}" alt="{_text(card.get("name", "卡牌"))}" loading="lazy" '
        'onerror="this.hidden=true">'
        if image_url
        else '<span></span>'
    )
    set_number = " ".join(
        str(value) for value in (card.get("expansion"), card.get("collection_number")) if value
    )
    return (
        f'<div class="deck-entry">{image}<span><span class="deck-card-name">'
        f'{_text(card.get("name", card.get("card_id", "-")))}</span>'
        f'<span class="deck-card-set">{_text(set_number)} · ID {_text(card.get("card_id", "-"))}</span></span>'
        f'<span class="deck-card-count">×{_text(card.get("count", 0))}</span></div>'
    )


def _league_quality_html(
    profile: Mapping[str, object], metrics: Mapping[str, object]
) -> str:
    if as_mapping(profile).get("id") != "league_deck_quality":
        return ""
    league = as_mapping(metrics.get("league_quality"))
    payload = as_mapping(league.get("payload"))
    deck_profile = as_mapping(payload.get("profile"))
    outcome = as_mapping(as_mapping(metrics.get("outcome")).get("payload"))
    turn_order = as_mapping(outcome.get("by_turn_order"))
    focus = set(deck_profile.get("focus_metrics", ()))
    specs = (
        ("first_attack_round", "首次攻击", "回合", "number"),
        ("attack_continuity", "攻击连续率", "首次攻击后的持续施压", "percentage"),
        ("missed_attack_opportunities", "错过攻击窗口", "总次数", "number"),
        ("prizes_per_attack", "每次攻击拿奖", "Prize / attack", "number"),
        ("multi_prize_turns", "多奖赏回合", "总次数", "number"),
        ("post_ko_attack_gap", "被击倒后断档", "平均回合", "number"),
        ("key_setup_round", "关键攻击手就位", "平均回合", "number"),
        ("max_evolved_pokemon", "进化场面峰值", "平均只数", "number"),
        ("max_bench", "Bench 峰值", "平均只数", "number"),
        ("max_attached_energy", "场上能量峰值", "平均张数", "number"),
        ("damage_events", "伤害事件", "含铺伤事件", "number"),
        ("opponent_attack_denial_rate", "压制后拒攻率", "对手未能攻击", "percentage"),
        ("minimum_deck_count", "最低牌库", "平均剩余张数", "number"),
        ("supporter_turn_rate", "Supporter 回合率", "资源执行", "percentage"),
        ("energy_attach_turn_rate", "附能回合率", "资源执行", "percentage"),
    )
    items = []
    for key, label, detail, kind in specs:
        value = payload.get(key)
        rendered = percentage(value) if kind == "percentage" else display_value(value)
        items.append(
            f'<div class="quality-item{" focus" if key in focus else ""}">'
            f'<div class="label">{_text(label)}</div><div class="quality-value">{_text(rendered)}</div>'
            f'<div class="quality-detail">{_text(detail)}</div></div>'
        )
    split = " · ".join(
        f'{label} {_text(percentage(as_mapping(turn_order.get(key)).get("value")))} '
        f'({_text(display_value(as_mapping(turn_order.get(key)).get("wins")))}/'
        f'{_text(display_value(as_mapping(turn_order.get(key)).get("denominator")))})'
        for key, label in (("first", "先手"), ("second", "后手"))
    )
    return (
        '<section><div class="league-contract"><div><h2>卡组质量诊断</h2>'
        f'<p class="muted">{_text(deck_profile.get("title", "League 通用画像"))}</p></div>'
        f'<div class="quality-callout"><strong>先后手结果</strong><p>{split or "-"}</p></div></div>'
        '<div class="quality-callout" style="margin-top:16px">'
        f'<strong>关注逻辑</strong><p>{_text(deck_profile.get("interpretation", "-"))}</p>'
        f'<p class="reward-warning">奖励边界：{_text(deck_profile.get("reward_warning", "-"))}</p></div>'
        f'<div class="quality-grid">{"".join(items)}</div>'
        '<p class="evidence-note">证据边界：每个 matchup 仅 10 局，单项胜率方差较大；'
        '过程指标用于解释策略行为，checkpoint 强弱仍以相同 Frozen Arena 合同下的 official-engine 结果为准。</p>'
        '</section>'
    )


def _profile_html(profile: Mapping[str, object]) -> str:
    if not profile:
        return ""
    values = as_mapping(profile)
    items = "".join(
        '<div class="profile-item">'
        f'<span class="label">{_text(label)}</span>'
        f'<div class="profile-value">{_text(value)}</div></div>'
        for label, value in (
            ("Metric profile", values.get("id", "-")),
            ("Revision", values.get("revision", "-")),
            ("Metrics", ", ".join(str(item) for item in values.get("metric_ids", ()) if item)),
        )
    )
    return f'<section><div class="profile-grid">{items}</div></section>'


def _summary_html(summary: Mapping[str, object]) -> str:
    cards = (
        ("总对局", display_value(summary.get("total_games"))),
        ("胜 / 负 / 平", _record(summary)),
        ("完成率", percentage(summary.get("completion_rate"))),
        ("胜率", percentage(summary.get("win_rate"))),
        ("错误数", display_value(summary.get("errors"))),
    )
    return "<section><h2>总体结果</h2><div class=\"summary\">" + "".join(
        f'<div class="card"><div class="label">{_text(label)}</div><div class="value">{_text(value)}</div></div>'
        for label, value in cards
    ) + "</div></section>"


def _matchup_html(
    summary: Mapping[str, object], manifest: Mapping[str, object]
) -> str:
    opponent_visuals = {
        str(item.get("name")): item
        for item in manifest.get("opponents", ())
        if isinstance(item, Mapping) and item.get("name")
    }
    matchup_rows = list(ordered_mapping(summary.get("by_opponent")))
    matchup_rows.sort(
        key=lambda item: int(
            as_mapping(as_mapping(opponent_visuals.get(item[0])).get("package_manifest")).get(
                "frozen_deck_number", 999
            )
        )
    )
    chart_rows = []
    for opponent, result in matchup_rows:
        values = as_mapping(result)
        rate = values.get("win_rate")
        visual = as_mapping(opponent_visuals.get(opponent))
        display_name = visual.get("display_name") or opponent
        package_manifest = as_mapping(visual.get("package_manifest"))
        deck_number = package_manifest.get("frozen_deck_number")
        report_href = package_manifest.get("frozen_report_href")
        thumbnails = []
        for card in visual.get("representative_cards", ()):
            card_values = as_mapping(card)
            image_url = card_values.get("image_url")
            if not image_url:
                continue
            thumbnails.append(
                f'<img class="opponent-thumb" src="{_text(image_url)}" '
                f'alt="{_text(card_values.get("name", "代表宝可梦"))}" loading="lazy" '
                'onerror="this.hidden=true">'
            )
        identity = (
            '<span class="opponent-identity">'
            f'<span class="opponent-thumbnails">{"".join(thumbnails)}</span>'
            + (f'<span class="deck-number">{_text(deck_number)}</span>' if deck_number else '')
            + f'<span class="opponent-name" title="{_text(display_name)}">'
            f'{_text(display_name)}</span></span>'
        )
        if report_href:
            identity = f'<a class="opponent-link" href="{_text(report_href)}">{identity}</a>'
        chart_rows.append(
            f'<div class="chart-row opponent-row">{identity}<span class="bar-track">'
            f'<span class="bar" style="width:{_width(rate)}"></span></span>'
            '<span class="chart-result">'
            f'<span class="chart-rate">{_text(percentage(rate))}</span>'
            f'<span class="chart-record">{_text(values.get("wins", 0))}-'
            f'{_text(values.get("losses", 0))}-{_text(values.get("draws", 0))}'
            f' / {_text(values.get("games", 0))}局</span></span></div>'
        )
    return (
        '<section class="matchup-section" id="opponent-matchups"><h2>对局胜率图</h2>'
        '<p class="muted">对 001–055 Exact Deck；每行均由本报告同一批 official-engine 对局按 opponent identity 聚合。</p>'
        '<div class="matchup-chart">'
        + "".join(chart_rows)
        + "</div></section>"
    )


def _meta_matchup_html(summary: Mapping[str, object]) -> str:
    rows = summary.get("by_meta_archetype")
    if not isinstance(rows, list | tuple):
        return ""
    chart_rows = []
    for raw in rows:
        values = as_mapping(raw)
        thumbnails = "".join(
            f'<img class="opponent-thumb" src="{_text(as_mapping(card).get("image_url"))}" '
            f'alt="{_text(as_mapping(card).get("name", "代表宝可梦"))}" loading="lazy" '
            'onerror="this.hidden=true">'
            for card in values.get("representative_cards", ())
            if as_mapping(card).get("image_url")
        )
        identity = (
            '<span class="opponent-identity">'
            f'<span class="opponent-thumbnails">{thumbnails}</span>'
            f'<span class="deck-number">{int(values.get("class_id", 0)) + 1:02d}</span>'
            f'<span class="opponent-name" title="{_text(values.get("display_name"))}">'
            f'{_text(values.get("display_name"))}</span></span>'
        )
        rate = values.get("win_rate")
        chart_rows.append(
            f'<div class="chart-row meta-row">{identity}<span class="bar-track">'
            f'<span class="bar" style="width:{_width(rate)}"></span></span>'
            '<span class="chart-result">'
            f'<span class="chart-rate">{_text(percentage(rate))}</span>'
            f'<span class="chart-record">{_text(values.get("wins", 0))}-'
            f'{_text(values.get("losses", 0))}-{_text(values.get("draws", 0))}'
            f' / {_text(values.get("games", 0))}局</span></span></div>'
        )
    other = as_mapping(summary.get("meta_archetype_other"))
    other_row = ""
    if other:
        rate = other.get("win_rate")
        other_row = (
            '<div class="chart-row meta-other-row">'
            '<span class="opponent-identity"><span class="opponent-thumbnails"></span>'
            '<span class="deck-number">15</span>'
            f'<span class="opponent-name" title="{_text(other.get("display_name", "Other"))}">'
            f'{_text(other.get("display_name", "Other"))}</span></span>'
            '<span class="bar-track">'
            f'<span class="bar" style="width:{_width(rate)}"></span></span>'
            '<span class="chart-result">'
            f'<span class="chart-rate">{_text(percentage(rate))}</span>'
            f'<span class="chart-record">{_text(other.get("wins", 0))}-'
            f'{_text(other.get("losses", 0))}-{_text(other.get("draws", 0))}'
            f' / {_text(other.get("games", 0))}局</span></span></div>'
        )
    return (
        '<section class="matchup-section" id="meta-archetype-matchups">'
        '<h2>对 14 种 Meta Archetype 的聚合胜率</h2>'
        '<p class="muted">按 opponent exact deck 的 priority-ordered trigger-card taxonomy 分类并按局数加权。</p>'
        f'<div class="matchup-chart">{"".join(chart_rows)}{other_row}</div></section>'
    )


def _semantic_metrics_html(
    profile: Mapping[str, object], metrics: Mapping[str, object]
) -> str:
    values = as_mapping(profile)
    groups = values.get("semantic_groups")
    semantics = values.get("metric_semantics")
    if not isinstance(groups, list | tuple) or not isinstance(semantics, list | tuple):
        return ""

    semantics_by_group: dict[str, list[Mapping[str, object]]] = {}
    for raw_semantic in semantics:
        semantic = as_mapping(raw_semantic)
        group_id = str(semantic.get("group_id", ""))
        if group_id:
            semantics_by_group.setdefault(group_id, []).append(semantic)

    sections: list[str] = []
    for raw_group in groups:
        group = as_mapping(raw_group)
        group_id = str(group.get("id", ""))
        title = group.get("title", group_id or "指标")
        description = group.get("description", "")
        rows = []
        for semantic in semantics_by_group.get(group_id, []):
            metric_id = str(semantic.get("metric_id", ""))
            metric = as_mapping(metrics.get(metric_id))
            if not metric:
                continue
            rows.append(_semantic_metric_row(semantic, metric))
        if not rows:
            rows.append('<tr><td colspan="5" class="muted">本轮未产出该组指标。</td></tr>')
        sections.append(
            "<section class=\"semantic-metric-group\">"
            f"<h2>{_text(title)}</h2>"
            f"<p class=\"muted\">{_text(description)}</p>"
            '<table><thead><tr><th>语义指标</th><th>metric_id</th><th>角色</th>'
            "<th>方向</th><th>语义值与追踪目标</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    return "".join(sections)


def _semantic_metric_row(
    semantic: Mapping[str, object], metric: Mapping[str, object]
) -> str:
    metric_id = str(semantic.get("metric_id", ""))
    role = _role_text(semantic.get("role"))
    direction = _direction_text(semantic.get("direction"))
    value, detail = _semantic_metric_value(semantic, metric)
    tracking_target = semantic.get("tracking_target", "")
    target = _text(tracking_target)
    detail_html = f"<div class=\"semantic-detail\">{_text(detail)}</div>" if detail else ""
    row = (
        "<tr>"
        f"<td><strong>{_text(semantic.get('title', metric_id))}</strong></td>"
        f'<td><span class="metric-id">{_text(metric_id)}</span></td>'
        f'<td><span class="tag">{_text(role)}</span></td>'
        f'<td><span class="tag">{_text(direction)}</span></td>'
        f'<td><strong class="metric-value">{_text(value)}</strong>{detail_html}'
        f'<div class="tracking-target">追踪目标：{target}</div></td>'
        "</tr>"
    )
    if semantic.get("display_kind") == "length_outcome_distribution":
        return row + _length_distribution_row(metric)
    return row


def _semantic_metric_value(
    semantic: Mapping[str, object], metric: Mapping[str, object]
) -> tuple[str, str]:
    payload = as_mapping(metric.get("payload"))
    kind = str(semantic.get("display_kind", "aggregate_ratio"))
    if kind == "outcome_turn_order":
        details = _turn_order_detail(payload)
        return _ratio(metric.get("numerator"), metric.get("denominator")), details
    if kind == "powerful_hand_turn_order":
        reached = _ratio(payload.get("reached_numerator"), payload.get("reached_denominator"))
        detail = f"实际到达二回合：{reached}；{_turn_order_detail(payload)}"
        return _ratio(metric.get("numerator"), metric.get("denominator")), detail
    if kind == "component_summary":
        components = as_mapping(as_mapping(payload.get("opening_four_components")).get("component_counts"))
        sample_games = as_mapping(payload.get("opening_four_components")).get("sample_games")
        detail = "；".join(
            f"{_component_title(key)} {display_value(value)}/{display_value(sample_games)}"
            for key, value in components.items()
        )
        return detail or "-", ""
    if kind == "payload_ratio":
        raw_value = _value_at_path(payload, str(semantic.get("value_source", "")))
        if isinstance(raw_value, Mapping):
            return _ratio(raw_value.get("numerator"), raw_value.get("denominator")), ""
        if semantic.get("semantic_id") == "attack_quality":
            value_source = str(semantic.get("value_source", ""))
            parent = _value_at_path(payload, value_source.rsplit(".", 1)[0])
            if isinstance(parent, Mapping):
                denominator = parent.get("denominator")
                detail = "无有效攻击" if _is_zero(denominator) else ""
                return _ratio(parent.get("numerator"), denominator), detail
        if raw_value is None:
            value_source = str(semantic.get("value_source", ""))
            parent = _value_at_path(payload, value_source.rsplit(".", 1)[0])
            if isinstance(parent, Mapping):
                return _ratio(parent.get("numerator"), parent.get("denominator")), ""
        return percentage(raw_value), ""
    if kind == "payload_success_ratio":
        opportunities = payload.get("opportunities")
        successes = payload.get("successes")
        rate = _value_at_path(payload, str(semantic.get("value_source", "")))
        if rate is None or _is_zero(opportunities):
            value = _ratio(successes, opportunities, empty_label="无机会")
        else:
            value = f"{_fraction(successes, opportunities)} = {percentage(rate)}"
        detail = _turn_order_detail(
            payload,
            numerator_key="successes",
            denominator_key="opportunities",
        )
        return value, detail
    if kind == "payload_scalar":
        value = _value_at_path(payload, str(semantic.get("value_source", "")))
        return f"{display_value(value)} 次" if value is not None else "-", ""
    if kind == "draw_summary":
        draws = as_mapping(payload.get("second_turn_draws"))
        all_games = as_mapping(draws.get("all_games"))
        reached = as_mapping(draws.get("reached_second_turn"))
        details = []
        if all_games.get("average") is not None:
            details.append(f"全部对局：{display_value(all_games.get('average'))} 张/局")
        if reached.get("average") is not None:
            details.append(f"实际到达二回合：{display_value(reached.get('average'))} 张/局")
        for key, label in (("first", "先手额外过牌"), ("second", "后手额外过牌")):
            turn_order = as_mapping(draws.get(key))
            if turn_order.get("average") is not None:
                details.append(f"{label}：{display_value(turn_order.get('average'))} 张/局")
        normal_draws = as_mapping(draws.get("normal_draw_cards"))
        normal_all_games = as_mapping(normal_draws.get("all_games"))
        if normal_all_games.get("average") is not None:
            details.append(
                f"正常回合抽牌审计：{display_value(normal_all_games.get('average'))} 张/局"
            )
        return (
            f"{display_value(all_games.get('average'))} 张/局",
            "；".join(details),
        )
    if kind == "length_outcome_distribution":
        outcomes = as_mapping(payload.get("by_outcome"))
        win = as_mapping(outcomes.get("win"))
        loss = as_mapping(outcomes.get("loss"))
        value = (
            f"胜利平均 {display_value(win.get('average'))} 回合；"
            f"失败平均 {display_value(loss.get('average'))} 回合"
        )
        detail = (
            f"胜利样本 {display_value(win.get('denominator'))} 局；"
            f"失败样本 {display_value(loss.get('denominator'))} 局；"
            f"正常完成总体平均 {display_value(metric.get('value'))} 回合"
        )
        return value, detail
    if kind == "powerful_hand_attack_ratio":
        powerful = as_mapping(payload.get("powerful_hand"))
        numerator = powerful.get("non_prize_attacks")
        denominator = int(powerful.get("resolved_attacks", 0)) - int(
            powerful.get("unknown_prize_attacks", 0)
        )
        detail = "无有效攻击" if not denominator else ""
        return _ratio(numerator, denominator), detail
    if kind == "scalar":
        return display_value(_value_at_path(metric, str(semantic.get("value_source", "")))), ""
    return _ratio(metric.get("numerator"), metric.get("denominator")), ""


def _length_distribution_row(metric: Mapping[str, object]) -> str:
    payload = as_mapping(metric.get("payload"))
    outcomes = as_mapping(payload.get("by_outcome"))
    charts = "".join(
        _length_distribution_chart(
            outcome,
            "胜利对局分布" if outcome == "win" else "失败对局分布",
            as_mapping(outcomes.get(outcome)),
        )
        for outcome in ("win", "loss")
    )
    return (
        '<tr class="length-distribution-row"><td colspan="5" '
        f'class="length-distribution-cell"><div class="length-distribution-grid">{charts}'
        "</div></td></tr>"
    )


def _length_distribution_chart(
    outcome: str,
    title: str,
    summary: Mapping[str, object],
) -> str:
    distribution = as_mapping(summary.get("distribution"))
    buckets = sorted(
        (
            (int(round_number), as_mapping(values))
            for round_number, values in distribution.items()
            if str(round_number).isdigit()
        ),
        key=lambda item: item[0],
    )
    max_count = max((int(values.get("count", 0)) for _, values in buckets), default=0)
    bars = []
    for round_number, values in buckets:
        count = int(values.get("count", 0))
        first = int(values.get("candidate_first", 0))
        second = int(values.get("candidate_second", 0))
        total_height = count / max_count * 100 if max_count else 0
        first_height = first / count * 100 if count else 0
        second_height = second / count * 100 if count else 0
        tooltip = (
            f"第 {round_number} 回合：{count} 局；"
            f"我们先攻 {first}；我们后攻 {second}"
        )
        bars.append(
            '<div class="length-bin">'
            f'<span class="length-count">{count}</span><div class="length-bar-space">'
            f'<div class="length-bar" title="{_text(tooltip)}" style="height:{total_height:.2f}%">'
            f'<span class="candidate-first" style="height:{first_height:.2f}%"></span>'
            f'<span class="candidate-second" style="height:{second_height:.2f}%"></span>'
            f'</div></div><span class="length-round">第{round_number}回合</span></div>'
        )
    empty = '<p class="muted">本轮没有可统计样本。</p>' if not bars else ""
    return (
        f'<div class="length-chart {_text(outcome)}"><div class="length-chart-head">'
        f'<strong>{_text(title)}</strong><span class="length-chart-summary">'
        f'平均 {_text(display_value(summary.get("average")))} 回合 · '
        f'{_text(display_value(summary.get("denominator")))} 局</span></div>'
        f'{empty}<div class="length-bars">{"".join(bars)}</div>'
        '<div class="length-legend"><span><i></i>我们先攻</span>'
        '<span class="second"><i></i>我们后攻</span></div></div>'
    )


def _value_at_path(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        if part in {"aggregate", "payload"}:
            continue
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _ratio(
    numerator: object,
    denominator: object,
    empty_label: str = "未定义（无有效样本）",
) -> str:
    if numerator is None or denominator is None:
        return "-"
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return _fraction(numerator, denominator)
    if not denominator_value:
        return empty_label
    return (
        f"{_number(numerator)}/{_number(denominator)} = "
        f"{numerator_value / denominator_value * 100:.2f}%"
    )


def _fraction(
    numerator: object, denominator: object, empty_label: str | None = None
) -> str:
    if numerator is None and denominator is None:
        return "-"
    if empty_label is not None and _is_zero(denominator):
        return empty_label
    return f"{display_value(numerator)}/{display_value(denominator)}"


def _number(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return display_value(value)


def _turn_order_detail(
    payload: Mapping[str, object],
    *,
    numerator_key: str = "numerator",
    denominator_key: str = "denominator",
) -> str:
    values = as_mapping(payload.get("by_turn_order"))
    return "；".join(
        f"{label}：{_ratio(as_mapping(values.get(key)).get(numerator_key), as_mapping(values.get(key)).get(denominator_key))}"
        for key, label in (("first", "先手"), ("second", "后手"))
        if values.get(key) is not None
    )


def _is_zero(value: object) -> bool:
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


def _component_title(key: str) -> str:
    return {
        "active_abra": "Active Abra",
        "rare_candy": "Rare Candy",
        "alakazam_or_search": "Alakazam/检索路线",
        "psychic_energy_or_hilda": "Psychic Energy/Hilda",
        "all_four": "四项同时满足",
    }.get(key, key)


def _role_text(value: object) -> str:
    return {
        "guardrail": "结果护栏",
        "target": "阶段目标",
        "diagnostic": "解释性观测",
        "penalty": "惩罚项",
        "health": "健康指标",
        "audit": "审计指标",
    }.get(str(value), str(value))


def _direction_text(value: object) -> str:
    return {
        "higher": "越高越好",
        "lower": "越低越好",
        "diagnostic": "仅作诊断",
    }.get(str(value), str(value))


def _metrics_html(metrics: Mapping[str, object]) -> str:
    rows = []
    for metric_id, metric in ordered_mapping(metrics):
        values = as_mapping(metric)
        rows.append(
            "<tr>"
            f"<td>{_text(metric_id)}</td><td>{_text(display_value(values.get('numerator')))}</td>"
            f"<td>{_text(display_value(values.get('denominator')))}</td><td>{_text(display_value(values.get('value')))}</td>"
            "</tr>"
        )
    return (
        "<section><h2>指标</h2><table><thead><tr><th>metric_id</th><th>分子</th><th>分母</th><th>达成值</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def _record(summary: Mapping[str, object]) -> str:
    return " / ".join(
        display_value(summary.get(key)) for key in ("wins", "losses", "draws")
    )


def _width(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "0%"
    safe_value = min(max(value, 0.0), 1.0)
    return f"{safe_value * 100:.2f}%"


def _text(value: object) -> str:
    return html_module.escape(str(value), quote=True)
