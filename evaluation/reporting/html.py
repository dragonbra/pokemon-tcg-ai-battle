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
  background:
    radial-gradient(circle at 12% 0%,rgba(58,155,112,.12),transparent 32rem),
    linear-gradient(180deg,#f8fbf9 0,var(--bg) 24rem);
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
  border-radius:18px;
  color:#fff;
  background:linear-gradient(135deg,var(--brand-dark),#23835d 68%,#3b9c71);
  box-shadow:0 18px 44px rgba(20,86,61,.2);
}
.eyebrow{margin:0 0 6px;color:#c8eadb;font-size:12px;font-weight:700;letter-spacing:.12em}
h1{margin:0;font-size:32px;line-height:1.2;letter-spacing:-.02em}
h2{margin:0 0 6px;font-size:20px;line-height:1.35;letter-spacing:-.01em}
h3{margin:0 0 10px}
.run-id{
  max-width:48%;
  padding:8px 12px;
  border:1px solid rgba(255,255,255,.22);
  border-radius:999px;
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
  border-radius:14px;
  background:rgba(255,255,255,.96);
  box-shadow:var(--shadow);
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
  background:linear-gradient(180deg,#fff,var(--surface-soft));
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
  grid-template-columns:minmax(190px,250px) 1fr 56px;
  gap:8px;
  align-items:center;
  padding:3px 5px;
  border-radius:6px;
}
.chart-row:hover{background:var(--surface-soft)}
.opponent-identity{display:flex;align-items:center;min-width:0;gap:7px}
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
.chart-rate{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
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
@media (max-width:760px){
  main{padding:18px 12px 40px}
  .hero{align-items:flex-start;flex-direction:column;padding:24px 20px;border-radius:14px}
  .run-id{max-width:100%}
  section{padding:17px 14px;border-radius:12px}
  .profile-grid{grid-template-columns:1fr}
  .matchup-chart{grid-template-columns:1fr}
  .chart-row{grid-template-columns:minmax(170px,1fr) 1fr 54px;gap:7px;padding:4px 0}
  .summary{grid-template-columns:repeat(2,minmax(0,1fr))}
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
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>评测报告 - {run_id}</title>",
            f"<style>{REPORT_STYLES}</style>",
            "</head>",
            "<body><main>",
            '<header class="hero"><div>'
            '<p class="eyebrow">POKÉMON TCG · EVALUATION</p>'
            '<h1>评测报告</h1></div>'
            f'<div class="run-id">{run_id}</div></header>',
            _profile_html(data.metric_profile),
            _summary_html(summary),
            _matchup_html(summary, manifest),
            visible_metrics,
            f'<script id="report-data" type="application/json">{document_data}</script>',
            "</main></body></html>",
        )
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
    chart_rows = []
    for opponent, result in ordered_mapping(summary.get("by_opponent")):
        values = as_mapping(result)
        rate = values.get("win_rate")
        visual = as_mapping(opponent_visuals.get(opponent))
        display_name = visual.get("display_name") or opponent
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
            f'<span class="opponent-name" title="{_text(display_name)}">'
            f'{_text(display_name)}</span></span>'
        )
        chart_rows.append(
            f'<div class="chart-row">{identity}<span class="bar-track">'
            f'<span class="bar" style="width:{_width(rate)}"></span></span>'
            f'<span class="chart-rate">{_text(percentage(rate))}</span></div>'
        )
    return (
        '<section class="matchup-section"><h2>对局胜率图</h2><div class="matchup-chart">'
        + "".join(chart_rows)
        + "</div></section>"
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
    return (
        "<tr>"
        f"<td><strong>{_text(semantic.get('title', metric_id))}</strong></td>"
        f'<td><span class="metric-id">{_text(metric_id)}</span></td>'
        f'<td><span class="tag">{_text(role)}</span></td>'
        f'<td><span class="tag">{_text(direction)}</span></td>'
        f'<td><strong class="metric-value">{_text(value)}</strong>{detail_html}'
        f'<div class="tracking-target">追踪目标：{target}</div></td>'
        "</tr>"
    )


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
