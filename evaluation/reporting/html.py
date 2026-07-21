from __future__ import annotations

import html as html_module
import json
from collections.abc import Mapping

from .models import (
    ReportData,
    as_mapping,
    case_evidence_steps,
    control_differences,
    display_value,
    failure_class_distribution,
    json_ready,
    ordered_mapping,
    percentage,
)


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
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>评测报告 - {run_id}</title>",
            "<style>"
            "body{margin:0;background:#f5f7f7;color:#172321;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
            "main{max-width:1120px;margin:0 auto;padding:24px}h1,h2,h3{margin:0 0 12px}"
            "section{margin:24px 0}table{width:100%;border-collapse:collapse;background:#fff}"
            "th,td{padding:8px 10px;border:1px solid #d7dfdc;text-align:left;vertical-align:top}th{background:#e9f0ed}"
            ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}"
            ".card,.case{background:#fff;border:1px solid #d7dfdc;padding:12px;border-radius:6px}"
            ".label{color:#52615d;font-size:12px}.value{font-size:21px;font-weight:700}.chart-row{display:grid;grid-template-columns:minmax(140px,220px) 1fr 70px;gap:10px;align-items:center;margin:6px 0}.bar-track{height:14px;background:#dce6e1}.bar{display:block;height:100%;background:#27885d}.case-list{display:grid;gap:10px}.semantic-detail{margin-top:4px}.muted{color:#52615d}"
            "</style>",
            "</head>",
            "<body><main>",
            f"<h1>评测报告</h1><p class=\"muted\">运行 ID：{run_id}</p>",
            _profile_html(data.metric_profile),
            _summary_html(summary),
            _matchup_html(summary),
            _semantic_metrics_html(data.metric_profile, data.metrics),
            _metrics_html(data.metrics),
            _failure_html(data.metrics),
            _cases_html(data.cases),
            _control_html(summary),
            _presentations_html(data),
            f'<script id="report-data" type="application/json">{document_data}</script>',
            "</main></body></html>",
        )
    )


def _profile_html(profile: Mapping[str, object]) -> str:
    if not profile:
        return ""
    values = as_mapping(profile)
    rows = "".join(
        f"<tr><th>{_text(label)}</th><td>{_text(value)}</td></tr>"
        for label, value in (
            ("profile", values.get("id", "-")),
            ("revision", values.get("revision", "-")),
            ("metrics", ", ".join(str(item) for item in values.get("metric_ids", ()) if item)),
        )
    )
    return f"<section><h2>Metric profile</h2><table>{rows}</table></section>"


def _presentations_html(data: ReportData) -> str:
    semantic_profile = bool(as_mapping(data.metric_profile).get("semantic_groups"))
    presentations = tuple(
        presentation
        for presentation in data.presentations.values()
        if not semantic_profile or not presentation.title.startswith("指标：")
    )
    sections = "".join(presentation.html for presentation in presentations)
    if semantic_profile and presentations:
        sections = f"<section><h2>插件审计明细</h2></section>{sections}"
    if data.presentation_errors:
        diagnostics = "".join(
            f"<li>{_text(item.get('metric_id', 'unknown'))}: {_text(item.get('error', 'unknown'))}</li>"
            for item in data.presentation_errors
        )
        sections += f"<section><h2>Presentation diagnostics</h2><ul>{diagnostics}</ul></section>"
    return sections


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


def _matchup_html(summary: Mapping[str, object]) -> str:
    rows = []
    chart_rows = []
    for opponent, result in ordered_mapping(summary.get("by_opponent")):
        values = as_mapping(result)
        rate = values.get("win_rate")
        rows.append(
            "<tr>"
            f"<td>{_text(opponent)}</td><td>{_text(display_value(values.get('wins')))}</td>"
            f"<td>{_text(display_value(values.get('losses')))}</td><td>{_text(display_value(values.get('draws')))}</td>"
            f"<td>{_text(display_value(values.get('errors')))}</td><td>{_text(display_value(values.get('unfinished')))}</td>"
            f"<td>{_text(percentage(rate))}</td></tr>"
        )
        chart_rows.append(
            f'<div class="chart-row"><span>{_text(opponent)}</span><span class="bar-track"><span class="bar" style="width:{_width(rate)}"></span></span><span>{_text(percentage(rate))}</span></div>'
        )
    return (
        "<section><h2>对局矩阵</h2><table><thead><tr><th>对手</th><th>胜</th><th>负</th><th>平</th><th>错误</th><th>未完成</th><th>胜率</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
        + '<section><h2>对局胜率图</h2><div class="matchup-chart">'
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
            rows.append('<tr><td colspan="6" class="muted">本轮未产出该组指标。</td></tr>')
        sections.append(
            "<section class=\"semantic-metric-group\">"
            f"<h2>{_text(title)}</h2>"
            f"<p class=\"muted\">{_text(description)}</p>"
            '<table><thead><tr><th>语义指标</th><th>metric_id</th><th>角色</th>'
            "<th>方向</th><th>分子 / 分母</th><th>语义值与追踪目标</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    return "".join(sections)


def _semantic_metric_row(
    semantic: Mapping[str, object], metric: Mapping[str, object]
) -> str:
    metric_id = str(semantic.get("metric_id", ""))
    role = _role_text(semantic.get("role"))
    direction = _direction_text(semantic.get("direction"))
    numerator = metric.get("numerator")
    denominator = metric.get("denominator")
    value, detail = _semantic_metric_value(semantic, metric)
    tracking_target = semantic.get("tracking_target", "")
    target = _text(tracking_target)
    detail_html = f"<div class=\"semantic-detail\">{_text(detail)}</div>" if detail else ""
    return (
        "<tr>"
        f"<td><strong>{_text(semantic.get('title', metric_id))}</strong></td>"
        f"<td><code>{_text(metric_id)}</code></td>"
        f"<td>{_text(role)}</td>"
        f"<td>{_text(direction)}</td>"
        f"<td>{_text(_semantic_fraction(semantic, metric))}</td>"
        f"<td><strong>{_text(value)}</strong>{detail_html}<div class=\"muted\">追踪目标：{target}</div></td>"
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
            value = _ratio(successes, opportunities)
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
        return _ratio(numerator, denominator), ""
    if kind == "scalar":
        return display_value(_value_at_path(metric, str(semantic.get("value_source", "")))), ""
    return _ratio(metric.get("numerator"), metric.get("denominator")), ""


def _semantic_fraction(
    semantic: Mapping[str, object], metric: Mapping[str, object]
) -> str:
    payload = as_mapping(metric.get("payload"))
    kind = str(semantic.get("display_kind", "aggregate_ratio"))
    if kind in {"outcome_turn_order", "powerful_hand_turn_order", "aggregate_ratio"}:
        return _fraction(
            metric.get("numerator"), metric.get("denominator"), empty_label="无有效样本"
        )
    if kind == "component_summary":
        return "多项观察"
    if kind == "payload_ratio":
        value_source = str(semantic.get("value_source", ""))
        raw_value = _value_at_path(payload, value_source)
        if isinstance(raw_value, Mapping):
            return _fraction(
                raw_value.get("numerator"),
                raw_value.get("denominator"),
                empty_label=_empty_fraction_label(semantic),
            )
        parent_path = value_source.rsplit(".", 1)[0]
        parent = _value_at_path(payload, parent_path)
        if isinstance(parent, Mapping):
            return _fraction(
                parent.get("numerator"),
                parent.get("denominator"),
                empty_label=_empty_fraction_label(semantic),
            )
        return "-"
    if kind == "payload_success_ratio":
        return _fraction(
            payload.get("successes"), payload.get("opportunities"), empty_label="无机会"
        )
    if kind == "draw_summary":
        draws = as_mapping(payload.get("second_turn_draws"))
        all_games = as_mapping(draws.get("all_games"))
        return _fraction(
            all_games.get("total"), all_games.get("games"), empty_label="无有效样本"
        )
    if kind == "powerful_hand_attack_ratio":
        powerful = as_mapping(payload.get("powerful_hand"))
        denominator = int(powerful.get("resolved_attacks", 0)) - int(
            powerful.get("unknown_prize_attacks", 0)
        )
        return _fraction(
            powerful.get("non_prize_attacks"), denominator, empty_label="无有效攻击"
        )
    if kind == "payload_scalar":
        return display_value(
            _value_at_path(payload, str(semantic.get("value_source", "")))
        )
    return _fraction(metric.get("numerator"), metric.get("denominator"))


def _empty_fraction_label(semantic: Mapping[str, object]) -> str:
    return {
        "dunsparce_bridge": "无机会",
        "attack_quality": "无有效攻击",
    }.get(str(semantic.get("semantic_id")), "无有效样本")


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


def _failure_html(metrics: Mapping[str, object]) -> str:
    rows = "".join(
        f"<tr><td>{_text(failure_class)}</td><td>{_text(display_value(count))}</td></tr>"
        for failure_class, count in ordered_mapping(failure_class_distribution(metrics))
    )
    return (
        "<section><h2>failure_class 分布</h2><table><thead><tr><th>failure_class</th><th>数量</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></section>"
    )


def _cases_html(cases: tuple) -> str:
    cards = []
    for index, raw_case in enumerate(cases[:3], start=1):
        case = as_mapping(raw_case)
        evidence = "".join(
            f"<li>步骤 {_text(display_value(item.get('step')))}：{_text(item.get('expected_reason', item.get('detail', '')))}</li>"
            for item in case_evidence_steps(case)
        )
        cards.append(
            '<article class="case">'
            f"<h3>案例 {index}：{_text(case.get('game_id', '-'))}</h3>"
            f"<p>对手：{_text(case.get('opponent', '-'))}</p>"
            f"<p>failure_class：{_text(case.get('failure_class', '-'))}</p>"
            f"<p>trace：<code>{_text(case.get('trace_path', '-'))}</code></p>"
            f"<ul>{evidence}</ul></article>"
        )
    content = "".join(cards) if cards else "<p>无重点案例。</p>"
    return f'<section><h2>重点案例</h2><div class="case-list">{content}</div></section>'


def _control_html(summary: Mapping[str, object]) -> str:
    differences = control_differences(summary)
    if not differences:
        return ""
    items = "".join(
        f"<li>{_text(key)}：{_text(value)}</li>" for key, value in ordered_mapping(differences)
    )
    return f"<section><h2>对照差异</h2><ul>{items}</ul></section>"


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
