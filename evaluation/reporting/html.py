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
            ".label{color:#52615d;font-size:12px}.value{font-size:21px;font-weight:700}.chart-row{display:grid;grid-template-columns:minmax(140px,220px) 1fr 70px;gap:10px;align-items:center;margin:6px 0}.bar-track{height:14px;background:#dce6e1}.bar{display:block;height:100%;background:#27885d}.case-list{display:grid;gap:10px}code{word-break:break-all}.muted{color:#52615d}"
            "</style>",
            "</head>",
            "<body><main>",
            f"<h1>评测报告</h1><p class=\"muted\">运行 ID：{run_id}</p>",
            _profile_html(data.metric_profile),
            _summary_html(summary),
            _matchup_html(summary),
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
    sections = "".join(presentation.html for presentation in data.presentations.values())
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
