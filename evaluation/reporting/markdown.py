from __future__ import annotations

from collections.abc import Mapping

from .models import (
    ReportData,
    as_mapping,
    case_evidence_steps,
    control_differences,
    display_value,
    failure_class_distribution,
    ordered_mapping,
    percentage,
)


def render_markdown(data: ReportData) -> str:
    """将已聚合的 ReportData 渲染为无需额外依赖的 Markdown。"""
    manifest = as_mapping(data.manifest)
    summary = as_mapping(data.summary)
    lines = ["# 评测报告", ""]
    if manifest.get("run_id") is not None:
        lines.extend((f"运行 ID：{_cell(manifest['run_id'])}", ""))

    lines.extend(_profile_section(data.metric_profile))
    lines.extend(_summary_section(summary))
    lines.extend(_matchup_section(summary))
    lines.extend(_metrics_section(data.metrics))
    lines.extend(_failure_section(data.metrics))
    lines.extend(_cases_section(data.cases))
    lines.extend(_control_section(summary))
    lines.extend(_presentation_sections(data))
    return "\n".join(lines).rstrip() + "\n"


def _profile_section(profile: Mapping[str, object]) -> list[str]:
    if not profile:
        return []
    values = as_mapping(profile)
    rows = (
        ("profile", values.get("id")),
        ("revision", values.get("revision")),
        ("metrics", ", ".join(str(item) for item in values.get("metric_ids", ()) if item)),
    )
    lines = ["## Metric profile", "", "| 项目 | 数值 |", "| --- | --- |"]
    lines.extend(f"| {_cell(label)} | {_cell(value)} |" for label, value in rows)
    return [*lines, ""]


def _presentation_sections(data: ReportData) -> list[str]:
    lines: list[str] = []
    for presentation in data.presentations.values():
        rendered = presentation.markdown.strip()
        if rendered:
            lines.extend((rendered, ""))
    if data.presentation_errors:
        lines.extend(("## Presentation diagnostics", ""))
        lines.extend(
            f"- {_cell(item.get('metric_id', 'unknown'))}: {_cell(item.get('error', 'unknown'))}"
            for item in data.presentation_errors
        )
        lines.append("")
    return lines


def _summary_section(summary: Mapping[str, object]) -> list[str]:
    rows = (
        ("总对局", display_value(summary.get("total_games"))),
        ("胜 / 负 / 平", _record(summary)),
        ("完成率", percentage(summary.get("completion_rate"))),
        ("胜率", percentage(summary.get("win_rate"))),
        ("错误数", display_value(summary.get("errors"))),
        ("未完成", display_value(summary.get("unfinished"))),
        ("已完成", display_value(summary.get("completed_games"))),
    )
    lines = ["## 总体结果", "", "| 项目 | 数值 |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return [*lines, ""]


def _matchup_section(summary: Mapping[str, object]) -> list[str]:
    lines = ["## 对局矩阵", "", "| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for opponent, result in ordered_mapping(summary.get("by_opponent")):
        values = as_mapping(result)
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(opponent),
                    display_value(values.get("wins")),
                    display_value(values.get("losses")),
                    display_value(values.get("draws")),
                    display_value(values.get("errors")),
                    display_value(values.get("unfinished")),
                    percentage(values.get("win_rate")),
                )
            )
            + " |"
        )
    return [*lines, ""]


def _metrics_section(metrics: Mapping[str, object]) -> list[str]:
    lines = ["## 指标", "", "| metric_id | 分子 | 分母 | value |", "| --- | ---: | ---: | ---: |"]
    for metric_id, metric in ordered_mapping(metrics):
        values = as_mapping(metric)
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(metric_id),
                    display_value(values.get("numerator")),
                    display_value(values.get("denominator")),
                    display_value(values.get("value")),
                )
            )
            + " |"
        )
    return [*lines, ""]


def _failure_section(metrics: Mapping[str, object]) -> list[str]:
    lines = ["## failure_class 分布", "", "| failure_class | 数量 |", "| --- | ---: |"]
    lines.extend(
        f"| {_cell(failure_class)} | {display_value(count)} |"
        for failure_class, count in ordered_mapping(failure_class_distribution(metrics))
    )
    return [*lines, ""]


def _cases_section(cases: tuple) -> list[str]:
    lines = ["## 重点案例", ""]
    for index, raw_case in enumerate(cases[:3], start=1):
        case = as_mapping(raw_case)
        lines.extend(
            (
                f"### 案例 {index}：{_cell(case.get('game_id', '-'))}",
                "",
                f"- 对手：{_cell(case.get('opponent', '-'))}",
                f"- failure_class：{_cell(case.get('failure_class', '-'))}",
                f"- trace：{_cell(case.get('trace_path', '-'))}",
            )
        )
        for evidence in case_evidence_steps(case):
            step = display_value(evidence.get("step"))
            detail = _cell(evidence.get("expected_reason", evidence.get("detail", "")))
            lines.append(f"- 证据步骤 {step}：{detail}")
        lines.append("")
    if not cases:
        lines.extend(("无重点案例。", ""))
    return lines


def _control_section(summary: Mapping[str, object]) -> list[str]:
    differences = control_differences(summary)
    if not differences:
        return []
    lines = ["## 对照差异", ""]
    lines.extend(
        f"- {_cell(key)}：{_cell(value)}" for key, value in ordered_mapping(differences)
    )
    return [*lines, ""]


def _record(summary: Mapping[str, object]) -> str:
    return " / ".join(
        display_value(summary.get(key)) for key in ("wins", "losses", "draws")
    )


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
