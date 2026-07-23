"""Render the evaluation-first INDEX section and standalone BC search report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable


COLORS = ("#176b5b", "#2463a6", "#9b5c16", "#8047a8", "#b23a48", "#3d7f25", "#57606a")


def render_campaign(repository_root: Path, experiment_root: Path) -> Path:
    repository_root = repository_root.resolve()
    experiment_root = experiment_root.resolve()
    search = _load_json(experiment_root / "search_manifest.json")
    comparison = _load_json(experiment_root / "comparison_manifest.json")
    output = experiment_root / "bc_capacity_search_report.html"
    output.write_text(
        _campaign_html(repository_root, experiment_root, search, comparison),
        encoding="utf-8",
    )
    _update_index(repository_root, experiment_root, comparison)
    return output


def _campaign_html(
    repository_root: Path,
    experiment_root: Path,
    search: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    trials = list(comparison.get("trials", []))
    completed = [trial for trial in trials if trial.get("status") == "completed"]
    selection = comparison.get("selection") or {}
    selected_version = str(selection.get("version", ""))
    selected_trial = next(
        (trial for trial in completed if trial["spec"]["version"] == selected_version),
        None,
    )
    budget = float(comparison.get("training_budget_seconds", 12_600))
    used = float(comparison.get("training_seconds", 0))
    report_rows = "".join(
        _report_trial_row(experiment_root, trial, selected_version=selected_version)
        for trial in trials
    )
    context_rows = _context_rows(completed)
    opponent_sections = "".join(_opponent_section(experiment_root, trial) for trial in completed)
    curve_exact = _line_chart(
        "Train / validation exact by epoch",
        completed,
        (
            ("train_exact", "train"),
            ("validation_exact", "validation"),
        ),
        maximum=1.0,
    )
    curve_loss = _line_chart(
        "Train / validation policy loss by epoch",
        completed,
        (
            ("train_policy_loss", "train"),
            ("validation_policy_loss", "validation"),
        ),
    )
    scatter_validation = _scatter_chart(
        "Validation exact vs parameter count",
        completed,
        lambda trial: float(trial["model"]["parameter_count"]),
        lambda trial: float(trial["offline"]["validation_exact"]),
        "parameters",
        "validation exact",
        y_maximum=1.0,
    )
    scatter_evaluation = _scatter_chart(
        "Evaluation win rate vs parameter count",
        completed,
        lambda trial: float(trial["model"]["parameter_count"]),
        lambda trial: float(trial["evaluation"]["win_rate"]),
        "parameters",
        "evaluation win rate",
        y_maximum=1.0,
    )
    scatter_relation = _scatter_chart(
        "Evaluation win rate vs validation exact",
        completed,
        lambda trial: float(trial["offline"]["validation_exact"]),
        lambda trial: float(trial["evaluation"]["win_rate"]),
        "validation exact",
        "evaluation win rate",
        x_maximum=1.0,
        y_maximum=1.0,
    )
    scatter_runtime = _scatter_chart(
        "Training runtime vs parameter count",
        completed,
        lambda trial: float(trial["model"]["parameter_count"]),
        lambda trial: float(trial["training"]["wall_seconds"]) / 60.0,
        "parameters",
        "training minutes",
    )
    scatter_memory = _scatter_chart(
        "Peak GPU memory vs parameter count",
        completed,
        lambda trial: float(trial["model"]["parameter_count"]),
        lambda trial: float(trial["training"]["peak_gpu_memory_bytes"]) / (1024**3),
        "parameters",
        "peak GiB",
    )
    observations = "".join(f"<li>{html.escape(item)}</li>" for item in _observations(completed))
    status = html.escape(str(comparison.get("status", "running")))
    if selected_trial is None:
        selection_summary = "搜索证据已冻结；尚未记录用户选择。"
    else:
        selected_evaluation = selected_trial["evaluation"]
        selection_summary = (
            f"搜索冻结后，用户选择 {selected_version} 作为当前 BC 模型："
            f"{selected_evaluation['wins']}-{selected_evaluation['losses']}，"
            f"{_pct(selected_evaluation['win_rate'])}，"
            f"{selected_trial['model']['parameter_count']:,} 参数。"
        )
    dataset = search["dataset"]
    source = search["source"]
    complete_count = len(completed)
    failure_count = len(trials) - complete_count
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(experiment_root.name)} BC Capacity Search</title>
  <style>
    :root {{ --ink:#20242b; --muted:#657080; --line:#d8dde4; --paper:#fff;
      --wash:#f3f5f7; --green:#176b5b; --blue:#2463a6; --amber:#9b5c16; --red:#a83c48; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--wash);
      font:14px/1.55 system-ui,sans-serif; }} header, main, footer {{ max-width:1440px;
      margin:auto; padding:24px 28px; }} header {{ padding-top:40px; }} h1 {{ font-size:34px;
      margin:4px 0 8px; }} h2 {{ margin:36px 0 12px; font-size:22px; }} h3 {{ font-size:16px; }}
    .eyebrow {{ color:var(--green); font-weight:750; text-transform:uppercase; }} .muted {{ color:var(--muted); }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:22px 0; }}
    .metric {{ background:var(--paper); border:1px solid var(--line); padding:14px; border-radius:6px; }}
    .metric strong {{ display:block; font-size:24px; margin-top:3px; }} .metric small {{ color:var(--muted); }}
    .scope {{ background:#eaf3f0; border-left:4px solid var(--green); padding:16px 18px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--paper); margin:12px 0 26px; }}
    th,td {{ border:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
    th {{ background:#e9edf1; position:sticky; top:0; }} .eval {{ font-size:17px; font-weight:750; }}
    .ok {{ color:var(--green); font-weight:750; }} .bad {{ color:var(--red); font-weight:750; }}
    code {{ background:#edf0f2; padding:2px 4px; }} a {{ color:var(--blue); }}
    .charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
    figure {{ margin:0; padding:12px; background:var(--paper); border:1px solid var(--line); }}
    figcaption {{ font-weight:750; margin-bottom:8px; }} svg {{ width:100%; height:auto; display:block; }}
    details {{ background:var(--paper); border:1px solid var(--line); margin:9px 0; padding:10px 13px; }}
    summary {{ cursor:pointer; font-weight:750; }} .facts {{ columns:2; }} footer {{ color:var(--muted); }}
    @media (max-width:900px) {{ .summary,.charts {{ grid-template-columns:1fr; }} .facts {{ columns:1; }}
      header,main,footer {{ padding-left:14px; padding-right:14px; }} .tableWrap {{ overflow:auto; }} }}
  </style>
</head>
<body>
<header>
  <div class="eyebrow">{html.escape(experiment_root.name)} · Frozen Yushin Ito BC capacity campaign</div>
  <h1>模型容量与超参数搜索</h1>
  <p class="muted">状态：<code>{status}</code>。搜索报告保持同口径证据与原始顺序；搜索结束后的用户选择单独记录，不改写搜索分支。</p>
  <div class="summary">
    <div class="metric"><small>训练预算</small><strong>{used / 60:.1f} / {budget / 60:.0f} min</strong><span>剩余 {max(0.0, budget-used)/60:.1f} min</span></div>
    <div class="metric"><small>Trial 完成性</small><strong>{complete_count} complete</strong><span>{failure_count} failed / incomplete</span></div>
    <div class="metric"><small>冻结数据</small><strong>{dataset['summary'].get('records', 86875):,}</strong><span>Yushin Ito decisions</span></div>
    <div class="metric"><small>闭环评测</small><strong>{sum(int(t['evaluation']['total_games']) for t in completed):,} games</strong><span>每个完成 trial 固定 18×10</span></div>
  </div>
  <div class="scope"><strong>实验边界</strong><br>所有 trial 从随机初始化开始，使用同一份 0003 人类专家 dataset、episode split、<code>ptcg_features_universal</code> 与 BC loss；没有续训 0003 checkpoint，没有新数据、self-play 或规则 fallback。搜索 runner 没有自动晋级或提交；正式归档与 Kaggle submission 只发生在完整搜索冻结、用户选择 V4 之后。</div>
</header>
<main>
  <h2>冻结证据</h2>
  <ul class="facts">
    <li>dataset SHA-256: <code>{html.escape(dataset['sha256'])}</code></li>
    <li>card metadata SHA-256: <code>{html.escape(dataset['metadata_sha256'])}</code></li>
    <li>source SHA-256: <code>{html.escape(source['combined_sha256'])}</code></li>
    <li>Git commit: <code>{html.escape(source['git_commit'])}</code></li>
    <li>train / validation / test: {dataset['summary']['records_by_split']['train']:,} / {dataset['summary']['records_by_split']['validation']:,} / {dataset['summary']['records_by_split']['test']:,} records</li>
    <li>搜索控制只读取 train / validation；test 与 evaluation 不参与分支。</li>
  </ul>

  <h2>Evaluation-first trial matrix</h2>
  <div class="tableWrap"><table>
    <thead><tr><th>Trial</th><th>Official evaluation</th><th>先手 / 后手</th><th>过程指标</th><th>模型</th><th>训练配置</th><th>BC 辅助</th><th>Artifacts</th></tr></thead>
    <tbody>{report_rows or '<tr><td colspan="8">No trial has completed yet.</td></tr>'}</tbody>
  </table></div>

  <h2>学习曲线与容量关系</h2>
  <div class="charts">{curve_exact}{curve_loss}{scatter_validation}{scatter_evaluation}{scatter_relation}{scatter_runtime}{scatter_memory}</div>

  <h2>按 selection context 的 validation BC 表现</h2>
  <p class="muted">micro 是 trial 总体 validation exact；macro 是各 context 等权平均。低于 100 条的 context 标记为低样本，不据此单独下结论。</p>
  <div class="tableWrap"><table><thead><tr><th>Trial</th><th>Context</th><th>Records</th><th>Exact</th><th>Micro</th><th>Macro</th><th>审计</th></tr></thead><tbody>{context_rows}</tbody></table></div>

  <h2>逐 trial 的 18-opponent 对战</h2>
  {opponent_sections or '<p>No completed evaluation.</p>'}

  <h2>中立观察与选择记录</h2>
  <ul>{observations or '<li>等待完成更多 trial 后生成容量与优化观察。</li>'}</ul>
  <p class="scope"><strong>用户决策记录：</strong>{html.escape(selection_summary)} 该选择发生在搜索结束后；搜索过程仍未使用 test/evaluation 进行超参分支。机器可读归档见 <a href="promotion.json">promotion.json</a>。</p>
</main>
<footer>Generated from <code>search_manifest.json</code>, <code>comparison_manifest.json</code>, each <code>trial_result.json</code>, checkpoint metadata and official evaluation artifacts.</footer>
</body>
</html>
"""


def _report_trial_row(
    experiment_root: Path,
    trial: dict[str, Any],
    *,
    selected_version: str = "",
) -> str:
    spec = trial["spec"]
    if trial.get("status") != "completed":
        return (
            f"<tr><td><code>{html.escape(spec['version'])}</code></td>"
            f"<td class='bad' colspan='7'>{html.escape(str(trial.get('failure', 'incomplete')))}</td></tr>"
        )
    evaluation = trial["evaluation"]
    offline = trial["offline"]
    model = trial["model"]
    training = trial["training"]
    first = evaluation.get("by_turn_order", {}).get("first", {})
    second = evaluation.get("by_turn_order", {}).get("second", {})
    metrics = evaluation["metrics"]
    report_link = _relative_to_experiment(experiment_root, evaluation["report_html"])
    decision_label = (
        "user selected after search" if spec["version"] == selected_version else "complete"
    )
    return f"""<tr>
<td><code>{html.escape(spec['version'])}</code><br>Phase {spec['phase']} · {spec['architecture']}<br><span class="ok">{decision_label}</span></td>
<td><span class="eval">{evaluation['wins']}-{evaluation['losses']}-{evaluation['draws']} · {_pct(evaluation['win_rate'])}</span><br>{evaluation['completed_games']}/{evaluation['total_games']} completed<br><span class="{'ok' if not evaluation['errors'] and not evaluation['unfinished'] else 'bad'}">{evaluation['errors']} errors · {evaluation['unfinished']} unfinished</span></td>
<td>先手 {_turn_order(first)}<br>后手 {_turn_order(second)}</td>
<td>Correctness {_metric(metrics,'correctness')}<br>Powerful Hand {_metric(metrics,'powerful_hand')}<br>Post-KO {_metric(metrics,'post_ko_relay')}<br>Attack penalty {_metric(metrics,'attack_quality')}<br>Library {_metric(metrics,'library_pressure')}</td>
<td>{model['parameter_count']:,} params<br>d={spec['d_model']} · L={spec['layers']} · hidden={spec['hidden_dim']}<br>{model['checkpoint_bytes']/1024/1024:.1f} MiB checkpoint</td>
<td>lr={spec['learning_rate']:g} · seed={spec['seed']}<br>dropout={spec['dropout']:g} · heads={spec['heads']}<br>{training['wall_seconds']/60:.1f} min · peak {training['peak_gpu_memory_bytes']/1024**3:.2f} GiB</td>
<td>val exact {_pct(offline['validation_exact'])}<br>train/val policy {offline['train_policy_loss']:.4f} / {offline['validation_policy_loss']:.4f}<br>gap {_pp(offline['exact_gap'])} · epoch {offline['best_epoch']}</td>
<td><a href="{html.escape(report_link)}">evaluation</a><br><a href="{html.escape(spec['version'])}/training_summary.json">summary</a><br><a href="{html.escape(spec['version'])}/training_metrics.jsonl">curves</a><br><a href="{html.escape(spec['version'])}/trial_result.json">trial JSON</a></td>
</tr>"""


def _context_rows(trials: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for trial in trials:
        offline = trial["offline"]
        for context, value in offline["validation_contexts"].items():
            records = int(value["records"])
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(trial['spec']['version'])}</code></td>"
                f"<td>{html.escape(context)}</td><td>{records}</td>"
                f"<td>{_pct(value['exact_action_rate'])}</td>"
                f"<td>{_pct(offline['validation_exact'])}</td>"
                f"<td>{_pct(offline['validation_context_macro_exact'])}</td>"
                f"<td>{'low sample' if records < 100 else ''}</td></tr>"
            )
    return "".join(rows)


def _opponent_section(experiment_root: Path, trial: dict[str, Any]) -> str:
    evaluation = trial["evaluation"]
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value['wins']}-{value['losses']}-{value['draws']}</td><td>{_pct(value['win_rate'])}</td><td>{value['errors']}</td><td>{value['unfinished']}</td></tr>"
        for name, value in evaluation["by_opponent"].items()
    )
    report_link = _relative_to_experiment(experiment_root, evaluation["report_html"])
    return f"""<details><summary><code>{html.escape(trial['spec']['version'])}</code> · {evaluation['wins']}-{evaluation['losses']}-{evaluation['draws']} · {_pct(evaluation['win_rate'])} · <a href="{html.escape(report_link)}">raw report</a></summary>
<div class="tableWrap"><table><thead><tr><th>Opponent</th><th>W-L-D</th><th>Win rate</th><th>Errors</th><th>Unfinished</th></tr></thead><tbody>{rows}</tbody></table></div></details>"""


def _observations(trials: list[dict[str, Any]]) -> list[str]:
    if not trials:
        return []
    observations: list[str] = []
    phase_a = [trial for trial in trials if trial["spec"]["phase"] == "A"]
    if phase_a:
        descriptions = ", ".join(
            f"lr={trial['spec']['learning_rate']:g}: {_pct(trial['offline']['validation_exact'])}"
            for trial in phase_a
        )
        observations.append(f"Baseline learning-rate validation exact: {descriptions}.")
    capacity = {
        trial["spec"]["architecture"]: trial
        for trial in trials
        if trial["spec"]["phase"] == "B"
    }
    baseline = next(
        (
            trial
            for trial in phase_a
            if trial["spec"]["learning_rate"]
            == _selected_baseline_lr_from_capacity(capacity, phase_a)
        ),
        None,
    )
    if baseline is not None:
        for architecture in ("S", "W", "D", "L"):
            trial = capacity.get(architecture)
            if trial is None:
                continue
            exact_delta = trial["offline"]["validation_exact"] - baseline["offline"]["validation_exact"]
            train_delta = trial["offline"]["train_exact"] - baseline["offline"]["train_exact"]
            observations.append(
                f"{architecture} vs calibrated B: validation exact {exact_delta*100:+.2f}pp, "
                f"train exact {train_delta*100:+.2f}pp, evaluation "
                f"{trial['evaluation']['wins']}-{trial['evaluation']['losses']}-{trial['evaluation']['draws']}."
            )
    seed_trials = [trial for trial in trials if int(trial["spec"]["seed"]) == 17]
    observations.append(
        "Seed confirmation is present for one structure."
        if seed_trials
        else "No seed-17 confirmation completed; observed differences retain seed uncertainty."
    )
    observations.append(
        "Exact Action Rate is an imitation-fit measure; equivalent legal actions and closed-loop "
        "state distribution make it non-interchangeable with evaluation win rate."
    )
    return observations


def _selected_baseline_lr_from_capacity(
    capacity: dict[str, dict[str, Any]],
    phase_a: list[dict[str, Any]],
) -> float:
    if capacity:
        return float(next(iter(capacity.values()))["spec"]["learning_rate"])
    return float(phase_a[0]["spec"]["learning_rate"])


def _line_chart(
    title: str,
    trials: list[dict[str, Any]],
    fields: tuple[tuple[str, str], ...],
    *,
    maximum: float | None = None,
) -> str:
    width, height, left, top, plot_width, plot_height = 680, 330, 52, 25, 600, 245
    all_values = [
        float(point[field])
        for trial in trials
        for point in trial["offline"]["curves"]
        for field, _ in fields
    ]
    y_max = maximum or max(all_values, default=1.0) * 1.05
    y_max = max(y_max, 0.001)
    elements = [_axes(width, height, left, top, plot_width, plot_height, 0, y_max)]
    legend: list[str] = []
    color_index = 0
    for trial in trials:
        curves = trial["offline"]["curves"]
        for field, suffix in fields:
            color = COLORS[color_index % len(COLORS)]
            color_index += 1
            points = " ".join(
                f"{left + (int(point['epoch'])-1)/19*plot_width:.1f},"
                f"{top + plot_height-float(point[field])/y_max*plot_height:.1f}"
                for point in curves
            )
            dash = " stroke-dasharray='5 4'" if suffix == "validation" else ""
            elements.append(
                f"<polyline points='{points}' fill='none' stroke='{color}' "
                f"stroke-width='2'{dash}/>"
            )
            legend.append(
                f"<span style='color:{color}'>{html.escape(trial['spec']['version'])} {suffix}</span>"
            )
    return _figure(title, width, height, "".join(elements), " · ".join(legend))


def _scatter_chart(
    title: str,
    trials: list[dict[str, Any]],
    x_value: Callable[[dict[str, Any]], float],
    y_value: Callable[[dict[str, Any]], float],
    x_label: str,
    y_label: str,
    *,
    x_maximum: float | None = None,
    y_maximum: float | None = None,
) -> str:
    width, height, left, top, plot_width, plot_height = 680, 330, 62, 25, 585, 240
    xs = [x_value(trial) for trial in trials]
    ys = [y_value(trial) for trial in trials]
    x_max = x_maximum or max(xs, default=1.0) * 1.05
    y_max = y_maximum or max(ys, default=1.0) * 1.08
    x_max, y_max = max(x_max, 0.001), max(y_max, 0.001)
    elements = [_axes(width, height, left, top, plot_width, plot_height, 0, y_max)]
    for index, trial in enumerate(trials):
        x = left + x_value(trial) / x_max * plot_width
        y = top + plot_height - y_value(trial) / y_max * plot_height
        color = COLORS[index % len(COLORS)]
        label = html.escape(trial["spec"]["version"].split("_", 1)[0])
        elements.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{color}'/>"
            f"<text x='{x+7:.1f}' y='{y-7:.1f}' font-size='11'>{label}</text>"
        )
    elements.append(
        f"<text x='{left+plot_width/2:.1f}' y='{height-10}' text-anchor='middle' "
        f"font-size='11'>{html.escape(x_label)}</text>"
        f"<text x='13' y='{top+plot_height/2:.1f}' transform='rotate(-90 13 "
        f"{top+plot_height/2:.1f})' text-anchor='middle' font-size='11'>"
        f"{html.escape(y_label)}</text>"
    )
    return _figure(title, width, height, "".join(elements), "")


def _axes(
    width: int,
    height: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    minimum: float,
    maximum: float,
) -> str:
    del width, height
    lines = [
        f"<rect x='{left}' y='{top}' width='{plot_width}' height='{plot_height}' "
        "fill='#fafbfc' stroke='#ccd2d9'/>"
    ]
    for tick in range(5):
        value = minimum + (maximum - minimum) * tick / 4
        y = top + plot_height - tick / 4 * plot_height
        lines.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left+plot_width}' y2='{y:.1f}' "
            "stroke='#e2e6ea'/><text x='46' y='"
            f"{y+4:.1f}' text-anchor='end' font-size='10'>{value:.2f}</text>"
        )
    return "".join(lines)


def _figure(title: str, width: int, height: int, body: str, legend: str) -> str:
    return (
        f"<figure><figcaption>{html.escape(title)}</figcaption>"
        f"<svg viewBox='0 0 {width} {height}' role='img'>{body}</svg>"
        f"<div class='muted' style='font-size:11px'>{legend}</div></figure>"
    )


def _update_index(
    repository_root: Path,
    experiment_root: Path,
    comparison: dict[str, Any],
) -> None:
    index_path = repository_root / "rl_runs/INDEX.html"
    content = index_path.read_text(encoding="utf-8")
    marker_start = f"<!-- BC_CAPACITY_SEARCH_START:{experiment_root.name} -->"
    marker_end = f"<!-- BC_CAPACITY_SEARCH_END:{experiment_root.name} -->"
    selection = comparison.get("selection") or {}
    selected_version = str(selection.get("version", ""))
    rows = "".join(
        _index_trial_row(experiment_root, trial, selected_version=selected_version)
        for trial in comparison["trials"]
    )
    selection_text = (
        f"用户已在搜索冻结后选择 <code>{html.escape(selected_version)}</code>，并完成正式归档与一次 Kaggle submission。"
        if selected_version
        else "等待用户选择结构。"
    )
    section = f"""{marker_start}
  <h2>{html.escape(experiment_root.name)} · BC capacity search</h2>
  <p class="subtle">Evaluation-first 210-minute campaign. 固定 0003 Yushin Ito 数据；当前状态 <code>{html.escape(comparison.get('status','running'))}</code>；{selection_text}</p>
  <table>
    <thead><tr><th>Trial</th><th>Evaluation 结果</th><th>先/后手与过程</th><th>模型结构</th><th>训练配置</th><th>BC 辅助</th><th>Artifacts</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="7">Campaign initialized; first trial is running.</td></tr>'}</tbody>
  </table>
  <p><a href="{html.escape(experiment_root.name)}/bc_capacity_search_report.html">打开完整 campaign 报告</a> · <a href="{html.escape(experiment_root.name)}/promotion.json">V4 promotion</a> · trial 矩阵保持原始顺序，用户选择不回写搜索分支。</p>
{marker_end}"""
    if marker_start in content and marker_end in content:
        before = content.split(marker_start, 1)[0]
        after = content.split(marker_end, 1)[1]
        content = before + section + after
    else:
        anchor = "  <h2>当前方向</h2>"
        if anchor not in content:
            anchor = "<h2>当前方向</h2>"
        if anchor not in content:
            raise RuntimeError("rl_runs/INDEX.html is missing the current-direction anchor")
        content = content.replace(anchor, section + "\n\n" + anchor, 1)
    index_path.write_text(content, encoding="utf-8")


def _index_trial_row(
    experiment_root: Path,
    trial: dict[str, Any],
    *,
    selected_version: str = "",
) -> str:
    spec = trial["spec"]
    if trial.get("status") != "completed":
        return (
            f"<tr><td><code>{html.escape(spec['version'])}</code></td>"
            f"<td class='reject' colspan='6'>{html.escape(str(trial.get('failure','incomplete')))}</td></tr>"
        )
    evaluation, offline = trial["evaluation"], trial["offline"]
    metrics = evaluation["metrics"]
    first = evaluation.get("by_turn_order", {}).get("first", {})
    second = evaluation.get("by_turn_order", {}).get("second", {})
    report = _relative_to_runs(experiment_root, evaluation["report_html"])
    decision_label = "user selected" if spec["version"] == selected_version else "complete"
    return f"""<tr>
<td><code>{html.escape(spec['version'])}</code><br>Phase {spec['phase']} · {spec['architecture']}<br><span class="keep">{decision_label}</span></td>
<td><strong>{evaluation['wins']}-{evaluation['losses']}-{evaluation['draws']} · {_pct(evaluation['win_rate'])}</strong><br>{evaluation['completed_games']}/180 · {evaluation['errors']} errors · {evaluation['unfinished']} unfinished</td>
<td>先手 {_turn_order(first)} · 后手 {_turn_order(second)}<br>Correct {_metric(metrics,'correctness')} · PH {_metric(metrics,'powerful_hand')} · Post-KO {_metric(metrics,'post_ko_relay')} · Attack {_metric(metrics,'attack_quality')}</td>
<td>{trial['model']['parameter_count']:,} params<br>d={spec['d_model']} · layers={spec['layers']} · hidden={spec['hidden_dim']}</td>
<td>lr={spec['learning_rate']:g} · seed={spec['seed']} · dropout={spec['dropout']:g}<br>{trial['training']['wall_seconds']/60:.1f} min · {trial['training']['peak_gpu_memory_bytes']/1024**3:.2f} GiB</td>
<td>val exact {_pct(offline['validation_exact'])}<br>policy {offline['train_policy_loss']:.4f}/{offline['validation_policy_loss']:.4f}<br>gap {_pp(offline['exact_gap'])} · epoch {offline['best_epoch']}</td>
<td><a href="{html.escape(report)}">evaluation</a> · <a href="{html.escape(experiment_root.name)}/{html.escape(spec['version'])}/trial_result.json">trial</a></td>
</tr>"""


def _relative_to_experiment(experiment_root: Path, repository_relative: str) -> str:
    absolute = experiment_root.parents[2] / repository_relative
    try:
        return str(absolute.resolve().relative_to(experiment_root))
    except ValueError:
        return repository_relative


def _relative_to_runs(experiment_root: Path, repository_relative: str) -> str:
    prefix = "rl_runs/"
    return repository_relative[len(prefix) :] if repository_relative.startswith(prefix) else repository_relative


def _turn_order(value: dict[str, Any]) -> str:
    games = int(value.get("games", value.get("denominator", 0)) or 0)
    wins = int(value.get("wins", value.get("numerator", 0)) or 0)
    rate = value.get("value")
    return f"{wins}/{games} ({_pct(rate)})" if games else "n/a"


def _metric(metrics: dict[str, Any], metric_id: str) -> str:
    value = metrics.get(metric_id, {}).get("value")
    return _number(value)


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return html.escape(str(value))


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _pp(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:+.2f}pp"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
