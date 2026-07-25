from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


_REPORT_DATA_START = '<script id="report-data" type="application/json">'
_REPORT_DATA_END = "</script>"
_VERSION_FILE = re.compile(r"^V(?P<version>[1-9]\d*)(?P<tag>.*)\.html$", re.IGNORECASE)


def write_evaluation_index(project_root: Path) -> Path:
    """Regenerate one project's browsable overview from its standalone reports."""
    project_root.mkdir(parents=True, exist_ok=True)
    reports = [_report_row(path) for path in _version_reports(project_root)]
    output_path = project_root / "index.html"
    temporary_path = project_root / ".index.html.tmp"
    temporary_path.write_text(_render_index(project_root.name, reports), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def _version_reports(project_root: Path) -> list[Path]:
    return sorted(
        (path for path in project_root.glob("V*.html") if path.is_file()),
        key=_version_sort_key,
    )


def _version_sort_key(path: Path) -> tuple[int, str]:
    match = _VERSION_FILE.fullmatch(path.name)
    if match is None:
        return (10**9, path.name.lower())
    return (int(match.group("version")), path.name.lower())


def _report_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"file": path.name, "version": path.stem, "error": None}
    try:
        payload = _embedded_report_data(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        row["error"] = str(exc)
        return row

    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    manifest = manifest if isinstance(manifest, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    candidate = manifest.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    row.update(
        {
            "run_id": manifest.get("run_id"),
            "candidate": candidate.get("display_name") or candidate.get("name"),
            "finished_at": manifest.get("finished_at"),
            "games": summary.get("total_games"),
            "wins": summary.get("wins"),
            "losses": summary.get("losses"),
            "draws": summary.get("draws"),
            "errors": summary.get("errors"),
            "unfinished": summary.get("unfinished"),
            "win_rate": summary.get("win_rate"),
            "completion_rate": summary.get("completion_rate"),
        }
    )
    return row


def _embedded_report_data(path: Path) -> dict[str, Any]:
    document = path.read_text(encoding="utf-8")
    try:
        payload = document.split(_REPORT_DATA_START, 1)[1].split(_REPORT_DATA_END, 1)[0]
    except IndexError as exc:
        raise ValueError("report-data payload is missing") from exc
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("report-data payload must be an object")
    return decoded


def _render_index(project_name: str, reports: list[dict[str, Any]]) -> str:
    total_games = sum(_integer(row.get("games")) for row in reports)
    zero_error_reports = sum(
        row.get("error") is None and _integer(row.get("errors")) == 0 for row in reports
    )
    rows = "".join(_render_row(row) for row in reports)
    if not rows:
        rows = '<tr><td colspan="10" class="empty">尚无 V*.html 正式评测报告</td></tr>'
    title = html.escape(project_name)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Evaluation 总览</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d9e0ea;
  --panel:#fff; --accent:#3157d5; --bg:#f4f7fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; }}
main {{ max-width:1320px; margin:0 auto; padding:32px 24px 56px; }}
h1 {{ margin:0 0 6px; font-size:28px; }}
.lead {{ margin:0 0 22px; color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.card strong {{ display:block; font-size:24px; }}
.card span {{ color:var(--muted); }}
.table-wrap {{ overflow:auto; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:right; }}
th {{ background:#eef3fb; color:#475467; font-size:12px; text-transform:uppercase; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }}
a {{ color:var(--accent); font-weight:650; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
.bad {{ color:#b42318; }} .muted,.empty {{ color:var(--muted); }}
@media (max-width:720px) {{ .cards {{ grid-template-columns:1fr; }} main {{ padding:20px 12px; }} }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<p class="lead">项目级 Evaluation 总览。每个版本仍以独立 HTML 保存；点击版本名查看完整报告。</p>
<section class="cards">
<div class="card"><strong>{len(reports)}</strong><span>版本报告</span></div>
<div class="card"><strong>{total_games}</strong><span>累计对局</span></div>
<div class="card"><strong>{zero_error_reports}</strong><span>零 error 报告</span></div>
</section>
<div class="table-wrap"><table>
<thead><tr><th>版本报告</th><th>candidate / run</th><th>对局</th><th>胜</th><th>负</th>
<th>平</th><th>error</th><th>unfinished</th><th>胜率</th><th>完成率</th></tr></thead>
<tbody>{rows}</tbody>
</table></div>
</main></body></html>
"""


def _render_row(row: dict[str, Any]) -> str:
    file_name = html.escape(str(row["file"]), quote=True)
    version = html.escape(str(row["version"]))
    if row.get("error") is not None:
        detail = html.escape(str(row["error"]))
        return (
            f'<tr><td><a href="{file_name}">{version}</a></td>'
            f'<td colspan="9" class="bad">无法解析：{detail}</td></tr>'
        )
    candidate = html.escape(str(row.get("candidate") or "—"))
    run_id = html.escape(str(row.get("run_id") or "—"))
    error_class = ' class="bad"' if _integer(row.get("errors")) else ""
    return (
        f'<tr><td><a href="{file_name}">{version}</a></td>'
        f'<td>{candidate}<br><span class="muted">{run_id}</span></td>'
        f'<td>{_display(row.get("games"))}</td><td>{_display(row.get("wins"))}</td>'
        f'<td>{_display(row.get("losses"))}</td><td>{_display(row.get("draws"))}</td>'
        f'<td{error_class}>{_display(row.get("errors"))}</td>'
        f'<td>{_display(row.get("unfinished"))}</td>'
        f'<td>{_percentage(row.get("win_rate"))}</td>'
        f'<td>{_percentage(row.get("completion_rate"))}</td></tr>'
    )


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _display(value: Any) -> str:
    return html.escape(str(value)) if isinstance(value, (int, float)) else "—"


def _percentage(value: Any) -> str:
    return f"{float(value):.2%}" if isinstance(value, (int, float)) else "—"
