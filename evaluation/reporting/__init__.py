from .html import render_html
from .markdown import render_markdown
from .models import ReportData, json_ready

from pathlib import Path


def write_report(data: ReportData, output_dir: Path) -> None:
    """写入默认唯一长期产物：独立 HTML 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.html").write_text(render_html(data), encoding="utf-8")


__all__ = ("ReportData", "json_ready", "render_html", "render_markdown", "write_report")
