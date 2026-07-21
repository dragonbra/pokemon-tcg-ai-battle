from .html import render_html
from .markdown import render_markdown
from .models import ReportData

from pathlib import Path


def write_report(data: ReportData, output_dir: Path) -> None:
    """写入同源的 Markdown 与独立 HTML 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(render_markdown(data), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(data), encoding="utf-8")


__all__ = ("ReportData", "render_html", "render_markdown", "write_report")
