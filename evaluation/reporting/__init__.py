from .html import render_html
from .index import write_evaluation_index
from .markdown import render_markdown
from .models import ReportData, json_ready

from pathlib import Path


def write_report(data: ReportData, output_dir: Path) -> None:
    """写入默认唯一长期产物：独立 HTML 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.html").write_text(render_html(data), encoding="utf-8")


def write_report_file(data: ReportData, output_path: Path) -> None:
    """Write one standalone report to an explicit immutable HTML path."""
    if output_path.suffix.lower() != ".html":
        raise ValueError("evaluation report path must end with .html")
    if output_path.exists():
        raise ValueError(f"evaluation report already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(data), encoding="utf-8")


__all__ = (
    "ReportData",
    "json_ready",
    "render_html",
    "render_markdown",
    "write_report",
    "write_report_file",
    "write_evaluation_index",
)
