"""Render lightweight HTML pages for the V8 semantic-recovery history."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CSS = """
:root { color-scheme: light; --ink:#1d2935; --muted:#64717e; --line:#d9e0e6; --accent:#0d6b68; --wash:#f5f8f8; --warn:#8a5a00; }
* { box-sizing:border-box; }
body { margin:0; background:#eef2f3; color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 20px 56px; }
header { border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:22px; }
h1 { margin:0 0 6px; font-size:28px; } h2 { font-size:18px; margin:0 0 14px; }
p { color:var(--muted); margin:5px 0; }
.cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:24px; }
.card, section { background:white; border:1px solid var(--line); border-radius:8px; }
.card { padding:16px; min-height:112px; } .card span,.card small { display:block; color:var(--muted); }
.card strong { display:block; font-size:25px; margin:10px 0 4px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-bottom:16px; }
section { padding:18px; overflow:auto; margin-bottom:16px; }
table { width:100%; border-collapse:collapse; } th,td { padding:8px 6px; border-bottom:1px solid #edf0f2; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; } a { color:var(--accent); } .tag { display:inline-block; padding:2px 8px; border-radius:12px; background:#e6f2f1; color:var(--accent); font-size:12px; }
.tag.focused { background:#fff3d8; color:var(--warn); } .notice { background:var(--wash); border-left:3px solid var(--accent); padding:10px 12px; color:#42515d; margin-top:12px; }
.muted { color:var(--muted); } .good { color:#176b42; } .bad { color:#9a2d2d; }
@media (max-width:860px) { .cards,.grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:600px) { .cards,.grid { grid-template-columns:1fr; } main { padding:20px 12px 40px; } }
"""


def load_summary(history_dir: Path) -> dict[str, Any]:
    value = json.loads((history_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("summary.json must contain an object with a records list")
    return value


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _ratio(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "—"
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        return "—"
    rate = f" ({numerator / denominator:.1%})" if denominator else ""
    return f"{numerator}/{denominator}{rate}"


def _rate(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "—"


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("each record needs a non-empty string id")
    return value


def _sample_label(record: Mapping[str, Any]) -> str:
    sample_type = record.get("sample_type")
    games = record.get("games")
    if sample_type == "focused":
        return f"focused · {games} 局"
    if games == 170 or sample_type == "full":
        return "full · 17×10 · 170 局"
    return f"{sample_type or record.get('kind', 'record')} · {games or '—'} 局"


def _wld(record: Mapping[str, Any]) -> str:
    values = [record.get(key) for key in ("wins", "losses", "draws")]
    return "/".join(str(value) if isinstance(value, int) else "—" for value in values)


def _post_ko(record: Mapping[str, Any]) -> tuple[str, str]:
    ready = record.get("post_ko_ready_attacker")
    if isinstance(ready, Mapping):
        return "立即 ready", _ratio(ready)
    zero_ready = record.get("post_ko_zero_ready")
    if isinstance(zero_ready, Mapping):
        return "zero-ready", _ratio(zero_ready)
    return "post-KO", "—"


def _markdown_link(history_dir: Path, record_id: str, *, prefix: str) -> str | None:
    candidates = (
        history_dir / f"{record_id}.md",
        history_dir / record_id / f"{record_id}.md",
        history_dir / record_id / "iteration.md",
        history_dir / record_id / "README.md",
    )
    for candidate in candidates:
        if candidate.exists():
            return f"{prefix}{candidate.relative_to(history_dir).as_posix()}"
    return None


def _style() -> str:
    return f"<style>{CSS}</style>"


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{_escape(title)}</title>{_style()}</head><body><main>{body}</main></body></html>"
    )


def _metric_card(label: str, value: str, detail: str = "") -> str:
    return (
        f"<article class=\"card\"><span>{_escape(label)}</span>"
        f"<strong>{_escape(value)}</strong><small>{_escape(detail)}</small></article>"
    )


def _record_row(record: Mapping[str, Any], history_dir: Path) -> str:
    record_id = _record_id(record)
    sample_type = "focused" if record.get("sample_type") == "focused" else "full"
    post_label, post_value = _post_ko(record)
    link = f"{record_id}/index.html"
    return (
        "<tr>"
        f"<td><a href=\"{_escape(link)}\">{_escape(record_id)}</a></td>"
        f"<td><span class=\"tag {'focused' if sample_type == 'focused' else ''}\">{sample_type}</span>"
        f"<br><span class=\"muted\">{_escape(_sample_label(record))}</span></td>"
        f"<td>{_escape(_wld(record))}</td><td>{_escape(_rate(record.get('win_rate')))}</td>"
        f"<td>{_escape(_rate(record.get('meta_weighted_win_rate')))}</td>"
        f"<td>{_escape(str(record.get('errors', '—')))}</td>"
        f"<td>{_escape(_ratio(record.get('second_turn_powerful_hand')))}</td>"
        f"<td>{_escape(post_label)}: {_escape(post_value)}</td>"
        f"<td>{_escape(str(record.get('decision', '—')))}</td></tr>"
    )


def _record_page(record: Mapping[str, Any], history_dir: Path) -> str:
    record_id = _record_id(record)
    post_label, post_value = _post_ko(record)
    markdown = _markdown_link(history_dir, record_id, prefix="../")
    markdown_cell = f"<a href=\"{_escape(markdown)}\">Markdown 详细记录</a>" if markdown else "—"
    sample = _sample_label(record)
    root_cause = record.get("root_cause", "—")
    body = (
        f"<header><h1>{_escape(record_id)}</h1><p><span class=\"tag {'focused' if record.get('sample_type') == 'focused' else ''}\">{_escape(sample)}</span></p>"
        f"<p><a href=\"../index.html\">返回 V8 history 总览</a> · {markdown_cell}</p></header>"
        "<div class=\"cards\">"
        + _metric_card("W / L / D", _wld(record), "胜 / 负 / 和")
        + _metric_card("总体胜率", _rate(record.get("win_rate")), f"{record.get('games', '—')} 局")
        + _metric_card("Meta 加权胜率", _rate(record.get("meta_weighted_win_rate")), "若有记录")
        + _metric_card("Agent error", str(record.get("errors", "—")), "correctness gate")
        + "</div>"
        "<div class=\"grid\"><section><h2>统一核心指标</h2><table>"
        f"<tr><th>指标</th><th>结果</th></tr><tr><td>评测矩阵</td><td>{_escape(sample)}</td></tr>"
        f"<tr><td>第二回合实际 attackId=1072</td><td>{_escape(_ratio(record.get('second_turn_powerful_hand')))}</td></tr>"
        f"<tr><td>{_escape(post_label)}</td><td>{_escape(post_value)}</td></tr>"
        f"<tr><td>post-KO 机会</td><td>{_escape(str(record.get('post_ko', '—')))}</td></tr>"
        f"<tr><td>出现打手断档的对局</td><td>{_escape(_ratio(record.get('game_level_breaks')))}</td></tr>"
        "</table></section><section><h2>本轮决策</h2><table>"
        f"<tr><th>字段</th><th>内容</th></tr><tr><td>类型</td><td>{_escape(record.get('kind', '—'))}</td></tr>"
        f"<tr><td>Decision</td><td>{_escape(record.get('decision', '—'))}</td></tr>"
        f"<tr><td>根因/备注</td><td>{_escape(root_cause)}</td></tr>"
        "</table></section></div>"
        "<div class=\"notice\">本页只展示轻量 summary。完整 trace 仍按 FIFO 策略保留在临时目录，未复制到 history。</div>"
    )
    return _page(record_id, body)


def _index_page(records: Sequence[Mapping[str, Any]], history_dir: Path) -> str:
    target = next((record for record in records if record.get("kind") == "target"), None)
    start = next((record for record in records if record.get("kind") == "start"), None)
    strategy = [record for record in records if record.get("kind") == "strategy" and record.get("games") == 170]
    best = max(strategy, key=lambda record: float(record.get("win_rate", -1)), default=None)
    body = (
        "<header><h1>V8 Semantic Recovery History</h1>"
        "<p>统一记录 Target Baseline、Start Baseline 与后续 Auto-Iteration Sample。</p>"
        "<p>Full Sample 固定 17×10=170 局；先后手只按实际 trace 统计。focused 只用于机制诊断。</p></header>"
        "<div class=\"cards\">"
        + _metric_card("Target Baseline", _wld(target or {}), "oracle · 170 局")
        + _metric_card("Start Baseline", _wld(start or {}), "重构现状 · 170 局")
        + _metric_card("当前最佳 full", _wld(best or {}), _record_id(best) if best else "尚无")
        + _metric_card("核心二回合事件", "attackId=1072", "Powerful Hand")
        + "</div>"
        "<section><h2>Iteration Timeline</h2><table><tr><th>记录</th><th>样本</th><th>W/L/D</th><th>胜率</th><th>Meta</th><th>errors</th><th>二回合 Powerful Hand</th><th>post-KO</th><th>Decision</th></tr>"
        + "".join(_record_row(record, history_dir) for record in records)
        + "</table></section>"
        "<section><h2>阅读规则</h2><div class=\"notice\">Target 与 Start 都是完整 170 局记录，使用同一指标结构展示；focused 记录的局数较小，只验证具体语义机制，不能替代 fresh 17×10 Sample 的 promotion 证据。</div></section>"
    )
    return _page("V8 Semantic Recovery History", body)


def build_history_pages(summary: Mapping[str, Any], history_dir: Path) -> dict[str, str]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("summary must contain a records list")
    typed_records = [record for record in records if isinstance(record, Mapping)]
    pages = {"index.html": _index_page(typed_records, history_dir)}
    for record in typed_records:
        record_id = _record_id(record)
        pages[f"{record_id}/index.html"] = _record_page(record, history_dir)
    return pages


def render_history(history_dir: Path, output_root: Path | None = None) -> list[Path]:
    summary = load_summary(history_dir)
    destination = output_root or history_dir
    pages = build_history_pages(summary, history_dir)
    written: list[Path] = []
    for relative_path, content in pages.items():
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    written = render_history(args.history_dir, args.output_root)
    print(f"rendered {len(written)} HTML pages under {args.output_root or args.history_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
