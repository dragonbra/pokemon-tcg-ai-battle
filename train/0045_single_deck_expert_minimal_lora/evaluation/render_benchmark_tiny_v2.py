"""Render a compact authoritative 0045 Benchmark Tiny V2 HTML report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path

from ..assets import AssetRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows(entries, key, labels=None):
    grouped = defaultdict(list)
    for row in entries:
        grouped[str(row[key])].append(row)
    result = []
    for identity, games in sorted(grouped.items()):
        wins = sum(row["outcome"] == 1 for row in games)
        losses = sum(row["outcome"] == -1 for row in games)
        draws = len(games) - wins - losses
        label = identity if labels is None else f"{identity} · {labels.get(identity, 'Unknown')}"
        result.append((label, len(games), wins, losses, draws, wins / len(games)))
    return result


def render(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("benchmark_id") != "Benchmark-Tiny-V2":
        raise RuntimeError("refusing to render a non-PASS Tiny V2 report")
    summary = report["summary"]
    audit = report["focal_policy_identity_audit"]
    deck_id = str(report["focal_deck_id"])
    deck_name = str(report["focal_deck_display_name"])
    checkpoint_update = int(report["focal_checkpoint_update"])
    title = f"Deck {deck_id} · U{checkpoint_update} · Benchmark Tiny V2"
    deck_names = {
        row.deck_id: row.name for row in AssetRegistry.load(PROJECT_ROOT).decks
    }
    def table(title, rows):
        body = "".join(
            f"<tr><td>{escape(identity)}</td><td>{games}</td><td>{wins}</td><td>{losses}</td><td>{draws}</td><td>{rate:.2%}</td></tr>"
            for identity, games, wins, losses, draws, rate in rows
        )
        return f"<h2>{title}</h2><table><tr><th>ID</th><th>Games</th><th>W</th><th>L</th><th>D</th><th>Win rate</th></tr>{body}</table>"
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{escape(title)}</title><style>body{{font:16px/1.55 system-ui;max-width:1180px;margin:32px auto;padding:0 24px;color:#18212b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#eef4fb;padding:14px;border-radius:10px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5df;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>{escape(title)}</h1>
<p>{escape(deck_name)}</p>
<p><strong>Status:</strong> PASS · official CUDA engine · greedy · 512/512 terminal · 0 error · 0 unfinished</p>
<div class='cards'><div class='card'><b>Win rate</b><br>{summary['win_rate']:.2%}</div><div class='card'><b>W–L–D</b><br>{summary['wins']}–{summary['losses']}–{summary['draws']}</div><div class='card'><b>First</b><br>{summary['focal_first_win_rate']:.2%} (n={summary['focal_first_games']})</div><div class='card'><b>Second</b><br>{summary['focal_second_win_rate']:.2%} (n={summary['focal_second_games']})</div></div>
<h2>Deployment identity</h2><p>Contract: <code>{escape(audit['contract_id'])}</code><br>Source FP32: <code>{escape(audit['source_checkpoint_sha256'])}</code><br>Portable FP16: <code>{escape(audit['portable_checkpoint_sha256'])}</code><br>Effective FP32 runtime: <code>{escape(audit['effective_candidate_sha256'])}</code></p>
{table('By opponent Meta archetype', _rows(report['entries'], 'opponent_meta_archetype_id'))}
{table('By exact opponent deck', _rows(report['entries'], 'opponent_id', deck_names))}
<h2>Evidence boundary</h2><p>This is a fixed 512-game Tiny diagnostic, not the full CUDA-2048 Promote benchmark. Sampled training rollout metrics must not be compared as greedy strength.</p>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
