"""Evaluate the external 0031 friend-0805 checkpoint on selected Frozen exact decks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from evaluation.reporting.index import _embedded_report_data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = REPOSITORY_ROOT / "evaluation/arena/frozen"
CHECKPOINT_ARCHIVE = REPOSITORY_ROOT / "pt0805.tar.gz"
CHECKPOINT = (
    REPOSITORY_ROOT
    / ".tmp/model_loading/0031_friend_0805/checkpoint/best_validation_loss.pt"
)
CG_SOURCE = REPOSITORY_ROOT / "evaluation/arena/frozen/_policy/cg"
PACKAGE_ROOT = REPOSITORY_ROOT / ".tmp/model_loading/0031_friend_0805/candidates"
TEMP_ROOT = REPOSITORY_ROOT / ".tmp/evaluation/0031_friend_0805_frozen_test"
OUTPUT_ROOT = REPOSITORY_ROOT / "evaluation/arena/combat_mat/0031_friend_0805_frozen_test"
PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"
FRIEND_VERSION_LABEL = "friend_0805_best_validation_loss"
EXPECTED_METADATA_VERSION = "V1_starter_daily_topn_20260729_20260803"
ARM = "rule_faithful_semantic"
POOL_ID = "0019_foundation_51_exact_decks_v4"
EXPECTED_GAMES = 510
PRIORITY_DECKS = (
    "dragapult_ex_dusknoir_001",
    "raging_bolt_ex_james_cox_henry_chao_001",
    "festival_lead_dipplin_001",
)

# The frozen manifest is the source of truth for the complete 51-deck queue.
# Priority decks are evaluated first; the remaining decks retain stable ID order.
DECKS = tuple(
    dict.fromkeys(
        (*PRIORITY_DECKS,
         *sorted(
             path.name
             for path in FROZEN_ROOT.iterdir()
             if path.is_dir() and path.name != "_policy"
         ))
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_hash(deck: list[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def ensure_checkpoint() -> Path:
    if CHECKPOINT.is_file():
        return CHECKPOINT
    if not CHECKPOINT_ARCHIVE.is_file():
        raise FileNotFoundError(CHECKPOINT_ARCHIVE)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", str(CHECKPOINT_ARCHIVE), "-C", str(CHECKPOINT.parent)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    return CHECKPOINT


def load_decks(path: Path = FROZEN_ROOT) -> list[dict[str, object]]:
    result = []
    for deck_id in DECKS:
        deck_root = path / deck_id
        record = json.loads((deck_root / "manifest.json").read_text(encoding="utf-8"))
        deck = [
            int(line)
            for line in (deck_root / "deck.csv").read_text(encoding="ascii").splitlines()
            if line.strip()
        ]
        if (
            record.get("deck_id") != deck_id
            or len(deck) != 60
            or not all(type(card_id) is int and card_id > 0 for card_id in deck)
            or record.get("deck_sha256") != _deck_hash(deck)
        ):
            raise ValueError(f"Frozen deck identity mismatch: {deck_id}")
        record["deck"] = deck
        result.append(record)
    return result


def validate_checkpoint(path: Path = CHECKPOINT) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    contract = metadata.get("model_config") if isinstance(metadata, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "state_dict", "metadata"}
        or payload.get("schema_version") != "0031_model_only_checkpoint_v1"
        or not isinstance(metadata, dict)
        or metadata.get("project_id") != PROJECT_ID
        or metadata.get("version") != EXPECTED_METADATA_VERSION
        or metadata.get("arm") != ARM
        or metadata.get("epoch") != 13
        or metadata.get("global_step") != 109135
        or not isinstance(contract, dict)
        or contract.get("model") != "SemanticPolicy"
    ):
        raise ValueError("checkpoint does not match the audited friend-0805 0031 contract")
    return metadata


def materialize_package(deck_record: dict[str, object], checkpoint: Path = CHECKPOINT) -> Path:
    deck_id = str(deck_record["deck_id"])
    output = PACKAGE_ROOT / deck_id
    checkpoint_sha256 = _sha256(checkpoint)
    if output.exists():
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if (
            manifest.get("checkpoint_sha256") == checkpoint_sha256
            and manifest.get("deck_sha256") == deck_record["deck_sha256"]
            and manifest.get("deck_id") == deck_id
        ):
            return output
        raise FileExistsError(f"conflicting candidate package already exists: {output}")
    deck_path = PACKAGE_ROOT.parent / "decks" / f"{deck_id}.csv"
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck_path.write_text(
        "".join(f"{card_id}\n" for card_id in deck_record["deck"]), encoding="ascii"
    )
    exporter = importlib.import_module(
        "train.0031_rule_faithful_semantic_foundation_pretraining.export_candidate"
    )
    exporter.export_candidate(
        checkpoint=checkpoint,
        deck_path=deck_path,
        cg_source=CG_SOURCE,
        output=output,
        deck_id=deck_id,
    )
    return output


def validate_report(
    report_path: Path,
    deck_record: dict[str, object],
    checkpoint_sha256: str,
    *,
    expected_games: int = EXPECTED_GAMES,
) -> dict[str, object]:
    payload = _embedded_report_data(report_path)
    manifest, summary, games = (
        payload.get("manifest"), payload.get("summary"), payload.get("games")
    )
    if not isinstance(manifest, dict) or not isinstance(summary, dict) or not isinstance(games, list):
        raise ValueError("evaluation report payload is incomplete")
    candidate = manifest.get("candidate")
    package = candidate.get("package_manifest") if isinstance(candidate, dict) else None
    pool = manifest.get("opponent_pool")
    if not isinstance(candidate, dict) or not isinstance(package, dict):
        raise ValueError("evaluation report lacks candidate identity")
    if not isinstance(pool, dict) or pool.get("pool_id") != POOL_ID:
        raise ValueError("evaluation report Frozen pool identity mismatch")
    if (
        package.get("project_id") != PROJECT_ID
        or package.get("version") != EXPECTED_METADATA_VERSION
        or package.get("arm") != ARM
        or package.get("checkpoint_selection") != "best_validation_loss"
        or package.get("checkpoint_sha256") != checkpoint_sha256
        or package.get("deck_id") != deck_record["deck_id"]
        or package.get("deck_sha256") != deck_record["deck_sha256"]
        or Counter(candidate.get("deck", [])) != Counter(deck_record["deck"])
    ):
        raise ValueError("evaluation report candidate provenance mismatch")
    if (
        manifest.get("games") != expected_games
        or len(manifest.get("opponents", [])) != 51
        or summary.get("total_games") != expected_games
        or summary.get("completed_games") != expected_games
        or summary.get("errors") != 0
        or summary.get("unfinished") != 0
        or len(games) != expected_games
    ):
        raise ValueError("evaluation report is partial or contains errors")
    turn_order = {
        "first": {"wins": 0, "losses": 0, "draws": 0},
        "second": {"wins": 0, "losses": 0, "draws": 0},
    }
    for game in games:
        if not isinstance(game, dict) or game.get("status") != "finished":
            raise ValueError("evaluation report contains an unfinished game")
        order = "first" if game.get("candidate_first") else "second"
        key = "wins" if game.get("winner") == 0 else "losses" if game.get("winner") == 1 else "draws"
        turn_order[order][key] += 1
    return {
        "deck_id": deck_record["deck_id"],
        "display_name": deck_record["display_name"],
        "deck_sha256": deck_record["deck_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "pool_id": POOL_ID,
        "run_id": manifest.get("run_id"),
        "wall_time_seconds": manifest.get("wall_time_seconds"),
        "games": expected_games,
        "wins": summary.get("wins"),
        "losses": summary.get("losses"),
        "draws": summary.get("draws"),
        "errors": 0,
        "completion_rate": summary.get("completion_rate"),
        "win_rate": summary.get("win_rate"),
        "turn_order": turn_order,
        "report_sha256": _sha256(report_path),
        "report": f"reports/{deck_record['deck_id']}.html",
    }


def publish_report(source: Path, deck_record: dict[str, object], checkpoint_sha256: str) -> dict[str, object]:
    record = validate_report(source, deck_record, checkpoint_sha256)
    target = OUTPUT_ROOT / str(record["report"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = validate_report(target, deck_record, checkpoint_sha256)
        if existing["report_sha256"] != record["report_sha256"]:
            raise FileExistsError(f"refusing to overwrite a different report: {target}")
        return existing
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def refresh_index(decks: list[dict[str, object]], checkpoint_sha256: str) -> list[dict[str, object]]:
    records = []
    for deck in decks:
        report = OUTPUT_ROOT / "reports" / f"{deck['deck_id']}.html"
        if report.is_file():
            records.append(validate_report(report, deck, checkpoint_sha256))
    manifest = {
        "schema_version": "0031_friend_0805_frozen_test_v1",
        "project_id": PROJECT_ID,
        "version_label": FRIEND_VERSION_LABEL,
        "metadata_version": EXPECTED_METADATA_VERSION,
        "checkpoint_archive": str(CHECKPOINT_ARCHIVE.relative_to(REPOSITORY_ROOT)),
        "checkpoint": str(CHECKPOINT.relative_to(REPOSITORY_ROOT)),
        "checkpoint_selection": "best_validation_loss",
        "checkpoint_epoch": 13,
        "checkpoint_global_step": 109135,
        "checkpoint_sha256": checkpoint_sha256,
        "source_note": "External friend 0805 checkpoint; intended as the next pretraining-model foundation.",
        "frozen_manifest_sha256": _sha256(FROZEN_ROOT / "manifest.json"),
        "pool_id": POOL_ID,
        "games_per_opponent": 10,
        "opponents_per_report": 51,
        "requested_decks": list(DECKS),
        "published_reports": len(records),
        "published_games": sum(int(record["games"]) for record in records),
        "reports": records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def format_duration(seconds: object) -> str:
        total = max(0, int(float(seconds or 0)))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours} 小时 {minutes} 分"
        if minutes:
            return f"{minutes} 分 {secs} 秒"
        return f"{secs} 秒"

    def seat_summary(record: dict[str, object], seat: str) -> tuple[int, int, float]:
        payload = record.get("turn_order", {}).get(seat, {})
        wins = int(payload.get("wins", 0))
        games = wins + int(payload.get("losses", 0)) + int(payload.get("draws", 0))
        return wins, games, wins / games if games else 0.0

    def win_rate_style(rate: float) -> str:
        hue = round(8 + max(0.0, min(1.0, rate)) * 132)
        lightness = 58 if rate < 0.18 else round(92 - abs(rate - 0.5) * 28)
        color = "#fff" if rate < 0.18 else "#173129"
        return f"background:hsl({hue} 58% {lightness}%);color:{color}"

    total_games = sum(int(record["games"]) for record in records)
    total_wins = sum(int(record["wins"]) for record in records)
    total_errors = sum(int(record["errors"]) for record in records)
    wall_time = sum(float(record["wall_time_seconds"] or 0) for record in records)
    completed = len(records)
    progress = completed / len(DECKS) if DECKS else 0.0
    first_wins = first_games = second_wins = second_games = 0
    for record in records:
        wins, games, _ = seat_summary(record, "first")
        first_wins += wins
        first_games += games
        wins, games, _ = seat_summary(record, "second")
        second_wins += wins
        second_games += games

    rows = "".join(
        (
            f'<tr data-default-order="{index}" data-win-rate="{float(record["win_rate"]):.8f}" '
            f'data-wall-time="{float(record["wall_time_seconds"] or 0):.3f}" '
            f'data-search="{html.escape((str(record["display_name"]) + " " + str(record["deck_id"])).lower())}">'
            f'<td><a class="deck-link" href="{html.escape(str(record["report"]))}">'
            f'{html.escape(str(record["display_name"]))}</a>'
            f'<small>{html.escape(str(record["deck_id"]))}</small></td>'
            f'<td class="record">{record["wins"]}-{record["losses"]}-{record["draws"]}</td>'
            f'<td><span class="win-rate-pill" style="{win_rate_style(float(record["win_rate"]))}">'
            f'{float(record["win_rate"]):.2%}</span></td>'
            f'<td>{seat_summary(record, "first")[2]:.2%}'
            f'<span class="sample">{seat_summary(record, "first")[0]}/{seat_summary(record, "first")[1]}</span></td>'
            f'<td>{seat_summary(record, "second")[2]:.2%}'
            f'<span class="sample">{seat_summary(record, "second")[0]}/{seat_summary(record, "second")[1]}</span></td>'
            f'<td>{record["games"]}</td><td>{record["errors"]}</td>'
            f'<td>{format_duration(record["wall_time_seconds"])}</td>'
            f'<td><code>{html.escape(str(record["run_id"]))}</code></td></tr>'
        )
        for index, record in enumerate(records)
    ) or '<tr class="empty-row"><td colspan="9">尚无已完成报告</td></tr>'
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>0031 Friend 0805 Frozen Test</title>
<style data-ui-version="0031-frozen-v2">
:root{{--bg:#f3f7f5;--surface:#fff;--soft:#f7faf8;--ink:#172b25;--muted:#60736c;--line:#dce7e2;--brand:#217a58;--brand-dark:#14563d;--shadow:0 12px 32px rgba(26,71,55,.08)}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0%,rgba(58,155,112,.12),transparent 32rem),linear-gradient(180deg,#f8fbf9 0,var(--bg) 24rem);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif}}main{{max-width:1480px;margin:auto;padding:36px 28px 64px}}.hero{{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:22px;padding:30px 32px;border-radius:18px;color:#fff;background:linear-gradient(135deg,var(--brand-dark),#23835d 68%,#3b9c71);box-shadow:0 18px 44px rgba(20,86,61,.2)}}.hero h1{{margin:0;font-size:34px;letter-spacing:0}}.hero p{{margin:4px 0 0;color:#d9f0e6}}.eyebrow{{font-size:12px!important;font-weight:700;letter-spacing:.12em}}.checkpoint{{max-width:520px;padding:9px 12px;border:1px solid rgba(255,255,255,.22);border-radius:8px;background:rgba(255,255,255,.1);font:11px/1.45 ui-monospace,monospace;overflow-wrap:anywhere}}section{{margin:18px 0;padding:22px;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.97);box-shadow:var(--shadow)}}h2{{margin:0 0 6px;font-size:20px;letter-spacing:0}}.note,.section-note{{margin:4px 0;color:var(--muted)}}.summary-grid{{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px;margin-top:14px}}.summary-card{{padding:15px 16px;border:1px solid var(--line);border-radius:11px;background:linear-gradient(180deg,#fff,var(--soft))}}.summary-card .label{{display:block;color:var(--muted);font-size:12px}}.summary-card strong{{display:block;margin-top:5px;font-size:22px;letter-spacing:0}}.progress-track{{height:10px;margin-top:18px;overflow:hidden;border-radius:999px;background:#e3ece7}}.progress-bar{{display:block;width:{progress:.4%};height:100%;border-radius:inherit;background:linear-gradient(90deg,#217a58,#51aa7f)}}.progress-copy{{display:flex;justify-content:space-between;gap:16px;margin-top:7px;color:var(--muted);font-size:12px}}.contract{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:16px}}.contract div{{min-width:0;padding:10px 12px;border-left:3px solid #7ab79a;background:var(--soft)}}.contract span{{display:block;color:var(--muted);font-size:11px}}.contract code{{display:block;margin-top:3px;overflow-wrap:anywhere;color:var(--brand-dark);font-size:11px}}.table-heading{{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}}.controls{{display:flex;align-items:center;justify-content:flex-end;flex-wrap:wrap;gap:8px}}.search{{width:min(260px,100%);padding:8px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink)}}button{{padding:7px 10px;border:1px solid var(--line);border-radius:8px;color:var(--brand-dark);background:#fff;cursor:pointer}}button:hover,button.active{{border-color:#70ad91;background:#e8f5ee}}.table-scroll{{overflow:auto;margin-top:14px;border:1px solid var(--line);border-radius:10px}}table{{width:100%;min-width:1050px;border-collapse:separate;border-spacing:0;background:#fff}}th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{position:sticky;top:0;z-index:1;background:var(--soft);color:#496159;font-size:12px}}th:first-child,td:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}tbody tr:hover td{{background:#fbfdfc}}.deck-link{{color:var(--brand);font-weight:700;text-decoration:none}}small{{display:block;margin-top:2px;color:var(--muted);font:11px/1.4 ui-monospace,monospace}}.record{{font-weight:700;font-variant-numeric:tabular-nums}}.win-rate-pill{{display:inline-block;min-width:68px;padding:4px 8px;border-radius:999px;background:#e8f5ee;color:var(--brand-dark);font-weight:700;text-align:center}}.sample{{display:block;color:var(--muted);font-size:11px}}td code{{display:block;max-width:170px;overflow:hidden;text-overflow:ellipsis;color:#496159;font-size:11px}}.empty-row td{{padding:40px;text-align:center;color:var(--muted)}}.hidden{{display:none}}@media(max-width:900px){{main{{padding:18px 12px 40px}}.hero,.table-heading{{align-items:flex-start;flex-direction:column}}.hero{{padding:24px 20px}}.hero h1{{font-size:28px}}.checkpoint{{max-width:100%}}.summary-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.contract{{grid-template-columns:1fr}}.controls{{justify-content:flex-start}}}}
</style></head><body><main>
<header class="hero"><div><p class="eyebrow">POKÉMON TCG · EVALUATION ARENA</p><h1>0031 Friend 0805 · Frozen Arena</h1><p>同一模型 checkpoint 在 51 套 exact deck 上的完整 Frozen 对局评测</p></div><div class="checkpoint">{checkpoint_sha256}</div></header>
<section><h2>评测总览</h2><div class="summary-grid">
<div class="summary-card"><span class="label">已完成卡组</span><strong>{completed} / {len(DECKS)}</strong></div>
<div class="summary-card"><span class="label">已发布对局</span><strong>{total_games:,}</strong></div>
<div class="summary-card"><span class="label">总体胜率</span><strong>{total_wins / total_games if total_games else 0:.2%}</strong></div>
<div class="summary-card"><span class="label">先攻 / 后攻</span><strong>{first_wins / first_games if first_games else 0:.1%} / {second_wins / second_games if second_games else 0:.1%}</strong></div>
<div class="summary-card"><span class="label">累计评测耗时</span><strong>{format_duration(wall_time)}</strong></div>
<div class="summary-card"><span class="label">错误</span><strong>{total_errors}</strong></div>
</div><div class="progress-track" aria-label="评测完成进度"><span class="progress-bar"></span></div><div class="progress-copy"><span>{progress:.1%} complete</span><span>每套 51 opponents × 10 games</span></div>
<div class="contract"><div><span>Checkpoint selection</span><code>best_validation_loss · epoch 13 · step 109135</code></div><div><span>Metadata version</span><code>{EXPECTED_METADATA_VERSION}</code></div><div><span>Frozen pool</span><code>{POOL_ID}</code></div></div>
<p class="note">所有结果来自 official engine runtime。每套报告固定包含 510 局；完成后才会进入本页，partial run 不计入汇总。</p></section>
<section><div class="table-heading"><div><h2>每套卡组的完整评测</h2><p class="section-note">点击卡组名称进入逐 opponent 的完整报告。</p></div><div class="controls"><input class="search" id="deck-search" type="search" placeholder="搜索卡组" aria-label="搜索卡组"><button type="button" data-sort="default" class="active">默认</button><button type="button" data-sort="win_rate">胜率</button><button type="button" data-sort="wall_time">耗时</button></div></div>
<div class="table-scroll"><table id="runs"><thead><tr><th>卡组 / 报告</th><th>W-L-D</th><th>胜率</th><th>先攻</th><th>后攻</th><th>对局</th><th>错误</th><th>耗时</th><th>Run ID</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<script>(()=>{{const body=document.querySelector('#runs tbody');const rows=[...body.querySelectorAll('tr[data-default-order]')];const search=document.querySelector('#deck-search');const buttons=[...document.querySelectorAll('[data-sort]')];let sort='default';function render(){{const query=search.value.trim().toLowerCase();const ordered=[...rows].sort((a,b)=>{{if(sort==='win_rate')return Number(b.dataset.winRate)-Number(a.dataset.winRate);if(sort==='wall_time')return Number(b.dataset.wallTime)-Number(a.dataset.wallTime);return Number(a.dataset.defaultOrder)-Number(b.dataset.defaultOrder)}});for(const row of ordered){{row.classList.toggle('hidden',query&&!row.dataset.search.includes(query));body.appendChild(row)}}}}search.addEventListener('input',render);for(const button of buttons)button.addEventListener('click',()=>{{sort=button.dataset.sort;for(const item of buttons)item.classList.toggle('active',item===button);render()}});render()}})();</script>
</main></body></html>'''
    (OUTPUT_ROOT / "index.html").write_text(document, encoding="utf-8")
    return records


def _run_evaluation(package: Path, deck_id: str, games: int, output: Path) -> Path:
    before = set(output.glob("run-*/report.html")) if output.exists() else set()
    subprocess.run(
        [
            sys.executable, "-m", "evaluation", "--pool", "frozen", "run",
            "--candidate", str(package), "--opponents", "all", "--games", str(games),
            "--output", str(output), "--workers", "8", "--worker-cpu-threads", "1",
            "--candidate-device", "cuda:0", "--opponent-device", "cuda:0",
            "--metric-profile", "league_deck_quality",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    created = set(output.glob("run-*/report.html")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new {deck_id} report, found {len(created)}")
    return created.pop()


def run_deck(deck_record: dict[str, object], checkpoint_sha256: str) -> dict[str, object]:
    existing = OUTPUT_ROOT / "reports" / f"{deck_record['deck_id']}.html"
    if existing.is_file():
        return validate_report(existing, deck_record, checkpoint_sha256)
    package = materialize_package(deck_record)
    subprocess.run(
        [sys.executable, "-m", "evaluation", "validate", str(package)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    source = _run_evaluation(
        package, str(deck_record["deck_id"]), 10, TEMP_ROOT / str(deck_record["deck_id"])
    )
    return publish_report(source, deck_record, checkpoint_sha256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", action="append", choices=DECKS)
    args = parser.parse_args(argv)
    ensure_checkpoint()
    validate_checkpoint()
    decks = load_decks()
    by_id = {str(deck["deck_id"]): deck for deck in decks}
    requested = args.deck or list(DECKS)
    checkpoint_sha256 = _sha256(CHECKPOINT)
    refresh_index(decks, checkpoint_sha256)
    for deck_id in requested:
        record = run_deck(by_id[deck_id], checkpoint_sha256)
        refresh_index(decks, checkpoint_sha256)
        print(
            f"0031_FRIEND_0805_COMPLETE deck={deck_id} run_id={record['run_id']} "
            f"record={record['wins']}-{record['losses']}-{record['draws']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
