"""CUDA-engine seeded-512 zero-shot evaluation for all Frozen-0806 decks."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
POOL_ROOT = ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
POLICY_ROOT = POOL_ROOT / "policies/policy_0806"
POLICY_MODEL = (
    ROOT / "archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/model.pt"
)
BENCHMARK = ROOT / "engine_cuda/tools/benchmark_0037_update32_cuda_rollout.py"
OUTPUT_ROOT = (
    ROOT
    / "evaluation/arena/combat_mat/policy_0806"
    / "0806_kaggle_top100_plus_v1_cuda_seeded_512_v1"
)
TEMP_ROOT = ROOT / ".tmp/evaluation/policy_0806_cuda_seeded512"
EVALUATION_SEED = 341_512_806
EXPECTED_DECKS = 55
EXPECTED_GAMES = 512
POLICY_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"


def _stable_seed(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return (value & 0x7FFFFFFF) or 1


def build_cuda_schedule(
    *,
    focal_deck_id: str,
    entries: Iterable[Any],
    evaluation_seed: int = EVALUATION_SEED,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    global_game_number = 0
    for entry in entries:
        for game_number in range(1, 2 * int(entry.games) + 1):
            global_game_number += 1
            pair_number = (game_number + 1) // 2
            jobs.append(
                {
                    "game_id": f"{entry.deck_id}-{game_number:03d}",
                    "opponent_id": str(entry.deck_id),
                    "engine_seed": _stable_seed(
                        evaluation_seed,
                        focal_deck_id,
                        entry.deck_id,
                        pair_number,
                    ),
                    "search_seed": _stable_seed(
                        evaluation_seed,
                        "search",
                        focal_deck_id,
                        entry.deck_id,
                        pair_number,
                    ),
                    "focal_first": global_game_number % 2 == 1,
                }
            )
    payload = {
        "schema": "policy_0806_cuda_seeded512_v1",
        "evaluation_seed": evaluation_seed,
        "focal_deck_id": focal_deck_id,
        "jobs": jobs,
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog() -> tuple[Any, tuple[Any, ...]]:
    from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog

    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    ordered = tuple(
        sorted(
            catalog.candidates,
            key=lambda item: int(item.package_manifest["frozen_deck_number"]),
        )
    )
    if len(ordered) != EXPECTED_DECKS:
        raise RuntimeError("Frozen-0806 catalog is not exactly 55 decks")
    return catalog, ordered


def _result_path(number: str) -> Path:
    return TEMP_ROOT / "results" / f"{number}.json"


def _schedule_path(number: str) -> Path:
    return TEMP_ROOT / "schedules" / f"{number}.json"


def _run_one(catalog: Any, candidate: Any) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    result_path = _result_path(number)
    schedule_path = _schedule_path(number)
    schedule = build_cuda_schedule(
        focal_deck_id=candidate.name,
        entries=catalog.pool.schedule,
    )
    if len(schedule["jobs"]) != EXPECTED_GAMES:
        raise RuntimeError(f"deck {number} schedule is not 512 games")
    _atomic_json(schedule_path, schedule)
    if not result_path.is_file():
        log_path = TEMP_ROOT / "logs" / f"{number}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(BENCHMARK),
            "--actor-mode",
            "0806",
            "--actor-package",
            str(POLICY_ROOT),
            "--opponent-model",
            str(POLICY_MODEL),
            "--focal-deck",
            str(candidate.root / "deck.csv"),
            "--deck-root",
            str(POOL_ROOT / "decks"),
            "--schedule",
            str(schedule_path),
            "--game-limit",
            str(EXPECTED_GAMES),
            "--check-interval",
            "8",
            "--routing-mode",
            "dense_masked",
            "--ability-repeat-limit",
            "20",
            "--output",
            str(result_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"CUDA evaluation failed for deck {number}; see {log_path}"
            )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("passed") is not True
        or result.get("collector", {}).get("completed_games") != EXPECTED_GAMES
        or result.get("collector", {}).get("errors") != 0
        or len(result.get("determinism", {}).get("game_results", []))
        != EXPECTED_GAMES
        or result.get("models", {}).get("actor_checkpoint_sha256") != POLICY_SHA256
        or result.get("models", {}).get("opponent_checkpoint_sha256") != POLICY_SHA256
        or result.get("schedule", {}).get("sha256") != _sha256(schedule_path)
    ):
        raise RuntimeError(f"CUDA result failed the seeded-512 contract: deck {number}")
    return result


def _summary(catalog: Any, candidate: Any, result: dict[str, Any]) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    schedule = json.loads(_schedule_path(number).read_text(encoding="utf-8"))
    jobs = schedule["jobs"]
    results = result["determinism"]["game_results"]
    by_opponent: dict[str, dict[str, int]] = {}
    first = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    second = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    wins = losses = draws = 0
    for job, game_result in zip(jobs, results, strict=True):
        focal_player = 0 if job["focal_first"] else 1
        won = (focal_player == 0 and game_result == 1) or (
            focal_player == 1 and game_result == 2
        )
        lost = (focal_player == 0 and game_result == 2) or (
            focal_player == 1 and game_result == 1
        )
        outcome = "wins" if won else "losses" if lost else "draws"
        wins += int(won)
        losses += int(lost)
        draws += int(not won and not lost)
        seat = first if job["focal_first"] else second
        seat["games"] += 1
        seat[outcome] += 1
        row = by_opponent.setdefault(
            job["opponent_id"],
            {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        )
        row["games"] += 1
        row[outcome] += 1
    if wins + losses + draws != EXPECTED_GAMES:
        raise RuntimeError(f"deck {number} outcome total is invalid")
    return {
        "deck_number": number,
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "games": EXPECTED_GAMES,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / EXPECTED_GAMES,
        "first": first,
        "second": second,
        "by_opponent": by_opponent,
        "progress_guard_forfeits": result["progress_guard"][
            "forfeit_schedule_indices"
        ],
        "wall_seconds": result["collector"]["wall_seconds"],
        "games_per_second": result["collector"]["games_per_second_wall"],
        "peak_reserved_bytes": result["memory"]["torch_peak_reserved_bytes"],
        "game_results_sha256": result["determinism"]["game_results_sha256"],
        "terminal_state_sha256": result["determinism"]["terminal_state_sha256"],
    }


def _card_rows(candidate: Any) -> str:
    from evaluation.cards import load_card_catalog

    cards = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    rows = []
    for card_id, count in sorted(
        Counter(candidate.deck).items(),
        key=lambda item: (cards[item[0]]["name"], item[0]),
    ):
        rows.append(
            f"<tr><td>{card_id}</td><td>{html.escape(cards[card_id]['name'])}</td>"
            f"<td>{count}</td></tr>"
        )
    return "".join(rows)


def _report_html(catalog: Any, candidate: Any, summary: dict[str, Any]) -> str:
    opponents = {item.name: item for item in catalog.opponents}
    matchup_rows = []
    for opponent in sorted(
        catalog.opponents,
        key=lambda item: int(item.package_manifest["frozen_deck_number"]),
    ):
        row = summary["by_opponent"][opponent.name]
        rate = row["wins"] / row["games"] if row["games"] else 0.0
        matchup_rows.append(
            "<tr>"
            f"<td>{opponent.package_manifest['frozen_deck_number']}</td>"
            f"<td>{html.escape(opponent.display_name or opponent.name)}</td>"
            f"<td>{row['games']}</td><td>{row['wins']}</td>"
            f"<td>{row['losses']}</td><td>{row['draws']}</td>"
            f"<td>{rate:.2%}</td></tr>"
        )
    embedded = json.dumps(
        {
            "schema": "policy_0806_cuda_seeded512_report_v1",
            "evidence_boundary": (
                "CUDA engine result; not official-CPU strength evidence until parity is established"
            ),
            "evaluation_seed": EVALUATION_SEED,
            "policy_sha256": POLICY_SHA256,
            "candidate": {
                "deck_number": summary["deck_number"],
                "deck_id": summary["deck_id"],
                "display_name": summary["display_name"],
                "exact_deck_sha256": summary["exact_deck_sha256"],
                "deck": list(candidate.deck),
            },
            "summary": summary,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{summary['deck_number']} · {html.escape(summary['display_name'])} · CUDA Seeded-512</title>
<style>body{{font:14px/1.5 system-ui;max-width:1280px;margin:28px auto;padding:0 18px;color:#17231f;background:#f4f7f5}}h1,h2{{margin:.4em 0}}section{{background:#fff;border:1px solid #d8e2dd;margin:16px 0;padding:18px}}.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}.stat{{background:#eaf2ee;padding:12px}}.stat b{{display:block;font-size:22px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:7px 9px;border-bottom:1px solid #dfe6e2;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}code{{font-size:11px}}.warn{{border-left:4px solid #b67a22}}@media(max-width:700px){{.stats{{grid-template-columns:1fr 1fr}}}}</style></head><body>
<h1>{summary['deck_number']} · {html.escape(summary['display_name'])}</h1>
<p>Policy-0806 CUDA engine zero-shot · fixed seed {EVALUATION_SEED} · 512 games · 256 first / 256 second</p>
<section class="warn"><strong>证据边界：</strong>这是 CUDA engine 评测；在 CUDA/official CPU parity 完成前，不作为旧 official-CPU 合同的可替代证据。</section>
<section><div class="stats"><div class="stat"><b>{summary['wins']}-{summary['losses']}-{summary['draws']}</b>W-L-D</div><div class="stat"><b>{summary['win_rate']:.2%}</b>胜率</div><div class="stat"><b>{summary['first']['wins']}/{summary['first']['games']}</b>先手胜局</div><div class="stat"><b>{summary['second']['wins']}/{summary['second']['games']}</b>后手胜局</div><div class="stat"><b>{summary['wall_seconds']:.1f}s</b>{summary['games_per_second']:.2f} games/s</div></div></section>
<section><h2>我的卡组构成</h2><table><thead><tr><th>Card ID</th><th>卡牌</th><th>数量</th></tr></thead><tbody>{_card_rows(candidate)}</tbody></table></section>
<section><h2>对 001–055 实际对局</h2><table><thead><tr><th>编号</th><th>对手卡组</th><th>对局</th><th>胜</th><th>负</th><th>平</th><th>胜率</th></tr></thead><tbody>{''.join(matchup_rows)}</tbody></table></section>
<section><p>Policy <code>{POLICY_SHA256}</code> · schedule <code>{summary['schedule_sha256']}</code> · result <code>{summary['game_results_sha256']}</code> · progress-guard forfeits {len(summary['progress_guard_forfeits'])}</p></section>
<script type="application/json" id="report-data">{embedded}</script></body></html>"""


def _refresh(catalog: Any, candidates: tuple[Any, ...]) -> list[dict[str, Any]]:
    records = []
    for candidate in candidates:
        number = str(candidate.package_manifest["frozen_deck_number"])
        result_path = _result_path(number)
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summary = _summary(catalog, candidate, result)
        report_name = str(candidate.package_manifest["frozen_report_href"])
        report_path = OUTPUT_ROOT / "reports" / report_name
        _atomic_text(report_path, _report_html(catalog, candidate, summary))
        summary["report"] = f"reports/{report_name}"
        summary["report_sha256"] = _sha256(report_path)
        records.append(summary)
    by_number = {record["deck_number"]: record for record in records}
    rows = []
    for candidate in candidates:
        number = str(candidate.package_manifest["frozen_deck_number"])
        record = by_number.get(number)
        if record is None:
            cells = "<td>-</td><td>-</td><td>-</td><td>待评测</td>"
            title = html.escape(candidate.display_name or candidate.name)
        else:
            cells = (
                f"<td>{record['wins']}-{record['losses']}-{record['draws']}</td>"
                f"<td>{record['win_rate']:.2%}</td>"
                f"<td>{record['wall_seconds']:.1f}s</td><td>完成</td>"
            )
            title = (
                f"<a href=\"{record['report']}\">"
                f"{html.escape(candidate.display_name or candidate.name)}</a>"
            )
        rows.append(f"<tr><td>{number}</td><td>{title}</td>{cells}</tr>")
    total_games = sum(record["games"] for record in records)
    index = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Policy-0806 CUDA Seeded-512</title><style>body{{font:14px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:0 18px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}a{{color:#176b4d;font-weight:700}}</style></head><body><h1>Policy-0806 · CUDA Seeded-512</h1><p>{len(records)}/55 decks · {total_games:,}/28,160 games · seed {EVALUATION_SEED}</p><p><strong>证据边界：</strong>CUDA engine zero-shot；尚不替代 official CPU strength contract。</p><table><thead><tr><th>编号</th><th>卡组</th><th>W-L-D</th><th>胜率</th><th>耗时</th><th>状态</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
    manifest = {
        "schema": "policy_0806_cuda_seeded512_index_v1",
        "evidence_boundary": "cuda_engine_not_official_cpu_parity",
        "evaluation_seed": EVALUATION_SEED,
        "policy_sha256": POLICY_SHA256,
        "expected_decks": EXPECTED_DECKS,
        "expected_games": EXPECTED_DECKS * EXPECTED_GAMES,
        "published_decks": len(records),
        "published_games": total_games,
        "complete": len(records) == EXPECTED_DECKS,
        "lane_topology": "static_512",
        "progress_guard_repeat_limit": 20,
        "reports": records,
    }
    _atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    _atomic_text(OUTPUT_ROOT / "index.html", index)
    policy_index = ROOT / "evaluation/arena/combat_mat/policy_0806/index.html"
    _atomic_text(
        policy_index,
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Policy-0806 Reports</title><body><h1>Policy-0806 Reports</h1><ul>'
        '<li><a href="0806_kaggle_top100_plus_v1_cuda_seeded_512_v1/index.html">CUDA Seeded-512 v1</a></li>'
        '<li><a href="0806_kaggle_top100_plus_v1/index.html">2026-08-07 legacy CPU 256（25/55）</a></li>'
        '</ul></body></html>\n',
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--deck-number", action="append", default=[])
    parser.add_argument("--reproduce-first", action="store_true")
    args = parser.parse_args(argv)
    catalog, candidates = _catalog()
    requested = (
        candidates
        if args.all
        else tuple(
            candidate
            for candidate in candidates
            if candidate.package_manifest["frozen_deck_number"]
            in set(args.deck_number)
        )
    )
    if not requested:
        parser.error("choose --all or at least one --deck-number NNN")
    _refresh(catalog, candidates)
    for candidate in requested:
        number = str(candidate.package_manifest["frozen_deck_number"])
        result = _run_one(catalog, candidate)
        if args.reproduce_first and number == "001":
            primary = _result_path(number)
            replay = TEMP_ROOT / "repro" / "001.json"
            replay.parent.mkdir(parents=True, exist_ok=True)
            replay.unlink(missing_ok=True)
            saved = _result_path(number)
            saved.replace(replay)
            try:
                rerun = _run_one(catalog, candidate)
                original = json.loads(replay.read_text(encoding="utf-8"))
                if (
                    original["determinism"]["game_results"]
                    != rerun["determinism"]["game_results"]
                    or original["determinism"]["terminal_state_sha256"]
                    != rerun["determinism"]["terminal_state_sha256"]
                    or original["progress_guard"]["forfeit_schedule_indices"]
                    != rerun["progress_guard"]["forfeit_schedule_indices"]
                ):
                    raise RuntimeError("deck 001 CUDA reproducibility check failed")
            finally:
                replay.replace(saved)
            result = json.loads(saved.read_text(encoding="utf-8"))
        records = _refresh(catalog, candidates)
        summary = next(row for row in records if row["deck_number"] == number)
        print(
            f"CUDA_POLICY_0806_COMPLETE deck={number} "
            f"record={summary['wins']}-{summary['losses']}-{summary['draws']} "
            f"wall={summary['wall_seconds']:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
