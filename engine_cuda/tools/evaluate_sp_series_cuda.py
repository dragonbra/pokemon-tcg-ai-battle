"""Evaluate archived SP01--SP09 decks with the strict CUDA Seeded-2048 contract."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine_cuda.tools import evaluate_policy_0806_cuda as base
from evaluation.cards import card_image_url, load_card_catalog
from evaluation.frozen_0806 import exact_deck_sha256


SERIES_ROOT = ROOT / "docs/reports/sp-series"
OUTPUT_ROOT = SERIES_ROOT / "seeded2048_cuda_v2"
TEMP_ROOT = ROOT / ".tmp/evaluation/sp_series_cuda_seeded2048_v2"
EXPECTED_IDS = tuple(f"SP{index:02d}_MAGA" for index in range(1, 10))


def _read_deck(path: Path) -> tuple[int, ...]:
    cards = tuple(int(value) for value in path.read_text(encoding="ascii").split())
    if len(cards) != 60:
        raise RuntimeError(f"{path} is not an exact 60-card deck")
    return cards


def load_sp_candidates(catalog: Any) -> tuple[Any, ...]:
    manifest = json.loads((SERIES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    records = {str(record["deck_id"]): record for record in manifest["decks"]}
    if tuple(sorted(records)) != EXPECTED_IDS:
        raise RuntimeError("SP archive must contain exactly SP01--SP09 MAGA")
    official = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    candidates = []
    for deck_id in EXPECTED_IDS:
        record = records[deck_id]
        deck_path = SERIES_ROOT / str(record["deck"])
        deck = _read_deck(deck_path)
        deck_hash = exact_deck_sha256(deck)
        if deck_hash != record["exact_deck_sha256"]:
            raise RuntimeError(f"{deck_id} exact-deck hash drifted")
        deck_manifest = json.loads(
            (SERIES_ROOT / str(record["deck_manifest"])).read_text(encoding="utf-8")
        )
        representative_ids = [743, 66]
        if 343 in deck:
            representative_ids.append(343)
        representative_cards = tuple(
            {
                "card_id": card_id,
                "name": official[card_id]["name"],
                "image_url": card_image_url(
                    official[card_id]["expansion"],
                    official[card_id]["collection_number"],
                ),
            }
            for card_id in representative_ids
        )
        package_manifest = dict(catalog.candidate_policy.package_manifest or {})
        package_manifest.update(
            {
                "exact_deck_sha256": deck_hash,
                "sp_deck_id": deck_id,
                "sp_deck_number": deck_id[:4],
            }
        )
        candidates.append(
            replace(
                catalog.candidate_policy,
                name=deck_id,
                deck=list(deck),
                deck_hash=hashlib.sha256(deck_path.read_bytes()).hexdigest(),
                display_name=str(deck_manifest["display_name"]),
                representative_cards=representative_cards,
                package_manifest=package_manifest,
            )
        )
    return tuple(candidates)


def _paths(deck_id: str) -> tuple[Path, Path, Path]:
    number = deck_id[:4]
    return (
        TEMP_ROOT / "schedules" / f"{number}.json",
        TEMP_ROOT / "results" / f"{number}.json",
        TEMP_ROOT / "logs" / f"{number}.log",
    )


def _strict_result(result: dict[str, Any], schedule_path: Path) -> bool:
    device = result.get("device", {})
    guard = result.get("progress_guard", {})
    return (
        result.get("passed") is True
        and result.get("schema_version")
        == "cuda_semantic0031_resident_refill_strict_fp32_v2"
        and result.get("collector", {}).get("completed_games") == base.EXPECTED_GAMES
        and result.get("collector", {}).get("errors") == 0
        and len(result.get("determinism", {}).get("game_results", []))
        == base.EXPECTED_GAMES
        and result.get("models", {}).get("actor_checkpoint_sha256")
        == base.POLICY_SHA256
        and result.get("models", {}).get("opponent_checkpoint_sha256")
        == base.POLICY_SHA256
        and result.get("schedule", {}).get("sha256") == base._sha256(schedule_path)
        and device.get("float32_matmul_precision") == "highest"
        and device.get("matmul_allow_tf32") is False
        and device.get("cudnn_allow_tf32") is False
        and guard.get("engine_turn_draw_limit") == 100
        and guard.get("full_round_draw_limit") == 50
        and isinstance(guard.get("turn_limit_draw_schedule_indices"), list)
    )


def run_one(catalog: Any, candidate: Any) -> dict[str, Any]:
    schedule_path, result_path, log_path = _paths(candidate.name)
    schedule = base.build_cuda_schedule(
        focal_deck_id=f"{candidate.name}:{candidate.package_manifest['exact_deck_sha256']}",
        entries=catalog.pool.schedule,
    )
    if len(schedule["jobs"]) != base.EXPECTED_GAMES:
        raise RuntimeError(f"{candidate.name} schedule is not 2048 games")
    base._atomic_json(schedule_path, schedule)
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if not _strict_result(cached, schedule_path):
            result_path.unlink()
    if not result_path.is_file():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        deck_path = SERIES_ROOT / "decks" / candidate.name / "deck.csv"
        command = [
            sys.executable,
            str(base.BENCHMARK),
            "--actor-mode",
            "0806",
            "--actor-package",
            str(base.POLICY_ROOT),
            "--opponent-model",
            str(base.POLICY_MODEL),
            "--focal-deck",
            str(deck_path),
            "--deck-root",
            str(base.POOL_ROOT / "decks"),
            "--schedule",
            str(schedule_path),
            "--game-limit",
            str(base.EXPECTED_GAMES),
            "--lane-count",
            str(base.CUDA_LANE_COUNT),
            "--check-interval",
            "8",
            "--ability-repeat-limit",
            "20",
            "--engine-turn-draw-limit",
            "100",
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
            result_path.unlink(missing_ok=True)
            raise RuntimeError(f"{candidate.name} failed; see {log_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not _strict_result(result, schedule_path):
        raise RuntimeError(f"{candidate.name} failed the strict Seeded-2048 contract")
    return result


def summarize(catalog: Any, candidate: Any, result: dict[str, Any]) -> dict[str, Any]:
    schedule_path, _, _ = _paths(candidate.name)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    first = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    second = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    by_opponent: dict[str, dict[str, int]] = {}
    wins = losses = draws = 0
    for job, game_result in zip(
        schedule["jobs"], result["determinism"]["game_results"], strict=True
    ):
        focal_player = 0 if job["focal_first"] else 1
        won = game_result == focal_player + 1
        lost = game_result in (1, 2) and not won
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
    if wins + losses + draws != base.EXPECTED_GAMES:
        raise RuntimeError(f"{candidate.name} outcome total is invalid")
    return {
        "deck_number": candidate.name[:4],
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "representative_cards": list(candidate.representative_cards),
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "games": base.EXPECTED_GAMES,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / base.EXPECTED_GAMES,
        "first": first,
        "second": second,
        "by_opponent": by_opponent,
        "progress_guard_forfeits": result["progress_guard"][
            "forfeit_schedule_indices"
        ],
        "turn_limit_draws": result["progress_guard"][
            "turn_limit_draw_schedule_indices"
        ],
        "turn_limit_draw_contract": True,
        "wall_seconds": result["collector"]["wall_seconds"],
        "games_per_second": result["collector"]["games_per_second_wall"],
        "lane_count": int(result["collector"].get("lane_count", 0)),
        "refill_events": int(result["collector"].get("refill_events", 0)),
        "peak_reserved_bytes": result["memory"]["torch_peak_reserved_bytes"],
        "game_results_sha256": result["determinism"]["game_results_sha256"],
        "terminal_state_sha256": result["determinism"]["terminal_state_sha256"],
    }


def publish(catalog: Any, candidates: tuple[Any, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        _, result_path, _ = _paths(candidate.name)
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summary = summarize(catalog, candidate, result)
        run_id = (
            "run-"
            + summary["schedule_sha256"][:12]
            + summary["game_results_sha256"][:12]
        )
        relative_report = Path("reports") / candidate.name / run_id / "report.html"
        report_path = OUTPUT_ROOT / relative_report
        base._atomic_text(
            report_path,
            base._report_html(
                catalog,
                candidate,
                summary,
                report_label="SP MAGA · Policy-0806 CUDA Seeded-2048",
                actor_label="Policy-0806",
                back_href="../../../index.html",
            ),
        )
        summary.update(
            {
                "run_id": run_id,
                "report": relative_report.as_posix(),
                "report_sha256": base._sha256(report_path),
            }
        )
        records.append(summary)
    rows = []
    best = max((record["win_rate"] for record in records), default=-1.0)
    by_id = {record["deck_id"]: record for record in records}
    for candidate in candidates:
        record = by_id.get(candidate.name)
        if record is None:
            result = "<td colspan=5>待评测</td>"
            link = html.escape(candidate.name)
            row_class = ""
        else:
            first_rate = record["first"]["wins"] / record["first"]["games"]
            second_rate = record["second"]["wins"] / record["second"]["games"]
            result = (
                f"<td>{record['wins']}-{record['losses']}-{record['draws']}</td>"
                f"<td><b>{record['win_rate']:.2%}</b></td>"
                f"<td>{first_rate:.2%}<small>{record['first']['wins']}/1024</small></td>"
                f"<td>{second_rate:.2%}<small>{record['second']['wins']}/1024</small></td>"
                f"<td>{record['wall_seconds']:.1f}s<small>{record['games_per_second']:.2f} games/s</small></td>"
            )
            link = f'<a href="{record["report"]}">{html.escape(candidate.name)}</a>'
            row_class = ' class="best"' if record["win_rate"] == best else ""
        images = "".join(
            f'<img src="{html.escape(str(card["image_url"]), quote=True)}" alt="{html.escape(str(card["name"]))}">' 
            for card in candidate.representative_cards
        )
        rows.append(
            f"<tr{row_class}><td><b>{candidate.name[:4]}</b></td>"
            f'<td><span class="art">{images}</span>{link}<small>{candidate.package_manifest["exact_deck_sha256"][:12]}</small></td>'
            f"{result}</tr>"
        )
    total_games = sum(record["games"] for record in records)
    index = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SP01–SP09 MAGA · CUDA Seeded-2048</title><style>
:root{{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}header{{padding:34px max(20px,calc((100vw - 1150px)/2));background:#18382d;color:#fff}}h1{{margin:0;font-size:30px}}header p{{color:#cfe1da}}main{{max-width:1150px;margin:auto;padding:20px}}.stats{{display:grid;grid-template-columns:repeat(5,1fr);background:#fff;border:1px solid var(--line)}}.stat{{padding:16px;border-right:1px solid var(--line)}}.stat b{{display:block;font-size:22px}}small,.stat span{{display:block;color:var(--muted)}}.table{{margin-top:18px;overflow:auto;border:1px solid var(--line);background:#fff}}table{{width:100%;min-width:900px;border-collapse:collapse}}th,td{{padding:11px;border-bottom:1px solid var(--line);text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#e9f0ec}}a{{color:var(--green);font-weight:800;text-decoration:none}}.art{{display:inline-flex;width:88px;vertical-align:middle}}.art img{{width:38px;height:53px;margin-right:-7px;border:1px solid #c9d5cf;border-radius:4px;object-fit:cover}}.best td{{background:#eef7f1}}.contract,.analysis{{margin-top:16px;padding:14px;border-left:4px solid var(--green);background:#fff}}.analysis a{{font-size:17px}}@media(max-width:700px){{.stats{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>SP01–SP09 MAGA · CUDA Seeded-2048</h1><p>Policy-0806 对 Policy-0806 · Frozen-0806 固定频率池 · strict FP32 · 每套先后手各 1024</p></header><main><div class="stats"><div class="stat"><b>{len(records)}/9</b><span>完成构筑</span></div><div class="stat"><b>{total_games:,}</b><span>完成对局</span></div><div class="stat"><b>{best:.2%}</b><span>当前最高胜率</span></div><div class="stat"><b>{sum(len(r['turn_limit_draws']) for r in records)}</b><span>50 回合平局</span></div><div class="stat"><b>{sum(len(r['progress_guard_forfeits']) for r in records)}</b><span>循环判负</span></div></div><div class="analysis"><a href="002_strength_analysis.html">查看 Frozen 002 独立复测与 Nighttime Mine 归因报告 →</a></div><div class="table"><table><thead><tr><th>编号</th><th>构筑 / 报告</th><th>W-L-D</th><th>胜率</th><th>先攻</th><th>后攻</th><th>耗时</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><div class="contract">Seed <code>{base.EVALUATION_SEED}</code> · 8×256 independent seeds · strict FP32 · 256 resident CUDA lanes · engine turn 100 记平局 · repeat-forfeit 20 · CUDA/official CPU parity 完成前不替代 official-CPU 强度合同。</div></main></body></html>'''
    manifest = {
        "schema": "sp_maga_policy_0806_cuda_seeded2048_v2",
        "evaluation_seed": base.EVALUATION_SEED,
        "policy_sha256": base.POLICY_SHA256,
        "opponent_pool_id": "0806_kaggle_top100_plus_v1",
        "expected_decks": 9,
        "expected_games_per_deck": base.EXPECTED_GAMES,
        "published_decks": len(records),
        "published_games": total_games,
        "complete": len(records) == 9,
        "reports": records,
    }
    base._atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    base._atomic_text(OUTPUT_ROOT / "index.html", index)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--deck-id", action="append", default=[])
    args = parser.parse_args(argv)
    catalog, _ = base._catalog()
    candidates = load_sp_candidates(catalog)
    requested = candidates if args.all else tuple(
        candidate for candidate in candidates if candidate.name in set(args.deck_id)
    )
    if not requested:
        parser.error("choose --all or at least one --deck-id SPXX_MAGA")
    publish(catalog, candidates)
    for candidate in requested:
        result = run_one(catalog, candidate)
        records = publish(catalog, candidates)
        summary = next(record for record in records if record["deck_id"] == candidate.name)
        print(
            f"CUDA_SP_COMPLETE deck={candidate.name} "
            f"record={summary['wins']}-{summary['losses']}-{summary['draws']} "
            f"wall={result['collector']['wall_seconds']:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
