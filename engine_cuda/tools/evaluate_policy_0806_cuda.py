"""CUDA-engine seeded-512 zero-shot evaluation for all Frozen-0806 decks."""

from __future__ import annotations

import argparse
import copy
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
CUDA_BATCH_SIZE = 128
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


def resolve_candidate_deck_path(candidate: Any, deck_root: Path = POOL_ROOT / "decks") -> Path:
    from evaluation.frozen_0806 import exact_deck_sha256

    number = str(candidate.package_manifest["frozen_deck_number"])
    matches = sorted(path for path in deck_root.glob(f"{number}_*") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"Frozen deck {number} does not resolve to exactly one directory")
    deck_path = matches[0] / "deck.csv"
    cards = tuple(
        int(value)
        for value in deck_path.read_text(encoding="ascii").splitlines()
        if value.strip()
    )
    expected_cards = tuple(int(value) for value in candidate.deck)
    expected_hash = str(candidate.package_manifest["exact_deck_sha256"])
    if cards != expected_cards or exact_deck_sha256(cards) != expected_hash:
        raise RuntimeError(f"Frozen deck {number} failed exact-deck identity validation")
    return deck_path


def merge_cuda_chunk_results(
    chunks: Iterable[dict[str, Any]], *, expected_games: int
) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda item: int(item["schedule"]["schedule_offset"]))
    if not ordered:
        raise ValueError("at least one CUDA chunk is required")
    cursor = 0
    game_results: list[int] = []
    terminal_hashes: list[str] = []
    forfeits: list[int] = []
    wall_seconds = gpu_seconds = 0.0
    routed_rows = completed_games = errors = 0
    peak_allocated = peak_reserved = 0
    for chunk in ordered:
        collector = chunk["collector"]
        schedule = chunk["schedule"]
        offset = int(schedule["schedule_offset"])
        games = int(collector["games"])
        results = list(chunk["determinism"]["game_results"])
        if (
            chunk.get("passed") is not True
            or offset != cursor
            or games != len(results)
            or int(collector["completed_games"]) != games
            or int(collector["errors"]) != 0
        ):
            raise ValueError("CUDA chunks are incomplete or non-contiguous")
        cursor += games
        completed_games += int(collector["completed_games"])
        errors += int(collector["errors"])
        wall_seconds += float(collector["wall_seconds"])
        gpu_seconds += float(collector["gpu_seconds"])
        routed_rows += int(collector["routed_ready_rows"])
        game_results.extend(int(value) for value in results)
        terminal_hashes.append(str(chunk["determinism"]["terminal_state_sha256"]))
        forfeits.extend(
            int(value)
            for value in chunk["progress_guard"]["forfeit_schedule_indices"]
        )
        peak_allocated = max(
            peak_allocated, int(chunk["memory"]["torch_peak_allocated_bytes"])
        )
        peak_reserved = max(
            peak_reserved, int(chunk["memory"]["torch_peak_reserved_bytes"])
        )
    if cursor != expected_games:
        raise ValueError(f"CUDA chunks cover {cursor}, expected {expected_games}")
    merged = copy.deepcopy(ordered[0])
    merged["schema_version"] = "cuda_semantic0031_rollout_chunked_v1"
    merged["collector"].update(
        {
            "games": expected_games,
            "completed_games": completed_games,
            "errors": errors,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "games_per_second_wall": expected_games / wall_seconds,
            "games_per_second_gpu": expected_games / gpu_seconds,
            "routed_ready_rows": routed_rows,
            "chunk_count": len(ordered),
            "cuda_batch_size": max(int(item["collector"]["games"]) for item in ordered),
        }
    )
    encoded_results = json.dumps(game_results, separators=(",", ":")).encode("ascii")
    encoded_terminal = json.dumps(terminal_hashes, separators=(",", ":")).encode(
        "ascii"
    )
    merged["determinism"].update(
        {
            "game_results": game_results,
            "game_results_sha256": hashlib.sha256(encoded_results).hexdigest(),
            "terminal_state_sha256": hashlib.sha256(encoded_terminal).hexdigest(),
            "terminal_state_chunk_sha256": terminal_hashes,
        }
    )
    merged["progress_guard"]["forfeit_schedule_indices"] = sorted(forfeits)
    merged["progress_guard"]["forfeit_count"] = len(forfeits)
    merged["schedule"].update(
        {"schedule_offset": 0, "used_jobs": expected_games}
    )
    merged["memory"].update(
        {
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_reserved_bytes": peak_reserved,
        }
    )
    return merged


def _run_one(catalog: Any, candidate: Any) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    focal_deck_path = resolve_candidate_deck_path(candidate)
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
        chunks = []
        with log_path.open("w", encoding="utf-8") as log:
            for offset in range(0, EXPECTED_GAMES, CUDA_BATCH_SIZE):
                chunk_path = TEMP_ROOT / "chunks" / number / f"{offset:03d}.json"
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
                    str(focal_deck_path),
                    "--deck-root",
                    str(POOL_ROOT / "decks"),
                    "--schedule",
                    str(schedule_path),
                    "--schedule-offset",
                    str(offset),
                    "--game-limit",
                    str(min(CUDA_BATCH_SIZE, EXPECTED_GAMES - offset)),
                    "--check-interval",
                    "8",
                    "--routing-mode",
                    "dense_masked",
                    "--ability-repeat-limit",
                    "20",
                    "--output",
                    str(chunk_path),
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"CUDA evaluation failed for deck {number} chunk {offset}; "
                        f"see {log_path}"
                    )
                chunks.append(json.loads(chunk_path.read_text(encoding="utf-8")))
        _atomic_json(
            result_path,
            merge_cuda_chunk_results(chunks, expected_games=EXPECTED_GAMES),
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
    schedule_entry = next(
        entry for entry in catalog.pool.schedule if entry.deck_id == candidate.name
    )
    return {
        "deck_number": number,
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "representative_cards": list(candidate.representative_cards),
        "schedule_games": int(schedule_entry.games),
        "best_rank": int(schedule_entry.best_rank),
        "observed_players": int(schedule_entry.observed_players),
        "segment": str(schedule_entry.segment),
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
    from evaluation.cards import card_image_url, load_card_catalog

    cards = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    rows = []
    for card_id, count in sorted(
        Counter(candidate.deck).items(),
        key=lambda item: (cards[item[0]]["name"], item[0]),
    ):
        card = cards[card_id]
        image_url = card_image_url(card["expansion"], card["collection_number"])
        image = (
            f'<img class="card-thumb" src="{html.escape(image_url, quote=True)}" '
            f'alt="{html.escape(card["name"], quote=True)}" loading="lazy">'
            if image_url
            else ""
        )
        rows.append(
            f"<tr><td>{image}</td><td>{card_id}</td>"
            f"<td>{html.escape(card['name'])}<small>{html.escape(card['expansion'])} "
            f"{html.escape(card['collection_number'])}</small></td><td>{count}</td></tr>"
        )
    return "".join(rows)


def _report_html(catalog: Any, candidate: Any, summary: dict[str, Any]) -> str:
    matchup_rows = []
    matchup_bars = []
    for opponent in sorted(
        catalog.opponents,
        key=lambda item: int(item.package_manifest["frozen_deck_number"]),
    ):
        row = summary["by_opponent"][opponent.name]
        rate = row["wins"] / row["games"] if row["games"] else 0.0
        images = "".join(
            f'<img class="opponent-thumb" src="{html.escape(str(card["image_url"]), quote=True)}" '
            f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
            for card in opponent.representative_cards
        )
        identity = (
            f'<span class="deck-number">{opponent.package_manifest["frozen_deck_number"]}</span>'
            f'<span class="opponent-thumbnails">{images}</span>'
            f'<span class="opponent-name">{html.escape(opponent.display_name or opponent.name)}</span>'
        )
        matchup_bars.append(
            f'<div class="chart-row"><div class="opponent-identity">{identity}</div>'
            f'<div class="bar-track"><span class="bar" style="width:{rate:.4%}"></span></div>'
            f'<div class="chart-result"><span class="chart-rate">{rate:.1%}</span>'
            f'<span class="chart-record">{row["wins"]}-{row["losses"]}-{row["draws"]}</span></div></div>'
        )
        matchup_rows.append(
            f'<tr><td>{opponent.package_manifest["frozen_deck_number"]}</td>'
            f'<td><div class="opponent-identity"><span class="opponent-thumbnails">{images}</span>'
            f'<span>{html.escape(opponent.display_name or opponent.name)}</span></div></td>'
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
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{summary['deck_number']} · {html.escape(summary['display_name'])} · Policy-0806 CUDA Seeded-512</title><style>
:root{{--bg:#f3f7f5;--surface:#fff;--soft:#f7faf8;--ink:#172b25;--muted:#60736c;--line:#dce7e2;--brand:#217a58;--dark:#14563d;--warn:#a66a18}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,"PingFang SC",sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 28px 64px}}
.hero{{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:22px;padding:30px 32px;border-radius:8px;color:#fff;background:var(--dark)}}.eyebrow{{margin:0 0 6px;color:#c8eadb;font-size:12px;font-weight:700;letter-spacing:.12em}}.back{{color:#d8eee5;text-decoration:none;font-weight:700}}h1{{margin:4px 0 0;font-size:32px;line-height:1.2}}h2{{margin:0 0 6px;font-size:20px}}section{{margin:18px 0;padding:22px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}.card{{min-height:92px;padding:15px 16px;border:1px solid var(--line);border-radius:11px;background:#fff}}.label,small{{display:block;color:var(--muted);font-size:12px}}.value{{margin-top:5px;font-size:23px;font-weight:750}}.warn{{border-left:4px solid var(--warn)}}
.matchup-chart{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px 20px;margin-top:14px;font-size:12px}}.chart-row{{display:grid;grid-template-columns:minmax(210px,270px) 1fr 125px;gap:8px;align-items:center;padding:4px 5px;border-radius:6px}}.chart-row:hover{{background:var(--soft)}}.opponent-identity{{display:flex;align-items:center;min-width:0;gap:7px}}.deck-number{{min-width:34px;padding:2px 5px;border:1px solid #b8cec4;border-radius:4px;background:#edf6f1;color:var(--dark);font-weight:800;text-align:center}}.opponent-thumbnails{{display:flex;padding-left:3px}}.opponent-thumb{{width:29px;height:38px;margin-left:-3px;object-fit:cover;border:1px solid #bdccc5;border-radius:4px;background:#e6eee9}}.opponent-name{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.bar-track{{height:7px;overflow:hidden;border-radius:999px;background:#e4ece8}}.bar{{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#2a8d65,#54b184)}}.chart-result{{text-align:right}}.chart-rate{{font-weight:800}}.chart-record{{margin-left:5px;color:var(--muted);font-size:11px}}
table{{width:100%;min-width:820px;margin-top:14px;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;vertical-align:middle}}th{{background:var(--soft);color:#496159;font-size:12px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#fbfdfc}}.card-thumb{{width:46px;height:64px;object-fit:cover;border-radius:5px}}code{{font-size:11px;overflow-wrap:anywhere}}@media(max-width:780px){{main{{padding:18px 12px}}.hero{{padding:22px 18px}}.matchup-chart{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="hero"><div><a class="back" href="../index.html">← 返回 Policy-0806 总览</a><p class="eyebrow">CUDA ENGINE · SEEDED-512 · DECK {summary['deck_number']}</p><h1>{html.escape(summary['display_name'])}</h1></div><div>4×128 CUDA lanes<br>{summary['games_per_second']:.2f} games/s</div></div>
<section class="warn"><strong>证据边界：</strong>CUDA engine zero-shot；在 CUDA/official CPU parity 完成前，不替代旧 official-CPU 强度合同。actor 与 opponent 均为同一 Policy-0806 checkpoint。</section>
<section class="summary"><div class="card"><span class="label">W-L-D</span><div class="value">{summary['wins']}-{summary['losses']}-{summary['draws']}</div></div><div class="card"><span class="label">总体胜率</span><div class="value">{summary['win_rate']:.2%}</div></div><div class="card"><span class="label">先攻</span><div class="value">{summary['first']['wins']}/{summary['first']['games']}</div><small>{summary['first']['wins']/summary['first']['games']:.2%}</small></div><div class="card"><span class="label">后攻</span><div class="value">{summary['second']['wins']}/{summary['second']['games']}</div><small>{summary['second']['wins']/summary['second']['games']:.2%}</small></div><div class="card"><span class="label">耗时</span><div class="value">{summary['wall_seconds']:.1f}s</div><small>{summary['games_per_second']:.2f} games/s</small></div><div class="card"><span class="label">循环判负</span><div class="value">{len(summary['progress_guard_forfeits'])}</div><small>同 Ability 第 20 次</small></div></section>
<section><h2>对 001–055 的表现</h2><p class="label">实际对局数按 Frozen-0806 频率分布；每个 engine seed 交换一次先后手。</p><div class="matchup-chart">{''.join(matchup_bars)}</div><table><thead><tr><th>编号</th><th>对手卡组</th><th>对局</th><th>胜</th><th>负</th><th>平</th><th>胜率</th></tr></thead><tbody>{''.join(matchup_rows)}</tbody></table></section>
<section><h2>我的 exact 60-card deck</h2><table><thead><tr><th>卡图</th><th>Card ID</th><th>卡牌</th><th>数量</th></tr></thead><tbody>{_card_rows(candidate)}</tbody></table></section>
<section><h2>可复现合同</h2><p>Seed <code>{EVALUATION_SEED}</code> · Policy <code>{POLICY_SHA256}</code> · schedule <code>{summary['schedule_sha256']}</code> · result <code>{summary['game_results_sha256']}</code></p></section>
<script type="application/json" id="report-data">{embedded}</script></main></body></html>"""


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
        schedule_entry = next(
            entry for entry in catalog.pool.schedule if entry.deck_id == candidate.name
        )
        images = "".join(
            f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
            f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
            for card in candidate.representative_cards
        )
        if record is None:
            result_cells = "<td>-</td><td>-</td><td>-</td><td>-</td><td>待评测</td>"
            title = f'<span>{html.escape(candidate.display_name or candidate.name)}</span>'
        else:
            first_rate = record["first"]["wins"] / record["first"]["games"]
            second_rate = record["second"]["wins"] / record["second"]["games"]
            result_cells = (
                f"<td>{record['wins']}-{record['losses']}-{record['draws']}</td>"
                f"<td class=\"strong\">{record['win_rate']:.2%}</td>"
                f"<td>{first_rate:.2%}<small>{record['first']['wins']}-{record['first']['losses']}-{record['first']['draws']}</small></td>"
                f"<td>{second_rate:.2%}<small>{record['second']['wins']}-{record['second']['losses']}-{record['second']['draws']}</small></td>"
                f"<td>{record['wall_seconds']:.1f}s<small>{record['games_per_second']:.2f} games/s</small></td>"
            )
            title = (
                f"<a href=\"{record['report']}\">"
                f"{html.escape(candidate.display_name or candidate.name)}</a>"
            )
        rows.append(
            f'<tr><td class="number">{number}</td><td><div class="deck"><span class="art">{images}</span>'
            f'<span>{title}<small>频率 {schedule_entry.games}/256 · 最佳名次 {schedule_entry.best_rank} · '
            f'{schedule_entry.observed_players} 人</small></span></div></td>{result_cells}</tr>'
        )
    total_games = sum(record["games"] for record in records)
    total_seconds = sum(record["wall_seconds"] for record in records)
    index = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Policy-0806 CUDA Seeded-512 · Frozen-0806 卡组强度</title><style>
:root{{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}header{{padding:28px max(20px,calc((100vw - 1500px)/2));background:#18382d;color:#fff}}h1{{margin:0;font-size:30px}}header p{{max-width:1050px;margin:7px 0 0;color:#cfe1da}}main{{max-width:1500px;margin:auto;padding:20px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:#fff}}.stat{{padding:15px 18px;border-right:1px solid var(--line)}}.stat:last-child{{border:0}}.stat b{{display:block;font-size:23px}}.stat span,small{{display:block;color:var(--muted)}}.tools{{display:flex;gap:10px;margin:18px 0}}input{{width:min(420px,100%);padding:9px 11px;border:1px solid #b9c9c1;border-radius:4px}}.table{{overflow:auto;border:1px solid var(--line);background:#fff}}table{{width:100%;min-width:1100px;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#e9f0ec;color:#486158;font-size:12px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f8fbf9}}a{{color:var(--green);font-weight:700;text-decoration:none}}.deck{{display:flex;align-items:center;gap:10px}}.art{{display:flex;width:68px}}.art img{{width:38px;height:53px;margin-right:-8px;border:1px solid #c9d5cf;border-radius:3px;object-fit:cover;background:#e4ebe7}}.strong{{font-size:16px;font-weight:750}}.number{{font-size:16px;font-weight:850;color:var(--green)}}.contract{{margin-top:16px;padding:14px 16px;border-left:4px solid var(--green);background:#fff;color:var(--muted)}}@media(max-width:760px){{.stats{{grid-template-columns:1fr 1fr}}header{{padding:22px 16px}}main{{padding:14px}}}}</style></head><body>
<header><h1>Policy-0806 · CUDA Seeded-512</h1><p>55 套 exact deck 均加载同一 Policy-0806 checkpoint，对战相同的两个 256 局固定单元；seed {EVALUATION_SEED}，先后手各 256。CUDA/official CPU parity 完成前，本页不替代旧 official-CPU 强度合同。</p></header><main><div class="stats"><div class="stat"><b>{len(records)}/55</b><span>完成卡组</span></div><div class="stat"><b>{total_games:,}</b><span>CUDA 对局</span></div><div class="stat"><b>{sum(r['wins'] for r in records):,}</b><span>Policy-0806 胜局</span></div><div class="stat"><b>{total_seconds/60:.1f}m</b><span>累计 wall time</span></div></div>
<div class="tools"><input id="search" type="search" placeholder="筛选编号或牌型"></div><div class="table"><table><thead><tr><th>编号</th><th>卡组 / 512 局报告</th><th>W-L-D</th><th>胜率</th><th>先攻</th><th>后攻</th><th>耗时</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><div class="contract">4×128 CUDA lanes · repeat-forfeit 20 · Policy <code>{POLICY_SHA256}</code></div><script>const q=document.querySelector('#search'),b=document.querySelector('tbody');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();for(const r of b.rows)r.hidden=!r.innerText.toLowerCase().includes(s)}});</script></main></body></html>"""
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
        "lane_topology": "4x128_sequential_cuda",
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
