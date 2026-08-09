"""Run and publish the 55-deck Frozen-0806 official-engine evaluation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import time
import uuid
from collections import Counter
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluation.frozen_0806 import POLICY_0019_SHA256, POLICY_0806_SHA256
from evaluation.frozen_0806_contract import (
    FROZEN_0806_CONTRACT_ID,
    FROZEN_0806_CPU_GAMES,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    evaluation_schedule_id,
)
from evaluation.frozen_0806_runtime import (
    Frozen0806RuntimeCatalog,
    load_frozen_0806_runtime_catalog,
)
from evaluation.reporting.index import _embedded_report_data
from evaluation.reporting.html import render_html
from evaluation.reporting.models import ReportData
from evaluation.metrics.base import MetricPresentation
from evaluation.runner.batch import (
    BatchConfig,
    _policy_inference_server,
    run_batch,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "evaluation/arena/combat_mat/policy_0019/0806_kaggle_top100_plus_v1"
)
LEGACY_OUTPUT_ROOT = (
    ROOT / "evaluation/arena/combat_mat/frozen/0806_kaggle_top100_plus_v1"
)
TEMP_ROOT = ROOT / ".tmp/evaluation/frozen_0806_cpu256_agent_choice_v3_full"
BENCHMARK_ROOT = ROOT / ".tmp/evaluation/frozen_0806_cpu256_agent_choice_v3_benchmark"
EXPECTED_DECKS = 55
EXPECTED_GAMES = FROZEN_0806_CPU_GAMES
DEFAULT_SEED = FROZEN_0806_EVALUATION_SEED
LEGACY_POLICY_0806_OUTPUT_ROOT = (
    ROOT / "evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1"
)
POLICY_0806_SEEDED2048_OUTPUT_ROOT = (
    ROOT
    / "evaluation/arena/combat_mat/policy_0806"
    / "0806_kaggle_top100_plus_v1_cpu_seeded_256_agent_choice_v3"
)


@dataclass(frozen=True)
class FrozenEvaluationTarget:
    key: str
    label: str
    policy_sha256: str
    output_root: Path
    legacy_output_root: Path | None = None


POLICY_0019_TARGET = FrozenEvaluationTarget(
    key="0019",
    label="Policy-0019",
    policy_sha256=POLICY_0019_SHA256,
    output_root=OUTPUT_ROOT,
    legacy_output_root=LEGACY_OUTPUT_ROOT,
)
POLICY_0806_TARGET = FrozenEvaluationTarget(
    key="0806",
    label="Policy-0806",
    policy_sha256=POLICY_0806_SHA256,
    output_root=POLICY_0806_SEEDED2048_OUTPUT_ROOT,
)


def evaluation_target(policy: str) -> FrozenEvaluationTarget:
    targets = {"0019": POLICY_0019_TARGET, "0806": POLICY_0806_TARGET}
    try:
        return targets[policy]
    except KeyError as exc:
        raise ValueError("opponent policy must be 0019 or 0806") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _schedule_counts(catalog: Frozen0806RuntimeCatalog) -> tuple[int, ...]:
    counts = tuple(int(entry.games) for entry in catalog.pool.schedule)
    if sum(counts) != FROZEN_0806_CPU_GAMES:
        raise ValueError("Frozen-0806 CPU schedule must contain exactly 256 games")
    return counts


def _turn_order(games: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {
        "first": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        "second": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
    }
    for game in games:
        key = "first" if game.get("candidate_first") else "second"
        row = result[key]
        row["games"] += 1
        winner = game.get("winner")
        outcome = "wins" if winner == 0 else "losses" if winner == 1 else "draws"
        row[outcome] += 1
    return result


def validate_report_payload(
    payload: dict[str, Any],
    candidate: Any,
    catalog: Frozen0806RuntimeCatalog,
    target: FrozenEvaluationTarget = POLICY_0019_TARGET,
) -> dict[str, Any]:
    manifest = payload.get("manifest")
    summary = payload.get("summary")
    games = payload.get("games")
    if not isinstance(manifest, dict) or not isinstance(summary, dict) or not isinstance(games, list):
        raise ValueError("Frozen-0806 report payload is incomplete")
    candidate_record = manifest.get("candidate")
    package_manifest = (
        candidate_record.get("package_manifest")
        if isinstance(candidate_record, dict)
        else None
    )
    pool = manifest.get("opponent_pool")
    expected_counts = list(_schedule_counts(catalog))
    if not isinstance(candidate_record, dict) or not isinstance(package_manifest, dict):
        raise ValueError("Frozen-0806 report lacks candidate identity")
    if (
        package_manifest.get("frozen_pool_id") != catalog.pool.pool_id
        or package_manifest.get("frozen_role") != "candidate"
        or package_manifest.get("deck_id") != candidate.name
        or package_manifest.get("exact_deck_sha256")
        != candidate.package_manifest["exact_deck_sha256"]
        or package_manifest.get("checkpoint_sha256") != POLICY_0806_SHA256
    ):
        raise ValueError("Frozen-0806 candidate policy or exact-deck identity mismatch")
    if Counter(candidate_record.get("deck", [])) != Counter(candidate.deck):
        raise ValueError("Frozen-0806 report candidate deck mismatch")
    if (
        not isinstance(pool, dict)
        or pool.get("pool_id") != catalog.pool.pool_id
        or pool.get("catalog_sha256") != catalog.pool.manifest_sha256
        or pool.get("policy_hash") != target.policy_sha256
        or manifest.get("opponent_schedule_id")
        != evaluation_schedule_id(
            catalog.pool.manifest["schedule_sha256"], evaluation_units=1
        )
        or manifest.get("first_player_contract")
        != FROZEN_0806_FIRST_PLAYER_CONTRACT
        or manifest.get("seed") != DEFAULT_SEED
        or manifest.get("games_per_opponent") != expected_counts
        or len(manifest.get("opponents", [])) != EXPECTED_DECKS
    ):
        raise ValueError("Frozen-0806 opponent policy or schedule identity mismatch")
    if (
        manifest.get("games") != EXPECTED_GAMES
        or summary.get("total_games") != EXPECTED_GAMES
        or summary.get("completed_games") != EXPECTED_GAMES
        or summary.get("errors") != 0
        or summary.get("unfinished") != 0
        or len(games) != EXPECTED_GAMES
        or any(game.get("status") != "finished" for game in games)
    ):
        raise ValueError("Frozen-0806 report is partial or contains errors")
    observed_counts = Counter(str(game.get("opponent")) for game in games)
    expected_by_name = {
        entry.deck_id: games
        for entry, games in zip(
            catalog.pool.schedule, _schedule_counts(catalog), strict=True
        )
    }
    if observed_counts != Counter(expected_by_name):
        raise ValueError("Frozen-0806 report game distribution mismatch")
    if any(type(game.get("candidate_won_toss")) is not bool for game in games):
        raise ValueError("Frozen-0806 report lacks seeded toss provenance")
    order = _turn_order(games)
    schedule = next(entry for entry in catalog.pool.schedule if entry.deck_id == candidate.name)
    return {
        "deck_id": candidate.name,
        "deck_number": str(package_manifest["frozen_deck_number"]),
        "display_name": candidate.display_name or candidate.name,
        "representative_cards": list(candidate.representative_cards),
        "exact_deck_sha256": package_manifest["exact_deck_sha256"],
        "segment": schedule.segment,
        "best_rank": schedule.best_rank,
        "observed_players": schedule.observed_players,
        "run_id": manifest.get("run_id"),
        "games": EXPECTED_GAMES,
        "wins": int(summary.get("wins", 0)),
        "losses": int(summary.get("losses", 0)),
        "draws": int(summary.get("draws", 0)),
        "win_rate": float(summary.get("win_rate", 0.0)),
        "turn_order": order,
        "wall_time_seconds": float(manifest.get("wall_time_seconds", 0.0)),
        "workers": manifest.get("workers"),
        "candidate_inference": manifest.get("candidate_inference"),
        "opponent_inference": manifest.get("opponent_inference"),
    }


def validate_report(
    path: Path,
    candidate: Any,
    catalog: Frozen0806RuntimeCatalog,
    target: FrozenEvaluationTarget = POLICY_0019_TARGET,
) -> dict[str, Any]:
    record = validate_report_payload(
        _embedded_report_data(path), candidate, catalog, target
    )
    record["report"] = f'reports/{candidate.package_manifest["frozen_report_href"]}'
    record["report_sha256"] = _sha256(path)
    return record


def _refresh_report_navigation(
    path: Path,
    candidate: Any,
    catalog: Frozen0806RuntimeCatalog,
    target: FrozenEvaluationTarget = POLICY_0019_TARGET,
) -> None:
    payload = _embedded_report_data(path)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"report manifest is missing: {path}")
    candidate_record = manifest.get("candidate")
    opponents = manifest.get("opponents")
    if not isinstance(candidate_record, dict) or not isinstance(opponents, list):
        raise ValueError(f"report package identities are missing: {path}")
    candidate_manifest = candidate_record.get("package_manifest")
    opponent_pool = manifest.get("opponent_pool")
    opponent_by_name = {opponent.name: opponent for opponent in catalog.opponents}
    already_numbered = (
        isinstance(candidate_manifest, dict)
        and candidate_manifest.get("frozen_deck_number")
        == candidate.package_manifest.get("frozen_deck_number")
        and isinstance(opponent_pool, dict)
        and opponent_pool.get("policy_label") == target.label
        and candidate_record.get("display_name") == candidate.display_name
        and tuple(candidate_record.get("representative_cards", ()))
        == tuple(candidate.representative_cards)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("package_manifest"), dict)
            and str(item.get("name")) in opponent_by_name
            and item["package_manifest"].get("frozen_deck_number")
            == opponent_by_name[str(item.get("name"))].package_manifest.get(
                "frozen_deck_number"
            )
            and item["package_manifest"].get("frozen_report_href")
            == opponent_by_name[str(item.get("name"))].package_manifest.get(
                "frozen_report_href"
            )
            and item.get("display_name")
            == opponent_by_name[str(item.get("name"))].display_name
            and tuple(item.get("representative_cards", ()))
            == tuple(opponent_by_name[str(item.get("name"))].representative_cards)
            for item in opponents
        )
    )
    if already_numbered:
        return

    candidate_record["package_manifest"] = dict(candidate.package_manifest or {})
    candidate_record["display_name"] = candidate.display_name
    candidate_record["representative_cards"] = list(candidate.representative_cards)
    if isinstance(opponent_pool, dict):
        opponent_pool["policy_label"] = target.label
    for item in opponents:
        if not isinstance(item, dict):
            continue
        package = opponent_by_name.get(str(item.get("name")))
        if package is not None:
            item["package_manifest"] = dict(package.package_manifest or {})
            item["display_name"] = package.display_name
            item["representative_cards"] = list(package.representative_cards)
    presentations = {
        str(metric_id): MetricPresentation(
            metric_id=str(values.get("metric_id", metric_id)),
            title=str(values.get("title", "")),
            markdown=str(values.get("markdown", "")),
            html=str(values.get("html", "")),
        )
        for metric_id, values in payload.get("presentations", {}).items()
        if isinstance(values, dict)
    }
    rendered = render_html(
        ReportData(
            manifest=manifest,
            summary=dict(payload.get("summary", {})),
            games=tuple(payload.get("games", ())),
            metrics=dict(payload.get("metrics", {})),
            cases=tuple(payload.get("cases", ())),
            metric_profile=dict(payload.get("metric_profile", {})),
            presentations=presentations,
            presentation_errors=tuple(payload.get("presentation_errors", ())),
        )
    )
    _atomic_text(path, rendered)


def _publish(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite Frozen-0806 report: {target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as read_handle, temporary.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_config(
    catalog: Frozen0806RuntimeCatalog,
    candidate: Any,
    target: FrozenEvaluationTarget,
    *,
    output_root: Path,
    workers: int,
    batch_size: int,
    batch_wait_ms: float,
    inference_dtype: str,
    candidate_socket: Path,
    opponent_socket: Path,
    opponents: tuple[Any, ...] | None = None,
    counts: tuple[int, ...] | None = None,
    engine_pool_size: int = 1,
) -> BatchConfig:
    selected_opponents = opponents or catalog.opponents
    selected_counts = counts or _schedule_counts(catalog)
    return BatchConfig(
        candidate=candidate,
        opponents=selected_opponents,
        games_per_opponent=1,
        games_by_opponent=selected_counts,
        opponent_schedule_id=evaluation_schedule_id(
            catalog.pool.manifest["schedule_sha256"], evaluation_units=1
        ),
        output_root=output_root,
        visualize=False,
        max_steps=2_500,
        metric_profile_id="league_deck_quality",
        workers=workers,
        worker_cpu_threads=1,
        worker_timeout_seconds=90.0,
        worker_crash_retries=1,
        candidate_inference_device="cuda:0",
        opponent_inference_root=catalog.opponent_policy.root,
        opponent_inference_device="cuda:0",
        candidate_inference_batch_size=batch_size,
        candidate_inference_batch_wait_ms=batch_wait_ms,
        candidate_inference_dtype=inference_dtype,
        opponent_inference_dtype=inference_dtype,
        candidate_inference_socket=candidate_socket,
        opponent_inference_socket=opponent_socket,
        opponent_pool_id=catalog.pool.pool_id,
        opponent_catalog_sha256=catalog.pool.manifest_sha256,
        opponent_policy_hash=target.policy_sha256,
        opponent_policy_label=target.label,
        share_policy_inference_server=target.key == "0806",
        seed=DEFAULT_SEED,
        inference_ability_repeat_limit=20,
        engine_turn_draw_limit=100,
        engine_pool_size=engine_pool_size,
        independent_engine_seeds=False,
        agent_selects_first_player=True,
        focal_seed_identity=candidate.name,
    )


def run_deck(
    catalog: Frozen0806RuntimeCatalog,
    candidate: Any,
    target: FrozenEvaluationTarget,
    *,
    workers: int,
    batch_size: int,
    batch_wait_ms: float,
    inference_dtype: str,
    candidate_socket: Path,
    opponent_socket: Path,
    engine_pool_size: int = 1,
) -> dict[str, Any]:
    report_path = target.output_root / "reports" / str(
        candidate.package_manifest["frozen_report_href"]
    )
    if report_path.is_file():
        return validate_report(report_path, candidate, catalog, target)
    result = run_batch(
        _batch_config(
            catalog,
            candidate,
            target,
            output_root=TEMP_ROOT / candidate.name,
            workers=workers,
            batch_size=batch_size,
            batch_wait_ms=batch_wait_ms,
            inference_dtype=inference_dtype,
            candidate_socket=candidate_socket,
            opponent_socket=opponent_socket,
            engine_pool_size=engine_pool_size,
        )
    )
    validate_report_payload(
        {
            "manifest": result.report_data.manifest,
            "summary": result.report_data.summary,
            "games": list(result.report_data.games),
        },
        candidate,
        catalog,
        target,
    )
    _publish(result.report_path, report_path)
    return validate_report(report_path, candidate, catalog, target)


def _rate(row: dict[str, int]) -> float:
    return row["wins"] / row["games"] if row["games"] else 0.0


def _index_row(record: dict[str, Any]) -> str:
    cards = "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
        for card in record["representative_cards"]
    )
    first = record["turn_order"]["first"]
    second = record["turn_order"]["second"]
    completed = bool(record.get("report"))
    title = (
        f'<a href="{html.escape(str(record["report"]), quote=True)}">'
        f'{html.escape(str(record["display_name"]))}</a>'
        if completed
        else f'<span>{html.escape(str(record["display_name"]))}</span>'
    )
    status = '<span class="done">已完成</span>' if completed else '<span class="pending">待评测</span>'
    result = f'{record["wins"]}-{record["losses"]}-{record["draws"]}' if completed else '-'
    win_rate = f'{record["win_rate"]:.2%}' if completed else '-'
    first_rate = (
        f'{_rate(first):.2%}<small>{first["wins"]}-{first["losses"]}-{first["draws"]}</small>'
        if completed else '-'
    )
    second_rate = (
        f'{_rate(second):.2%}<small>{second["wins"]}-{second["losses"]}-{second["draws"]}</small>'
        if completed else '-'
    )
    return (
        f'<tr data-rate="{record["win_rate"]:.9f}" data-rank="{record["best_rank"]}">'
        f'<td class="number">{record["deck_number"]}</td>'
        f'<td><div class="deck"><span class="art">{cards}</span><span>{title}'
        f'<small>{status} · 频率 {record["schedule_games"]}/256</small></span></div></td>'
        f'<td>{result}</td><td class="strong">{win_rate}</td>'
        f'<td>{first_rate}</td><td>{second_rate}</td>'
        f'<td>{record["best_rank"]}</td><td>{record["observed_players"]}</td>'
        f'<td>{html.escape(str(record["segment"]))}</td>'
        f'<td>{record["wall_time_seconds"]:.1f}s</td></tr>'
    )


def _index_html(
    records: list[dict[str, Any]],
    catalog: Frozen0806RuntimeCatalog,
    target: FrozenEvaluationTarget = POLICY_0019_TARGET,
) -> str:
    completed_by_id = {record["deck_id"]: record for record in records}
    ordered = []
    for candidate, schedule in zip(catalog.candidates, catalog.pool.schedule, strict=True):
        record = completed_by_id.get(candidate.name)
        if record is None:
            record = {
                "deck_id": candidate.name,
                "deck_number": candidate.package_manifest["frozen_deck_number"],
                "display_name": candidate.display_name or candidate.name,
                "representative_cards": list(candidate.representative_cards),
                "schedule_games": schedule.games,
                "best_rank": schedule.best_rank,
                "observed_players": schedule.observed_players,
                "segment": schedule.segment,
                "wins": 0, "losses": 0, "draws": 0, "win_rate": -1.0,
                "wall_time_seconds": 0.0,
                "turn_order": {
                    "first": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
                    "second": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
                },
            }
        else:
            record["schedule_games"] = schedule.games
        ordered.append(record)
    ordered.sort(key=lambda record: int(record["deck_number"]))
    rows = "".join(_index_row(record) for record in ordered)
    total_games = sum(record["games"] for record in records)
    total_seconds = sum(record["wall_time_seconds"] for record in records)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{target.label} CPU Seeded-256 Agent-Choice Report · Frozen-0806 卡组强度</title><style>
:root{{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d;--red:#a54343}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}
header{{padding:28px max(20px,calc((100vw - 1500px)/2));background:#18382d;color:#fff}}h1{{margin:0;font-size:30px;letter-spacing:0}}header p{{max-width:980px;margin:7px 0 0;color:#cfe1da}}
main{{max-width:1500px;margin:auto;padding:20px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--paper)}}
.stat{{padding:15px 18px;border-right:1px solid var(--line)}}.stat:last-child{{border:0}}.stat b{{display:block;font-size:23px}}.stat span,small{{display:block;color:var(--muted)}}
.tools{{display:flex;gap:10px;margin:18px 0}}input{{width:min(420px,100%);padding:9px 11px;border:1px solid #b9c9c1;border-radius:4px;background:#fff}}
.table{{overflow:auto;border:1px solid var(--line);background:#fff}}table{{width:100%;min-width:1180px;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#e9f0ec;color:#486158;font-size:12px;cursor:pointer}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f8fbf9}}a{{color:var(--green);font-weight:700;text-decoration:none}}
.deck{{display:flex;align-items:center;gap:10px}}.art{{display:flex;width:68px}}.art img{{width:38px;height:53px;margin-right:-8px;border:1px solid #c9d5cf;border-radius:3px;object-fit:cover;background:#e4ebe7}}.strong{{font-size:16px;font-weight:750}}code{{font-size:11px}}
.number{{font-size:16px;font-weight:850;color:var(--green);font-variant-numeric:tabular-nums}}.done{{color:var(--green)}}.pending{{color:#8a6b28}}
.contract{{margin-top:16px;padding:14px 16px;border-left:4px solid var(--green);background:#fff;color:var(--muted)}}
@media(max-width:760px){{.stats{{grid-template-columns:1fr 1fr}}.stat:nth-child(2){{border-right:0}}header{{padding:22px 16px}}main{{padding:14px}}}}
</style></head><body><header><h1>{target.label} CPU Seeded-256 · Agent 选择先后手</h1><p>55 套 exact deck 均加载 Policy-0806，对战一个固定 256 局频率单元；evaluation seed 固定为 {DEFAULT_SEED}。抛硬币赢家由对应 Agent 处理 context 41 并自主选择先后手，不做人为坐席分配。胜负来自 official engine；达到 50 个完整回合仍未结束时记为平局。</p></header><main>
<div class="stats"><div class="stat"><b>{len(records)}/{EXPECTED_DECKS}</b><span>完成卡组</span></div><div class="stat"><b>{total_games:,}</b><span>正式对局</span></div><div class="stat"><b>{sum(r['wins'] for r in records):,}</b><span>Policy-0806 胜局</span></div><div class="stat"><b>{total_seconds/3600:.2f}h</b><span>累计 wall time</span></div></div>
<div class="tools"><input id="search" type="search" placeholder="筛选编号或牌型"></div>
<div class="table"><table id="results"><thead><tr><th>编号</th><th>卡组 / 256 局报告</th><th>W-L-D</th><th>胜率</th><th>实际先攻</th><th>实际后攻</th><th>最佳名次</th><th>观察人数</th><th>来源</th><th>耗时</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="contract">Contract <code>{FROZEN_0806_CONTRACT_ID}</code> · first-player <code>{FROZEN_0806_FIRST_PLAYER_CONTRACT}</code> · seed <code>{DEFAULT_SEED}</code> · Pool <code>{catalog.pool.pool_id}</code> · Schedule <code>{evaluation_schedule_id(catalog.pool.manifest['schedule_sha256'], evaluation_units=1)}</code> · Candidate Policy-0806 <code>{POLICY_0806_SHA256}</code> · Opponent {target.label} <code>{target.policy_sha256}</code></div>
<script>
const q=document.querySelector('#search'),body=document.querySelector('tbody');
q.addEventListener('input',()=>{{
  const s=q.value.toLowerCase();
  for(const r of body.rows)r.hidden=!r.innerText.toLowerCase().includes(s);
}});
const publishedDecks={len(records)},evaluationComplete={str(len(records) == EXPECTED_DECKS).lower()};
if(!evaluationComplete)setInterval(async()=>{{
  try{{
    const response=await fetch(`manifest.json?updated=${{Date.now()}}`,{{cache:'no-store'}});
    const manifest=await response.json();
    if(manifest.published_decks>publishedDecks)location.reload();
  }}catch(_error){{}}
}},5000);
</script>
</main></body></html>"""


def refresh_index(
    catalog: Frozen0806RuntimeCatalog,
    *,
    target: FrozenEvaluationTarget = POLICY_0019_TARGET,
    candidate_concurrency: int = 2,
    workers_per_candidate: int = 16,
) -> list[dict[str, Any]]:
    if (
        target.legacy_output_root is not None
        and target.legacy_output_root.is_dir()
        and not target.output_root.exists()
    ):
        target.output_root.parent.mkdir(parents=True, exist_ok=True)
        target.legacy_output_root.replace(target.output_root)
    records = []
    for candidate in catalog.candidates:
        path = target.output_root / "reports" / str(
            candidate.package_manifest["frozen_report_href"]
        )
        legacy_path = target.output_root / "reports" / f"{candidate.name}.html"
        if legacy_path.is_file() and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.replace(path)
        if not path.exists():
            number = candidate.package_manifest["frozen_deck_number"]
            numbered_legacy = list(
                (target.output_root / "reports").glob(f"{number}_*.html")
            )
            if len(numbered_legacy) == 1:
                numbered_legacy[0].replace(path)
        if path.is_file():
            _refresh_report_navigation(path, candidate, catalog, target)
            records.append(validate_report(path, candidate, catalog, target))
    manifest = {
        "schema_version": "frozen_0806_cpu256_agent_first_player_v3",
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": DEFAULT_SEED,
        "pool_id": catalog.pool.pool_id,
        "pool_manifest_sha256": catalog.pool.manifest_sha256,
        "base_schedule_sha256": catalog.pool.manifest["schedule_sha256"],
        "schedule_sha256": evaluation_schedule_id(
            catalog.pool.manifest["schedule_sha256"], evaluation_units=1
        ),
        "policy_0806_sha256": POLICY_0806_SHA256,
        "opponent_policy_label": target.label,
        "opponent_policy_sha256": target.policy_sha256,
        "expected_decks": EXPECTED_DECKS,
        "games_per_deck": EXPECTED_GAMES,
        "expected_games": EXPECTED_DECKS * EXPECTED_GAMES,
        "published_decks": len(records),
        "published_games": sum(record["games"] for record in records),
        "complete": len(records) == EXPECTED_DECKS,
        "candidate_concurrency": candidate_concurrency,
        "workers_per_candidate": workers_per_candidate,
        "maximum_game_workers": candidate_concurrency * workers_per_candidate,
        "reports": records,
    }
    _atomic_json(target.output_root / "manifest.json", manifest)
    _atomic_text(
        target.output_root / "index.html", _index_html(records, catalog, target)
    )
    if target.key == "0806":
        _atomic_text(
            ROOT / "evaluation/arena/combat_mat/policy_0806/index.html",
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<title>Policy-0806 Reports</title><body><h1>Policy-0806 Reports</h1><ul>'
            '<li><a href="0806_kaggle_top100_plus_v1_cuda_seeded_2048_agent_choice_v3/index.html">'
            'CUDA Seeded-2048 Agent Choice v3（正式强度合同）</a></li>'
            '<li><a href="0806_kaggle_top100_plus_v1_cpu_seeded_256_agent_choice_v3/index.html">'
            'Official CPU Seeded-256 Agent Choice v3（部署复核合同）</a></li>'
            '</ul></body></html>\n',
        )
    _atomic_text(
        ROOT / "evaluation/arena/combat_mat/index.html",
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>Frozen-0806 Reports</title><body><h1>Frozen-0806 Reports</h1><ul>'
        '<li><a href="policy_0806/index.html">Policy-0806 Reports</a></li>'
        '<li><a href="policy_0019/0806_kaggle_top100_plus_v1/index.html">'
        'Policy-0019 Report</a></li></ul></body></html>\n',
    )
    return records


def benchmark(
    catalog: Frozen0806RuntimeCatalog,
    target: FrozenEvaluationTarget,
    *,
    worker_counts: Iterable[int],
    games: int,
    batch_size: int,
    batch_wait_ms: float,
    inference_dtype: str,
    engine_pool_size: int = 1,
) -> list[dict[str, Any]]:
    if games < 2 or games % 2:
        raise ValueError("benchmark games must be an even integer of at least two")
    opponents = catalog.opponents[: games // 2]
    counts = (2,) * len(opponents)
    candidate = catalog.candidates[0]
    records = []
    with ExitStack() as stack:
        candidate_socket = stack.enter_context(
            _policy_inference_server(
                root=catalog.candidate_policy.root,
                device="cuda:0",
                batch_size=batch_size,
                batch_wait_ms=batch_wait_ms,
                label="candidate",
                ability_repeat_limit=20,
                inference_dtype=inference_dtype,
            )
        )
        opponent_socket = candidate_socket
        if target.key != "0806":
            opponent_socket = stack.enter_context(
                _policy_inference_server(
                    root=catalog.opponent_policy.root,
                    device="cuda:0",
                    batch_size=batch_size,
                    batch_wait_ms=batch_wait_ms,
                    label="opponent",
                    ability_repeat_limit=20,
                    inference_dtype=inference_dtype,
                )
            )
        assert candidate_socket is not None and opponent_socket is not None
        for workers in worker_counts:
            started = time.perf_counter()
            result = run_batch(
                _batch_config(
                    catalog,
                    candidate,
                    target,
                    output_root=BENCHMARK_ROOT / f"workers_{workers}",
                    workers=workers,
                    batch_size=batch_size,
                    batch_wait_ms=batch_wait_ms,
                    inference_dtype=inference_dtype,
                    candidate_socket=candidate_socket,
                    opponent_socket=opponent_socket,
                    opponents=opponents,
                    counts=counts,
                    engine_pool_size=engine_pool_size,
                )
            )
            summary = result.report_data.summary
            wall = time.perf_counter() - started
            record = {
                "workers": workers,
                "games": games,
                "errors": summary["errors"],
                "unfinished": summary["unfinished"],
                "wall_time_seconds": wall,
                "games_per_second": games / wall,
                "report": str(result.report_path.relative_to(ROOT)),
            }
            records.append(record)
            _atomic_json(BENCHMARK_ROOT / "benchmark.json", {"trials": records})
    return records


def run_full(
    catalog: Frozen0806RuntimeCatalog,
    candidates: Iterable[Any],
    target: FrozenEvaluationTarget,
    *,
    workers: int,
    candidate_concurrency: int,
    batch_size: int,
    batch_wait_ms: float,
    inference_dtype: str,
    engine_pool_size: int = 1,
) -> list[dict[str, Any]]:
    if candidate_concurrency < 1:
        raise ValueError("candidate_concurrency must be at least one")
    published = refresh_index(
        catalog,
        target=target,
        candidate_concurrency=candidate_concurrency,
        workers_per_candidate=workers,
    )
    published_ids = {record["deck_id"] for record in published}
    with ExitStack() as stack:
        candidate_socket = stack.enter_context(
            _policy_inference_server(
                root=catalog.candidate_policy.root,
                device="cuda:0",
                batch_size=batch_size,
                batch_wait_ms=batch_wait_ms,
                label="candidate",
                ability_repeat_limit=20,
                inference_dtype=inference_dtype,
            )
        )
        opponent_socket = candidate_socket
        if target.key != "0806":
            opponent_socket = stack.enter_context(
                _policy_inference_server(
                    root=catalog.opponent_policy.root,
                    device="cuda:0",
                    batch_size=batch_size,
                    batch_wait_ms=batch_wait_ms,
                    label="opponent",
                    ability_repeat_limit=20,
                    inference_dtype=inference_dtype,
                )
            )
        assert candidate_socket is not None and opponent_socket is not None
        candidate_list = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.name not in published_ids
            ),
            key=lambda candidate: int(
                candidate.package_manifest["frozen_deck_number"]
            ),
        )
        with ThreadPoolExecutor(max_workers=candidate_concurrency) as executor:
            for offset in range(0, len(candidate_list), candidate_concurrency):
                batch = candidate_list[offset : offset + candidate_concurrency]
                futures = {
                    executor.submit(
                        run_deck,
                        catalog,
                        candidate,
                        target,
                        workers=workers,
                        batch_size=batch_size,
                        batch_wait_ms=batch_wait_ms,
                        inference_dtype=inference_dtype,
                        candidate_socket=candidate_socket,
                        opponent_socket=opponent_socket,
                        engine_pool_size=engine_pool_size,
                    ): candidate
                    for candidate in batch
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    record = future.result()
                    refresh_index(
                        catalog,
                        target=target,
                        candidate_concurrency=candidate_concurrency,
                        workers_per_candidate=workers,
                    )
                    print(
                        "FROZEN_0806_COMPLETE "
                        f"deck={candidate.name} run_id={record['run_id']} "
                        f"record={record['wins']}-{record['losses']}-{record['draws']}",
                        flush=True,
                    )
    return refresh_index(
        catalog,
        target=target,
        candidate_concurrency=candidate_concurrency,
        workers_per_candidate=workers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--deck", action="append", default=[])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-games", type=int, default=24)
    parser.add_argument("--benchmark-workers", default="8,12,16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--engine-pool-size", type=int, default=1)
    parser.add_argument("--candidate-concurrency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--inference-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--opponent-policy", choices=("0019", "0806"), default="0019")
    args = parser.parse_args(argv)
    target = evaluation_target(args.opponent_policy)
    catalog = load_frozen_0806_runtime_catalog(
        opponent_policy_label=args.opponent_policy
    )
    if len(catalog.candidates) != EXPECTED_DECKS or catalog.pool.total_games != 256:
        raise RuntimeError("Frozen-0806 runtime catalog does not match the formal contract")
    if args.benchmark:
        worker_counts = tuple(int(value) for value in args.benchmark_workers.split(","))
        records = benchmark(
            catalog,
            target,
            worker_counts=worker_counts,
            games=args.benchmark_games,
            batch_size=args.batch_size,
            batch_wait_ms=args.batch_wait_ms,
            inference_dtype=args.inference_dtype,
            engine_pool_size=args.engine_pool_size,
        )
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    by_id = {candidate.name: candidate for candidate in catalog.candidates}
    requested = list(by_id) if args.all else args.deck
    if not requested:
        parser.error("choose --all or at least one --deck")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        parser.error(f"unknown Frozen-0806 deck: {unknown[0]}")
    run_full(
        catalog,
        (by_id[deck_id] for deck_id in requested),
        target,
        workers=args.workers,
        candidate_concurrency=args.candidate_concurrency,
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
        inference_dtype=args.inference_dtype,
        engine_pool_size=args.engine_pool_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
