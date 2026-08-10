"""Seeded 512-game comparison of selected 0034 V6 decoder checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

from evaluation.frozen_0806_full_evaluation import POLICY_0806_TARGET, _batch_config
from evaluation.frozen_0806_contract import (
    FROZEN_0806_EVALUATION_GAMES as EVALUATION_GAMES,
    FROZEN_0806_EVALUATION_SEED as EVALUATION_SEED,
    evaluation_counts,
    evaluation_schedule_id,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.reporting.index import write_evaluation_index
from evaluation.runner.batch import _game_jobs, _policy_inference_server, run_batch
from evaluation.runtime.seeded import build_seeded_runtime
from evaluation.traces.store import TraceStore

from .export_full_semantic_candidate import export_candidate


ROOT = Path(__file__).resolve().parents[2]
PROJECT = "0042_full_model_design"
VERSION = "V6_exact007_0023_selection_lambda095"
VERSION_ROOT = ROOT / "rl_runs" / PROJECT / "versions" / VERSION
METRICS_PATH = VERSION_ROOT / "artifact/training_metrics.jsonl"
CHECKPOINT_DIR = VERSION_ROOT / "checkpoint"
SOURCE_CANDIDATE = ROOT / "evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot"
OUTPUT_ROOT = ROOT / ".tmp/evaluation/0034_seeded_512_checkpoint_comparison"
FORMAL_REPORT = ROOT / "experiments" / PROJECT / "evaluation" / f"{VERSION}.html"
EVALUATION_RECORD = VERSION_ROOT / "artifact/evaluation.json"
COMPARISON_UPDATES = (0, 10, 20, 80, 100, 123)
_REPORT_DATA_START = '<script id="report-data" type="application/json">'


@dataclass(frozen=True, slots=True)
class ComparisonArm:
    update: int
    label: str
    selection_reason: str
    prior_evidence: str
    prior_win_rate: float
    source_policy_update: int | None
    checkpoint: Path
    checkpoint_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def select_comparison_arms(metrics_path: Path, checkpoint_dir: Path) -> tuple[ComparisonArm, ...]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    by_update: dict[int, dict[str, Any]] = {}
    for row in rows:
        update = row.get("trainer/update")
        if type(update) is not int:
            continue
        if update in by_update:
            raise ValueError(f"duplicate training metric update: {update}")
        by_update[update] = row
    reasons = {
        0: "zero-shot reference",
        10: "early frozen-greedy high point",
        20: "early frozen-greedy regression control",
        80: "middle/late frozen-greedy high point",
        100: "late frozen-greedy high point",
        123: "final user-stopped checkpoint",
    }
    arms = []
    for update in COMPARISON_UPDATES:
        checkpoint = checkpoint_dir / f"update-{update:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        row = by_update.get(update)
        if row is None:
            raise ValueError(f"metrics have no update {update}")
        if "eval/win_rate" in row:
            evidence = "historical_frozen_greedy_probe"
            rate = float(row["eval/win_rate"])
            source_update = None
        elif "rollout/win_rate" in row:
            evidence = "sampled_rollout_context"
            rate = float(row["rollout/win_rate"])
            source_update = int(row["rollout/source_policy_update"])
        else:
            raise ValueError(f"selected update {update} has no win-rate context")
        arms.append(ComparisonArm(
            update=update,
            label=f"update_{update:03d}",
            selection_reason=reasons[update],
            prior_evidence=evidence,
            prior_win_rate=rate,
            source_policy_update=source_update,
            checkpoint=checkpoint,
            checkpoint_sha256=_sha256(checkpoint),
        ))
    return tuple(arms)


def comparison_config(catalog: Any, *, output_root: Path) -> Any:
    counts = evaluation_counts(catalog.pool.schedule)
    base = _batch_config(
        catalog,
        catalog.candidates[0],
        POLICY_0806_TARGET,
        output_root=output_root,
        workers=16,
        batch_size=128,
        batch_wait_ms=1.0,
        inference_dtype="fp16",
        candidate_socket=Path("/external-candidate"),
        opponent_socket=Path("/external-opponent"),
        counts=counts,
        engine_pool_size=6,
    )
    return replace(
        base,
        seed=EVALUATION_SEED,
        opponent_schedule_id=evaluation_schedule_id(
            catalog.pool.manifest["schedule_sha256"]
        ),
        share_policy_inference_server=False,
        worker_local_compiler=True,
        worker_compiler_backend="policy_stateless",
        compiler_workers=1,
        async_h2d=False,
        inference_profile=False,
        worker_timeout_seconds=180.0,
        worker_crash_retries=1,
        candidate_inference_socket=Path("/external-candidate"),
        opponent_inference_socket=Path("/external-opponent"),
    )


def schedule_commitment(jobs: Iterable[tuple[Any, Path]]) -> dict[str, Any]:
    rows = [{
        "game_id": request.game_id,
        "opponent": request.opponent.name,
        "candidate_first": request.candidate_first,
        "engine_seed": request.seed,
        "policy_seed": request.policy_seed,
        "search_seed": request.search_seed,
    } for request, _ in jobs]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "games": len(rows),
        "engine_seed_pairs": len({(row["opponent"], row["engine_seed"]) for row in rows}),
        "candidate_first": sum(row["candidate_first"] for row in rows),
        "candidate_second": sum(not row["candidate_first"] for row in rows),
        "request_schedule_sha256": hashlib.sha256(encoded).hexdigest(),
        "requests": rows,
    }


def _build_schedule(catalog: Any) -> dict[str, Any]:
    config = comparison_config(catalog, output_root=OUTPUT_ROOT / "schedule")
    store = TraceStore(OUTPUT_ROOT / "schedule-traces", OUTPUT_ROOT / "schedule-report")
    try:
        return schedule_commitment(_game_jobs(config, "seeded-512-contract", store))
    finally:
        store.cleanup()


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("invalid binomial counts")
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return centre - spread, centre + spread


def _arm_payload(arm: ComparisonArm, result: Any, export_manifest: dict[str, Any]) -> dict[str, Any]:
    summary = result.report_data.summary
    games = list(result.report_data.games)
    wins = int(summary["wins"])
    low, high = wilson_interval(wins, EVALUATION_GAMES)
    first = [row for row in games if row.get("candidate_first")]
    second = [row for row in games if not row.get("candidate_first")]
    def win_count(rows: list[dict[str, Any]]) -> int:
        return sum(row.get("winner") == 0 for row in rows)
    return {
        "arm": {**asdict(arm), "checkpoint": str(arm.checkpoint)},
        "export_manifest": export_manifest,
        "run_id": result.run_id,
        "report": str(result.report_path.relative_to(ROOT)),
        "wins": wins,
        "losses": int(summary["losses"]),
        "draws": int(summary["draws"]),
        "win_rate": wins / EVALUATION_GAMES,
        "wilson_95": [low, high],
        "first_win_rate": win_count(first) / len(first),
        "second_win_rate": win_count(second) / len(second),
        "performance": summary.get("performance", {}),
        "games": games,
        "manifest": result.report_data.manifest,
    }


def _render(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['arm']['label'])}</td>"
        f"<td>{row['arm']['update']}</td>"
        f"<td>{row['arm']['prior_win_rate']:.2%}</td>"
        f"<td><strong>{row['win_rate']:.2%}</strong></td>"
        f"<td>{row['wins']}-{row['losses']}-{row['draws']}</td>"
        f"<td>{row['first_win_rate']:.2%}</td><td>{row['second_win_rate']:.2%}</td>"
        f"<td>{row['wilson_95'][0]:.2%}–{row['wilson_95'][1]:.2%}</td>"
        f"<td><a href='{html.escape(row['report'])}'>source</a></td></tr>"
        for row in arms
    )
    embedded = json.dumps(payload["report_data"], ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>0034 V6 · Seeded 512 checkpoint comparison</title>
<style>body{{font:15px/1.55 system-ui;margin:32px auto;max-width:1280px;padding:0 20px;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #d9e0ea;text-align:right}}
th:first-child,td:first-child{{text-align:left}}code{{background:#f2f4f7;padding:2px 5px}}a{{color:#3157d5}}</style></head><body>
<h1>0034 V6 · Seeded 512 checkpoint comparison</h1>
<p>六个 decoder 使用同一 exact007 主视角卡组、同一 Frozen-0806 对手分布、同一 256 组 engine seed 正反先后手配对。旧 win rate 仅用于预先选点；本表的 official-engine frozen greedy 才是强度证据。</p>
<p>Schedule <code>{payload['contract']['schedule']['request_schedule_sha256']}</code> · evaluation seed <code>{EVALUATION_SEED}</code></p>
<table><thead><tr><th>Arm</th><th>Update</th><th>旧参考</th><th>新胜率</th><th>W-L-D</th><th>先攻</th><th>后攻</th><th>Wilson 95%</th><th>源报告</th></tr></thead><tbody>{rows}</tbody></table>
{_REPORT_DATA_START}{embedded}</script></body></html>"""


def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    arms = select_comparison_arms(METRICS_PATH, CHECKPOINT_DIR)
    schedule = _build_schedule(catalog)
    runtime = build_seeded_runtime()
    contract = {
        "schema_version": "0034_seeded_512_checkpoint_comparison_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "project": PROJECT,
        "version": VERSION,
        "evaluation_seed": EVALUATION_SEED,
        "training_seed": 330_031_001,
        "updates": list(COMPARISON_UPDATES),
        "schedule": schedule,
        "pool_manifest_sha256": catalog.pool.manifest_sha256,
        "seeded_runtime": runtime.json_payload(),
        "topology": {"workers": 16, "engine_pool_size": 6, "batch_size": 128,
                     "batch_wait_ms": 1.0, "dtype": "fp16", "async_h2d": False},
    }
    _atomic_json(output_root / "contract.json", contract)
    results = []
    with _policy_inference_server(
            root=catalog.opponent_policy.root, device="cuda:0", batch_size=128,
            batch_wait_ms=1.0, label="0034-comparison-opponent", ability_repeat_limit=8,
            inference_dtype="fp16") as opponent_socket:
        for arm in arms:
            arm_root = output_root / "arms" / arm.label
            result_path = arm_root / "result.json"
            if result_path.is_file():
                existing = json.loads(result_path.read_text())
                if (existing["arm"]["checkpoint_sha256"] != arm.checkpoint_sha256 or
                        existing["schedule_sha256"] != schedule["request_schedule_sha256"]):
                    raise RuntimeError(f"stale arm result: {arm.label}")
                results.append(existing)
                continue
            candidate = arm_root / "candidate"
            if candidate.exists():
                shutil.rmtree(candidate)
            export_manifest = export_candidate(
                source=SOURCE_CANDIDATE, checkpoint=arm.checkpoint, output=candidate
            )
            with _policy_inference_server(
                root=candidate, device="cuda:0", batch_size=128, batch_wait_ms=1.0,
                label=f"0034-{arm.label}", ability_repeat_limit=8,
                inference_dtype="fp16") as candidate_socket:
                config = comparison_config(catalog, output_root=arm_root)
                config = replace(config, candidate_inference_socket=candidate_socket,
                                 opponent_inference_socket=opponent_socket)
                result = run_batch(config)
            summary = result.report_data.summary
            if (summary["completed_games"] != EVALUATION_GAMES or summary["errors"] or
                    summary["unfinished"]):
                raise RuntimeError(f"arm did not complete cleanly: {arm.label}")
            payload = _arm_payload(arm, result, export_manifest)
            payload["schedule_sha256"] = schedule["request_schedule_sha256"]
            _atomic_json(result_path, payload)
            results.append(payload)

    ranked = sorted(results, key=lambda row: (row["win_rate"], row["arm"]["update"]), reverse=True)
    champion = ranked[0]
    report_data = {
        "manifest": {"run_id": "0034-v6-seeded-512-series", "finished_at": datetime.now(UTC).isoformat(),
                     "candidate": {"name": champion["arm"]["label"], "display_name": f"0034 update {champion['arm']['update']}"}},
        "summary": {"total_games": EVALUATION_GAMES * len(results), "wins": champion["wins"],
                    "losses": champion["losses"], "draws": champion["draws"], "errors": 0,
                    "unfinished": 0, "win_rate": champion["win_rate"], "completion_rate": 1.0},
        "games": [], "metrics": {}, "cases": [], "metric_profile": {},
        "presentations": {}, "presentation_errors": [],
    }
    aggregate = {"schema_version": contract["schema_version"], "contract": contract,
                 "arms": results, "ranking": [row["arm"]["label"] for row in ranked],
                 "champion": champion["arm"]["label"], "report_data": report_data}
    _atomic_json(output_root / "comparison.json", aggregate)
    document = _render(aggregate)
    (output_root / "index.html").write_text(document, encoding="utf-8")
    if FORMAL_REPORT.exists():
        raise FileExistsError(FORMAL_REPORT)
    FORMAL_REPORT.write_text(document, encoding="utf-8")
    write_evaluation_index(FORMAL_REPORT.parent)
    _atomic_json(EVALUATION_RECORD, {
        "schema_version": "0034_checkpoint_series_evaluation_link_v1",
        "version": VERSION, "report": str(FORMAL_REPORT.relative_to(ROOT)),
        "report_sha256": _sha256(FORMAL_REPORT), "champion": champion["arm"]["label"],
        "request_schedule_sha256": schedule["request_schedule_sha256"],
    })
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run(args.output_root)
    print(json.dumps({"champion": result["champion"], "ranking": result["ranking"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
