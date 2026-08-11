"""Run the immutable 2,048-game official-engine evaluation for 0042 update 0."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import html
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

from evaluation.reporting.index import write_evaluation_index
from evaluation.runtime.seeded import build_seeded_runtime

from ..initialization import ALLOCATION_HEAD_CHECKPOINT, build_preset_from_common_update0
from ..integrated.presets import preset
from ..rollout import FullSemanticRolloutCollector
from ..training.run_full_semantic import (
    FOCAL_DECK_ID,
    OPPONENT_POLICY_ID,
    _episode_focal_first,
    focal_deck,
    load_frozen_opponent,
    runtime_root,
)
from ..league import load_frozen_catalog
from ..policy.opponent_archetype import OpponentArchetypeTaxonomy
from .frozen_jobs import build_frozen_jobs
from .frozen_panel import wilson_interval


ROOT = Path(__file__).resolve().parents[3]
VERSION = "V3_update0_chance_boundary_fallback"
ARTIFACT = ROOT / "rl_runs/0042_full_model_design/versions" / VERSION / "artifact"
REPORT = ROOT / "experiments/0042_full_model_design/evaluation" / f"{VERSION}.html"
RESULTS = ARTIFACT / "frozen_update0_2048_results.json"


@dataclass(frozen=True, slots=True)
class _MetaCalibration:
    accuracy: float
    ece: float


def _calibration(logits: torch.Tensor, labels: torch.Tensor) -> _MetaCalibration:
    probabilities = logits.softmax(-1)
    confidence, prediction = probabilities.max(-1)
    ece = logits.new_zeros(())
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = confidence.ge(low) & (
            confidence.lt(high) if index < 9 else confidence.le(high)
        )
        if mask.any():
            ece += mask.float().mean() * (
                confidence[mask].mean()
                - prediction[mask].eq(labels[mask]).float().mean()
            ).abs()
    return _MetaCalibration(
        float(prediction.eq(labels).float().mean()), float(ece)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _rate(rows: list[Any]) -> float:
    return sum(item.reward == 1.0 for item in rows) / len(rows) if rows else 0.0


def _longest_streak(events: list[dict[str, Any]]) -> int:
    current = best = 0
    for event in sorted(events, key=lambda item: int(item.get("own_turn_index", 0))):
        current = current + 1 if event.get("attacked") else 0
        best = max(best, current)
    return best


def summarize(episodes: list[Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_seed = {int(row["seed"]): row for row in manifest["entries"]}
    if len(episodes) != 2048 or len(by_seed) != 2048:
        raise RuntimeError("update-0 Frozen evaluation requires 2,048 unique games")
    invalid = [item for item in episodes if not item.valid]
    rows: list[dict[str, Any]] = []
    matchup: dict[str, list[Any]] = defaultdict(list)
    shard: dict[int, list[Any]] = defaultdict(list)
    first_prize_turns: list[int] = []
    focal_prizes: list[int] = []
    five_prize_losses = 0
    second_turn_attacks = second_turn_opportunities = second_turn_conversions = 0
    attack_streaks: list[int] = []
    meta_logits: list[torch.Tensor] = []
    meta_labels: list[int] = []
    meta_vocabulary = OpponentArchetypeTaxonomy.load()
    for episode in episodes:
        contract = by_seed[episode.job.seed]
        matchup[contract["opponent_archetype"]].append(episode)
        shard[int(contract["shard_id"])].append(episode)
        prizes = sum(int(item.metadata.get("focal_prizes_taken", 0))
                     for item in episode.policy_transitions)
        focal_prizes.append(prizes)
        prize_turns = [int(item.metadata.get("turn", 0)) for item in episode.policy_transitions
                       if int(item.metadata.get("focal_prizes_taken", 0)) > 0]
        if prize_turns:
            first_prize_turns.append(min(prize_turns))
        if episode.reward != 1.0 and prizes >= 5:
            five_prize_losses += 1
        tempo = list(episode.diagnostics.get("tempo_events") or [])
        second = [item for item in tempo if int(item.get("own_turn_index", 0)) == 2]
        attacked = any(bool(item.get("attacked")) for item in second)
        opportunity = any(bool(item.get("attack_legal")) for item in second)
        second_turn_attacks += int(attacked)
        second_turn_opportunities += int(opportunity)
        second_turn_conversions += int(attacked and opportunity)
        attack_streaks.append(_longest_streak(tempo))
        label = meta_vocabulary.classify_target(episode.job.opponent_deck).value
        for decision in episode.decisions:
            logits = decision.auxiliary_values.get("opponent_meta_logits")
            if isinstance(logits, (torch.Tensor, list, tuple)):
                meta_logits.append(torch.as_tensor(logits).detach().cpu().flatten())
                meta_labels.append(label)
        outcome = 1 if episode.reward == 1.0 else -1 if episode.reward == -1.0 else 0
        rows.append({
            "game_id": episode.job.game_id, "seed": episode.job.seed,
            "shard_id": contract["shard_id"], "opponent_deck": episode.job.opponent_id,
            "opponent_archetype": contract["opponent_archetype"],
            "focal_won_toss": episode.job.focal_won_toss,
            "focal_first": _episode_focal_first(episode), "outcome": outcome,
            "turns": episode.turns, "valid": episode.valid, "error": episode.error,
            "fallback": int(episode.diagnostics.get("macro_fallback", 0)),
            "fallback_reason": episode.diagnostics.get("macro_fallback_reason"),
            "forced_shortcuts": int(episode.diagnostics.get("forced_shortcuts", 0)),
            "macro_actions": sum(item.canonical_macro_action.family == "phantom_dive"
                                 for item in episode.policy_transitions),
            "policy_transitions": len(episode.policy_transitions),
            "focal_prizes": prizes, "second_turn_attack": attacked,
            "second_turn_attack_legal": opportunity, "longest_attack_streak": attack_streaks[-1],
        })
    wins = sum(item.reward == 1.0 for item in episodes)
    losses = sum(item.reward == -1.0 for item in episodes)
    draws = len(episodes) - wins - losses
    shard_rates = [_rate(shard[index]) for index in range(8)]
    meta = _calibration(torch.stack(meta_logits), torch.tensor(meta_labels)) if meta_logits else None
    summary = {
        "total_games": len(episodes), "wins": wins, "losses": losses, "draws": draws,
        "errors": len(invalid), "unfinished": len(invalid), "win_rate": wins / len(episodes),
        "completion_rate": (len(episodes) - len(invalid)) / len(episodes),
        "wilson_95": list(wilson_interval(wins, len(episodes))),
        "first_win_rate": _rate([item for item in episodes if _episode_focal_first(item)]),
        "second_win_rate": _rate([item for item in episodes if not _episode_focal_first(item)]),
        "mean_turn": sum(item.turns for item in episodes) / len(episodes),
        "shard_win_rates": shard_rates, "shard_mean": statistics.fmean(shard_rates),
        "shard_variance": statistics.pvariance(shard_rates),
        "mean_first_prize_turn": statistics.fmean(first_prize_turns) if first_prize_turns else None,
        "mean_focal_prizes": statistics.fmean(focal_prizes),
        "five_prize_failure_rate": five_prize_losses / len(episodes),
        "second_turn_attack_rate": second_turn_attacks / len(episodes),
        "second_turn_attack_opportunity_rate": second_turn_opportunities / len(episodes),
        "second_turn_opportunity_conversion": (
            second_turn_conversions / second_turn_opportunities
            if second_turn_opportunities else 0.0
        ),
        "mean_longest_attack_streak": statistics.fmean(attack_streaks),
        "forced_shortcuts": sum(row["forced_shortcuts"] for row in rows),
        "macro_actions": sum(row["macro_actions"] for row in rows),
        "fallbacks": sum(row["fallback"] for row in rows),
        "policy_transitions": sum(row["policy_transitions"] for row in rows),
        "per_matchup": {
            name: {"games": len(items), "win_rate": _rate(items)}
            for name, items in sorted(matchup.items())
        },
        "meta": (
            {"accuracy": meta.accuracy, "ece": meta.ece}
            if meta is not None else None
        ),
        "unavailable_without_extra_trace_or_forward": [
            "immediate_prize_opportunity", "chosen_vs_max_immediate_prizes"
        ],
    }
    return summary, rows


def _render(report_data: dict[str, Any]) -> str:
    summary = report_data["summary"]
    embedded = json.dumps(report_data, ensure_ascii=False).replace("</", "<\\/")
    matchups = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{row['games']}</td><td>{row['win_rate']:.2%}</td></tr>"
        for name, row in summary["per_matchup"].items()
    )
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>0038 update-0 Frozen 2048</title><style>body{{font:15px/1.55 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#f2f4f7;padding:2px 5px}}</style></head><body>
<h1>0038 update-0 · Frozen 2,048</h1>
<p>8×256、2,048 个唯一固定 seed；official engine、greedy、0038 macro protocol。该规模用于约 2pp 方向与稳定性判断，微小差异不能自动宣称显著收益。</p>
<p><strong>{summary['wins']}-{summary['losses']}-{summary['draws']}</strong> · win rate {summary['win_rate']:.2%} · Wilson 95% {summary['wilson_95'][0]:.2%}–{summary['wilson_95'][1]:.2%} · error {summary['errors']} · fallback {summary['fallbacks']}</p>
<p>先攻 {summary['first_win_rate']:.2%} · 后攻 {summary['second_win_rate']:.2%} · shard variance {summary['shard_variance']:.6f} · transitions {summary['policy_transitions']}</p>
<h2>Matchup</h2><table><thead><tr><th>archetype</th><th>games</th><th>win rate</th></tr></thead><tbody>{matchups}</tbody></table>
<script id="report-data" type="application/json">{embedded}</script></body></html>"""


def run(*, workers: int = 16, engines_per_worker: int = 8,
        inference_channels_per_role: int = 8, device_name: str = "cuda:0") -> dict[str, Any]:
    raise RuntimeError(
        "FATAL: this historical CPU update-0 evaluator is disabled for 0042. "
        "Formal Frozen-0809 evaluation must use training.run_full_semantic so the "
        "candidate is exported as FP16 storage, rematerialized as FP32 runtime, "
        "identity-audited, and evaluated through CUDA-2048."
    )
    if REPORT.exists() or RESULTS.exists():
        raise FileExistsError("update-0 Frozen evaluation output already exists")
    device = torch.device(device_name)
    model, identity = build_preset_from_common_update0(
        focal_deck(), preset("FULL_MODEL"), device=device
    )
    opponent = load_frozen_opponent(device, OPPONENT_POLICY_ID)
    opponent_audit = opponent._policy_identity_audit
    if opponent_audit.status != "PASS":
        raise RuntimeError("FATAL: formal Frozen evaluation policy identity audit failed")
    jobs, schedule_sha = build_frozen_jobs(
        focal_deck_id=FOCAL_DECK_ID,
        focal_deck=focal_deck(), runtime_root=runtime_root(),
        source_policy_update=0,
        focal_deployment_identity="Candidate-0042-update000000",
        opponent_effective_policy_sha256=opponent_audit.effective_policy_sha256,
    )
    names = {item.deck_id: item.display_name for item in load_frozen_catalog()}
    manifest = {
        "frozen_panel_version": "0042_frozen_0809_seeded_agent_first_player_v1",
        "master_seed": 341512806,
        "game_list_sha256": schedule_sha,
        "entries": [
            {
                "seed": job.seed,
                "shard_id": index // 256,
                "opponent_archetype": names[job.opponent_id],
            }
            for index, job in enumerate(jobs)
        ],
    }
    collector = FullSemanticRolloutCollector(
        model, opponent, device=device, worker_processes=workers,
        engines_per_worker=engines_per_worker,
        inference_channels_per_role=inference_channels_per_role,
        mode="greedy", coalesce_ms=5.0,
    )
    started = time.perf_counter()
    episodes = collector.collect(jobs)
    wall = time.perf_counter() - started
    summary, rows = summarize(episodes, manifest)
    runtime = build_seeded_runtime()
    evaluation_config = {
        "workers": workers,
        "engines_per_worker": engines_per_worker,
        "inference_channels_per_role": inference_channels_per_role,
        "device": device_name,
    }
    evaluation_config_sha256 = hashlib.sha256(
        json.dumps(
            evaluation_config, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    payload = {
        "schema_version": "0038_update0_frozen_2048_report_v1",
        "manifest": {
            "run_id": "0038-v2-update0-frozen-2048", "finished_at": datetime.now(UTC).isoformat(),
            "candidate": {
                "name": VERSION,
                "display_name": "0038 Zero-Shot Action Boundary update-0",
                "policy_id": "Candidate-0042-update000000",
                "base_policy_id": "Policy-0809",
            },
            "frozen_panel_version": manifest["frozen_panel_version"],
            "canonical_schedule_sha256": schedule_sha,
            "game_list_sha256": manifest["game_list_sha256"],
            "allocation_head_checkpoint": str(ALLOCATION_HEAD_CHECKPOINT.relative_to(ROOT)),
            "allocation_head_checkpoint_sha256": _sha256(ALLOCATION_HEAD_CHECKPOINT),
            "source_policy_checkpoint_sha256": identity.checkpoint_sha256,
            "opponent_policy_id": OPPONENT_POLICY_ID,
            "opponent_effective_policy_sha256": (
                opponent_audit.effective_policy_sha256
            ),
            "policy_identity_audit": opponent_audit.to_manifest(),
            "v5_or_rl_updated_weights_loaded": False,
            "engine": runtime.json_payload(), "action_protocol": "0038_macro_v1",
            "execution_path": "official_cpu_engine_gpu_batched_inference",
            "lane_count": workers * engines_per_worker,
            "selection_mode": "greedy",
            "greedy": True, "preset": "INTEGRATED",
            "master_seed": 341512806,
            "per_game_seeds": [job.seed for job in jobs],
            "matchup_distribution": {
                name: sum(job.opponent_id == deck_id for job in jobs)
                for deck_id, name in names.items()
            },
            "repo_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "evaluation_config": evaluation_config,
            "evaluation_config_sha256": evaluation_config_sha256,
            "topology": {"workers": workers, "engines_per_worker": engines_per_worker,
                         "inference_channels_per_role": inference_channels_per_role},
        },
        "summary": summary,
        "performance": {"evaluation_wall_seconds": wall, "games_per_second": len(episodes) / wall,
                        **collector.metrics()},
        "games": rows,
    }
    _atomic_text(RESULTS, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(REPORT, _render(payload))
    write_evaluation_index(REPORT.parent)
    _atomic_text(ARTIFACT / "evaluation.json", json.dumps({
        "schema_version": "0038_evaluation_link_v1", "version": VERSION,
        "report": str(REPORT.relative_to(ROOT)), "report_sha256": _sha256(REPORT),
        "results": str(RESULTS.relative_to(ROOT)), "results_sha256": _sha256(RESULTS),
    }, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "performance": result["performance"]}, indent=2))
