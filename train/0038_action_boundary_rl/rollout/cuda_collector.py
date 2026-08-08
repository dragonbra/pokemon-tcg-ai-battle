"""CUDA-resident official-engine collector for 0038 compound PPO."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Literal

import torch
from torch import nn

from .cuda_action_boundary import CudaActionBoundaryAdapter
from .protocol import (
    CanonicalMacroAction,
    EpisodeTrajectory,
    PolicyTransition,
    RolloutJob,
    TrajectoryDecision,
)


ROOT = Path(__file__).resolve().parents[3]
CUDA_PYTHON = ROOT / "engine_cuda/python"
if str(CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(CUDA_PYTHON))

from ptcg_cuda_engine.semantic0031_bridge import Semantic0031DeviceAdapter
from ptcg_cuda_engine.semantic0031_resident import ResidentJob, run_resident_greedy_jobs
from ptcg_cuda_engine.semantic0031_router import Semantic0031ResidentRouter


@dataclass(frozen=True)
class _Staged:
    job_indices: torch.Tensor
    features: dict[str, torch.Tensor]
    actions: torch.Tensor
    lengths: torch.Tensor
    stopped: torch.Tensor
    root_logprob: torch.Tensor
    root_entropy: torch.Tensor
    values: torch.Tensor
    prize_values: torch.Tensor
    meta_logits: torch.Tensor | None
    prize_counts: torch.Tensor
    turns: torch.Tensor
    metadata: tuple[dict[str, Any] | None, ...]


def _resident_jobs(jobs: list[RolloutJob]) -> tuple[ResidentJob, ...]:
    return tuple(
        ResidentJob(
            schedule_index=index,
            decks=(job.focal_deck, job.opponent_deck)
            if job.focal_first else (job.opponent_deck, job.focal_deck),
            engine_seed=job.seed,
            focal_player=0 if job.focal_first else 1,
            opponent_id=job.opponent_id,
            policy_seed=int(job.policy_seed or job.seed),
        )
        for index, job in enumerate(jobs)
    )


def _engine_turn_draw_limit(jobs: list[RolloutJob]) -> int:
    """Convert the common full-round contract to the official turn index."""

    round_limits = {job.full_round_draw_limit for job in jobs}
    if len(round_limits) != 1:
        raise ValueError("resident batch must use one full-round draw limit")
    full_round_limit = next(iter(round_limits))
    return 2 * full_round_limit - 1 if full_round_limit > 0 else 0


def _termination_status(
    index: int, *, forfeits: set[int], turn_limit_draws: set[int]
) -> str:
    if index in forfeits:
        return "repeat_forfeit"
    if index in turn_limit_draws:
        return "turn_limit_draw"
    return "terminal"


def _trajectory_job_indices(
    jobs: list[RolloutJob], budget: int | None
) -> frozenset[int]:
    """Uniform deterministic episode sample, stratified over 256-game units."""

    if budget is None or budget >= len(jobs):
        return frozenset(range(len(jobs)))
    if budget < 0:
        raise ValueError("trajectory game budget cannot be negative")
    units = [range(start, min(start + 256, len(jobs))) for start in range(0, len(jobs), 256)]
    base, remainder = divmod(budget, len(units))
    selected: set[int] = set()
    for unit_index, unit in enumerate(units):
        quota = base + int(unit_index < remainder)
        if quota > len(unit):
            raise ValueError("trajectory budget cannot be stratified over rollout units")
        ranked = sorted(
            unit,
            key=lambda index: hashlib.sha256(
                f"{jobs[index].source_policy_update}:{jobs[index].game_id}".encode()
            ).digest(),
        )
        selected.update(ranked[:quota])
    if len(selected) != budget:
        raise RuntimeError("trajectory sampling did not produce the configured budget")
    return frozenset(selected)


class CudaFullSemanticRolloutCollector:
    """Collect true strategic boundaries while the official clone stays on GPU."""

    def __init__(
        self,
        model: nn.Module,
        opponent: nn.Module,
        *,
        device: torch.device,
        rules_path: Path,
        extension_dir: Path,
        lane_count: int = 256,
        mode: Literal["sample", "greedy"] = "sample",
        max_select: int = 64,
        max_decisions: int = 8192,
        check_interval: int = 8,
        ability_repeat_limit: int = 20,
        record_trajectory: bool = True,
        record_job_indices: set[int] | frozenset[int] | None = None,
    ) -> None:
        if device.type != "cuda" or lane_count < 1:
            raise ValueError("0038 CUDA collector requires CUDA and positive lanes")
        extension = extension_dir.resolve() / "_ptcg_cuda.so"
        if not extension.is_file() or not rules_path.is_file():
            raise FileNotFoundError(extension if not extension.is_file() else rules_path)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if str(extension_dir.resolve()) not in sys.path:
            sys.path.insert(0, str(extension_dir.resolve()))
        __import__("_ptcg_cuda")
        self.model = model.eval()
        self.opponent = opponent.eval().requires_grad_(False)
        self.device = device
        self.rules_path = rules_path.resolve()
        self.rules = self.rules_path.read_bytes()
        self.extension = extension.resolve()
        self.lane_count = int(lane_count)
        self.mode = mode
        self.max_select = int(max_select)
        self.max_decisions = int(max_decisions)
        self.check_interval = int(check_interval)
        self.ability_repeat_limit = int(ability_repeat_limit)
        self.record_trajectory = bool(record_trajectory)
        self.record_job_indices = (
            None if record_job_indices is None
            else frozenset(int(index) for index in record_job_indices)
        )
        self._metrics: dict[str, float] = {}

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        if not jobs:
            return []
        if len({job.game_id for job in jobs}) != len(jobs):
            raise ValueError("rollout game IDs must be unique")
        if len({job.focal_deck for job in jobs}) != 1:
            raise ValueError("resident collector requires one exact focal deck")
        if len({job.source_policy_update for job in jobs}) != 1:
            raise ValueError("resident batch must use one behavior-policy update")
        if any(job.action_boundary_mode != "enabled" for job in jobs):
            raise ValueError("0038 CUDA PPO only accepts the enabled action contract")
        if self.record_job_indices is not None and any(
            index < 0 or index >= len(jobs) for index in self.record_job_indices
        ):
            raise ValueError("trajectory job index is outside the resident batch")

        resident_jobs = _resident_jobs(jobs)
        engine_turn_draw_limit = _engine_turn_draw_limit(jobs)
        adapter = Semantic0031DeviceAdapter(
            self.model.actor, jobs[0].focal_deck, max_select=self.max_select
        )
        transformer = self.opponent.option_encoder.cross_attention_transformer
        router = Semantic0031ResidentRouter(
            focal_adapter=adapter,
            opponent_last_option_layer=transformer.layers[1],
            opponent_option_norm=transformer.norm,
            opponent_decoder=self.opponent.action_decoder,
            same_policy=False,
            focal_summary_fn=self.model.actor_summary,
        )
        boundary = CudaActionBoundaryAdapter(
            self.model,
            resident_jobs,
            greedy=self.mode == "greedy",
            max_select=self.max_select,
        )
        expected = tuple(sorted(self.model.actor.expected_batch_keys))
        staged: list[_Staged] = []
        staged_bytes = 0
        captured_jobs = torch.ones(
            len(jobs), dtype=torch.bool, device=self.device
        ) if self.record_job_indices is None else torch.tensor(
            [index in self.record_job_indices for index in range(len(jobs))],
            dtype=torch.bool,
            device=self.device,
        )
        capture_trajectory = self.record_trajectory and bool(captured_jobs.any())

        def value_fn(validated: Any, state: Any, options: torch.Tensor) -> dict[str, torch.Tensor]:
            value, auxiliary = self.model.value_and_aux_from_encoded(validated, state, options)
            return {"value": value, **auxiliary}

        def sink(**payload: Any) -> None:
            nonlocal staged_bytes
            route = payload["focal_route"].bool() & captured_jobs.index_select(
                0, payload["lane_job"].long()
            )
            rows = route.nonzero(as_tuple=False).flatten()
            semantic = payload["semantic"]
            routed = payload["routed"]
            values = payload["values"]
            all_metadata = payload.get("decision_metadata") or [None] * route.numel()
            metadata = tuple(all_metadata[int(row)] for row in rows.tolist())
            meta_logits = values.get("opponent_meta_logits")
            block = _Staged(
                job_indices=payload["lane_job"].index_select(0, rows).detach().clone(),
                features={
                    name: semantic[name].index_select(0, rows).detach().clone()
                    for name in expected
                },
                actions=routed.actions.index_select(0, rows).detach().clone(),
                lengths=routed.lengths.index_select(0, rows).detach().clone(),
                stopped=routed.stopped.index_select(0, rows).detach().clone(),
                root_logprob=routed.focal_logprob.index_select(0, rows).detach().clone(),
                root_entropy=routed.focal_entropy.index_select(0, rows).detach().clone(),
                values=values["value"].index_select(0, rows).detach().clone(),
                prize_values=(
                    values["v_prize"].index_select(0, rows).detach().clone()
                    if "v_prize" in values else torch.zeros(rows.numel(), device=self.device)
                ),
                meta_logits=(
                    meta_logits.index_select(0, rows).detach().clone()
                    if meta_logits is not None else None
                ),
                prize_counts=semantic["global_num"].index_select(0, rows)[:, 6:8].detach().clone(),
                turns=payload["turns"].index_select(0, rows).detach().clone(),
                metadata=metadata,
            )
            staged.append(block)
            tensors = [
                *block.features.values(), block.job_indices, block.actions, block.lengths,
                block.stopped, block.root_logprob, block.root_entropy, block.values,
                block.prize_values, block.prize_counts, block.turns,
            ]
            if block.meta_logits is not None:
                tensors.append(block.meta_logits)
            staged_bytes += sum(value.numel() * value.element_size() for value in tensors)

        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = run_resident_greedy_jobs(
            jobs=resident_jobs,
            rules=self.rules,
            router=router,
            lane_count=min(self.lane_count, len(jobs)),
            device=self.device,
            device_index=self.device.index or 0,
            max_select=self.max_select,
            max_decisions=self.max_decisions,
            check_interval=self.check_interval,
            ability_repeat_limit=self.ability_repeat_limit,
            engine_turn_draw_limit=engine_turn_draw_limit,
            focal_greedy=self.mode == "greedy",
            focal_value_fn=value_fn if capture_trajectory else None,
            focal_decision_sink=sink if capture_trajectory else None,
            compact_prefixes=True,
            action_adapter=boundary,
        )
        hot_seconds = time.perf_counter() - started
        hot_allocated = torch.cuda.max_memory_allocated(self.device)
        hot_reserved = torch.cuda.max_memory_reserved(self.device)

        episodes = [EpisodeTrajectory(job) for job in jobs]
        materialize_started = time.perf_counter()
        for block in staged:
            job_indices = block.job_indices.cpu().tolist()
            features = {name: value.cpu() for name, value in block.features.items()}
            actions = block.actions.cpu()
            lengths = block.lengths.cpu()
            stopped = block.stopped.cpu()
            root_logprob = block.root_logprob.cpu()
            root_entropy = block.root_entropy.cpu()
            values = block.values.cpu()
            prize_values = block.prize_values.cpu()
            prize_counts = block.prize_counts.cpu()
            turns = block.turns.cpu()
            meta_logits = block.meta_logits.cpu() if block.meta_logits is not None else None
            for row, job_index in enumerate(job_indices):
                length = int(lengths[row])
                metadata = block.metadata[row] or {}
                parameter_logprob = float(metadata.get("parameter_logprob", 0.0))
                parameter_entropy = float(metadata.get("parameter_entropy", 0.0))
                auxiliary: dict[str, Any] = {
                    "v_prize": float(prize_values[row]),
                    "own_prize_count": int(round(float(prize_counts[row, 0]))),
                    "opponent_prize_count": int(round(float(prize_counts[row, 1]))),
                }
                if meta_logits is not None:
                    auxiliary["opponent_meta_logits"] = meta_logits[row].float().tolist()
                episodes[job_index].decisions.append(TrajectoryDecision(
                    features={name: value[row : row + 1] for name, value in features.items()},
                    indices=tuple(int(value) for value in actions[row, :length].tolist()),
                    stopped=bool(stopped[row]),
                    log_prob=float(root_logprob[row]) + parameter_logprob,
                    entropy=float(root_entropy[row]) + parameter_entropy,
                    value=float(values[row]),
                    policy_update=jobs[job_index].source_policy_update,
                    turn=int(turns[row]),
                    parameter_log_prob=parameter_logprob,
                    parameter_entropy=parameter_entropy,
                    macro_action=metadata.get("macro_action"),
                    auxiliary_values=auxiliary,
                ))
        materialize_seconds = time.perf_counter() - materialize_started

        forfeits = set(result.forfeit_schedule_indices)
        turn_limit_draws = set(result.turn_limit_draw_schedule_indices)
        for index, episode in enumerate(episodes):
            focal_player = 0 if episode.job.focal_first else 1
            game_result = result.game_results[index]
            reward = 1.0 if game_result == focal_player + 1 else -1.0 if game_result else 0.0
            episode.finish(reward, result.terminal_turns[index])
            job_stats = boundary.per_job[index]
            invalid_reason = boundary.invalid_jobs.get(index)
            episode.diagnostics = {
                "engine_selections": result.engine_selections[index],
                "termination_status": _termination_status(
                    index,
                    forfeits=forfeits,
                    turn_limit_draws=turn_limit_draws,
                ),
                "source_policy_update": episode.job.source_policy_update,
                "cuda_resident": True,
                **job_stats,
                "macro_fallback": int(invalid_reason is not None),
                "macro_fallback_reason": invalid_reason,
            }
            decisions = episode.decisions
            own_turns = {
                turn: position + 1
                for position, turn in enumerate(sorted({item.turn for item in decisions}))
            }
            final_absolute = result.terminal_prize_counts[index]
            final_prizes = (
                final_absolute[focal_player], final_absolute[1 - focal_player]
            )
            for row, decision in enumerate(decisions):
                final = row == len(decisions) - 1
                next_decision = None if final else decisions[row + 1]
                before = (
                    int(decision.auxiliary_values["own_prize_count"]),
                    int(decision.auxiliary_values["opponent_prize_count"]),
                )
                after = final_prizes if final else (
                    int(next_decision.auxiliary_values["own_prize_count"]),
                    int(next_decision.auxiliary_values["opponent_prize_count"]),
                )
                root_logprob = decision.log_prob - decision.parameter_log_prob
                macro = decision.macro_action
                episode.policy_transitions.append(PolicyTransition(
                    pre_action_features=decision.features,
                    canonical_macro_action=CanonicalMacroAction(
                        "phantom_dive" if macro else "root",
                        decision.indices,
                        decision.stopped,
                        macro or {},
                    ),
                    root_old_logprob=root_logprob,
                    parameter_old_logprob=decision.parameter_log_prob,
                    joint_old_logprob=decision.log_prob,
                    pre_action_value=decision.value,
                    accumulated_reward=reward if final else 0.0,
                    accumulated_discount=1.0,
                    post_commit_observation=None,
                    next_value=0.0 if final else next_decision.value,
                    done=final,
                    engine_event_span=(-1, -1),
                    valid=invalid_reason is None,
                    metadata={
                        "policy_update": decision.policy_update,
                        "turn": decision.turn,
                        "parameter_entropy": decision.parameter_entropy,
                        "pre_action_prize_value": decision.auxiliary_values.get("v_prize", 0.0),
                        "focal_prizes_taken": max(0, before[0] - after[0]),
                        "opponent_prizes_taken": max(0, before[1] - after[1]),
                        "own_turn_index": own_turns.get(decision.turn, 0),
                        "opponent_meta_logits": decision.auxiliary_values.get("opponent_meta_logits"),
                        "cuda_resident": True,
                    },
                ))
            episode.decisions.clear()

        adapter_metrics = result.action_adapter_metrics
        focal_decisions = int(adapter_metrics.get("strategic_decisions", 0))
        stored_decisions = sum(len(episode.policy_transitions) for episode in episodes)
        total_seconds = hot_seconds + materialize_seconds
        self._metrics = {
            "rollout/inference_batches": float(result.decisions),
            "rollout/inference_requests": float(result.routed_ready_rows),
            "rollout/focal_requests": float(focal_decisions),
            "rollout/max_batch_size": float(min(self.lane_count, len(jobs))),
            "rollout/mean_batch_size": result.routed_ready_rows / max(1, result.decisions),
            "rollout/inference_seconds": result.gpu_seconds,
            "rollout/cuda_hot_loop_seconds": hot_seconds,
            "rollout/cuda_materialize_seconds": materialize_seconds,
            "rollout/cuda_games_per_second": len(jobs) / max(total_seconds, 1e-9),
            "rollout/cuda_hot_loop_games_per_second": len(jobs) / max(hot_seconds, 1e-9),
            "rollout/strategic_decisions_per_second": focal_decisions / max(total_seconds, 1e-9),
            "rollout/cuda_lane_count": float(min(self.lane_count, len(jobs))),
            "rollout/cuda_refill_events": float(result.refill_events),
            "rollout/cuda_repeat_forfeits": float(len(forfeits)),
            "rollout/cuda_turn_limit_draws": float(len(turn_limit_draws)),
            "rollout/cuda_staged_trajectory_bytes": float(staged_bytes),
            "rollout/cuda_trajectory_games": float(sum(
                bool(episode.policy_transitions) for episode in episodes
            )),
            "rollout/cuda_stored_policy_transitions": float(stored_decisions),
            "rollout/cuda_peak_allocated_bytes": float(hot_allocated),
            "rollout/cuda_peak_reserved_bytes": float(hot_reserved),
            "rollout/forced_shortcuts": float(adapter_metrics.get("forced_shortcuts", 0)),
            "rollout/macro_actions": float(adapter_metrics.get("macro_actions", 0)),
            "rollout/macro_callbacks": float(adapter_metrics.get("macro_callbacks", 0)),
            "rollout/invalid_macros": float(adapter_metrics.get("invalid_macros", 0)),
            "rollout/worker_processes": 0.0,
            "rollout/engines_per_worker": 0.0,
            "rollout/inference_channels_per_role": 0.0,
        }
        return episodes

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)


class ChunkedCudaRolloutCollector:
    """Bound trajectory staging memory without changing the rollout contract."""

    def __init__(self, model: nn.Module, opponent: nn.Module, *, rollout_batch_size: int,
                 trajectory_games_per_update: int | None = None,
                 **collector_kwargs: Any) -> None:
        if rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be positive")
        self.model = model
        self.opponent = opponent
        self.rollout_batch_size = int(rollout_batch_size)
        self.trajectory_games_per_update = (
            None if trajectory_games_per_update is None
            else int(trajectory_games_per_update)
        )
        self.collector_kwargs = collector_kwargs
        self._metrics: dict[str, float] = {}

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        if self.trajectory_games_per_update is not None and self.trajectory_games_per_update < 0:
            raise ValueError("trajectory game budget cannot be negative")
        episodes: list[EpisodeTrajectory] = []
        chunks: list[dict[str, float]] = []
        trajectory_budget = (
            min(self.trajectory_games_per_update, len(jobs))
            if (
                self.collector_kwargs.get("record_trajectory", True)
                and self.trajectory_games_per_update is not None
            )
            else 0
        )
        selected_jobs = _trajectory_job_indices(jobs, trajectory_budget)
        started = time.perf_counter()
        for begin in range(0, len(jobs), self.rollout_batch_size):
            chunk_started = time.perf_counter()
            kwargs = dict(self.collector_kwargs)
            if kwargs.get("record_trajectory", True) and self.trajectory_games_per_update is not None:
                stop = min(begin + self.rollout_batch_size, len(jobs))
                kwargs["record_job_indices"] = {
                    global_index - begin
                    for global_index in range(begin, stop)
                    if global_index in selected_jobs
                }
            collector = CudaFullSemanticRolloutCollector(
                self.model, self.opponent, **kwargs
            )
            episodes.extend(collector.collect(jobs[begin : begin + self.rollout_batch_size]))
            chunks.append(collector.metrics())
            completed = min(begin + self.rollout_batch_size, len(jobs))
            print(
                f"[0038 CUDA] games {completed}/{len(jobs)} "
                f"({completed / max(1, len(jobs)):.1%}) "
                f"chunk={time.perf_counter() - chunk_started:.1f}s",
                flush=True,
            )
            del collector
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        additive = {
            "rollout/inference_batches", "rollout/inference_requests",
            "rollout/focal_requests", "rollout/inference_seconds",
            "rollout/cuda_hot_loop_seconds", "rollout/cuda_materialize_seconds",
            "rollout/cuda_refill_events", "rollout/cuda_repeat_forfeits",
            "rollout/cuda_turn_limit_draws",
            "rollout/cuda_staged_trajectory_bytes", "rollout/cuda_trajectory_games",
            "rollout/cuda_stored_policy_transitions", "rollout/forced_shortcuts",
            "rollout/macro_actions", "rollout/macro_callbacks", "rollout/invalid_macros",
        }
        maxima = {
            "rollout/max_batch_size", "rollout/cuda_lane_count",
            "rollout/cuda_peak_allocated_bytes", "rollout/cuda_peak_reserved_bytes",
        }
        metrics: dict[str, float] = {}
        for name in additive:
            metrics[name] = sum(row.get(name, 0.0) for row in chunks)
        for name in maxima:
            metrics[name] = max((row.get(name, 0.0) for row in chunks), default=0.0)
        metrics.update({
            "rollout/cuda_chunks": float(len(chunks)),
            "rollout/cuda_games_per_second": len(jobs) / max(elapsed, 1e-9),
            "rollout/strategic_decisions_per_second": metrics["rollout/focal_requests"] / max(elapsed, 1e-9),
            "rollout/mean_batch_size": metrics["rollout/inference_requests"]
            / max(1.0, metrics["rollout/inference_batches"]),
            "rollout/worker_processes": 0.0,
            "rollout/engines_per_worker": 0.0,
            "rollout/inference_channels_per_role": 0.0,
        })
        self._metrics = metrics
        return episodes

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)


__all__ = [
    "ChunkedCudaRolloutCollector", "CudaFullSemanticRolloutCollector",
    "_trajectory_job_indices",
]
