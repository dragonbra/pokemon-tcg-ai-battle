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
from .deck_routing import audit_exact_deck_rows, exact_deck_sha256
from .protocol import (
    CanonicalMacroAction,
    EpisodeTrajectory,
    PolicyTransition,
    RolloutJob,
    TrajectoryDecision,
    require_opponent_policy_binding,
)


ROOT = Path(__file__).resolve().parents[3]
CUDA_PYTHON = ROOT / "engine_cuda_2_0/python"
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
            if (job.focal_won_toss if job.focal_won_toss is not None else job.focal_first)
            else (job.opponent_deck, job.focal_deck),
            engine_seed=job.seed,
            focal_player=0
            if (job.focal_won_toss if job.focal_won_toss is not None else job.focal_first)
            else 1,
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
        agent_selects_first_player: bool = False,
        decision_trace_sink: Any | None = None,
        opponent_policy_id: str | None = None,
        opponent_identity_audit: Any | None = None,
        opponent_own_archetype_ids: torch.Tensor | None = None,
        role_compacted: bool = True,
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
        # Project-local loaders return complete portable policy wrappers. The
        # resident bridge consumes their actor module, while Champion-G1 keeps
        # the wrapper for its effective Value/strategy adapter path.
        self.opponent = opponent
        self.opponent_actor = (
            opponent.model if hasattr(opponent, "model") else opponent.actor
        ).eval().requires_grad_(False)
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
        self.agent_selects_first_player = bool(agent_selects_first_player)
        self.decision_trace_sink = decision_trace_sink
        if opponent_policy_id is None or opponent_identity_audit is None:
            raise RuntimeError(
                "FATAL: CUDA rollout requires a resolved opponent policy identity"
            )
        self.opponent_policy_id = opponent_policy_id
        self.opponent_identity_audit = opponent_identity_audit
        self.opponent_own_archetype_ids = opponent_own_archetype_ids
        self.role_compacted = bool(role_compacted)
        self._metrics: dict[str, float] = {}

    def _opponent_strategy_fn(
        self, validated: Any, state: Any, options: torch.Tensor,
        job_indices: torch.Tensor | None,
    ):
        if job_indices is None or self.opponent_own_archetype_ids is None:
            raise RuntimeError("compound champion CUDA routing lacks exact-deck own IDs")
        from ..policy.strategy_adapters import build_defined_strategy_context

        policy = self.opponent
        own_ids = self.opponent_own_archetype_ids.index_select(
            0, job_indices.to(self.opponent_own_archetype_ids.device)
        ).to(options.device)
        value, auxiliary = policy.value_and_aux_from_encoded(
            validated, state, options, own_archetype_id=own_ids
        )
        rows, context = build_defined_strategy_context(
            relative_first_player=validated.global_cat[:, 2],
            z_meta=auxiliary["z_meta"], meta_logits=auxiliary["meta_logits"],
            value=value, own_archetype_id=own_ids,
        )

        def readout(hidden: torch.Tensor) -> torch.Tensor:
            if context is None:
                return hidden
            adapted = policy.policy_strategy_adapter(
                hidden.index_select(0, rows), context
            )[0]
            return hidden.index_copy(0, rows, adapted)

        return readout, {"value": value, **auxiliary}

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        if not jobs:
            return []
        require_opponent_policy_binding(
            jobs, materialized_policy_id=self.opponent_policy_id
        )
        if len({job.game_id for job in jobs}) != len(jobs):
            raise ValueError("rollout game IDs must be unique")
        if len({job.source_policy_update for job in jobs}) != 1:
            raise ValueError("resident batch must use one behavior-policy update")
        if any(job.action_boundary_mode != "enabled" for job in jobs):
            raise ValueError("0038 CUDA PPO only accepts the enabled action contract")
        if self.agent_selects_first_player and any(
            job.focal_won_toss is None for job in jobs
        ):
            raise ValueError(
                "Agent-owned first-player choice requires a seeded toss winner for every job"
            )
        if self.record_job_indices is not None and any(
            index < 0 or index >= len(jobs) for index in self.record_job_indices
        ):
            raise ValueError("trajectory job index is outside the resident batch")

        resident_jobs = _resident_jobs(jobs)
        engine_turn_draw_limit = _engine_turn_draw_limit(jobs)
        adapter = Semantic0031DeviceAdapter(
            self.model.actor, None, max_select=self.max_select
        )
        opponent_adapter = Semantic0031DeviceAdapter(
            self.opponent_actor, None, max_select=self.max_select
        )
        focal_own_ids = torch.tensor(
            [job.focal_own_archetype_id for job in jobs],
            dtype=torch.long,
            device=self.device,
        )

        def focal_strategy_fn(
            validated: Any,
            state: Any,
            options: torch.Tensor,
            job_indices: torch.Tensor,
        ):
            from ..policy.strategy_adapters import build_defined_strategy_context

            self.model.set_runtime_own_archetype_ids(
                focal_own_ids.index_select(0, job_indices.long())
            )
            value, auxiliary = self.model.value_and_aux_from_encoded(
                validated, state, options
            )
            strategy_rows, context = build_defined_strategy_context(
                relative_first_player=validated.global_cat[:, 2],
                z_meta=auxiliary["z_meta"],
                meta_logits=auxiliary["meta_logits"],
                value=value,
                own_archetype_id=self.model.own_archetype_ids(
                    value.shape[0], value.device
                ),
            )

            def readout(hidden: torch.Tensor) -> torch.Tensor:
                if context is None:
                    return hidden
                adapted = self.model.policy_strategy_adapter(
                    hidden.index_select(0, strategy_rows), context
                )[0]
                return hidden.index_copy(0, strategy_rows, adapted)

            return readout, {"value": value, **auxiliary}

        router = Semantic0031ResidentRouter(
            focal_adapter=adapter,
            same_policy=False,
            opponent_adapter=opponent_adapter,
            requested_opponent_policy_id=self.opponent_policy_id,
            opponent_identity_audit=self.opponent_identity_audit,
            focal_summary_fn=self.model.actor_summary,
            focal_strategy_fn=focal_strategy_fn,
            opponent_strategy_fn=(
                self._opponent_strategy_fn
                if hasattr(self.opponent, "policy_strategy_adapter") else None
            ),
            role_compacted=self.role_compacted,
        )
        boundary = CudaActionBoundaryAdapter(
            self.model,
            resident_jobs,
            greedy=self.mode == "greedy",
            max_select=self.max_select,
            agent_selects_first_player=self.agent_selects_first_player,
        )
        expected = tuple(sorted(self.model.actor.expected_batch_keys))
        staged: list[_Staged] = []
        staged_bytes = 0
        staged_feature_bytes = 0
        captured_jobs = torch.ones(
            len(jobs), dtype=torch.bool, device=self.device
        ) if self.record_job_indices is None else torch.tensor(
            [index in self.record_job_indices for index in range(len(jobs))],
            dtype=torch.bool,
            device=self.device,
        )
        capture_trajectory = self.record_trajectory and bool(captured_jobs.any())

        def value_fn(
            validated: Any,
            state: Any,
            options: torch.Tensor,
            job_indices: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            self.model.set_runtime_own_archetype_ids(
                focal_own_ids.index_select(0, job_indices.long())
            )
            value, auxiliary = self.model.value_and_aux_from_encoded(validated, state, options)
            return {"value": value, **auxiliary}

        def sink(**payload: Any) -> None:
            nonlocal staged_bytes, staged_feature_bytes
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
            staged_feature_bytes += sum(
                value.numel() * value.element_size() for value in block.features.values()
            )

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
            decision_trace_sink=self.decision_trace_sink,
            compact_prefixes=True,
            action_adapter=boundary,
        )
        hot_seconds = time.perf_counter() - started
        hot_allocated = torch.cuda.max_memory_allocated(self.device)
        hot_reserved = torch.cuda.max_memory_reserved(self.device)
        routing_audit = boundary.routing_audit_manifest()

        episodes = [EpisodeTrajectory(job) for job in jobs]
        materialize_started = time.perf_counter()
        for block in staged:
            job_indices = block.job_indices.cpu().tolist()
            # Keep the high-volume canonical feature tensors on CUDA through PPO
            # collation. Only compact Python control data crosses to the host.
            features = block.features
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
            focal_player = resident_jobs[index].focal_player
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
                "first_player_choice": boundary.first_player_choices.get(index),
                "focal_won_toss": episode.job.focal_won_toss,
                "opponent_policy_id": self.opponent_policy_id,
                "opponent_effective_policy_sha256": (
                    self.opponent_identity_audit.effective_policy_sha256
                    if hasattr(self.opponent_identity_audit, "effective_policy_sha256")
                    else self.opponent_identity_audit["effective_policy_sha256"]
                ),
                "opponent_exact_deck_sha256": exact_deck_sha256(
                    episode.job.opponent_deck
                ),
                "lane_routing_audit_status": "PASS",
                "focal_prizes_remaining": int(result.terminal_prize_counts[index][focal_player]),
                "opponent_prizes_remaining": int(result.terminal_prize_counts[index][1 - focal_player]),
                "lane_routing_audit_sha256": routing_audit[
                    "routing_audit_sha256"
                ],
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
                        "focal_deck_id": episode.job.focal_deck_id,
                        "focal_own_archetype_id": episode.job.focal_own_archetype_id,
                        "turn": decision.turn,
                        "parameter_entropy": decision.parameter_entropy,
                        "pre_action_prize_value": decision.auxiliary_values.get("v_prize", 0.0),
                        "focal_prizes_taken": max(0, before[0] - after[0]),
                        "opponent_prizes_taken": max(0, before[1] - after[1]),
                        "own_turn_index": own_turns.get(decision.turn, 0),
                        "opponent_meta_logits": decision.auxiliary_values.get("opponent_meta_logits"),
                        "cuda_resident": True,
                        "opponent_policy_id": self.opponent_policy_id,
                        "opponent_effective_policy_sha256": episode.diagnostics[
                            "opponent_effective_policy_sha256"
                        ],
                        "opponent_exact_deck_sha256": episode.diagnostics[
                            "opponent_exact_deck_sha256"
                        ],
                        "lane_routing_audit_status": "PASS",
                        "lane_routing_audit_sha256": routing_audit[
                            "routing_audit_sha256"
                        ],
                    },
                ))
            episode.decisions.clear()

        adapter_metrics = result.action_adapter_metrics
        focal_decisions = int(adapter_metrics.get("strategic_decisions", 0))
        stored_decisions = sum(len(episode.policy_transitions) for episode in episodes)
        total_seconds = hot_seconds + materialize_seconds
        scalar_d2h_bytes = staged_bytes - staged_feature_bytes
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
            "rollout/cuda_staged_feature_bytes": float(staged_feature_bytes),
            "rollout/cuda_feature_d2h_bytes": 0.0,
            "rollout/cuda_scalar_d2h_bytes": float(scalar_d2h_bytes),
            "rollout/cuda_features_device_resident": 1.0,
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
            "rollout/lane_routing_audit_pass": 1.0,
            "rollout/lane_routing_audit_failures": 0.0,
            "rollout/lane_routing_roles_audited": float(
                routing_audit["roles_audited"]
            ),
            "rollout/unique_opponent_exact_decks": float(
                routing_audit["unique_opponent_exact_decks"]
            ),
            "rollout/deck_static_cache_misses": float(
                routing_audit["unique_opponent_exact_decks"]
            ),
            "rollout/deck_static_cache_hits": float(
                len(jobs) - routing_audit["unique_opponent_exact_decks"]
            ),
            "rollout/worker_processes": 0.0,
            "rollout/engines_per_worker": 0.0,
            "rollout/inference_channels_per_role": 0.0,
            "rollout/role_compacted_routing": float(self.role_compacted),
        }
        return episodes

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def policy_identity_audit(self) -> dict[str, Any]:
        audit = self.opponent_identity_audit
        if hasattr(audit, "to_manifest"):
            return audit.to_manifest()
        return dict(audit)


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
        wins = losses = draws = 0
        source_update = jobs[0].source_policy_update if jobs else -1
        for begin in range(0, len(jobs), self.rollout_batch_size):
            chunk_started = time.perf_counter()
            kwargs = dict(self.collector_kwargs)
            stop = min(begin + self.rollout_batch_size, len(jobs))
            opponent_own_ids = kwargs.get("opponent_own_archetype_ids")
            if opponent_own_ids is not None:
                if opponent_own_ids.shape != (len(jobs),):
                    raise ValueError(
                        "opponent own-archetype IDs must align with all chunked jobs"
                    )
                kwargs["opponent_own_archetype_ids"] = opponent_own_ids[begin:stop]
            if kwargs.get("record_trajectory", True) and self.trajectory_games_per_update is not None:
                kwargs["record_job_indices"] = {
                    global_index - begin
                    for global_index in range(begin, stop)
                    if global_index in selected_jobs
                }
            collector = CudaFullSemanticRolloutCollector(
                self.model, self.opponent, **kwargs
            )
            chunk_episodes = collector.collect(jobs[begin : begin + self.rollout_batch_size])
            episodes.extend(chunk_episodes)
            wins += sum(episode.reward == 1.0 for episode in chunk_episodes)
            losses += sum(episode.reward == -1.0 for episode in chunk_episodes)
            draws += sum(episode.reward == 0.0 for episode in chunk_episodes)
            chunks.append(collector.metrics())
            completed = min(begin + self.rollout_batch_size, len(jobs))
            print(
                f"[0043 CUDA][update {source_update:04d}] games {completed}/{len(jobs)} "
                f"({completed / max(1, len(jobs)):.1%}) "
                f"W-L-D={wins}-{losses}-{draws} win_rate={wins / completed:.3f} "
                f"chunk={time.perf_counter() - chunk_started:.1f}s "
                f"throughput={completed / max(time.perf_counter() - started, 1e-9):.2f} games/s",
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
            "rollout/cuda_staged_trajectory_bytes", "rollout/cuda_staged_feature_bytes",
            "rollout/cuda_feature_d2h_bytes", "rollout/cuda_scalar_d2h_bytes",
            "rollout/cuda_trajectory_games",
            "rollout/cuda_stored_policy_transitions", "rollout/forced_shortcuts",
            "rollout/macro_actions", "rollout/macro_callbacks", "rollout/invalid_macros",
            "rollout/lane_routing_audit_failures",
            "rollout/lane_routing_roles_audited",
            "rollout/deck_static_cache_misses", "rollout/deck_static_cache_hits",
        }
        maxima = {
            "rollout/max_batch_size", "rollout/cuda_lane_count",
            "rollout/cuda_peak_allocated_bytes", "rollout/cuda_peak_reserved_bytes",
            "rollout/unique_opponent_exact_decks",
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
            "rollout/cuda_features_device_resident": float(
                bool(chunks) and all(
                    row.get("rollout/cuda_features_device_resident") == 1.0
                    for row in chunks
                )
            ),
            "rollout/lane_routing_audit_pass": float(
                bool(chunks) and all(
                    row.get("rollout/lane_routing_audit_pass") == 1.0
                    for row in chunks
                )
            ),
            "rollout/role_compacted_routing": float(
                bool(chunks) and all(
                    row.get("rollout/role_compacted_routing") == 1.0
                    for row in chunks
                )
            ),
            "rollout/source_policy_update": float(source_update),
            "rollout/wins": float(wins),
            "rollout/losses": float(losses),
            "rollout/draws": float(draws),
            "rollout/win_rate": wins / max(1, len(jobs)),
        })
        self._metrics = metrics
        return episodes

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def policy_identity_audit(self) -> dict[str, Any]:
        audit = self.collector_kwargs.get("opponent_identity_audit")
        if audit is None:
            raise RuntimeError("FATAL: chunked CUDA collector has no opponent identity audit")
        if hasattr(audit, "to_manifest"):
            return audit.to_manifest()
        return dict(audit)


__all__ = [
    "ChunkedCudaRolloutCollector", "CudaFullSemanticRolloutCollector",
    "_trajectory_job_indices", "audit_exact_deck_rows", "exact_deck_sha256",
]
