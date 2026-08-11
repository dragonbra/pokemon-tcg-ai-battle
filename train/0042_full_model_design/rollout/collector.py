"""Central CUDA batching around persistent multi-battle official-engine workers."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Literal

import torch
from torch import nn

from ..policy.action_distribution import sample_actions_with_encoding
from ..policy.actor_critic import SemanticActorCritic
from ..policy.batching import cpu_batch, move_batch
from ..semantic_policy.features.collate import collate_canonical_records
from .protocol import (
    CanonicalMacroAction, EpisodeTrajectory, PolicyTransition, RolloutJob,
    TrajectoryDecision, require_opponent_policy_binding,
)
from .pool_worker import run_engine_pool
from ..action_boundary.dragapult import (
    PHANTOM_DIVE_ATTACK_ID, PHANTOM_DIVE_MAX_TARGETS, StableTargetIdentity,
)
from ..action_boundary.macro_planner import MacroPlanner
from ..action_boundary.macro_protocol import PendingMacroTransaction
from ..action_boundary.public_card_features import card_prize_counts, with_public_prize
from ..data.allocation_dataset import AllocationBCSample


CHANCE_BOUNDARY_FALLBACK = "chance_boundary_before_allocation"


def phantom_macro_eligible(
    *, phantom_root: int | None, target_count: int, action_boundary_mode: str,
    chance_before_allocation: bool,
) -> bool:
    return (
        phantom_root is not None
        and 1 <= target_count <= PHANTOM_DIVE_MAX_TARGETS
        and action_boundary_mode == "enabled"
        and not chance_before_allocation
    )


@dataclass
class _Session:
    job: RolloutJob
    policy_generator: torch.Generator
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    last_decision: dict[str, object] | None = None
    last_action: tuple[int, ...] | None = None
    recent_actions: list[dict[str, object]] = field(default_factory=list)
    pending_shadow_allocation: dict[str, object] | None = None
    allocation_bc_samples: list[AllocationBCSample] = field(default_factory=list)
    allocation_bc_invalid: int = 0
    allocation_bc_invalid_reasons: dict[str, int] = field(default_factory=dict)
    invalid_macro_episode: bool = False
    macro_fallback_reason: str | None = None

    def invalidate_allocation_chain(self, reason: str) -> None:
        self.pending_shadow_allocation = None
        self.allocation_bc_invalid += 1
        self.allocation_bc_invalid_reasons[reason] = (
            self.allocation_bc_invalid_reasons.get(reason, 0) + 1
        )


@dataclass
class _PoolProcess:
    process: mp.Process
    result_connection: Connection
    inference_connections: list[Connection]
    jobs: list[RolloutJob]


class FullSemanticRolloutCollector:
    def __init__(
        self,
        model: SemanticActorCritic,
        opponent: nn.Module,
        *,
        device: torch.device,
        workers: int | None = None,
        worker_processes: int | None = None,
        engines_per_worker: int = 1,
        inference_channels_per_role: int = 1,
        mode: Literal["sample", "greedy"] = "sample",
        coalesce_ms: float = 0.5,
        timeout_seconds: float = 120.0,
        record_trajectory: bool = True,
    ) -> None:
        if worker_processes is None:
            worker_processes = workers if workers is not None else 8
        if (
            worker_processes < 1
            or engines_per_worker < 1
            or inference_channels_per_role < 1
            or inference_channels_per_role > engines_per_worker
            or coalesce_ms < 0
            or timeout_seconds <= 0
        ):
            raise ValueError("invalid collector configuration")
        self.model = model.eval()
        self.opponent = opponent.eval().requires_grad_(False)
        self.device = device
        self.worker_processes = worker_processes
        self.engines_per_worker = engines_per_worker
        self.inference_channels_per_role = inference_channels_per_role
        self.mode = mode
        self.record_trajectory = bool(record_trajectory)
        self.prizes = card_prize_counts(
            Path(__file__).resolve().parents[1]
            / "semantic_policy/assets/official_full_engine_prototypes_v2.json"
        )
        self.coalesce_seconds = coalesce_ms / 1000.0
        self.timeout_seconds = timeout_seconds
        self.context = mp.get_context("spawn")
        self.inference_batches = 0
        self.inference_requests = 0
        self.focal_requests = 0
        self.opponent_requests = 0
        self.max_batch_size = 0
        self.inference_seconds = 0.0
        self.focal_encode_seconds = 0.0
        self.focal_collate_move_seconds = 0.0
        self.focal_model_seconds = 0.0
        self.focal_transformer_seconds = 0.0
        self.focal_value_seconds = 0.0
        self.focal_gru_seconds = 0.0
        self.opponent_encode_seconds = 0.0
        self.opponent_collate_move_seconds = 0.0
        self.opponent_model_seconds = 0.0
        self.peak_process_tree_rss_bytes = 0
        self._focal_prototype_start = self.model.actor.prototype_cache_stats()
        self._opponent_prototype_start = self.opponent.prototype_cache_stats()
        self.macro_planner = MacroPlanner(self.model.allocation_head)
        self.phantom_macros = 0
        self.forced_choices_observed = 0
        self.forced_shortcuts = 0
        self.strategic_decisions = 0

    def _session(self, job: RolloutJob) -> _Session:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(job.policy_seed or job.seed)
        return _Session(
            job=job,
            policy_generator=generator,
        )

    def _start_pools(self, jobs: list[RolloutJob]) -> list[_PoolProcess]:
        shard_count = min(self.worker_processes, len(jobs))
        shards = [jobs[index::shard_count] for index in range(shard_count)]
        pools: list[_PoolProcess] = []
        for worker_index, shard in enumerate(shards):
            result_parent, result_child = self.context.Pipe(duplex=False)
            focal_pairs = [self.context.Pipe(duplex=True) for _ in range(
                self.inference_channels_per_role
            )]
            opponent_pairs = [self.context.Pipe(duplex=True) for _ in range(
                self.inference_channels_per_role
            )]
            focal_parents = [pair[0] for pair in focal_pairs]
            focal_children = [pair[1] for pair in focal_pairs]
            opponent_parents = [pair[0] for pair in opponent_pairs]
            opponent_children = [pair[1] for pair in opponent_pairs]
            process = self.context.Process(
                target=run_engine_pool,
                args=(
                    result_child,
                    focal_children,
                    opponent_children,
                    shard,
                    self.engines_per_worker,
                ),
                name=f"0038-action-boundary-pool-{worker_index:02d}",
            )
            process.start()
            result_child.close()
            for connection in focal_children + opponent_children:
                connection.close()
            pools.append(_PoolProcess(
                process=process,
                result_connection=result_parent,
                inference_connections=focal_parents + opponent_parents,
                jobs=shard,
            ))
        return pools

    @staticmethod
    def _finish(item: _Session, message: dict, worker_exitcode: int | None) -> EpisodeTrajectory:
        if int(message.get("macro_fallback", 0)):
            item.invalid_macro_episode = True
            item.macro_fallback_reason = str(
                message.get("macro_fallback_reason") or "worker_macro_fallback"
            )
        episode = EpisodeTrajectory(item.job, decisions=item.decisions)
        if message.get("valid") and message.get("reward") is not None:
            episode.finish(float(message["reward"]), int(message.get("final_turn") or 0))
        else:
            detail = message.get("error") or message.get("status")
            episode.error = (
                f"{detail}; game_id={item.job.game_id}; opponent={item.job.opponent_id}; "
                f"focal_first={item.job.focal_first}; seed={item.job.seed}; "
                f"worker_exitcode={worker_exitcode}; last_decision={item.last_decision!r}; "
                f"last_action={item.last_action!r}; recent_actions={item.recent_actions!r}"
            )
        episode.diagnostics = {
            "engine_selections": int(message.get("engine_selections", 0)),
            "termination_status": message.get("status"),
            "termination_error": message.get("error"),
            "source_policy_update": item.job.source_policy_update,
            "worker_exitcode": worker_exitcode,
            "last_decision": item.last_decision,
            "last_action": item.last_action,
            "engine_event_trace": message.get("engine_event_trace", []),
            "forced_shortcuts": int(message.get("forced_shortcuts", 0)),
            "forced_choices_observed": int(message.get("forced_choices_observed", 0)),
            "strategic_decisions": int(message.get("strategic_decisions", 0)),
            "allocation_bc_samples": item.allocation_bc_samples,
            "allocation_bc_invalid": item.allocation_bc_invalid,
            "allocation_bc_invalid_reasons": item.allocation_bc_invalid_reasons,
            "macro_fallback": int(item.invalid_macro_episode),
            "macro_fallback_reason": item.macro_fallback_reason,
            "tempo_events": message.get("tempo_events", []),
        }
        if episode.valid and episode.reward is not None:
            trace = episode.diagnostics["engine_event_trace"]
            own_turns = {
                turn: index + 1 for index, turn in enumerate(sorted({
                    decision.turn for decision in item.decisions if decision.turn is not None
                }))
            }

            def prize_counts(event_index: int) -> tuple[int, int] | None:
                if not trace:
                    return None
                event = trace[min(max(0, event_index), len(trace) - 1)]
                counts = event.get("public_prize_counts")
                if counts is None:
                    players = ((event.get("observation") or {}).get("current") or {}).get("players") or []
                    counts = [len(player.get("prize") or []) for player in players]
                if len(counts) != 2 or not all(isinstance(value, int) for value in counts):
                    return None
                return int(counts[0]), int(counts[1])

            for index, decision in enumerate(item.decisions):
                final = index == len(item.decisions) - 1
                next_value = 0.0 if final else item.decisions[index + 1].value
                start = decision.engine_event_index
                end = (
                    item.decisions[index + 1].engine_event_index
                    if not final else len(trace)
                )
                macro = decision.macro_action
                if macro:
                    commit_index = min(start + 7, len(trace) - 1)
                else:
                    commit_index = min(start + 1, len(trace) - 1)
                post = trace[commit_index].get("observation") if trace else None
                if ((post or {}).get("current") or {}).get("yourIndex") != 0:
                    post = None  # never inject an opponent-private observation
                root_logprob = decision.log_prob - decision.parameter_log_prob
                before_prizes = prize_counts(start)
                after_prizes = prize_counts(end)
                focal_taken = opponent_taken = 0
                if before_prizes is not None and after_prizes is not None:
                    focal_taken = max(0, before_prizes[0] - after_prizes[0])
                    opponent_taken = max(0, before_prizes[1] - after_prizes[1])
                episode.policy_transitions.append(PolicyTransition(
                    pre_action_features=decision.features,
                    canonical_macro_action=CanonicalMacroAction(
                        "phantom_dive" if macro else "root",
                        decision.indices, decision.stopped, macro or {},
                    ),
                    root_old_logprob=root_logprob,
                    parameter_old_logprob=decision.parameter_log_prob,
                    joint_old_logprob=decision.log_prob,
                    pre_action_value=decision.value,
                    accumulated_reward=float(episode.reward) if final else 0.0,
                    accumulated_discount=1.0,
                    post_commit_observation=post,
                    next_value=next_value,
                    done=final,
                    engine_event_span=(start, end),
                    valid=not item.invalid_macro_episode,
                    metadata={
                        "policy_update": decision.policy_update,
                        "turn": decision.turn,
                        "parameter_entropy": decision.parameter_entropy,
                        "pre_action_prize_value": decision.auxiliary_values.get("v_prize", 0.0),
                        "focal_prizes_taken": focal_taken,
                        "opponent_prizes_taken": opponent_taken,
                        "own_turn_index": own_turns.get(decision.turn, 0),
                        "opponent_meta_logits": decision.auxiliary_values.get(
                            "opponent_meta_logits"
                        ),
                    },
                ))
        return episode

    def _infer_focal(self, requests: list[tuple[_Session, dict]]) -> list[object]:
        started = time.perf_counter()
        cpu_features = collate_canonical_records(
            [message["record"] for _, message in requests]
        )
        encoded = time.perf_counter()
        batch = move_batch(cpu_features, self.device)
        prepared = time.perf_counter()
        actions, validated, state, option_embeddings, _, auxiliary, component_timing = sample_actions_with_encoding(
            self.model, batch, greedy=self.mode != "sample",
            generators=[item.policy_generator for item, _ in requests],
        )
        inferred = time.perf_counter()
        self.focal_encode_seconds += encoded - started
        self.focal_collate_move_seconds += prepared - encoded
        self.focal_model_seconds += inferred - prepared
        self.focal_transformer_seconds += component_timing["transformer_seconds"]
        self.focal_value_seconds += component_timing["value_seconds"]
        self.focal_gru_seconds += component_timing["gru_seconds"]
        routed: list[tuple[object, dict | None, dict[str, float]]] = []
        per_request_timing = {
            key: value / max(1, len(requests)) for key, value in component_timing.items()
        }
        for index, (item, message) in enumerate(requests):
            action = actions[index]
            turn = message.get("turn")
            if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
                raise RuntimeError(f"official observation has invalid turn: {turn!r}")
            parameter_log_prob = 0.0
            parameter_entropy = 0.0
            macro_action = None
            macro_wire = None
            root_options = (message.get("select") or {}).get("option") or []
            phantom_root = next((
                selected for selected in action.indices
                if 0 <= selected < len(root_options)
                and isinstance(root_options[selected], dict)
                and root_options[selected].get("attackId") == PHANTOM_DIVE_ATTACK_ID
            ), None)
            raw_bench = message.get("visible_opponent_bench") or []
            if item.job.action_boundary_mode == "shadow":
                self._observe_shadow_allocation(
                    item, message, action, cpu_features, index, phantom_root, raw_bench
                )
            chance_before_allocation = bool(message.get("chance_before_phantom_allocation"))
            if phantom_macro_eligible(
                phantom_root=phantom_root, target_count=len(raw_bench),
                action_boundary_mode=item.job.action_boundary_mode,
                chance_before_allocation=chance_before_allocation,
            ):
                identities = [
                    StableTargetIdentity(1 - int(message["actor"]), int(raw["serial"]),
                                         int(raw["id"]), slot)
                    for slot, raw in enumerate(raw_bench)
                ]
                identities.sort()
                raw_by_serial = {int(raw["serial"]): {**raw, "benchSlot": slot}
                                 for slot, raw in enumerate(raw_bench)}
                visible = [
                    with_public_prize(raw_by_serial[target.serial], self.prizes)
                    for target in identities
                ]
                target_rows = []
                for target in identities:
                    card_mask = (
                        validated.card_mask[index]
                        & validated.card_cat[index, :, 2].eq(2)
                        & validated.card_cat[index, :, 3].eq(6)
                        & validated.card_cat[index, :, 4].eq(target.initial_bench_slot + 1)
                    )
                    positions = card_mask.nonzero(as_tuple=False).flatten()
                    if positions.numel() != 1:
                        raise RuntimeError("Phantom target does not map to exactly one state token")
                    target_rows.append(state.cards[index, int(positions[0])])
                planned = self.macro_planner.plan(
                    state_summary=state.summary[index],
                    root_option=option_embeddings[index, phantom_root],
                    target_embeddings=torch.stack(target_rows),
                    target_identities=identities,
                    visible_targets=visible,
                    greedy=self.mode != "sample",
                    generator=item.policy_generator,
                )
                parameter_log_prob = planned.allocation_logprob
                parameter_entropy = planned.allocation_entropy
                macro_action = {
                    "family": "phantom_dive", "allocation_index": planned.allocation_index,
                    "root_index": phantom_root,
                    "targets": [target.serial for target in planned.allocation.target_ids],
                    "counters": list(planned.allocation.counters),
                }
                macro_wire = PendingMacroTransaction(
                    item.job.game_id, int(message["actor"]), planned.allocation
                ).to_wire()
                self.phantom_macros += 1
            elif (phantom_root is not None and item.job.action_boundary_mode == "enabled"
                  and chance_before_allocation):
                item.invalid_macro_episode = True
                item.macro_fallback_reason = CHANCE_BOUNDARY_FALLBACK
            elif (phantom_root is not None and item.job.action_boundary_mode == "enabled"
                  and len(raw_bench) > PHANTOM_DIVE_MAX_TARGETS):
                item.invalid_macro_episode = True
                item.macro_fallback_reason = "phantom_target_count_above_v1_limit"
            if self.record_trajectory:
                item.decisions.append(
                    TrajectoryDecision(
                        cpu_batch({
                            name: value[index : index + 1]
                            for name, value in cpu_features.items()
                        }),
                        tuple(action.indices),
                        bool(action.stopped),
                        float(action.log_prob) + parameter_log_prob,
                        float(action.entropy) + parameter_entropy,
                        float(action.value),
                        item.job.source_policy_update,
                        turn, parameter_log_prob, parameter_entropy, macro_action,
                        int(message.get("selection_index", -1)),
                        {
                            name: (float(value[index]) if value.ndim == 1
                                   else value[index].detach().float().cpu().tolist())
                            for name, value in auxiliary.items()
                        },
                    )
                )
            routed.append((action, macro_wire, per_request_timing))
        return routed

    @staticmethod
    def _observe_shadow_allocation(
        item: _Session,
        message: dict,
        action,
        cpu_features: dict[str, torch.Tensor],
        batch_index: int,
        phantom_root: int | None,
        raw_bench: list[dict],
    ) -> None:
        """Reconstruct only labels from the unchanged sequential teacher contract."""
        selection = message.get("select") or {}
        context = selection.get("context")
        if phantom_root is not None:
            identities = tuple(sorted(
                StableTargetIdentity(1 - int(message["actor"]), int(raw["serial"]),
                                     int(raw["id"]), slot)
                for slot, raw in enumerate(raw_bench)
            ))
            if not 1 <= len(identities) <= PHANTOM_DIVE_MAX_TARGETS:
                item.invalidate_allocation_chain("target_count_outside_1_8")
                return
            item.pending_shadow_allocation = {
                "features": cpu_batch({name: value[batch_index: batch_index + 1]
                                       for name, value in cpu_features.items()}),
                "root_index": phantom_root,
                "turn": int(message["turn"]),
                "identities": identities,
                "visible": {int(raw["serial"]): raw for raw in raw_bench},
                "order": [],
                "primitive_action_prefix": tuple(
                    tuple(int(value) for value in action)
                    for action in message.get("primitive_action_prefix", [])
                ) + (tuple(action.indices),),
            }
            return
        pending = item.pending_shadow_allocation
        if pending is None:
            return
        if context not in {14, "DamageCounterAny"}:
            item.invalidate_allocation_chain("context_drift")
            return
        indices = tuple(action.indices)
        options = selection.get("option") or []
        if len(indices) != 1 or not 0 <= indices[0] < len(options):
            item.invalidate_allocation_chain("invalid_teacher_action")
            return
        option = options[indices[0]]
        if not isinstance(option, dict):
            item.invalidate_allocation_chain("invalid_teacher_option")
            return
        bench = raw_bench
        slot = option.get("index")
        if option.get("playerIndex") != 1 - int(message["actor"]) or not isinstance(slot, int) \
                or not 0 <= slot < len(bench):
            item.invalidate_allocation_chain("target_slot_drift")
            return
        serial = int(bench[slot]["serial"])
        if serial not in {target.serial for target in pending["identities"]}:
            item.invalidate_allocation_chain("target_identity_drift")
            return
        order = pending["order"]
        order.append(serial)
        if len(order) < 6:
            return
        identities = pending["identities"]
        counters = tuple(order.count(target.serial) for target in identities)
        if sum(counters) != 6:
            item.invalidate_allocation_chain("counter_sum_drift")
            return
        visible = pending["visible"]
        ko = prizes = 0
        for target, count in zip(identities, counters, strict=True):
            raw = visible[target.serial]
            hp = float(raw.get("hp", raw.get("maxHp", 0)))
            if count and count * 10 >= hp:
                ko += 1
                prizes += int(raw.get("prize", raw.get("prizeCount", 1)))
        item.allocation_bc_samples.append(AllocationBCSample(
            battle_id=item.job.game_id, seed=item.job.seed,
            opponent_id=item.job.opponent_id, focal_first=item.job.focal_first,
            turn=pending["turn"], pre_action_features=pending["features"],
            root_index=pending["root_index"],
            target_ids=tuple({"player_index": target.player_index, "serial": target.serial,
                              "card_id": target.card_id,
                              "initial_bench_slot": target.initial_bench_slot}
                             for target in identities),
            counters=counters, primitive_order=tuple(order),
            immediate_ko_targets=ko, immediate_prizes=prizes,
            search_seed=item.job.search_seed,
            primitive_action_prefix=pending["primitive_action_prefix"],
        ))
        item.pending_shadow_allocation = None

    def _infer_opponent(self, requests: list[tuple[_Session, dict]]) -> list[tuple[int, ...]]:
        started = time.perf_counter()
        cpu_features = collate_canonical_records(
            [message["record"] for _, message in requests]
        )
        encoded = time.perf_counter()
        batch = move_batch(cpu_features, self.device)
        prepared = time.perf_counter()
        with torch.inference_mode():
            decoded = self.opponent.deterministic_action_tensors(batch)
        if not bool(decoded.legal.all()):
            raise RuntimeError("Frozen Policy-0809 opponent produced an illegal action")
        actions = [
            tuple(int(value) for value in decoded.sequences[index, : decoded.lengths[index]].tolist())
            for index in range(len(requests))
        ]
        inferred = time.perf_counter()
        self.opponent_encode_seconds += encoded - started
        self.opponent_collate_move_seconds += prepared - encoded
        self.opponent_model_seconds += inferred - prepared
        return actions

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        if not jobs:
            return []
        require_opponent_policy_binding(
            jobs,
            materialized_policy_id=str(getattr(self.opponent, "_policy_id", "")),
        )
        if len({job.game_id for job in jobs}) != len(jobs):
            raise ValueError("rollout game IDs must be unique")
        sessions = {job.game_id: self._session(job) for job in jobs}
        pools = self._start_pools(jobs)
        result_owners = {pool.result_connection: pool for pool in pools}
        inference_owners = {
            connection: pool
            for pool in pools
            for connection in pool.inference_connections
        }
        active = set(result_owners) | set(inference_owners)
        messages: dict[str, tuple[dict, _PoolProcess]] = {}

        def sample_rss() -> None:
            total = 0
            for pid in (os.getpid(), *(pool.process.pid for pool in pools)):
                if pid is None:
                    continue
                try:
                    status = (Path(f"/proc/{pid}/status")).read_text()
                    line = next(row for row in status.splitlines() if row.startswith("VmRSS:"))
                    total += int(line.split()[1]) * 1024
                except (FileNotFoundError, StopIteration, ValueError, PermissionError):
                    pass
            self.peak_process_tree_rss_bytes = max(
                self.peak_process_tree_rss_bytes, total
            )

        try:
            while len(messages) < len(jobs):
                sample_rss()
                ready = list(wait(list(active), timeout=self.timeout_seconds))
                if not ready:
                    raise TimeoutError("official CPU engine workers stopped responding")
                deadline = time.perf_counter() + self.coalesce_seconds
                while len(ready) < len(active) and time.perf_counter() < deadline:
                    more = wait(
                        [connection for connection in active if connection not in ready],
                        timeout=max(0.0, deadline - time.perf_counter()),
                    )
                    if not more:
                        break
                    ready.extend(more)
                focal: list[tuple[_Session, dict, Connection]] = []
                opponent: list[tuple[_Session, dict, Connection]] = []
                for connection in ready:
                    try:
                        message = connection.recv()
                    except EOFError:
                        active.discard(connection)
                        connection.close()
                        continue
                    if message.get("kind") == "decision":
                        game_id = message.get("game_id")
                        if game_id not in sessions:
                            raise RuntimeError(f"unknown pooled rollout session: {game_id!r}")
                        item = sessions[game_id]
                        selection = message.get("select") or {}
                        options = selection.get("option") or []
                        item.last_decision = {
                            "selection_index": int(message.get("selection_index", -1)),
                            "role": message.get("role"),
                            "actor": message.get("actor"),
                            "turn": message.get("turn"),
                            "select_type": selection.get("type"),
                            "select_context": selection.get("context"),
                            "option_count": len(options),
                            "min_count": selection.get("minCount"),
                            "max_count": selection.get("maxCount"),
                            "options": [
                                {
                                    key: option.get(key)
                                    for key in ("type", "id", "card", "skill", "source", "target")
                                    if key in option
                                }
                                for option in options
                                if isinstance(option, dict)
                            ],
                        }
                        if message.get("role") == "focal":
                            self.focal_encode_seconds += float(message.get("compile_seconds", 0.0))
                        else:
                            self.opponent_encode_seconds += float(message.get("compile_seconds", 0.0))
                        request = (item, message, connection)
                        (focal if message.get("role") == "focal" else opponent).append(request)
                    elif message.get("kind") == "result":
                        game_id = message.get("game_id")
                        if game_id not in sessions or game_id in messages:
                            raise RuntimeError(f"invalid duplicate pooled result: {game_id!r}")
                        messages[game_id] = (message, result_owners[connection])
                        self.forced_choices_observed += int(message.get("forced_choices_observed", 0))
                        self.forced_shortcuts += int(message.get("forced_shortcuts", 0))
                        self.strategic_decisions += int(message.get("strategic_decisions", 0))
                    elif message.get("kind") == "worker_done":
                        active.discard(connection)
                        connection.close()
                    else:
                        raise RuntimeError("unknown official CPU worker message")
                started = time.perf_counter()
                routed: list[tuple[_Session, tuple[int, ...], Connection, dict | None, dict | None]] = []
                if focal:
                    routed.extend(
                        (item, tuple(action.indices), connection, macro, timing)
                        for (item, _, connection), (action, macro, timing) in zip(
                            focal,
                            self._infer_focal([(item, message) for item, message, _ in focal]),
                            strict=True,
                        )
                    )
                if opponent:
                    routed.extend(
                        (item, action, connection, None, None)
                        for (item, _, connection), action in zip(
                            opponent,
                            self._infer_opponent([(item, message) for item, message, _ in opponent]),
                            strict=True,
                        )
                    )
                if routed:
                    elapsed = time.perf_counter() - started
                    self.inference_batches += int(bool(focal)) + int(bool(opponent))
                    self.inference_requests += len(routed)
                    self.focal_requests += len(focal)
                    self.opponent_requests += len(opponent)
                    self.max_batch_size = max(self.max_batch_size, len(focal), len(opponent))
                    self.inference_seconds += elapsed
                    for item, action, connection, macro, timing in routed:
                        item.last_action = action
                        item.recent_actions.append(
                            {"decision": item.last_decision, "action": action}
                        )
                        del item.recent_actions[:-12]
                        response = {"kind": "action", "action": list(action)}
                        if macro is not None:
                            response["macro_transaction"] = macro
                        if timing is not None:
                            response["telemetry"] = timing
                        connection.send(response)
        finally:
            for connection in active:
                try:
                    connection.close()
                except OSError:
                    pass
            for pool in pools:
                pool.process.join(timeout=5)
                if pool.process.is_alive():
                    pool.process.terminate()
                    pool.process.join(timeout=2)
        results: list[EpisodeTrajectory] = []
        for job in jobs:
            payload = messages.get(job.game_id)
            if payload is None:
                message = {
                    "valid": False,
                    "status": "worker_eof",
                    "error": "pool worker exited without a game result",
                }
                owner = next(pool for pool in pools if job in pool.jobs)
            else:
                message, owner = payload
            results.append(self._finish(
                sessions[job.game_id], message, owner.process.exitcode
            ))
        return results

    def metrics(self) -> dict[str, float]:
        focal_cache = self.model.actor.prototype_cache_stats()
        opponent_cache = self.opponent.prototype_cache_stats()
        return {
            "rollout/inference_batches": float(self.inference_batches),
            "rollout/inference_requests": float(self.inference_requests),
            "rollout/focal_requests": float(self.focal_requests),
            "rollout/opponent_requests": float(self.opponent_requests),
            "rollout/max_batch_size": float(self.max_batch_size),
            "rollout/mean_batch_size": (
                float(self.inference_requests) / self.inference_batches
                if self.inference_batches else 0.0
            ),
            "rollout/inference_seconds": self.inference_seconds,
            "rollout/focal_encode_seconds": self.focal_encode_seconds,
            "rollout/focal_collate_move_seconds": self.focal_collate_move_seconds,
            "rollout/focal_model_seconds": self.focal_model_seconds,
            "rollout/focal_transformer_seconds": self.focal_transformer_seconds,
            "rollout/focal_value_seconds": self.focal_value_seconds,
            "rollout/focal_gru_seconds": self.focal_gru_seconds,
            "rollout/opponent_encode_seconds": self.opponent_encode_seconds,
            "rollout/opponent_collate_move_seconds": self.opponent_collate_move_seconds,
            "rollout/opponent_model_seconds": self.opponent_model_seconds,
            "rollout/phantom_dive_macros": float(self.phantom_macros),
            "rollout/forced_choices_observed": float(self.forced_choices_observed),
            "rollout/forced_shortcuts": float(self.forced_shortcuts),
            "rollout/strategic_decisions_gate": float(self.strategic_decisions),
            "rollout/peak_process_tree_rss_bytes": float(
                self.peak_process_tree_rss_bytes
            ),
            "rollout/worker_processes": float(self.worker_processes),
            "rollout/engines_per_worker": float(self.engines_per_worker),
            "rollout/inference_channels_per_role": float(
                self.inference_channels_per_role
            ),
            **{
                f"rollout/focal_prototype_cache_{name}": float(
                    focal_cache[name] - self._focal_prototype_start[name]
                )
                for name in ("builds", "hits", "invalidations")
            },
            **{
                f"rollout/opponent_prototype_cache_{name}": float(
                    opponent_cache[name] - self._opponent_prototype_start[name]
                )
                for name in ("builds", "hits", "invalidations")
            },
        }


__all__ = ["FullSemanticRolloutCollector"]
