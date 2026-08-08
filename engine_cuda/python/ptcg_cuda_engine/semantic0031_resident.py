"""Finite job scheduling for refillable CUDA-resident official-engine lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Sequence

from .progress_guard import DeviceRepeatForfeitGuard, apply_loop_forfeits
from .semantic0031_bridge import (
    semantic0031_v2_ready_batch,
)


@dataclass(frozen=True, slots=True)
class ResidentJob:
    schedule_index: int
    decks: tuple[tuple[int, ...], tuple[int, ...]]
    engine_seed: int
    focal_player: int
    opponent_id: str
    policy_seed: int = 0


@dataclass(frozen=True)
class LaneRefill:
    lane_indices: Any
    job_indices: Any
    retired_lane_indices: Any


@dataclass(frozen=True)
class ResidentActionBypass:
    """Primitive actions that provably do not require a policy forward."""

    actions: Any
    lengths: Any
    bypass_mask: Any
    forced_mask: Any
    macro_mask: Any

    def validate(self, *, lane_count: int) -> None:
        import torch

        masks = (self.bypass_mask, self.forced_mask, self.macro_mask)
        if any(mask.shape != (lane_count,) or mask.dtype != torch.bool for mask in masks):
            raise ValueError("resident bypass masks must be bool [lane_count]")
        if self.actions.ndim != 2 or self.actions.shape[0] != lane_count:
            raise ValueError("resident bypass actions must be [lane_count, width]")
        if self.lengths.shape != (lane_count,):
            raise ValueError("resident bypass lengths must be [lane_count]")
        if bool((self.forced_mask & self.macro_mask).any()):
            raise ValueError("forced and macro bypass masks must not overlap")
        if not torch.equal(self.bypass_mask, self.forced_mask | self.macro_mask):
            raise ValueError("bypass mask must equal forced | macro")
        if bool((self.lengths[self.bypass_mask] < 0).any()):
            raise ValueError("resident bypass action lengths cannot be negative")


def select_semantic_rows(
    batch: dict[str, Any], rows: Any, *, batch_size: int
) -> dict[str, Any]:
    """Select device batch rows while preserving scalar/schema metadata."""

    output: dict[str, Any] = {}
    for name, value in batch.items():
        if hasattr(value, "ndim") and value.ndim > 0 and value.shape[0] == batch_size:
            output[name] = value.index_select(0, rows)
        else:
            output[name] = value
    return output


def merge_resident_actions(
    *,
    bypass: ResidentActionBypass,
    policy_rows: Any,
    policy_actions: Any,
    policy_lengths: Any,
    lane_count: int,
    action_width: int,
    device: Any,
) -> tuple[Any, Any]:
    """Scatter compact policy rows and bypass primitives into lane order."""

    import torch

    bypass.validate(lane_count=lane_count)
    width = max(action_width, int(bypass.actions.shape[1]), int(policy_actions.shape[1]))
    actions = torch.zeros((lane_count, width), dtype=torch.long, device=device)
    lengths = torch.zeros(lane_count, dtype=torch.long, device=device)
    if policy_rows.numel():
        actions[policy_rows, : policy_actions.shape[1]] = policy_actions
        lengths.index_copy_(0, policy_rows, policy_lengths.long())
    actions[:, : bypass.actions.shape[1]] = torch.where(
        bypass.bypass_mask[:, None], bypass.actions.long(),
        actions[:, : bypass.actions.shape[1]],
    )
    lengths = torch.where(bypass.bypass_mask, bypass.lengths.long(), lengths)
    return actions.contiguous(), lengths.contiguous()


class ResidentLaneQueue:
    """Map a finite ordered job queue onto reusable device lanes."""

    def __init__(self, jobs: Sequence[ResidentJob], *, lane_count: int, device: Any) -> None:
        import torch

        if not jobs or lane_count < 1:
            raise ValueError("resident jobs and lane_count must be positive")
        if [job.schedule_index for job in jobs] != list(range(len(jobs))):
            raise ValueError("resident jobs must use contiguous schedule indices")
        self.jobs = tuple(jobs)
        self.lane_count = int(lane_count)
        self.device = torch.device(device)
        self.lane_job = torch.full(
            (lane_count,), -1, dtype=torch.long, device=self.device
        )
        self.results = torch.zeros(len(jobs), dtype=torch.uint8, device=self.device)
        self.finished = torch.zeros(len(jobs), dtype=torch.bool, device=self.device)
        self.next_job = 0

    @property
    def completed(self) -> int:
        return int(self.finished.long().sum().item())

    def initial(self) -> LaneRefill:
        import torch

        count = min(self.lane_count, len(self.jobs))
        lanes = torch.arange(count, dtype=torch.long, device=self.device)
        jobs = torch.arange(count, dtype=torch.long, device=self.device)
        self.lane_job[:count] = jobs
        self.next_job = count
        retired = torch.arange(
            count, self.lane_count, dtype=torch.long, device=self.device
        )
        return LaneRefill(lanes, jobs, retired)

    def complete(self, terminal_mask: Any, game_results: Any) -> LaneRefill:
        import torch

        terminal = terminal_mask.bool().view(self.lane_count) & self.lane_job.ge(0)
        lanes = terminal.nonzero(as_tuple=False).flatten()
        if not lanes.numel():
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return LaneRefill(empty, empty, empty)
        completed_jobs = self.lane_job.index_select(0, lanes)
        if bool(self.finished.index_select(0, completed_jobs).any()):
            raise RuntimeError("resident lane attempted to complete one job twice")
        self.results.index_copy_(0, completed_jobs, game_results.index_select(0, lanes))
        self.finished.index_fill_(0, completed_jobs, True)
        self.lane_job.index_fill_(0, lanes, -1)

        available = len(self.jobs) - self.next_job
        refill_count = min(int(lanes.numel()), available)
        refill_lanes = lanes[:refill_count]
        refill_jobs = torch.arange(
            self.next_job,
            self.next_job + refill_count,
            dtype=torch.long,
            device=self.device,
        )
        if refill_count:
            self.lane_job.index_copy_(0, refill_lanes, refill_jobs)
            self.next_job += refill_count
        return LaneRefill(refill_lanes, refill_jobs, lanes[refill_count:])


def apply_turn_limit_draws(engine: Any, draw_mask: Any) -> None:
    """Mark selected resident lanes terminal with the canonical draw result."""

    import torch

    mask = draw_mask.bool().view(-1)
    statuses = engine.statuses()
    results = engine.game_results()
    if mask.shape != statuses.shape or results.shape != statuses.shape:
        raise ValueError("turn-limit mask and engine result tensors disagree")
    results.copy_(torch.where(mask, torch.zeros_like(results), results))
    statuses.copy_(
        torch.where(mask, torch.full_like(statuses, 2), statuses)
    )


_PREFIX_FAMILIES = {
    "card": ("card_cat", "card_num", "card_state", "card_mask", "card_parent"),
    "resource": ("resource_cat", "resource_num", "resource_state", "resource_mask"),
    "event": (
        "event_cat", "event_num", "event_state", "event_mask", "event_source",
        "event_target", "event_before", "event_after",
    ),
    "option": (
        "option_cat", "option_num", "option_state", "option_mask", "option_source",
        "option_target", "option_context", "option_effect_card",
    ),
    "option_skill": (
        "option_skill_id", "option_skill_role", "option_skill_parent", "option_skill_mask",
    ),
    "option_effect": (
        "option_effect_id", "option_effect_role", "option_effect_parent", "option_effect_mask",
    ),
}
_PREFIX_MASKS = {
    "card": "card_mask",
    "resource": "resource_mask",
    "event": "event_mask",
    "option": "option_mask",
    "option_skill": "option_skill_mask",
    "option_effect": "option_effect_mask",
}


def compact_semantic_prefixes(batch: dict[str, Any]) -> dict[str, Any]:
    """Trim globally masked suffixes without changing row order."""

    import torch

    families = tuple(_PREFIX_FAMILIES)
    maxima = torch.stack(
        [batch[_PREFIX_MASKS[family]].long().sum(dim=1).amax() for family in families]
    ).cpu().tolist()
    widths = {
        family: max(1, int(maximum))
        for family, maximum in zip(families, maxima, strict=True)
    }
    output = dict(batch)
    for family, names in _PREFIX_FAMILIES.items():
        width = widths[family]
        for name in names:
            output[name] = batch[name][:, :width]
    return output


@dataclass(frozen=True)
class ResidentRunResult:
    game_results: tuple[int, ...]
    terminal_state_bytes: bytes
    forfeit_schedule_indices: tuple[int, ...]
    turn_limit_draw_schedule_indices: tuple[int, ...]
    terminal_turns: tuple[int, ...]
    engine_selections: tuple[int, ...]
    terminal_prize_counts: tuple[tuple[int, int], ...]
    decisions: int
    routed_ready_rows: int
    refill_events: int
    wall_seconds: float
    gpu_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    action_adapter_metrics: dict[str, int] = field(default_factory=dict)


def run_resident_greedy_jobs(
    *,
    jobs: Sequence[ResidentJob],
    rules: bytes,
    router: Any,
    lane_count: int,
    device: Any,
    device_index: int = 0,
    max_select: int = 64,
    max_decisions: int = 8192,
    check_interval: int = 8,
    ability_repeat_limit: int = 20,
    engine_turn_draw_limit: int = 100,
    focal_greedy: bool = True,
    focal_value_fn: Any | None = None,
    focal_decision_sink: Any | None = None,
    compact_prefixes: bool = True,
    action_adapter: Any | None = None,
) -> ResidentRunResult:
    """Run finite routed jobs while immediately reusing terminal lanes."""

    import torch
    from .native import create_official_engine

    if min(lane_count, max_select, max_decisions, check_interval) < 1:
        raise ValueError("resident rollout dimensions must be positive")
    if engine_turn_draw_limit < 0:
        raise ValueError("engine_turn_draw_limit cannot be negative")
    device = torch.device(device)
    queue = ResidentLaneQueue(jobs, lane_count=lane_count, device=device)
    job_decks = torch.tensor(
        [job.decks for job in jobs], dtype=torch.int32, device=device
    )
    job_seeds = torch.tensor(
        [job.engine_seed for job in jobs], dtype=torch.int64, device=device
    )
    job_focal = torch.tensor(
        [job.focal_player for job in jobs], dtype=torch.long, device=device
    )
    job_policy_seeds = torch.tensor(
        [job.policy_seed for job in jobs], dtype=torch.int64, device=device
    )
    lane_decks = torch.zeros((lane_count, 2, 60), dtype=torch.int32, device=device)
    lane_seeds = torch.ones(lane_count, dtype=torch.int64, device=device)
    lane_focal = torch.zeros(lane_count, dtype=torch.long, device=device)
    lane_policy_seeds = torch.zeros(lane_count, dtype=torch.int64, device=device)
    lane_focal_decisions = torch.zeros(lane_count, dtype=torch.int64, device=device)
    lane_forfeit = torch.zeros(lane_count, dtype=torch.bool, device=device)
    lane_turn_limit_draw = torch.zeros(lane_count, dtype=torch.bool, device=device)
    completed_forfeit = torch.zeros(len(jobs), dtype=torch.bool, device=device)
    completed_turn_limit_draw = torch.zeros(
        len(jobs), dtype=torch.bool, device=device
    )
    completed_turns = torch.zeros(len(jobs), dtype=torch.int32, device=device)
    completed_selections = torch.zeros(len(jobs), dtype=torch.int32, device=device)
    completed_prizes = torch.zeros((len(jobs), 2), dtype=torch.int16, device=device)
    lane_turn = torch.zeros(lane_count, dtype=torch.int32, device=device)
    lane_selections = torch.zeros(lane_count, dtype=torch.int32, device=device)
    lanes = torch.arange(lane_count, dtype=torch.int32, device=device)
    engine = create_official_engine(rules, batch_size=lane_count, device_index=device_index)
    terminal_states = torch.zeros(
        (len(jobs), engine.state_bytes().shape[1]), dtype=torch.uint8, device=device
    )
    guard = DeviceRepeatForfeitGuard(
        batch_size=lane_count,
        limit=ability_repeat_limit,
        device=device,
    )

    def assign(refill: LaneRefill) -> None:
        if not refill.lane_indices.numel():
            return
        lane_decks.index_copy_(
            0, refill.lane_indices, job_decks.index_select(0, refill.job_indices)
        )
        lane_seeds.index_copy_(
            0, refill.lane_indices, job_seeds.index_select(0, refill.job_indices)
        )
        lane_focal.index_copy_(
            0, refill.lane_indices, job_focal.index_select(0, refill.job_indices)
        )
        lane_policy_seeds.index_copy_(
            0, refill.lane_indices, job_policy_seeds.index_select(0, refill.job_indices)
        )
        lane_focal_decisions.index_fill_(0, refill.lane_indices, 0)
        lane_forfeit.index_fill_(0, refill.lane_indices, False)
        lane_turn_limit_draw.index_fill_(0, refill.lane_indices, False)
        lane_turn.index_fill_(0, refill.lane_indices, 0)
        lane_selections.index_fill_(0, refill.lane_indices, 0)
        reset_mask = torch.zeros(lane_count, dtype=torch.bool, device=device)
        reset_mask.index_fill_(0, refill.lane_indices, True)
        guard.reset(reset_mask)
        engine.reset_seeded_interactive_semantic_masked(
            lane_decks, lane_seeds, reset_mask
        )
        if action_adapter is not None and hasattr(action_adapter, "on_refill"):
            action_adapter.on_refill(
                lane_indices=refill.lane_indices,
                job_indices=refill.job_indices,
            )

    initial = queue.initial()
    assign(initial)
    engine.advance_to_decision()
    completed = 0
    refill_events = 1
    routed_rows = torch.zeros((), dtype=torch.int64, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    started.record()

    with torch.inference_mode():
        for step in range(max_decisions):
            statuses = engine.statuses()
            active_lanes = queue.lane_job.ge(0)
            lane_turn.copy_(engine.turns())
            turn_limit_draws = (
                active_lanes
                & ~statuses.eq(2)
                & lane_turn.ge(engine_turn_draw_limit)
                if engine_turn_draw_limit > 0
                else torch.zeros_like(active_lanes)
            )
            lane_turn_limit_draw |= turn_limit_draws
            apply_turn_limit_draws(engine, turn_limit_draws)
            ready = statuses.eq(1) & active_lanes & ~turn_limit_draws
            force_terminal_check = bool(turn_limit_draws.any().item())
            semantic = semantic0031_v2_ready_batch(
                engine.encode_semantic0031_v2_lanes(lanes),
                max_action_steps=max_select,
            )
            if compact_prefixes:
                semantic = compact_semantic_prefixes(semantic)
            # Semantic0031 is actor-relative by design.  global_cat[3] is the
            # supporter-played flag, not an absolute seat.  Routing must use the
            # official state field so a supporter play cannot silently swap the
            # focal and opponent policies.
            actor = engine.decision_actors().long()
            focal_route = ready & actor.eq(lane_focal)
            opponent_route = ready & ~focal_route
            action_width = max(
                1,
                min(
                    max_select,
                    int(torch.where(ready, semantic["max_count"], 0).amax().item()),
                ),
            )
            bypass = None
            if action_adapter is not None:
                bypass = action_adapter.pre_route(
                    semantic=semantic,
                    ready=ready,
                    focal_route=focal_route,
                    opponent_route=opponent_route,
                    lane_job=queue.lane_job,
                    turns=lane_turn,
                    selections=lane_selections,
                    max_select=max_select,
                )
                bypass.validate(lane_count=lane_count)
                if bool((bypass.bypass_mask & ~ready).any()):
                    raise RuntimeError("action adapter attempted to bypass a non-ready lane")
                policy_ready = ready & ~bypass.bypass_mask
                policy_lanes = policy_ready.nonzero(as_tuple=False).flatten()
                policy_semantic = select_semantic_rows(
                    semantic, policy_lanes, batch_size=lane_count
                )
                routed = None
                decision_metadata = None
                if policy_lanes.numel():
                    local_focal = focal_route.index_select(0, policy_lanes)
                    local_opponent = opponent_route.index_select(0, policy_lanes)
                    routed = router.route(
                        policy_semantic,
                        focal_route=local_focal,
                        opponent_route=local_opponent,
                        max_select=action_width,
                        focal_greedy=focal_greedy,
                        compute_stats=not focal_greedy,
                        focal_sampling_seeds=lane_policy_seeds.index_select(0, policy_lanes),
                        focal_sampling_counters=lane_focal_decisions.index_select(0, policy_lanes),
                    )
                    if hasattr(action_adapter, "post_route"):
                        decision_metadata = action_adapter.post_route(
                            semantic=policy_semantic,
                            routed=routed,
                            focal_route=local_focal,
                            opponent_route=local_opponent,
                            lane_indices=policy_lanes,
                            lane_job=queue.lane_job.index_select(0, policy_lanes),
                            turns=lane_turn.index_select(0, policy_lanes),
                        )
                    actions, routed_lengths = merge_resident_actions(
                        bypass=bypass,
                        policy_rows=policy_lanes,
                        policy_actions=routed.actions,
                        policy_lengths=routed.lengths,
                        lane_count=lane_count,
                        action_width=action_width,
                        device=device,
                    )
                else:
                    empty_actions = torch.empty(
                        (0, action_width), dtype=torch.long, device=device
                    )
                    empty_lengths = torch.empty(0, dtype=torch.long, device=device)
                    actions, routed_lengths = merge_resident_actions(
                        bypass=bypass,
                        policy_rows=policy_lanes,
                        policy_actions=empty_actions,
                        policy_lengths=empty_lengths,
                        lane_count=lane_count,
                        action_width=action_width,
                        device=device,
                    )
                routed_rows.add_(policy_ready.long().sum())
            else:
                policy_lanes = ready.nonzero(as_tuple=False).flatten()
                policy_semantic = semantic
                local_focal = focal_route
                routed = router.route(
                    semantic,
                    focal_route=focal_route,
                    opponent_route=opponent_route,
                    max_select=action_width,
                    focal_greedy=focal_greedy,
                    compute_stats=not focal_greedy,
                    focal_sampling_seeds=lane_policy_seeds,
                    focal_sampling_counters=lane_focal_decisions,
                )
                actions, routed_lengths = routed.actions, routed.lengths
                decision_metadata = None
                routed_rows.add_(ready.long().sum())
            lane_selections.add_(ready.to(dtype=torch.int32))
            strategic_focal = focal_route if bypass is None else focal_route & ~bypass.bypass_mask
            lane_focal_decisions.add_(strategic_focal.to(dtype=torch.int64))
            if (
                focal_decision_sink is not None
                and routed is not None
                and bool(local_focal.any().item())
            ):
                focal_value = (
                    focal_value_fn(
                        routed.validated,
                        routed.state,
                        routed.focal_options,
                    )
                    if focal_value_fn is not None
                    else None
                )
                focal_decision_sink(
                    semantic=policy_semantic,
                    routed=routed,
                    focal_route=local_focal,
                    lane_job=queue.lane_job.index_select(0, policy_lanes)
                    if bypass is not None else queue.lane_job,
                    lane_indices=policy_lanes,
                    turns=lane_turn.index_select(0, policy_lanes)
                    if bypass is not None else lane_turn,
                    values=focal_value,
                    decision_metadata=decision_metadata,
                )
            new_forfeits = guard.observe(
                option_cat=semantic["option_cat"],
                selection_type=semantic["global_cat"][:, 0],
                turn=semantic["global_num"][:, 0],
                actor=actor,
                actions=actions,
                lengths=routed_lengths,
                ready=ready,
            )
            lane_forfeit |= new_forfeits
            apply_loop_forfeits(engine, new_forfeits, actor)
            lengths = torch.where(new_forfeits, 0, routed_lengths)
            engine.pack_actions(actions, lengths)
            engine.apply_packed_actions()
            engine.advance_to_decision()
            # Result value 0 is both the engine's non-terminal sentinel and the
            # scheduler's draw value.  Advance reclassifies it as running, so the
            # scheduler-owned terminal status must be restored before collection.
            apply_turn_limit_draws(engine, turn_limit_draws)
            decisions = step + 1

            if decisions % check_interval != 0 and not force_terminal_check:
                continue
            checked = engine.statuses()
            active_errors = checked.eq(3) & queue.lane_job.ge(0)
            if bool(active_errors.any().item()):
                error_rows = active_errors.nonzero(as_tuple=False).flatten()
                error_lanes = error_rows.cpu().tolist()
                error_lengths = lengths.index_select(0, error_rows).long().cpu().tolist()
                error_actions = actions.index_select(0, error_rows).long().cpu()
                diagnostic = []
                for offset, lane in enumerate(error_lanes):
                    action_length = int(error_lengths[offset])
                    diagnostic.append({
                        "lane": int(lane),
                        "schedule_index": int(queue.lane_job[lane].item()),
                        "turn": int(engine.turns()[lane].item()),
                        "actor": int(actor[lane].item()),
                        "selection_type": int(
                            semantic["global_cat"][lane, 0].item()
                        ),
                        "selection_context": int(
                            semantic["global_cat"][lane, 1].item()
                        ),
                        "min_count": int(semantic["min_count"][lane].item()),
                        "max_count": int(semantic["max_count"][lane].item()),
                        "option_count": int(
                            semantic["option_mask"][lane].long().sum().item()
                        ),
                        "repeat_forfeit": bool(new_forfeits[lane].item()),
                        "action": error_actions[offset, :action_length].tolist(),
                    })
                raise RuntimeError(
                    f"resident CUDA engine errors: {diagnostic}"
                )
            terminal = checked.eq(2) & queue.lane_job.ge(0)
            terminal_lanes = terminal.nonzero(as_tuple=False).flatten()
            if terminal_lanes.numel():
                completed_jobs = queue.lane_job.index_select(0, terminal_lanes)
                if action_adapter is not None and hasattr(action_adapter, "on_terminal"):
                    action_adapter.on_terminal(
                        lane_indices=terminal_lanes,
                        job_indices=completed_jobs,
                    )
                terminal_states.index_copy_(
                    0,
                    completed_jobs,
                    engine.state_bytes().index_select(0, terminal_lanes),
                )
                completed_forfeit.index_copy_(
                    0, completed_jobs, lane_forfeit.index_select(0, terminal_lanes)
                )
                completed_turn_limit_draw.index_copy_(
                    0,
                    completed_jobs,
                    lane_turn_limit_draw.index_select(0, terminal_lanes),
                )
                completed_turns.index_copy_(
                    0, completed_jobs, lane_turn.index_select(0, terminal_lanes)
                )
                completed_selections.index_copy_(
                    0, completed_jobs, lane_selections.index_select(0, terminal_lanes)
                )
                completed_prizes.index_copy_(
                    0, completed_jobs, engine.prize_counts().index_select(0, terminal_lanes)
                )
                refill = queue.complete(terminal, engine.game_results())
                completed += int(terminal_lanes.numel())
                if refill.lane_indices.numel():
                    assign(refill)
                    engine.advance_to_decision()
                    refill_events += 1
            if completed == len(jobs):
                break
        else:
            active = queue.lane_job.ge(0)
            active_rows = active.nonzero(as_tuple=False).flatten()
            diagnostic = {
                "jobs": queue.lane_job.index_select(0, active_rows).cpu().tolist(),
                "turns": engine.turns().index_select(0, active_rows).cpu().tolist(),
                "lane_turns": lane_turn.index_select(0, active_rows).cpu().tolist(),
                "draw_mask": turn_limit_draws
                .index_select(0, active_rows)
                .cpu()
                .tolist(),
                "draw_limit": engine_turn_draw_limit,
                "actors": engine.decision_actors()
                .index_select(0, active_rows)
                .cpu()
                .tolist(),
                "selections": lane_selections
                .index_select(0, active_rows)
                .cpu()
                .tolist(),
                "repeat_max": guard.counts.amax(dim=(1, 2))
                .index_select(0, active_rows)
                .cpu()
                .tolist(),
            }
            raise RuntimeError(
                f"resident rollout exhausted {max_decisions} decisions at "
                f"{completed}/{len(jobs)}; active={diagnostic}"
            )

    ended.record()
    ended.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    gpu_seconds = started.elapsed_time(ended) / 1000.0
    game_results = tuple(int(value) for value in queue.results.cpu().tolist())
    forfeit_indices = tuple(
        int(value)
        for value in completed_forfeit.nonzero(as_tuple=False).flatten().cpu().tolist()
    )
    turn_limit_draw_indices = tuple(
        int(value)
        for value in completed_turn_limit_draw.nonzero(as_tuple=False)
        .flatten()
        .cpu()
        .tolist()
    )
    terminal_bytes = terminal_states.contiguous().cpu().numpy().tobytes()
    return ResidentRunResult(
        game_results=game_results,
        terminal_state_bytes=terminal_bytes,
        forfeit_schedule_indices=forfeit_indices,
        turn_limit_draw_schedule_indices=turn_limit_draw_indices,
        terminal_turns=tuple(int(value) for value in completed_turns.cpu().tolist()),
        engine_selections=tuple(
            int(value) for value in completed_selections.cpu().tolist()
        ),
        terminal_prize_counts=tuple(
            (int(row[0]), int(row[1])) for row in completed_prizes.cpu().tolist()
        ),
        decisions=decisions,
        routed_ready_rows=int(routed_rows.item()),
        refill_events=refill_events,
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        action_adapter_metrics=(
            {str(key): int(value) for key, value in action_adapter.metrics().items()}
            if action_adapter is not None and hasattr(action_adapter, "metrics") else {}
        ),
    )


__all__ = [
    "LaneRefill",
    "ResidentActionBypass",
    "ResidentJob",
    "ResidentLaneQueue",
    "ResidentRunResult",
    "apply_turn_limit_draws",
    "compact_semantic_prefixes",
    "merge_resident_actions",
    "run_resident_greedy_jobs",
    "select_semantic_rows",
]
