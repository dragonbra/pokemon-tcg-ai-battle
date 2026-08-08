"""Device-resident DecisionGate and Phantom Dive protocol adapter for 0038."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..action_boundary.dragapult import (
    PHANTOM_DIVE_ATTACK_ID,
    StableTargetIdentity,
)
from ..action_boundary.macro_planner import MacroPlanner


ROOT = Path(__file__).resolve().parents[3]
PROTOTYPES = (
    ROOT / "train/0038_action_boundary_rl/semantic_policy/assets/"
    "official_full_engine_prototypes_v2.json"
)


@dataclass(slots=True)
class _Pending:
    actor: int
    root_index: int
    targets: list[StableTargetIdentity]
    macro_action: dict[str, Any]


def _card_prizes() -> dict[int, int]:
    payload = json.loads(PROTOTYPES.read_text(encoding="utf-8"))
    result: dict[int, int] = {}
    for row in payload["cards"]:
        name = str(row.get("name_en") or "").lower()
        if "mega " in name and " ex" in name:
            prizes = 3
        elif " ex" in name:
            prizes = 2
        else:
            prizes = 1
        if row.get("no_prize"):
            prizes = 0
        result[int(row["card_id"])] = prizes
    return result


class CudaActionBoundaryAdapter:
    """Bypass forced callbacks and cache one hierarchical Phantom allocation.

    The adapter consumes the already-produced Semantic0031 observation.  It never
    changes the primitive action ABI: the resident runner still submits one packed
    official selection for every ready callback.
    """

    def __init__(
        self,
        model: Any,
        jobs: Sequence[Any],
        *,
        greedy: bool,
        max_select: int,
    ) -> None:
        self.model = model
        self.jobs = tuple(jobs)
        self.greedy = bool(greedy)
        self.max_select = int(max_select)
        self.planner = MacroPlanner(model.allocation_head)
        self.pending: dict[int, _Pending] = {}
        self.invalid_jobs: dict[int, str] = {}
        self.per_job: dict[int, dict[str, int]] = {
            index: {
                "forced_shortcuts": 0,
                "macro_actions": 0,
                "macro_callbacks": 0,
                "strategic_decisions": 0,
            }
            for index in range(len(jobs))
        }
        self.generators: dict[int, torch.Generator] = {}
        for index, job in enumerate(jobs):
            generator = torch.Generator(device=model.device)
            generator.manual_seed(int(job.policy_seed))
            self.generators[index] = generator
        self.prizes = _card_prizes()

    def _invalidate(self, job: int, reason: str) -> None:
        self.pending.pop(job, None)
        self.invalid_jobs.setdefault(job, reason)

    @staticmethod
    def _option_target_serials(semantic: dict[str, torch.Tensor]) -> torch.Tensor:
        relation = semantic["option_target"].long()
        safe = (relation - 1).clamp(min=0, max=semantic["card_cat"].shape[1] - 1)
        serial = semantic["card_cat"][..., 1].long() - 1
        gathered = serial.gather(1, safe)
        return torch.where(relation.gt(0), gathered, torch.full_like(gathered, -1))

    def pre_route(
        self,
        *,
        semantic: dict[str, torch.Tensor],
        ready: torch.Tensor,
        focal_route: torch.Tensor,
        opponent_route: torch.Tensor,
        lane_job: torch.Tensor,
        turns: torch.Tensor,
        selections: torch.Tensor,
        max_select: int,
    ) -> Any:
        from ptcg_cuda_engine.semantic0031_resident import ResidentActionBypass

        lane_count = ready.numel()
        device = ready.device
        actions = torch.zeros((lane_count, max_select), dtype=torch.long, device=device)
        lengths = torch.zeros(lane_count, dtype=torch.long, device=device)
        forced = torch.zeros(lane_count, dtype=torch.bool, device=device)
        macro = torch.zeros_like(forced)

        option_count = semantic["option_mask"].long().sum(dim=1)
        minimum = semantic["min_count"].long()
        maximum = semantic["max_count"].long()
        sole_type = semantic["option_cat"][:, 0, 0].long() - 1
        # No/decline (2) and End/pass (14) remain policy choices.  This is
        # deliberately conservative: equivalent multi-option aliases are not
        # shortcut until a device canonicalizer can prove their equivalence.
        forced |= (
            ready & option_count.eq(1) & minimum.eq(1) & maximum.eq(1)
            & sole_type.ne(2) & sole_type.ne(14)
        )
        empty_pass = ready & option_count.eq(0) & minimum.eq(0) & maximum.eq(0)
        forced |= empty_pass
        actions[forced & ~empty_pass, 0] = 0
        lengths[forced & ~empty_pass] = 1

        ready_lanes_tensor = ready.nonzero(as_tuple=False).flatten()
        ready_lanes = ready_lanes_tensor.tolist()
        ready_jobs = lane_job.index_select(0, ready_lanes_tensor).tolist()
        ready_forced = forced.index_select(0, ready_lanes_tensor).tolist()
        pending_records: list[tuple[int, int, _Pending]] = []
        for lane, job, is_forced in zip(
            ready_lanes, ready_jobs, ready_forced, strict=True
        ):
            job = int(job)
            pending = self.pending.get(job)
            if pending is None:
                if is_forced:
                    self.per_job[job]["forced_shortcuts"] += 1
                continue
            pending_records.append((lane, job, pending))

        if pending_records:
            pending_lanes = torch.tensor(
                [item[0] for item in pending_records], dtype=torch.long, device=device
            )
            pending_focal = focal_route.index_select(0, pending_lanes).tolist()
            pending_context = (semantic["global_cat"].index_select(0, pending_lanes)[:, 1] - 1).tolist()
            pending_remaining = semantic["global_num"].index_select(0, pending_lanes)[:, 13].round().long().tolist()
            pending_masks = semantic["option_mask"].index_select(0, pending_lanes).cpu()
            pending_cats = semantic["option_cat"].index_select(0, pending_lanes).cpu()

        for position, (lane, job, pending) in enumerate(pending_records):
            # Pending macro actions own these callbacks even when the current
            # option set happens to contain one target.
            forced[lane] = False
            if not pending_focal[position]:
                self._invalidate(job, "actor_or_priority_drift")
                continue
            context = int(pending_context[position])
            remaining = int(pending_remaining[position])
            expected_remaining = len(pending.targets)
            if context != 14 or remaining != expected_remaining:
                self._invalidate(job, "context_or_remaining_counter_drift")
                continue
            expected = pending.targets[0]
            option_cat = pending_cats[position]
            candidates = (
                pending_masks[position]
                & option_cat[:, 5].eq(expected.card_id)
                & option_cat[:, 13].eq(expected.initial_bench_slot + 1)
            ).nonzero(as_tuple=False).flatten()
            if candidates.numel() != 1:
                cats = option_cat[pending_masks[position]].tolist()
                self._invalidate(
                    job,
                    f"stable_target_identity_drift:expected={expected.serial}:cat={cats}",
                )
                continue
            actions[lane, 0] = int(candidates[0])
            lengths[lane] = 1
            macro[lane] = True
            pending.targets.pop(0)
            self.per_job[job]["macro_callbacks"] += 1
            if not pending.targets:
                self.pending.pop(job, None)

        return ResidentActionBypass(actions, lengths, forced | macro, forced, macro)

    def _visible_target(
        self,
        card_cat: torch.Tensor,
        card_num: torch.Tensor,
        row: int,
    ) -> dict[str, Any]:
        card_id = int(card_cat[row, 0])
        current_hp = float(card_num[row, 0])
        maximum_hp = float(card_num[row, 1])
        return {
            "serial": int(card_cat[row, 1]) - 1,
            "id": card_id,
            "hp": current_hp,
            "maxHp": maximum_hp,
            "prize": self.prizes.get(card_id, 1),
            "benchSlot": int(card_cat[row, 4]) - 1,
            "energyCards": [0] * max(0, int(round(float(card_num[row, 2])))),
            "statusBits": max(0, int(card_cat[row, 6]) - 1),
            "preEvolution": [0] * max(0, int(round(float(card_num[row, 5])))),
        }

    def post_route(
        self,
        *,
        semantic: dict[str, torch.Tensor],
        routed: Any,
        focal_route: torch.Tensor,
        opponent_route: torch.Tensor,
        lane_indices: torch.Tensor,
        lane_job: torch.Tensor,
        turns: torch.Tensor,
    ) -> list[dict[str, Any] | None]:
        metadata: list[dict[str, Any] | None] = [None] * int(lane_indices.numel())
        focal_rows_tensor = focal_route.nonzero(as_tuple=False).flatten()
        focal_jobs = lane_job.index_select(0, focal_rows_tensor).tolist()
        for job in focal_jobs:
            job = int(job)
            self.per_job[job]["strategic_decisions"] += 1
        action_width = routed.actions.shape[1]
        safe_actions = routed.actions.clamp(min=0, max=semantic["option_cat"].shape[1] - 1)
        selected_attack = semantic["option_cat"][..., 7].gather(1, safe_actions)
        selected_valid = torch.arange(
            action_width, device=focal_route.device
        ).unsqueeze(0).lt(routed.lengths.unsqueeze(1))
        phantom_count = (selected_attack.eq(PHANTOM_DIVE_ATTACK_ID) & selected_valid).sum(dim=1)
        phantom_rows = (focal_route & phantom_count.gt(0)).nonzero(as_tuple=False).flatten().tolist()
        for row in phantom_rows:
            job = int(lane_job[row])
            count = int(phantom_count[row])
            if count != 1:
                self._invalidate(job, "multiple_phantom_roots")
                continue
            length = int(routed.lengths[row])
            chosen = routed.actions[row, :length]
            phantom = [
                int(index) for index in chosen.tolist()
                if int(semantic["option_cat"][row, int(index), 7]) == PHANTOM_DIVE_ATTACK_ID
            ]
            # Confusion creates a chance boundary before allocation.  Preserve
            # the legacy callbacks and exclude this episode from macro PPO.
            own_status = max(0, int(semantic["global_cat"][row, 7]) - 1)
            if own_status & (1 << 2):
                self._invalidate(job, "chance_boundary_before_allocation")
                continue
            target_mask = (
                routed.validated.card_mask[row]
                & routed.validated.card_cat[row, :, 2].eq(2)
                & routed.validated.card_cat[row, :, 3].eq(6)
            )
            target_rows = target_mask.nonzero(as_tuple=False).flatten()
            if target_rows.numel() == 0:
                # Phantom Dive remains a normal attack when no opposing Bench
                # exists; the official engine emits no allocation callbacks.
                continue
            if target_rows.numel() > 5:
                serials = [
                    int(routed.validated.card_cat[row, index, 1]) - 1
                    for index in target_rows.tolist()
                ]
                self._invalidate(job, f"phantom_target_count_outside_v1:serials={serials}")
                continue
            ordered = sorted(
                target_rows.tolist(),
                key=lambda index: (
                    int(routed.validated.card_cat[row, index, 1]),
                    int(routed.validated.card_cat[row, index, 0]),
                    int(routed.validated.card_cat[row, index, 4]),
                ),
            )
            focal_player = int(self.jobs[job].focal_player)
            identities = tuple(
                StableTargetIdentity(
                    1 - focal_player,
                    int(routed.validated.card_cat[row, index, 1]) - 1,
                    int(routed.validated.card_cat[row, index, 0]),
                    int(routed.validated.card_cat[row, index, 4]) - 1,
                )
                for index in ordered
            )
            visible = [
                self._visible_target(
                    routed.validated.card_cat[row],
                    routed.validated.card_num[row],
                    index,
                )
                for index in ordered
            ]
            root = phantom[0]
            planned = self.planner.plan(
                state_summary=routed.state.summary[row],
                root_option=routed.focal_options[row, root],
                target_embeddings=routed.state.cards[row, ordered],
                target_identities=identities,
                visible_targets=visible,
                greedy=self.greedy,
                generator=self.generators[job],
            )
            macro_action = {
                "family": "phantom_dive",
                "allocation_index": planned.allocation_index,
                "root_index": root,
                "targets": [target.serial for target in planned.allocation.target_ids],
                "counters": list(planned.allocation.counters),
                # Public, pre-action inputs used by the conditional head.  PPO
                # must replay the exact behavior distribution rather than
                # reconstructing card attributes with lossy defaults.
                "visible_targets": visible,
            }
            self.pending[job] = _Pending(
                actor=focal_player,
                root_index=root,
                targets=[
                    target
                    for target, count in zip(
                        planned.allocation.target_ids,
                        planned.allocation.counters,
                        strict=True,
                    )
                    for _ in range(count)
                ],
                macro_action=macro_action,
            )
            self.per_job[job]["macro_actions"] += 1
            metadata[row] = {
                "parameter_logprob": planned.allocation_logprob,
                "parameter_entropy": planned.allocation_entropy,
                "macro_action": macro_action,
            }
        return metadata

    def on_refill(self, *, lane_indices: torch.Tensor, job_indices: torch.Tensor) -> None:
        for job in job_indices.tolist():
            self.pending.pop(int(job), None)

    def on_terminal(self, *, lane_indices: torch.Tensor, job_indices: torch.Tensor) -> None:
        for job in job_indices.tolist():
            job = int(job)
            if job in self.pending:
                self._invalidate(job, "terminal_inside_macro_transaction")

    def metrics(self) -> dict[str, int]:
        totals = {
            "forced_shortcuts": 0,
            "macro_actions": 0,
            "macro_callbacks": 0,
            "strategic_decisions": 0,
            "invalid_macros": len(self.invalid_jobs),
        }
        for row in self.per_job.values():
            for name in totals:
                if name != "invalid_macros":
                    totals[name] += row.get(name, 0)
        return totals


__all__ = ["CudaActionBoundaryAdapter"]
