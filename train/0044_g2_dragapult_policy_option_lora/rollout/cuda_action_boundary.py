"""Device-resident DecisionGate and Phantom Dive protocol adapter for 0038."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ..semantic_runtime.action_boundary.dragapult import (
    PHANTOM_DIVE_ATTACK_ID,
    PHANTOM_DIVE_MAX_TARGETS,
    StableTargetIdentity,
)
from ..semantic_runtime.action_boundary.macro_planner import MacroPlanner
from ..semantic_runtime.action_boundary.macro_protocol import MacroProtocolError
from ..semantic_runtime.action_boundary.public_card_features import card_prize_counts
from .deck_routing import audit_exact_deck_rows, exact_deck_sha256


ROOT = Path(__file__).resolve().parents[3]
PROTOTYPES = (
    ROOT / "train/0044_g2_dragapult_policy_option_lora/semantic_runtime/assets/"
    "official_full_engine_prototypes_v2.json"
)


def _first_player_harness_mask(
    semantic: dict[str, torch.Tensor], ready: torch.Tensor
) -> torch.Tensor:
    option_count = semantic["option_mask"].long().sum(dim=1)
    minimum = semantic["min_count"].long()
    maximum = semantic["max_count"].long()
    context = semantic["global_cat"][:, 1].long() - 1
    mask = ready & context.eq(41)
    invalid = mask & ~(option_count.eq(2) & minimum.eq(1) & maximum.eq(1))
    if bool(invalid.any()):
        rows = invalid.nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"first-player harness contract drift on resident lanes {rows}"
        )
    return mask


@dataclass(slots=True)
class _Pending:
    actor: int
    root_index: int
    targets: list[StableTargetIdentity]
    macro_action: dict[str, Any]


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
        agent_selects_first_player: bool = False,
    ) -> None:
        self.model = model
        self.jobs = tuple(jobs)
        self.greedy = bool(greedy)
        self.max_select = int(max_select)
        self.agent_selects_first_player = bool(agent_selects_first_player)
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
        self.first_player_choices: dict[int, dict[str, int | bool]] = {}
        self.audited_deck_roles: set[tuple[int, str]] = set()
        for index, job in enumerate(jobs):
            generator = torch.Generator(device=model.device)
            generator.manual_seed(int(job.policy_seed))
            self.generators[index] = generator
        self.prizes = card_prize_counts(PROTOTYPES)

    def _invalidate(self, job: int, reason: str) -> None:
        self.pending.pop(job, None)
        self.invalid_jobs.setdefault(job, reason)

    @staticmethod
    def _option_source_serials(semantic: dict[str, torch.Tensor]) -> torch.Tensor:
        """Resolve official type-3 in-play choices through option_source.

        Phantom Dive target callbacks are represented by the official ABI as
        ``{type: 3, area: Bench, index: ..., playerIndex: ...}``.  The CPU
        feature compiler therefore places the chosen in-play entity in
        ``option_source``; ``option_target`` is intentionally empty because no
        ``inPlayArea``/``inPlayIndex`` fields exist on this callback.
        """
        relation = semantic["option_source"].long()
        safe = (relation - 1).clamp(min=0, max=semantic["card_cat"].shape[1] - 1)
        serial = semantic["card_cat"][..., 1].long() - 1
        gathered = serial.gather(1, safe)
        return torch.where(relation.gt(0), gathered, torch.full_like(gathered, -1))

    @classmethod
    def _matching_target_options(
        cls,
        semantic: dict[str, torch.Tensor],
        *,
        row: int,
        expected: StableTargetIdentity,
    ) -> torch.Tensor:
        option_cat = semantic["option_cat"][row]
        target_serials = cls._option_source_serials(semantic)[row]
        return (
            semantic["option_mask"][row]
            & option_cat[:, 5].eq(expected.card_id)
            & target_serials.eq(expected.serial)
        ).nonzero(as_tuple=False).flatten()

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

        self.audited_deck_roles.update(audit_exact_deck_rows(
            semantic=semantic,
            ready=ready,
            focal_route=focal_route,
            lane_job=lane_job,
            jobs=self.jobs,
            already_audited=self.audited_deck_roles,
        ))

        option_count = semantic["option_mask"].long().sum(dim=1)
        minimum = semantic["min_count"].long()
        maximum = semantic["max_count"].long()
        # Resident jobs reorder decks so physical player 0 is always the
        # requested first player.  Match the official CPU harness by submitting
        # Yes/index 0 directly; seat assignment is not a learned decision.
        first_player_harness = _first_player_harness_mask(semantic, ready)
        sole_type = semantic["option_cat"][:, 0, 0].long() - 1
        # No/decline (2) and End/pass (14) remain policy choices.  This is
        # deliberately conservative: equivalent multi-option aliases are not
        # shortcut until a device canonicalizer can prove their equivalence.
        forced |= (
            ready & option_count.eq(1) & minimum.eq(1) & maximum.eq(1)
            & sole_type.ne(2) & sole_type.ne(14)
        )
        if not self.agent_selects_first_player:
            forced |= first_player_harness
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
                reason = "actor_or_priority_drift"
                self._invalidate(job, reason)
                raise MacroProtocolError(f"job {job}: {reason}")
            context = int(pending_context[position])
            remaining = int(pending_remaining[position])
            expected_remaining = len(pending.targets)
            if context != 14 or remaining != expected_remaining:
                reason = "context_or_remaining_counter_drift"
                self._invalidate(job, reason)
                raise MacroProtocolError(f"job {job}: {reason}")
            expected = pending.targets[0]
            option_cat = pending_cats[position]
            candidates = self._matching_target_options(
                semantic, row=lane, expected=expected
            )
            if candidates.numel() != 1:
                cats = option_cat[pending_masks[position]].tolist()
                target_serials = self._option_source_serials(semantic)[
                    lane, pending_masks[position].to(device)
                ].tolist()
                reason = (
                    "stable_target_identity_drift:"
                    f"expected_serial={expected.serial}:"
                    f"expected_card={expected.card_id}:"
                    f"option_source_serials={target_serials}:cat={cats}"
                )
                self._invalidate(job, reason)
                raise MacroProtocolError(f"job {job}: {reason}")
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
        if card_id not in self.prizes:
            raise ValueError(
                f"visible CUDA target is absent from public Prize map: {card_id}"
            )
        current_hp = float(card_num[row, 0])
        maximum_hp = float(card_num[row, 1])
        return {
            "serial": int(card_cat[row, 1]) - 1,
            "id": card_id,
            "hp": current_hp,
            "maxHp": maximum_hp,
            "prize": self.prizes[card_id],
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
        contexts = semantic["global_cat"][:, 1].long() - 1
        first_rows = contexts.eq(41).nonzero(as_tuple=False).flatten().tolist()
        for row in first_rows:
            job = int(lane_job[row])
            length = int(routed.lengths[row])
            if length != 1:
                raise RuntimeError(
                    f"job {job}: first-player policy returned {length} selections"
                )
            action_index = int(routed.actions[row, 0])
            if action_index not in (0, 1):
                raise RuntimeError(
                    f"job {job}: invalid first-player option index {action_index}"
                )
            resident_focal = int(self.jobs[job].focal_player)
            chooser = resident_focal if bool(focal_route[row]) else 1 - resident_focal
            actual_first = chooser if action_index == 0 else 1 - chooser
            self.first_player_choices[job] = {
                "chooser": chooser,
                "chooser_is_focal": chooser == resident_focal,
                "action_index": action_index,
                "actual_first": actual_first,
                "focal_first": actual_first == resident_focal,
            }
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
            if target_rows.numel() > PHANTOM_DIVE_MAX_TARGETS:
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

    def routing_audit_manifest(self) -> dict[str, Any]:
        expected = {
            (job, role)
            for job in range(len(self.jobs))
            for role in ("focal", "opponent")
        }
        missing = sorted(expected - self.audited_deck_roles)
        if missing:
            raise RuntimeError(
                "FATAL: incomplete per-lane exact-deck routing audit: "
                f"missing={missing[:16]} count={len(missing)}"
            )
        rows = [
            {
                "job_index": index,
                "focal_player": int(job.focal_player),
                "focal_exact_deck_sha256": exact_deck_sha256(
                    job.decks[int(job.focal_player)]
                ),
                "opponent_exact_deck_sha256": exact_deck_sha256(
                    job.decks[1 - int(job.focal_player)]
                ),
                "roles_audited": ["focal", "opponent"],
                "status": "PASS",
            }
            for index, job in enumerate(self.jobs)
        ]
        import hashlib
        import json

        digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        return {
            "schema": "0044_policy0809_exact_deck_lane_routing_v1",
            "status": "PASS",
            "same_policy": False,
            "jobs": len(rows),
            "roles_audited": len(self.audited_deck_roles),
            "unique_opponent_exact_decks": len({
                row["opponent_exact_deck_sha256"] for row in rows
            }),
            "routing_audit_sha256": digest,
            "rows": rows,
        }


__all__ = ["CudaActionBoundaryAdapter", "_first_player_harness_mask"]
