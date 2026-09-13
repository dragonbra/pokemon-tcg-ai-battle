"""Public-information-only routing rules for the Policy-0814 deck router V3."""

from __future__ import annotations

from typing import Any

import torch


PUBLIC_POLICY_ID = "Experimental-Public-DeckRouter-V3-Policy0814"
ROUTED_UPDATES = (5, 25, 125, 140, 145, 160, 165, 170)
DECK_TO_UPDATE = {
    "001": 165,
    "002": 170,
    "003": 25,
    "007": 125,
    "008": 5,
    "009": 140,
    "011": 160,
    "012": 25,
    "071": 145,
}

FAMILY_CARDS = {
    1: frozenset((646, 647, 648)),
    2: frozenset((741, 742, 743)),
    3: frozenset((848, 849)),
    7: frozenset((119, 120, 121)),
    9: frozenset((673, 674, 675, 676, 677, 678)),
    11: frozenset((344, 345, 756)),
    12: frozenset((1030, 1031)),
    71: frozenset((93, 149, 150, 346, 709, 710, 917, 918, 920)),
}
OGERPON = frozenset((96,))
TRIGGER_CARD_IDS = tuple(sorted(OGERPON | frozenset().union(*FAMILY_CARDS.values())))
_TRIGGER_INDEX = {card_id: index for index, card_id in enumerate(TRIGGER_CARD_IDS)}


def route_manifest(default_update: int) -> dict[str, Any]:
    """Return the immutable public-input routing contract for one default choice."""

    _validate_default_update(default_update)
    return {
        "schema_version": "0045_public_deck_router_v3_rules_v1",
        "policy_id": PUBLIC_POLICY_ID,
        "default_update": int(default_update),
        "routed_updates": list(ROUTED_UPDATES),
        "deck_to_update": dict(DECK_TO_UPDATE),
        "family_cards": {
            f"{deck_code:03d}": sorted(cards)
            for deck_code, cards in FAMILY_CARDS.items()
        },
        "ogerpon_provisional": {
            "cards": sorted(OGERPON),
            "predicted_deck": "008",
            "update": DECK_TO_UPDATE["008"],
            "override_on_071_family": DECK_TO_UPDATE["071"],
        },
        "conflict_behavior": f"fail_closed_to_default_u{default_update}",
        "public_inputs": [
            "semantic.card_cat.card_id",
            "semantic.card_cat.relative_owner",
            "semantic.card_cat.identity_knowledge",
            "semantic.card_mask",
        ],
        "forbidden_inputs": [
            "opponent_exact_deck_id",
            "hidden_opponent_cards",
            "identity_knowledge_candidate",
            "critic_outputs",
        ],
    }


def _validate_default_update(default_update: int) -> None:
    if int(default_update) not in ROUTED_UPDATES:
        raise ValueError(f"default update must be one of {ROUTED_UPDATES}")


class PublicDeckMemory:
    """Per-game monotonic public evidence with a configurable fail-closed route."""

    def __init__(
        self, job_count: int, device: torch.device, *, default_update: int
    ) -> None:
        if job_count < 1:
            raise ValueError("public deck memory requires at least one job")
        _validate_default_update(default_update)
        self.job_count = int(job_count)
        self.device = device
        self.default_update = int(default_update)
        self.seen = torch.zeros(
            (job_count, len(TRIGGER_CARD_IDS)), dtype=torch.bool, device=device
        )
        self.predicted_deck_codes = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.provisional = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.conflict = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.route_updates = torch.full(
            (job_count,), self.default_update, dtype=torch.long, device=device
        )
        self.observation_count = torch.zeros(job_count, dtype=torch.long, device=device)
        self.first_classification_observation = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.first_specialist_route_observation = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.route_change_count = torch.zeros(job_count, dtype=torch.long, device=device)
        self.route_decision_counts = torch.zeros(
            (job_count, len(ROUTED_UPDATES)), dtype=torch.long, device=device
        )
        self._trigger_ids = torch.tensor(
            TRIGGER_CARD_IDS, dtype=torch.long, device=device
        )
        self._family_indices = {
            code: torch.tensor(
                [_TRIGGER_INDEX[card_id] for card_id in sorted(cards)],
                dtype=torch.long,
                device=device,
            )
            for code, cards in FAMILY_CARDS.items()
        }
        self._ogerpon_indices = torch.tensor(
            [_TRIGGER_INDEX[card_id] for card_id in sorted(OGERPON)],
            dtype=torch.long,
            device=device,
        )

    def _any(self, rows: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.seen.index_select(0, rows).index_select(1, indices).any(dim=1)

    def observe(self, validated: Any, job_indices: torch.Tensor) -> torch.Tensor:
        job_indices = job_indices.long().to(self.device)
        if job_indices.ndim != 1 or job_indices.numel() != validated.card_cat.shape[0]:
            raise ValueError("public deck job indices do not align with semantic rows")
        if bool(job_indices.lt(0).any()) or bool(job_indices.ge(self.job_count).any()):
            raise ValueError("public deck job index outside allocated memory")
        if job_indices.unique().numel() != job_indices.numel():
            raise ValueError("one resident job appeared twice in a decision batch")

        card_ids = validated.card_cat[..., 0].long()
        owner = validated.card_cat[..., 2].long()
        knowledge = validated.card_cat[..., 8].long()
        certain_public = (
            validated.card_mask.bool()
            & owner.eq(2)
            & (knowledge.eq(1) | knowledge.eq(2))
        )
        observed = (
            card_ids.unsqueeze(-1).eq(self._trigger_ids.view(1, 1, -1))
            & certain_public.unsqueeze(-1)
        ).any(dim=1)
        self.seen.index_copy_(
            0, job_indices, self.seen.index_select(0, job_indices) | observed
        )

        counts = self.observation_count.index_select(0, job_indices) + 1
        self.observation_count.index_copy_(0, job_indices, counts)
        codes = tuple(FAMILY_CARDS)
        family_hits = torch.stack(
            [self._any(job_indices, self._family_indices[code]) for code in codes],
            dim=1,
        )
        family_count = family_hits.sum(dim=1)
        ogerpon = self._any(job_indices, self._ogerpon_indices)
        only_code = torch.full_like(family_count, -1)
        for index, code in enumerate(codes):
            only_code = torch.where(
                family_count.eq(1) & family_hits[:, index],
                torch.full_like(only_code, code),
                only_code,
            )
        new_conflict = family_count.gt(1) | (
            family_count.eq(1) & ogerpon & only_code.ne(71)
        )
        conflict = self.conflict.index_select(0, job_indices) | new_conflict
        self.conflict.index_copy_(0, job_indices, conflict)

        prediction = torch.where(
            family_count.eq(1),
            only_code,
            torch.where(ogerpon, torch.full_like(only_code, 8), only_code),
        )
        prediction = torch.where(conflict, torch.full_like(prediction, -2), prediction)
        previous_prediction = self.predicted_deck_codes.index_select(0, job_indices)
        self.predicted_deck_codes.index_copy_(0, job_indices, prediction)
        self.provisional.index_copy_(0, job_indices, prediction.eq(8) & ~conflict)

        first_classification = self.first_classification_observation.index_select(
            0, job_indices
        )
        newly_classified = previous_prediction.eq(-1) & prediction.ge(0)
        self.first_classification_observation.index_copy_(
            0,
            job_indices,
            torch.where(newly_classified, counts, first_classification),
        )

        routes = torch.full_like(prediction, self.default_update)
        for deck_id, update in DECK_TO_UPDATE.items():
            routes = torch.where(
                prediction.eq(int(deck_id)), torch.full_like(routes, update), routes
            )
        previous_routes = self.route_updates.index_select(0, job_indices)
        changed = routes.ne(previous_routes)
        self.route_updates.index_copy_(0, job_indices, routes)
        self.route_change_count.index_copy_(
            0,
            job_indices,
            self.route_change_count.index_select(0, job_indices) + changed.long(),
        )
        first_specialist = self.first_specialist_route_observation.index_select(
            0, job_indices
        )
        newly_specialist = (
            previous_routes.eq(self.default_update) & routes.ne(self.default_update)
        )
        self.first_specialist_route_observation.index_copy_(
            0,
            job_indices,
            torch.where(newly_specialist & first_specialist.lt(0), counts, first_specialist),
        )
        decision_counts = self.route_decision_counts.index_select(0, job_indices)
        for index, update in enumerate(ROUTED_UPDATES):
            decision_counts[:, index] += routes.eq(update).long()
        self.route_decision_counts.index_copy_(0, job_indices, decision_counts)
        return routes

    def routes(self, job_indices: torch.Tensor) -> torch.Tensor:
        return self.route_updates.index_select(0, job_indices.long().to(self.device))

    def telemetry(self) -> dict[str, Any]:
        seen = self.seen.detach().cpu()
        return {
            "default_update": self.default_update,
            "route_updates": self.route_updates.detach().cpu().tolist(),
            "predicted_deck_codes": self.predicted_deck_codes.detach().cpu().tolist(),
            "provisional": self.provisional.detach().cpu().tolist(),
            "conflict": self.conflict.detach().cpu().tolist(),
            "observation_count": self.observation_count.detach().cpu().tolist(),
            "first_classification_observation": (
                self.first_classification_observation.detach().cpu().tolist()
            ),
            "first_specialist_route_observation": (
                self.first_specialist_route_observation.detach().cpu().tolist()
            ),
            "route_change_count": self.route_change_count.detach().cpu().tolist(),
            "route_decision_counts": [
                {
                    str(update): int(row[index])
                    for index, update in enumerate(ROUTED_UPDATES)
                }
                for row in self.route_decision_counts.detach().cpu().tolist()
            ],
            "seen_trigger_card_ids": [
                [TRIGGER_CARD_IDS[index] for index in row.nonzero().flatten().tolist()]
                for row in seen
            ],
        }


__all__ = [
    "DECK_TO_UPDATE",
    "PUBLIC_POLICY_ID",
    "PublicDeckMemory",
    "ROUTED_UPDATES",
    "TRIGGER_CARD_IDS",
    "route_manifest",
]
