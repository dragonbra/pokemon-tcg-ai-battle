"""Public-information-only routing rules for the Policy-0814 deck router V2."""

from __future__ import annotations

from typing import Any

import torch


PUBLIC_POLICY_ID = "Experimental-Public-DeckRouter-V2-Policy0814"
DEFAULT_UPDATE = 70
ROUTED_UPDATES = (5, 15, 25, 70, 95)
DECK_TO_UPDATE = {
    "001": 5,
    "002": 15,
    "003": 25,
    "007": 95,
    "008": 5,
    "009": 70,
    "011": 70,
    "071": 70,
}

FAMILY_CARDS = {
    1: frozenset((646, 647, 648)),
    2: frozenset((741, 742, 743)),
    3: frozenset((848, 849)),
    7: frozenset((119, 120, 121)),
    9: frozenset((673, 674, 675, 676, 677, 678)),
    11: frozenset((344, 345, 756)),
    71: frozenset((93, 149, 150, 346, 709, 710, 917, 918, 920)),
}
OGERPON = frozenset((96,))
TRIGGER_CARD_IDS = tuple(sorted(OGERPON | frozenset().union(*FAMILY_CARDS.values())))
_TRIGGER_INDEX = {card_id: index for index, card_id in enumerate(TRIGGER_CARD_IDS)}

ROUTE_MANIFEST = {
    "schema_version": "0045_public_deck_router_v2_rules_v1",
    "policy_id": PUBLIC_POLICY_ID,
    "default_update": DEFAULT_UPDATE,
    "routed_updates": list(ROUTED_UPDATES),
    "deck_to_update": DECK_TO_UPDATE,
    "family_cards": {
        f"{deck_code:03d}": sorted(cards)
        for deck_code, cards in FAMILY_CARDS.items()
    },
    "ogerpon_provisional": {
        "cards": sorted(OGERPON),
        "predicted_deck": "008",
        "update": 5,
        "override_on_071_family": 70,
    },
    "conflict_behavior": "fail_closed_to_default_u70",
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


class PublicDeckMemory:
    """Per-game monotonic evidence with provisional and fail-closed routes."""

    def __init__(self, job_count: int, device: torch.device) -> None:
        if job_count < 1:
            raise ValueError("public deck memory requires at least one job")
        self.job_count = int(job_count)
        self.device = device
        self.seen = torch.zeros(
            (job_count, len(TRIGGER_CARD_IDS)), dtype=torch.bool, device=device
        )
        self.predicted_deck_codes = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.provisional = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.conflict = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.route_updates = torch.full(
            (job_count,), DEFAULT_UPDATE, dtype=torch.long, device=device
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
            family_count.eq(1), only_code,
            torch.where(ogerpon, torch.full_like(only_code, 8), only_code),
        )
        prediction = torch.where(conflict, torch.full_like(prediction, -2), prediction)
        previous_prediction = self.predicted_deck_codes.index_select(0, job_indices)
        self.predicted_deck_codes.index_copy_(0, job_indices, prediction)
        provisional = prediction.eq(8) & ~conflict
        self.provisional.index_copy_(0, job_indices, provisional)

        first_classification = self.first_classification_observation.index_select(
            0, job_indices
        )
        newly_classified = previous_prediction.eq(-1) & prediction.ge(0)
        self.first_classification_observation.index_copy_(
            0,
            job_indices,
            torch.where(newly_classified, counts, first_classification),
        )

        routes = torch.full_like(prediction, DEFAULT_UPDATE)
        for deck_code, update in ((1, 5), (2, 15), (3, 25), (7, 95), (8, 5)):
            routes = torch.where(
                prediction.eq(deck_code), torch.full_like(routes, update), routes
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
        newly_specialist = previous_routes.eq(DEFAULT_UPDATE) & routes.ne(DEFAULT_UPDATE)
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
    "DEFAULT_UPDATE",
    "DECK_TO_UPDATE",
    "PUBLIC_POLICY_ID",
    "PublicDeckMemory",
    "ROUTED_UPDATES",
    "ROUTE_MANIFEST",
    "TRIGGER_CARD_IDS",
]
