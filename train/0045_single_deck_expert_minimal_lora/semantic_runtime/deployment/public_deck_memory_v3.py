"""Self-contained public-information Deck Router V3 memory for deployment."""

from __future__ import annotations

from typing import Any

import torch


PUBLIC_POLICY_ID = "Experimental-Public-DeckRouter-V3-Policy0814"
ROUTED_UPDATES = (5, 25, 125, 140, 145, 160, 165, 170)
DECK_TO_UPDATE = {
    "001": 165, "002": 170, "003": 25, "007": 125,
    "008": 5, "009": 140, "011": 160, "071": 145,
    "012": 25,
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


def route_manifest(default_update: int) -> dict[str, Any]:
    if int(default_update) not in ROUTED_UPDATES:
        raise ValueError(f"default update must be one of {ROUTED_UPDATES}")
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


class PublicDeckMemory:
    """Single/multi-battle deployment memory with monotonic public evidence."""

    def __init__(
        self, job_count: int, device: torch.device, *, default_update: int
    ) -> None:
        if job_count < 1:
            raise ValueError("public deck memory requires at least one job")
        route_manifest(default_update)
        self.job_count = int(job_count)
        self.device = device
        self.default_update = int(default_update)
        self.seen = [set() for _ in range(job_count)]
        self.predicted_deck_codes = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.provisional = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.conflict = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.route_updates = torch.full(
            (job_count,), self.default_update, dtype=torch.long, device=device
        )
        self.observation_count = torch.zeros(job_count, dtype=torch.long, device=device)

    def observe(self, validated: Any, job_indices: torch.Tensor) -> torch.Tensor:
        rows = job_indices.long().to(self.device)
        if rows.ndim != 1 or rows.numel() != validated.card_cat.shape[0]:
            raise ValueError("public deck job indices do not align with semantic rows")
        if bool(rows.lt(0).any()) or bool(rows.ge(self.job_count).any()):
            raise ValueError("public deck job index outside allocated memory")
        if rows.unique().numel() != rows.numel():
            raise ValueError("one resident job appeared twice in a decision batch")
        for batch_row, job_tensor in enumerate(rows):
            job = int(job_tensor)
            self.observation_count[job] += 1
            cards = validated.card_cat[batch_row]
            public = (
                validated.card_mask[batch_row].bool()
                & cards[:, 2].eq(2)
                & (cards[:, 8].eq(1) | cards[:, 8].eq(2))
            )
            self.seen[job].update(
                int(card) for card in cards[public, 0].detach().cpu().tolist()
                if int(card) in TRIGGER_CARD_IDS
            )
            hits = [
                code for code, family in FAMILY_CARDS.items()
                if self.seen[job] & family
            ]
            ogerpon = bool(self.seen[job] & OGERPON)
            conflict = bool(self.conflict[job]) or len(hits) > 1 or (
                len(hits) == 1 and ogerpon and hits[0] != 71
            )
            self.conflict[job] = conflict
            if conflict:
                prediction = -2
            elif len(hits) == 1:
                prediction = hits[0]
            elif ogerpon:
                prediction = 8
            else:
                prediction = -1
            self.predicted_deck_codes[job] = prediction
            self.provisional[job] = prediction == 8
            route = {
                int(deck_id): update for deck_id, update in DECK_TO_UPDATE.items()
            }.get(prediction, self.default_update)
            self.route_updates[job] = route
        return self.route_updates.index_select(0, rows)


__all__ = [
    "DECK_TO_UPDATE",
    "PUBLIC_POLICY_ID",
    "PublicDeckMemory",
    "ROUTED_UPDATES",
    "TRIGGER_CARD_IDS",
    "route_manifest",
]
