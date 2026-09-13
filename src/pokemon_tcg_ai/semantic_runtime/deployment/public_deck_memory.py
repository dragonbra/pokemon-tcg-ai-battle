"""Public-information-only Deck Router V2 memory for packaged inference."""

from __future__ import annotations

from typing import Any

import torch


PUBLIC_POLICY_ID = "Experimental-Public-DeckRouter-V2-Policy0814"
DEFAULT_UPDATE = 70
ROUTED_UPDATES = (5, 15, 25, 70, 95)
DECK_TO_UPDATE = {
    "001": 5, "002": 15, "003": 25, "007": 95,
    "008": 5, "009": 70, "011": 70, "071": 70,
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

RULE_MANIFEST = {
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
        "cards": sorted(OGERPON), "predicted_deck": "008", "update": 5,
        "override_on_071_family": 70,
    },
    "conflict_behavior": "fail_closed_to_default_u70",
    "public_inputs": [
        "semantic.card_cat.card_id", "semantic.card_cat.relative_owner",
        "semantic.card_cat.identity_knowledge", "semantic.card_mask",
    ],
    "forbidden_inputs": [
        "opponent_exact_deck_id", "hidden_opponent_cards",
        "identity_knowledge_candidate", "critic_outputs",
    ],
}


class PublicDeckMemory:
    """Per-game public card memory with provisional and fail-closed routing."""

    def __init__(self, job_count: int, device: torch.device) -> None:
        if job_count < 1:
            raise ValueError("public deck memory requires at least one job")
        self.job_count = int(job_count)
        self.device = device
        self.seen = [set() for _ in range(job_count)]
        self.predicted_deck_codes = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.provisional = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.conflict = torch.zeros(job_count, dtype=torch.bool, device=device)
        self.route_updates = torch.full(
            (job_count,), DEFAULT_UPDATE, dtype=torch.long, device=device
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
            mask = validated.card_mask[batch_row].bool()
            cards = validated.card_cat[batch_row]
            public = mask & cards[:, 2].eq(2) & (
                cards[:, 8].eq(1) | cards[:, 8].eq(2)
            )
            self.seen[job].update(
                int(card) for card in cards[public, 0].detach().cpu().tolist()
                if int(card) in TRIGGER_CARD_IDS
            )
            hits = [code for code, family in FAMILY_CARDS.items()
                    if self.seen[job] & family]
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
            route = {1: 5, 2: 15, 3: 25, 7: 95, 8: 5}.get(
                prediction, DEFAULT_UPDATE
            )
            self.route_updates[job] = route
        return self.route_updates.index_select(0, rows)


__all__ = [
    "DEFAULT_UPDATE", "DECK_TO_UPDATE", "PUBLIC_POLICY_ID",
    "PublicDeckMemory", "ROUTED_UPDATES", "RULE_MANIFEST",
    "TRIGGER_CARD_IDS",
]
