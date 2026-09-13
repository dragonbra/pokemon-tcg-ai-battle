"""Shared CPU/CUDA public-opponent Pokémon memory and routing rules."""

from __future__ import annotations

from typing import Any

import torch


PUBLIC_POLICY_ID = "Experimental-Public-MetaRouter-V3-GrassFoldU200"
DEFAULT_UPDATE = 282
ROUTED_UPDATES = (40, 90, 200, 282)

DRAGAPULT = frozenset((119, 120, 121))
LOPUNNY = frozenset((848, 849))
GRIMMSNARL = frozenset((646, 647, 648))
ALAKAZAM = frozenset((741, 742, 743))
FESTIVAL_CORE = frozenset((89, 90))
FESTIVAL_DIPPLIN = frozenset((92, 93))
# Public Pokemon that identify one of the folded 08/27/28 grass-family decks.
# Teal Mask Ogerpon is deliberately included: all of its ambiguous destinations
# use the same U200 expert, so routing need not wait for finer classification.
GRASS_BROTHERS = frozenset((
    96,                    # Teal Mask Ogerpon ex
    149, 150, 346,         # Hydrapple line
    402, 403, 404,         # Arboliva line
    650, 651, 652,         # Mega Venusaur line
    655,                   # Celebi
    708, 709, 710,         # Chikorita / Bayleef / Meganium
    756,                   # Mega Kangaskhan ex (meta 28)
    917, 918, 919,         # alternate Meganium line
    920,                   # Tapu Bulu
))
U200_EARLY_ROUTE = FESTIVAL_CORE | FESTIVAL_DIPPLIN | GRASS_BROTHERS
STARMIE = frozenset((1030, 1031))
DUSKNOIR = frozenset((131, 132, 133))

TRIGGER_CARD_IDS = tuple(sorted(
    DRAGAPULT | LOPUNNY | GRIMMSNARL | ALAKAZAM | U200_EARLY_ROUTE
    | STARMIE | DUSKNOIR
))
_TRIGGER_INDEX = {card_id: index for index, card_id in enumerate(TRIGGER_CARD_IDS)}


def _indices(cards: frozenset[int]) -> tuple[int, ...]:
    return tuple(_TRIGGER_INDEX[card_id] for card_id in sorted(cards))


RULE_MANIFEST = {
    "00_dragapult_family": {
        "any": sorted(DRAGAPULT), "meta_id": 0, "update": 282,
        "folded_original_meta_ids": [0, 15, 16],
    },
    "01_lopunny_family": {
        "any": sorted(LOPUNNY), "meta_id": 1, "update": 40,
        "routing_semantics": "all_public_lopunny_lines_fold_to_01",
    },
    "02_grimmsnarl": {"any": sorted(GRIMMSNARL), "meta_id": 2, "update": 40},
    "03_alakazam": {"any": sorted(ALAKAZAM), "meta_id": 3, "update": 90},
    "06_festival": {
        "any": sorted(FESTIVAL_CORE), "meta_id": 6, "update": 200,
        "early_route_also_accepts": sorted(FESTIVAL_DIPPLIN),
    },
    "08_27_28_grass_family": {
        "any": sorted(GRASS_BROTHERS), "meta_id": 8, "update": 200,
        "folded_original_meta_ids": [8, 27, 28],
    },
    "u200_early_route": {
        "any": sorted(U200_EARLY_ROUTE), "update": 200,
        "routing_semantics": "route_before_fine_meta_disambiguation",
    },
    "17_starmie_dusknoir": {
        "all_groups": [sorted(STARMIE), sorted(DUSKNOIR)],
        "meta_id": 17, "update": 40,
    },
    "12_starmie_provisional": {
        "any": sorted(STARMIE), "meta_id": 12, "update": 282, "lock": False,
    },
}


class PublicMetaMemory:
    """Small monotonic tensor memory, indexed by battle/job ID."""

    def __init__(self, job_count: int, device: torch.device) -> None:
        if job_count < 1:
            raise ValueError("public Meta memory requires at least one job")
        self.job_count = int(job_count)
        self.device = device
        self.seen = torch.zeros(
            (job_count, len(TRIGGER_CARD_IDS)), dtype=torch.bool, device=device
        )
        self.locked_meta = torch.full((job_count,), -1, dtype=torch.long, device=device)
        self.provisional_meta = torch.full((job_count,), -1, dtype=torch.long, device=device)
        self.route_updates = torch.full(
            (job_count,), DEFAULT_UPDATE, dtype=torch.long, device=device
        )
        self.observation_count = torch.zeros(job_count, dtype=torch.long, device=device)
        self.first_lock_observation = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self._trigger_ids = torch.tensor(
            TRIGGER_CARD_IDS, dtype=torch.long, device=device
        )
        self._rule_indices = {
            cards: torch.tensor(_indices(cards), dtype=torch.long, device=device)
            for cards in (
                DRAGAPULT, LOPUNNY, GRIMMSNARL, ALAKAZAM, FESTIVAL_CORE,
                GRASS_BROTHERS, U200_EARLY_ROUTE, STARMIE, DUSKNOIR,
            )
        }

    def _any(self, rows: torch.Tensor, cards: frozenset[int]) -> torch.Tensor:
        return self.seen.index_select(0, rows).index_select(
            1, self._rule_indices[cards]
        ).any(dim=1)

    def observe(self, validated: Any, job_indices: torch.Tensor) -> torch.Tensor:
        job_indices = job_indices.long().to(self.device)
        if job_indices.ndim != 1 or job_indices.numel() != validated.card_cat.shape[0]:
            raise ValueError("public Meta job indices do not align with semantic rows")
        if bool(job_indices.lt(0).any()) or bool(job_indices.ge(self.job_count).any()):
            raise ValueError("public Meta job index outside allocated memory")
        if job_indices.unique().numel() != job_indices.numel():
            raise ValueError("one resident job appeared twice in a decision batch")
        card_ids = validated.card_cat[..., 0].long()
        owner = validated.card_cat[..., 2].long()
        knowledge = validated.card_cat[..., 8].long()
        certain_public = (
            validated.card_mask.bool() & owner.eq(2)
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
        locked = self.locked_meta.index_select(0, job_indices)
        unresolved = locked.lt(0)
        candidate = torch.full_like(locked, -1)

        def assign(condition: torch.Tensor, meta_id: int) -> None:
            nonlocal candidate
            take = unresolved & candidate.lt(0) & condition
            candidate = torch.where(take, torch.full_like(candidate, meta_id), candidate)

        assign(self._any(job_indices, DRAGAPULT), 0)
        assign(self._any(job_indices, LOPUNNY), 1)
        assign(self._any(job_indices, GRIMMSNARL), 2)
        assign(self._any(job_indices, ALAKAZAM), 3)
        assign(self._any(job_indices, FESTIVAL_CORE), 6)
        assign(self._any(job_indices, GRASS_BROTHERS), 8)
        assign(
            self._any(job_indices, STARMIE) & self._any(job_indices, DUSKNOIR), 17
        )
        newly_locked = unresolved & candidate.ge(0)
        locked = torch.where(newly_locked, candidate, locked)
        self.locked_meta.index_copy_(0, job_indices, locked)
        self.first_lock_observation.index_copy_(
            0, job_indices,
            torch.where(
                newly_locked, counts,
                self.first_lock_observation.index_select(0, job_indices),
            ),
        )
        route = self.route_updates.index_select(0, job_indices)
        route = torch.where(
            route.eq(DEFAULT_UPDATE) & newly_locked
            & (candidate.eq(1) | candidate.eq(2) | candidate.eq(17)),
            torch.full_like(route, 40), route,
        )
        route = torch.where(
            route.eq(DEFAULT_UPDATE) & newly_locked & candidate.eq(3),
            torch.full_like(route, 90), route,
        )
        route = torch.where(
            route.eq(DEFAULT_UPDATE) & self._any(job_indices, U200_EARLY_ROUTE),
            torch.full_like(route, 200), route,
        )
        self.route_updates.index_copy_(0, job_indices, route)
        star = self._any(job_indices, STARMIE)
        self.provisional_meta.index_copy_(
            0, job_indices,
            torch.where(
                locked.ge(0), locked,
                torch.where(
                    star, torch.full_like(locked, 12), torch.full_like(locked, -1)
                ),
            ),
        )
        return route

    def routes(self, job_indices: torch.Tensor) -> torch.Tensor:
        return self.route_updates.index_select(0, job_indices.long().to(self.device))

    def telemetry(self) -> dict[str, Any]:
        seen = self.seen.detach().cpu()
        return {
            "route_updates": self.route_updates.detach().cpu().tolist(),
            "locked_meta": self.locked_meta.detach().cpu().tolist(),
            "provisional_meta": self.provisional_meta.detach().cpu().tolist(),
            "first_lock_observation": self.first_lock_observation.detach().cpu().tolist(),
            "observation_count": self.observation_count.detach().cpu().tolist(),
            "seen_trigger_card_ids": [
                [TRIGGER_CARD_IDS[index] for index in row.nonzero().flatten().tolist()]
                for row in seen
            ],
        }


__all__ = [
    "DEFAULT_UPDATE", "PUBLIC_POLICY_ID", "PublicMetaMemory", "ROUTED_UPDATES",
    "RULE_MANIFEST", "TRIGGER_CARD_IDS",
]
