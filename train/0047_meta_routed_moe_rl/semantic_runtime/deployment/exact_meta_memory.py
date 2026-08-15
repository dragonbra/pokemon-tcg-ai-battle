"""Deployable exact-29 Meta identification from public opponent Pokemon only."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTIFIER_VERSION = "0047_public_exact_deck_candidates_v1"
UNKNOWN_META_ID = -1


@dataclass(frozen=True, slots=True)
class ExactMetaCatalog:
    deck_ids: tuple[str, ...]
    meta_ids: tuple[int, ...]
    pokemon_by_deck: tuple[frozenset[int], ...]
    max_card_id: int

    @classmethod
    def load(cls, project_root: Path = PROJECT_ROOT) -> "ExactMetaCatalog":
        registry = json.loads(
            (project_root / "assets/decks/registry.json").read_text(encoding="utf-8")
        )
        mapping = json.loads(
            (project_root / "assets/taxonomy/deck_own_archetype_mapping_v2.json")
            .read_text(encoding="utf-8")
        )
        prototypes = json.loads(
            (project_root / "semantic_runtime/assets/official_full_engine_prototypes_v2.json")
            .read_text(encoding="utf-8")
        )
        pokemon = {
            int(row["card_id"])
            for row in prototypes["cards"]
            if int(row["card_type"]) == 0
        }
        meta_by_deck = {
            str(row["deck_id"]): int(row["archetype_id"])
            for row in mapping["decks"]
        }
        deck_ids: list[str] = []
        meta_ids: list[int] = []
        cards_by_deck: list[frozenset[int]] = []
        for row in registry["decks"]:
            deck_id = str(row["deck_id"])
            cards = {
                int(value)
                for value in (project_root / row["deck_path"])
                .read_text(encoding="utf-8").splitlines()
            }
            deck_ids.append(deck_id)
            meta_ids.append(meta_by_deck[deck_id])
            cards_by_deck.append(frozenset(cards & pokemon))
        if deck_ids != [f"{index:03d}" for index in range(1, 71)]:
            raise RuntimeError("0047 exact-Meta catalog must contain decks 001-070")
        return cls(
            tuple(deck_ids), tuple(meta_ids), tuple(cards_by_deck),
            max(int(row["card_id"]) for row in prototypes["cards"]),
        )


class ExactPublicMetaMemory:
    """Monotonic UNKNOWN→confirmed memory indexed by resident game/job."""

    def __init__(
        self, job_count: int, device: torch.device,
        catalog: ExactMetaCatalog | None = None,
    ) -> None:
        if job_count < 1:
            raise ValueError("exact public Meta memory requires at least one job")
        self.catalog = catalog or ExactMetaCatalog.load()
        self.job_count = int(job_count)
        self.device = device
        card_width = self.catalog.max_card_id + 1
        membership = torch.zeros(
            (len(self.catalog.deck_ids), card_width), dtype=torch.bool, device=device
        )
        for row, cards in enumerate(self.catalog.pokemon_by_deck):
            if cards:
                membership[row, torch.tensor(sorted(cards), device=device)] = True
        self.deck_membership = membership
        self.deck_meta = torch.tensor(self.catalog.meta_ids, dtype=torch.long, device=device)
        self.candidates = torch.ones(
            (job_count, len(self.catalog.deck_ids)), dtype=torch.bool, device=device
        )
        self.seen = torch.zeros((job_count, card_width), dtype=torch.bool, device=device)
        self.confirmed_meta = torch.full(
            (job_count,), UNKNOWN_META_ID, dtype=torch.long, device=device
        )
        self.decision_count = torch.zeros(job_count, dtype=torch.long, device=device)
        self.first_identification_decision = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )

    def observe(self, validated: Any, job_indices: torch.Tensor) -> torch.Tensor:
        jobs = job_indices.long().to(self.device)
        if jobs.ndim != 1 or jobs.numel() != validated.card_cat.shape[0]:
            raise ValueError("exact public Meta jobs do not align with semantic rows")
        if bool(jobs.lt(0).any()) or bool(jobs.ge(self.job_count).any()):
            raise ValueError("exact public Meta job index is outside memory")
        if jobs.unique().numel() != jobs.numel():
            raise ValueError("one resident job appeared twice at one decision boundary")

        card_ids = validated.card_cat[..., 0].long()
        owner = validated.card_cat[..., 2].long()
        knowledge = validated.card_cat[..., 8].long()
        public_opponent = (
            validated.card_mask.bool() & owner.eq(2)
            & (knowledge.eq(1) | knowledge.eq(2))
            & card_ids.gt(0) & card_ids.le(self.catalog.max_card_id)
        )
        observed_count = torch.zeros(
            (jobs.numel(), self.catalog.max_card_id + 1),
            dtype=torch.int16, device=self.device,
        )
        observed_count.scatter_add_(
            1, card_ids.clamp(0, self.catalog.max_card_id), public_opponent.to(torch.int16)
        )
        observed = observed_count.gt(0)
        previous_seen = self.seen.index_select(0, jobs)
        accumulated = previous_seen | observed
        self.seen.index_copy_(0, jobs, accumulated)

        counts = self.decision_count.index_select(0, jobs) + 1
        self.decision_count.index_copy_(0, jobs, counts)
        locked = self.confirmed_meta.index_select(0, jobs)
        unresolved = locked.eq(UNKNOWN_META_ID)
        # A candidate deck remains possible iff it contains every publicly seen
        # opponent Pokemon. No hidden zone, deck ID, evaluator label, or Critic
        # prediction enters this computation.
        absent = (~self.deck_membership).to(torch.float32)
        incompatible = accumulated.to(torch.float32).matmul(absent.t()).gt(0)
        candidates = self.candidates.index_select(0, jobs) & ~incompatible
        self.candidates.index_copy_(0, jobs, candidates)
        candidate_count = candidates.sum(dim=1)
        minimum = torch.where(
            candidates, self.deck_meta.unsqueeze(0),
            torch.full_like(candidates, 29, dtype=torch.long),
        ).min(dim=1).values
        maximum = torch.where(
            candidates, self.deck_meta.unsqueeze(0),
            torch.full_like(candidates, -1, dtype=torch.long),
        ).max(dim=1).values
        newly_confirmed = unresolved & candidate_count.gt(0) & minimum.eq(maximum)
        locked = torch.where(newly_confirmed, minimum, locked)
        self.confirmed_meta.index_copy_(0, jobs, locked)
        first = self.first_identification_decision.index_select(0, jobs)
        self.first_identification_decision.index_copy_(
            0, jobs, torch.where(newly_confirmed, counts, first)
        )
        return locked

    def telemetry(self) -> dict[str, Any]:
        confirmed = self.confirmed_meta.detach().cpu()
        identified = confirmed.ge(0)
        timings = self.first_identification_decision.detach().cpu()
        return {
            "identifier_version": IDENTIFIER_VERSION,
            "confirmed_meta": confirmed.tolist(),
            "first_identification_decision": timings.tolist(),
            "decision_count": self.decision_count.detach().cpu().tolist(),
            "candidate_count": self.candidates.sum(dim=1).detach().cpu().tolist(),
            "identified_game_fraction": float(identified.float().mean()),
            "mean_identification_decision": float(
                timings[identified].float().mean() if bool(identified.any()) else 0.0
            ),
        }


__all__ = [
    "ExactMetaCatalog", "ExactPublicMetaMemory", "IDENTIFIER_VERSION",
    "UNKNOWN_META_ID",
]
