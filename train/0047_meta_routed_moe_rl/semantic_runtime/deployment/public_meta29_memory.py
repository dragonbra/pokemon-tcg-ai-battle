"""Public-Pokemon priority rules for the deployable 29-class Meta router."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = PROJECT_ROOT / "assets/meta/public_meta29_rules_v1.json"
IDENTIFIER_VERSION = "0047_public_meta29_priority_rules_v1"
UNKNOWN_META_ID = -1


@dataclass(frozen=True, slots=True)
class PublicMeta29Rule:
    meta_id: int
    name: str
    all_groups: tuple[frozenset[int], ...]


@dataclass(frozen=True, slots=True)
class PublicMeta29Rulebook:
    rules: tuple[PublicMeta29Rule, ...]
    class_count: int
    other_class_id: int

    @classmethod
    def load(cls, path: Path = DEFAULT_RULES) -> "PublicMeta29Rulebook":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != IDENTIFIER_VERSION:
            raise RuntimeError("0047 public Meta-29 rule schema mismatch")
        rules = tuple(
            PublicMeta29Rule(
                meta_id=int(row["meta_id"]),
                name=str(row["name"]),
                all_groups=tuple(
                    frozenset(int(card_id) for card_id in group)
                    for group in row["all_groups"]
                ),
            )
            for row in raw["rules"]
        )
        class_count = int(raw["class_count"])
        other_class_id = int(raw["other_class_id"])
        expected = set(range(class_count)) - {other_class_id}
        if {rule.meta_id for rule in rules} != expected or len(rules) != len(expected):
            raise RuntimeError("0047 public Meta-29 rules must cover every non-Other class once")
        if any(not rule.all_groups or any(not group for group in rule.all_groups) for rule in rules):
            raise RuntimeError("0047 public Meta-29 rule groups must be non-empty")
        taxonomy = json.loads(
            (PROJECT_ROOT / "assets/taxonomy/own_archetypes_v2.json")
            .read_text(encoding="utf-8")
        )
        names = {int(row["archetype_id"]): str(row["name"]) for row in taxonomy["classes"]}
        if int(taxonomy["other_class_id"]) != other_class_id or len(names) != class_count:
            raise RuntimeError("0047 public Meta-29 taxonomy shape mismatch")
        if any(names[rule.meta_id] != rule.name for rule in rules):
            raise RuntimeError("0047 public Meta-29 rule names disagree with taxonomy")
        return cls(rules=rules, class_count=class_count, other_class_id=other_class_id)

    @property
    def trigger_card_ids(self) -> tuple[int, ...]:
        return tuple(sorted({card for rule in self.rules for group in rule.all_groups for card in group}))

    def classify(self, cards: set[int] | frozenset[int]) -> int:
        present = set(int(card) for card in cards)
        for rule in self.rules:
            if all(present.intersection(group) for group in rule.all_groups):
                return rule.meta_id
        return self.other_class_id


class PublicMeta29Memory:
    """Accumulate public triggers and recompute the ordered 29-class rule."""

    def __init__(
        self, job_count: int, device: torch.device,
        rulebook: PublicMeta29Rulebook | None = None,
    ) -> None:
        if job_count < 1:
            raise ValueError("public Meta-29 memory requires at least one job")
        self.job_count = int(job_count)
        self.device = device
        self.rulebook = rulebook or PublicMeta29Rulebook.load()
        self.trigger_card_ids = self.rulebook.trigger_card_ids
        self._trigger_ids = torch.tensor(self.trigger_card_ids, dtype=torch.long, device=device)
        self._trigger_index = {
            card_id: index for index, card_id in enumerate(self.trigger_card_ids)
        }
        self._rule_groups = tuple(
            tuple(
                torch.tensor(
                    [self._trigger_index[card] for card in sorted(group)],
                    dtype=torch.long, device=device,
                )
                for group in rule.all_groups
            )
            for rule in self.rulebook.rules
        )
        self.seen = torch.zeros(
            (job_count, len(self.trigger_card_ids)), dtype=torch.bool, device=device
        )
        self.current_meta = torch.full(
            (job_count,), UNKNOWN_META_ID, dtype=torch.long, device=device
        )
        self.decision_count = torch.zeros(job_count, dtype=torch.long, device=device)
        self.first_identification_decision = torch.full(
            (job_count,), -1, dtype=torch.long, device=device
        )
        self.classification_change_count = torch.zeros(
            job_count, dtype=torch.long, device=device
        )

    def observe(self, validated: Any, job_indices: torch.Tensor) -> torch.Tensor:
        jobs = job_indices.long().to(self.device)
        if jobs.ndim != 1 or jobs.numel() != validated.card_cat.shape[0]:
            raise ValueError("public Meta-29 jobs do not align with semantic rows")
        if bool(jobs.lt(0).any()) or bool(jobs.ge(self.job_count).any()):
            raise ValueError("public Meta-29 job index is outside memory")
        if jobs.unique().numel() != jobs.numel():
            raise ValueError("one resident job appeared twice at one decision boundary")

        card_ids = validated.card_cat[..., 0].long()
        owner = validated.card_cat[..., 2].long()
        knowledge = validated.card_cat[..., 8].long()
        certain_public_opponent = (
            validated.card_mask.bool() & owner.eq(2)
            & (knowledge.eq(1) | knowledge.eq(2))
        )
        observed = (
            card_ids.unsqueeze(-1).eq(self._trigger_ids.view(1, 1, -1))
            & certain_public_opponent.unsqueeze(-1)
        ).any(dim=1)
        accumulated = self.seen.index_select(0, jobs) | observed
        self.seen.index_copy_(0, jobs, accumulated)

        counts = self.decision_count.index_select(0, jobs) + 1
        self.decision_count.index_copy_(0, jobs, counts)
        classified = torch.full(
            (jobs.numel(),), UNKNOWN_META_ID, dtype=torch.long, device=self.device
        )
        for rule, groups in zip(self.rulebook.rules, self._rule_groups, strict=True):
            matches = torch.ones(jobs.numel(), dtype=torch.bool, device=self.device)
            for group in groups:
                matches &= accumulated.index_select(1, group).any(dim=1)
            take = classified.eq(UNKNOWN_META_ID) & matches
            classified = torch.where(
                take, torch.full_like(classified, rule.meta_id), classified
            )

        previous = self.current_meta.index_select(0, jobs)
        newly_identified = previous.eq(UNKNOWN_META_ID) & classified.ge(0)
        changed = previous.ge(0) & classified.ge(0) & previous.ne(classified)
        self.classification_change_count.index_add_(0, jobs, changed.long())
        first = self.first_identification_decision.index_select(0, jobs)
        self.first_identification_decision.index_copy_(
            0, jobs, torch.where(newly_identified, counts, first)
        )
        self.current_meta.index_copy_(0, jobs, classified)
        return classified

    def telemetry(self) -> dict[str, Any]:
        current = self.current_meta.detach().cpu()
        identified = current.ge(0)
        timings = self.first_identification_decision.detach().cpu()
        seen = self.seen.detach().cpu()
        return {
            "identifier_version": IDENTIFIER_VERSION,
            "current_meta": current.tolist(),
            "first_identification_decision": timings.tolist(),
            "decision_count": self.decision_count.detach().cpu().tolist(),
            "classification_change_count": self.classification_change_count.detach().cpu().tolist(),
            "seen_trigger_card_ids": [
                [self.trigger_card_ids[index] for index in row.nonzero().flatten().tolist()]
                for row in seen
            ],
            "identified_game_fraction": float(identified.float().mean()),
            "mean_identification_decision": float(
                timings[timings.ge(0)].float().mean() if bool(timings.ge(0).any()) else 0.0
            ),
        }


__all__ = [
    "IDENTIFIER_VERSION", "UNKNOWN_META_ID", "PublicMeta29Memory",
    "PublicMeta29Rule", "PublicMeta29Rulebook",
]
