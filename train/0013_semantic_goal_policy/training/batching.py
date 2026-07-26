"""Compile and collate serialized decision records into policy tensors."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..features.card_semantics import CardSemanticRegistry
from ..features.compiler import compile_features, tensorize_compiled
from ..knowledge.state import CausalKnowledgeState


def _deck_cards(row: Mapping[str, Any]) -> tuple[int, ...]:
    manifest = row.get("deck_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("record deck_manifest must be a mapping")
    counts = manifest.get("counts")
    if not isinstance(counts, Sequence):
        raise ValueError("record deck counts must be a sequence")
    cards: list[int] = []
    for item in counts:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("record deck count entry must be [card_id, count]")
        card_id, count = item
        if isinstance(card_id, bool) or not isinstance(card_id, int):
            raise ValueError("record deck card ID must be an integer")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("record deck multiplicity must be positive")
        cards.extend([card_id] * count)
    if len(cards) != 60:
        raise ValueError("record registered deck must contain 60 cards")
    return tuple(cards)


def _stream_key(row: Mapping[str, Any], fallback: int) -> tuple[object, ...]:
    identity = row.get("identity")
    if isinstance(identity, Mapping):
        source = identity.get("source")
        if isinstance(source, Mapping):
            return (
                source.get("date"), source.get("episode_id"), source.get("player_index"),
            )
        required = ("date", "episode_id", "player_index")
        if all(name in identity for name in required):
            return tuple(identity[name] for name in required)
    return ("isolated", fallback)


def iter_prepare_record_stream(
    records: Iterable[Mapping[str, Any]],
    *,
    registry: CardSemanticRegistry | None = None,
    include_digest: bool = True,
) -> Iterator[dict[str, Any]]:
    """Compile records chronologically before a DataLoader is allowed to shuffle them."""
    sessions: dict[tuple[object, ...], CausalKnowledgeState] = {}
    for index, row in enumerate(records):
        observation = row.get("actor_observation")
        if not isinstance(observation, Mapping):
            raise ValueError("record actor_observation must be a mapping")
        current = observation.get("current")
        actor = current.get("yourIndex") if isinstance(current, Mapping) else None
        if isinstance(actor, bool) or not isinstance(actor, int):
            raise ValueError("record actor index must be an integer")
        deck = _deck_cards(row)
        key = _stream_key(row, index)
        session = sessions.setdefault(key, CausalKnowledgeState.new_game(actor, deck))
        if session.actor != actor or session.initial != Counter(deck):
            raise ValueError("record stream registration changed within an episode-player group")
        knowledge = session.consume(observation)
        compiled = compile_features(
            observation,
            registered_deck=deck,
            knowledge=knowledge,
            registry=registry,
        )
        prepared = dict(row)
        prepared["_compiled_features"] = tensorize_compiled(compiled)
        if include_digest:
            prepared["_compiled_digest"] = compiled.digest
        yield prepared
        action = row.get("ordered_action")
        if not isinstance(action, Sequence):
            raise ValueError("record ordered_action must be a sequence")
        session.record_pending(tuple(int(item) for item in action), knowledge.decision_index)


def prepare_record_stream(
    records: Iterable[Mapping[str, Any]],
    *,
    registry: CardSemanticRegistry | None = None,
) -> list[dict[str, Any]]:
    return list(iter_prepare_record_stream(records, registry=registry))


def _copy_matrix(target: Tensor, row: int, values: Sequence[Sequence[float | int]]) -> None:
    if not values:
        return
    tensor = torch.tensor(values, dtype=target.dtype)
    target[row, : tensor.size(0), : tensor.size(1)] = tensor


def collate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    registry: CardSemanticRegistry | None = None,
    permute_options: bool = False,
    generator: torch.Generator | None = None,
    permutations: Sequence[Sequence[int]] | None = None,
) -> dict[str, Tensor]:
    if not records:
        raise ValueError("records must be nonempty")
    if permutations is not None and len(permutations) != len(records):
        raise ValueError("permutation batch length mismatch")
    prepared = list(records)
    if any("_compiled_features" not in row for row in prepared):
        prepared = prepare_record_stream(prepared, registry=registry)
    features = [row["_compiled_features"] for row in prepared]
    batch = len(prepared)
    max_entities = max(1, max(len(item["entities_cat"]) for item in features))
    max_deck = max(1, max(len(item["deck_card_ids"]) for item in features))
    max_ledger = max(1, max(len(item["ledger_cat"]) for item in features))
    max_events = max(1, max(len(item["events_cat"]) for item in features))
    max_options = max(1, max(len(item["options_cat"]) for item in features))
    max_steps = max(1, max(
        len(row["ordered_action"]) + (0 if row["action_termination"] == "forced_max" else 1)
        for row in prepared
    ))
    out = {
        "state_cat": torch.zeros(batch, 8, dtype=torch.long),
        "state_num": torch.zeros(batch, 16),
        "entities_cat": torch.zeros(batch, max_entities, 8, dtype=torch.long),
        "entities_num": torch.zeros(batch, max_entities, 12),
        "entity_semantic": torch.zeros(batch, max_entities, 64),
        "entity_mask": torch.zeros(batch, max_entities, dtype=torch.bool),
        "deck_card_ids": torch.zeros(batch, max_deck, dtype=torch.long),
        "deck_multiplicity": torch.zeros(batch, max_deck),
        "deck_semantic": torch.zeros(batch, max_deck, 64),
        "deck_mask": torch.zeros(batch, max_deck, dtype=torch.bool),
        "ledger_cat": torch.zeros(batch, max_ledger, 6, dtype=torch.long),
        "ledger_num": torch.zeros(batch, max_ledger, 12),
        "ledger_semantic": torch.zeros(batch, max_ledger, 64),
        "ledger_mask": torch.zeros(batch, max_ledger, dtype=torch.bool),
        "events_cat": torch.zeros(batch, max_events, 6, dtype=torch.long),
        "events_num": torch.zeros(batch, max_events, 8),
        "event_semantic": torch.zeros(batch, max_events, 64),
        "event_mask": torch.zeros(batch, max_events, dtype=torch.bool),
        "relations": torch.zeros(batch, max_entities, max_entities, dtype=torch.long),
        "options_cat": torch.zeros(batch, max_options, 12, dtype=torch.long),
        "options_num": torch.zeros(batch, max_options, 8),
        "option_semantic": torch.zeros(batch, max_options, 64),
        "option_mask": torch.zeros(batch, max_options, dtype=torch.bool),
        "min_count": torch.zeros(batch, dtype=torch.long),
        "max_count": torch.zeros(batch, dtype=torch.long),
        "targets": torch.full((batch, max_steps), max_options, dtype=torch.long),
        "target_mask": torch.zeros(batch, max_steps, dtype=torch.bool),
        "option_permutation_old_to_new": torch.full((batch, max_options), -1, dtype=torch.long),
    }
    for row_index, (row, item) in enumerate(zip(prepared, features)):
        out["state_cat"][row_index] = torch.tensor(item["state_cat"])
        out["state_num"][row_index] = torch.tensor(item["state_num"])
        for prefix, feature_name, mask_name in (
            ("entity", "entities", "entity_mask"),
            ("ledger", "ledger", "ledger_mask"),
            ("event", "events", "event_mask"),
            ("option", "options", "option_mask"),
        ):
            count = len(item[f"{feature_name}_cat"])
            _copy_matrix(out[f"{feature_name}_cat"], row_index, item[f"{feature_name}_cat"])
            _copy_matrix(out[f"{feature_name}_num"], row_index, item[f"{feature_name}_num"])
            _copy_matrix(out[f"{prefix}_semantic"], row_index, item[f"{prefix}_semantic"])
            out[mask_name][row_index, :count] = True
        deck_count = len(item["deck_card_ids"])
        out["deck_card_ids"][row_index, :deck_count] = torch.tensor(item["deck_card_ids"])
        out["deck_multiplicity"][row_index, :deck_count] = torch.tensor(item["deck_multiplicity"])
        _copy_matrix(out["deck_semantic"], row_index, item["deck_semantic"])
        out["deck_mask"][row_index, :deck_count] = True
        for source, target, kind in item["relations"]:
            out["relations"][row_index, source, target] = kind
        out["min_count"][row_index] = item["min_count"]
        out["max_count"][row_index] = item["max_count"]

        option_count = len(item["options_cat"])
        if permutations is not None:
            new_to_old = torch.tensor(permutations[row_index], dtype=torch.long)
        elif permute_options:
            new_to_old = torch.randperm(option_count, generator=generator)
        else:
            new_to_old = torch.arange(option_count)
        if new_to_old.numel() != option_count or sorted(new_to_old.tolist()) != list(range(option_count)):
            raise ValueError("option permutation must be a bijection")
        old_to_new = torch.empty(option_count, dtype=torch.long)
        old_to_new[new_to_old] = torch.arange(option_count)
        out["option_permutation_old_to_new"][row_index, :option_count] = old_to_new
        for name in ("options_cat", "options_num", "option_semantic", "option_mask"):
            original = out[name][row_index, :option_count].clone()
            out[name][row_index, :option_count] = original[new_to_old]

        action = [old_to_new[int(index)].item() for index in row["ordered_action"]]
        targets = action if row["action_termination"] == "forced_max" else action + [max_options]
        out["targets"][row_index, :len(targets)] = torch.tensor(targets)
        out["target_mask"][row_index, :len(targets)] = True
    return out


def default_registry() -> CardSemanticRegistry:
    return CardSemanticRegistry.from_official_csv(Path("data/official/EN_Card_Data.csv"))


__all__ = [
    "collate_records", "default_registry", "iter_prepare_record_stream", "prepare_record_stream",
]
