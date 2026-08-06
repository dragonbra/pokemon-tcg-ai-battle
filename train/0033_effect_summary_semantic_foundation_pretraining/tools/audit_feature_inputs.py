"""Audit raw replay facts through compiler, batch validation, and model forward."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..contracts.batch import DecisionBatch
from ..data.replay_contract import (
    DeckManifest,
    decision_frames,
    project_observation,
    registration_decks,
    validate_ordered_action,
)
from ..domain.prototypes import PrototypeIndex
from ..features.audit import audit_compiled_feature_input
from ..features.collate import collate_canonical_records
from ..features.compiler import compile_canonical_row
from ..knowledge.state import CausalKnowledge
from ..model import ModelConfig, SemanticPolicy


ENERGY_CARD_TYPES = frozenset({5, 6})
SPECIAL_ENERGY_CARD_TYPE = 6


def _winner(payload: Mapping[str, Any]) -> int:
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    if not isinstance(rewards, list) or not isinstance(statuses, list):
        raise ValueError("replay has no terminal rewards/statuses")
    winners = [
        index
        for index, reward in enumerate(rewards)
        if index < len(statuses)
        and statuses[index] == "DONE"
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and reward > 0
    ]
    if len(winners) != 1:
        raise ValueError(f"replay must have one completed positive-reward winner: {winners}")
    return winners[0]


def _source_rows(
    payload: Mapping[str, Any], actor: int
) -> list[dict[str, Any]]:
    deck = registration_decks(payload)[actor]
    deck_manifest = DeckManifest.from_card_ids(deck).as_dict()
    output: list[dict[str, Any]] = []
    for decision_index, (frame_index, raw_observation, target, _) in enumerate(
        decision_frames(payload, actor)
    ):
        observation = project_observation(raw_observation)
        select = observation["select"]
        ordered, termination = validate_ordered_action(
            target,
            option_count=len(select["option"]),
            min_count=select["minCount"],
            max_count=select["maxCount"],
            capacity=64,
        )
        output.append(
            {
                "identity": {
                    "date": "feature-audit",
                    "episode_id": int(payload.get("info", {}).get("EpisodeId", -1)),
                    "player_index": actor,
                    "episode_step": frame_index,
                },
                "split": "audit",
                "deck_manifest": deck_manifest,
                "actor_observation": observation,
                "ordered_action": list(ordered),
                "action_termination": termination,
                "event_cursor": {
                    "visual_frame_index": frame_index,
                    "actor_decision_index": decision_index,
                    "incoming_log_count": len(observation.get("logs", [])),
                },
            }
        )
    if not output:
        raise ValueError("winner replay has no decision frames")
    return output


def _audit_energy_prototypes(prototypes: PrototypeIndex) -> dict[str, int]:
    energy_cards = [
        card
        for card in prototypes.engine_cards.values()
        if int(card["card_type"]) in ENERGY_CARD_TYPES
    ]
    special = [
        card for card in energy_cards if int(card["card_type"]) == SPECIAL_ENERGY_CARD_TYPE
    ]
    for card in energy_cards:
        mask = int(card["energy_type_mask"])
        count = int(card["energy_count"])
        if not 0 <= mask < (1 << 9) or count < 1:
            raise ValueError(f"invalid engine Energy prototype: {card['card_id']}")
    special_with_skill = sum(
        any(int(card[name]) > 0 for name in ("ability_skill_id", "play_skill_id", "delay_skill_id"))
        for card in special
    )
    if special_with_skill != len(special):
        raise ValueError("special Energy prototype has no Card-to-Skill relation")
    return {
        "energy_cards": len(energy_cards),
        "special_energy_cards": len(special),
        "special_energy_cards_with_skill": special_with_skill,
    }


def audit_episode(
    replay_path: Path,
    prototype_path: Path,
    *,
    forward_batch: int = 8,
    device: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    actor = _winner(payload)
    sources = _source_rows(payload, actor)
    prototypes = PrototypeIndex.load(prototype_path)
    knowledge = CausalKnowledge(actor, registration_decks(payload)[actor])

    records: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for source in sources:
        snapshot = knowledge.consume(source["actor_observation"], source["event_cursor"])
        record = compile_canonical_row(source, snapshot, prototypes)
        totals.update(audit_compiled_feature_input(source, record, snapshot))
        for name in (
            "card_parent",
            "event_source",
            "event_target",
            "option_source",
            "option_target",
        ):
            relation_counts[name] += sum(value > 0 for value in record["actor"][name])
        records.append(record)

    validated = DecisionBatch.from_mapping(collate_canonical_records(records))
    max_shapes = {
        "cards": int(validated.card_mask.sum(dim=1).max()),
        "resources": int(validated.resource_mask.sum(dim=1).max()),
        "events": int(validated.event_mask.sum(dim=1).max()),
        "options": int(validated.option_mask.sum(dim=1).max()),
    }

    forward: dict[str, Any] = {"rows": 0}
    if forward_batch > 0:
        count = min(forward_batch, len(records))
        forward_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        forward_inputs = DecisionBatch.from_mapping(
            collate_canonical_records(records[:count])
        ).to(forward_device)
        model = SemanticPolicy(ModelConfig(), prototypes).to(forward_device).eval()
        with torch.inference_mode():
            logits = model.teacher_logits(forward_inputs)
        if not torch.isfinite(logits).all():
            raise ValueError("model forward produced non-finite logits")
        forward = {
            "rows": count,
            "device": str(forward_device),
            "logits_shape": list(logits.shape),
            "finite": True,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }

    return {
        "status": "passed",
        "episode_id": int(payload.get("info", {}).get("EpisodeId", -1)),
        "winner": actor,
        "decisions": len(records),
        "schema_version": records[0]["schema_version"],
        "facts": dict(totals),
        "nonzero_relations": dict(relation_counts),
        "max_sequence_lengths": max_shapes,
        "energy_prototypes": _audit_energy_prototypes(prototypes),
        "forward": forward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets"
        / "official_public_prototypes_v1.json",
    )
    parser.add_argument("--forward-batch", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.forward_batch < 0:
        parser.error("--forward-batch cannot be negative")
    report = audit_episode(
        args.replay,
        args.prototypes,
        forward_batch=args.forward_batch,
        device=args.device,
    )
    encoded = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
