from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
DEFAULT_DECK_ROOT = WORKSPACE_ROOT / "train" / "0022_league_training" / "deck40"
DEFAULT_RULES_JSON = (
    CUDA_ENGINE_ROOT
    / "generated"
    / "private"
    / "official_3aaeaa92"
    / "official_rules.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map test-only effect-offset coverage back to reachable 0022 deck40 "
            "skill and attack branches."
        )
    )
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument("--rules-json", type=Path, default=DEFAULT_RULES_JSON)
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_deck_card_ids(deck_root: Path) -> set[int]:
    deck_paths = sorted(deck_root.glob("*/deck.csv"))
    if len(deck_paths) != 40:
        raise ValueError(f"expected 40 deck files, found {len(deck_paths)}")
    card_ids: set[int] = set()
    for deck_path in deck_paths:
        cards = [
            int(line.strip())
            for line in deck_path.read_text().splitlines()
            if line.strip()
        ]
        if len(cards) != 60:
            raise ValueError(f"deck must contain 60 card IDs: {deck_path}")
        card_ids.update(cards)
    return card_ids


def effect_semantic_signature(effect: dict[str, Any]) -> str:
    normalized = {
        key: value
        for key, value in effect.items()
        if key not in {"linked_skill_id", "linked_attack_id"}
    }
    canonical = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def build_effect_inventory(
    payload: dict[str, Any],
    deck_card_ids: set[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    cards = {int(card["id"]): card for card in payload["cards"]}
    skills = {int(skill["id"]): skill for skill in payload["skills"]}
    attacks = {int(attack["id"]): attack for attack in payload["attacks"]}

    skill_effects: dict[int, list[tuple[int, str, int, dict[str, Any]]]] = {}
    attack_effects: dict[int, list[tuple[int, str, int, dict[str, Any]]]] = {}
    absolute = 0
    for skill in payload["skills"]:
        rows: list[tuple[int, str, int, dict[str, Any]]] = []
        for index, effect in enumerate(skill["effects"]):
            rows.append((absolute, "effect", index, effect))
            absolute += 1
        skill_effects[int(skill["id"])] = rows
    for attack in payload["attacks"]:
        rows = []
        for phase in ("pre_effects", "post_effects"):
            for index, effect in enumerate(attack[phase]):
                rows.append((absolute, phase, index, effect))
                absolute += 1
        attack_effects[int(attack["id"])] = rows

    queue: deque[tuple[str, int]] = deque()
    missing_card_ids: list[int] = []
    for card_id in sorted(deck_card_ids):
        card = cards.get(card_id)
        if card is None:
            missing_card_ids.append(card_id)
            continue
        for skill_id in (card["ability_id"], card["play_id"], card["delay_id"]):
            if int(skill_id) > 0:
                queue.append(("skill", int(skill_id)))
        for attack_id in card["attack_ids"]:
            if int(attack_id) > 0:
                queue.append(("attack", int(attack_id)))

    seen_skills: set[int] = set()
    seen_attacks: set[int] = set()
    missing_owner_ids: list[dict[str, Any]] = []
    inventory: dict[int, dict[str, Any]] = {}
    while queue:
        owner_kind, owner_id = queue.popleft()
        if owner_kind == "skill":
            if owner_id in seen_skills:
                continue
            seen_skills.add(owner_id)
            owner = skills.get(owner_id)
            rows = skill_effects.get(owner_id)
        else:
            if owner_id in seen_attacks:
                continue
            seen_attacks.add(owner_id)
            owner = attacks.get(owner_id)
            rows = attack_effects.get(owner_id)
        if owner is None or rows is None:
            missing_owner_ids.append({"kind": owner_kind, "id": owner_id})
            continue

        for effect_offset, phase, effect_index, effect in rows:
            is_condition = (int(effect["flags"]) & 1) != 0
            inventory[effect_offset] = {
                "effect_offset": effect_offset,
                "owner_kind": owner_kind,
                "owner_id": owner_id,
                "owner_card_id": int(owner["card_id"]),
                "phase": phase,
                "effect_index": effect_index,
                "effect_type": int(effect["type"]),
                "condition_type": int(effect["condition_type"]),
                "is_condition": is_condition,
                "semantic_signature": effect_semantic_signature(effect),
            }
            linked_skill_id = int(effect["linked_skill_id"])
            linked_attack_id = int(effect["linked_attack_id"])
            if linked_skill_id > 0:
                queue.append(("skill", linked_skill_id))
            if linked_attack_id > 0:
                queue.append(("attack", linked_attack_id))

    metadata = {
        "flattened_effect_count": absolute,
        "deck_unique_card_count": len(deck_card_ids),
        "reachable_skill_count": len(seen_skills),
        "reachable_attack_count": len(seen_attacks),
        "missing_card_ids": missing_card_ids,
        "missing_owner_ids": missing_owner_ids,
    }
    return inventory, metadata


def load_coverage(
    paths: list[Path],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    fields = (
        "effect_offsets_reached",
        "effect_offsets_applied",
        "effect_offsets_condition_true",
        "effect_offsets_condition_false",
    )
    combined = {field: set() for field in fields}
    evidence_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("passed") is not True:
            raise ValueError(f"coverage evidence did not pass: {path}")
        if payload.get("branch_coverage_enabled") is not True:
            raise ValueError(f"branch coverage is not enabled: {path}")
        for field in fields:
            combined[field].update(int(value) for value in payload.get(field, []))
        evidence_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "policy": payload.get("policy") or payload.get("scope"),
                "battles_compared": int(
                    payload.get("battles_compared", payload.get("seed_count", 0))
                ),
                "decisions_compared": int(payload.get("decisions_compared", 0)),
            }
        )
    return combined, evidence_rows


def main() -> None:
    args = parse_args()
    rules_path = args.rules_json.resolve()
    deck_root = args.deck_root.resolve()
    evidence_paths = [path.resolve() for path in args.evidence]
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    inventory, metadata = build_effect_inventory(payload, load_deck_card_ids(deck_root))
    coverage, evidence_rows = load_coverage(evidence_paths)

    reachable = set(inventory)
    reached = coverage["effect_offsets_reached"] & reachable
    applied = coverage["effect_offsets_applied"] & reachable
    condition_true = coverage["effect_offsets_condition_true"] & reachable
    condition_false = coverage["effect_offsets_condition_false"] & reachable
    condition_offsets = {
        offset for offset, row in inventory.items() if row["is_condition"]
    }
    non_condition_offsets = reachable - condition_offsets
    reachable_signatures = {
        inventory[offset]["semantic_signature"] for offset in reachable
    }
    reached_signatures = {inventory[offset]["semantic_signature"] for offset in reached}
    uncovered_signature_offsets = {
        offset
        for offset in reachable
        if inventory[offset]["semantic_signature"] not in reached_signatures
    }

    def rows(offsets: set[int]) -> list[dict[str, Any]]:
        return [inventory[offset] for offset in sorted(offsets)]

    report = {
        "passed": not metadata["missing_card_ids"]
        and not metadata["missing_owner_ids"],
        "contract": "0022_deck40_effect_branch_coverage_v1",
        **metadata,
        "reachable_effect_count": len(reachable),
        "reachable_condition_count": len(condition_offsets),
        "reachable_non_condition_count": len(non_condition_offsets),
        "reachable_semantic_signature_count": len(reachable_signatures),
        "reached_effect_count": len(reached),
        "applied_non_condition_count": len(applied & non_condition_offsets),
        "condition_true_count": len(condition_true),
        "condition_false_count": len(condition_false),
        "condition_dual_outcome_count": len(condition_true & condition_false),
        "never_reached_count": len(reachable - reached),
        "never_reached": rows(reachable - reached),
        "uncovered_semantic_signature_count": len(
            reachable_signatures - reached_signatures
        ),
        "uncovered_semantic_signatures": rows(uncovered_signature_offsets),
        "non_condition_not_applied_count": len(non_condition_offsets - applied),
        "non_condition_not_applied": rows(non_condition_offsets - applied),
        "condition_never_true_count": len(condition_offsets - condition_true),
        "condition_never_true": rows(condition_offsets - condition_true),
        "condition_never_false_count": len(condition_offsets - condition_false),
        "condition_never_false": rows(condition_offsets - condition_false),
        "extra_reached_offsets": sorted(coverage["effect_offsets_reached"] - reachable),
        "rules_json_sha256": sha256_file(rules_path),
        "evidence": evidence_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in {
                    "never_reached",
                    "uncovered_semantic_signatures",
                    "non_condition_not_applied",
                    "condition_never_true",
                    "condition_never_false",
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
