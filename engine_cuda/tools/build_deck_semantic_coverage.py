from __future__ import annotations

"""Build a deck-scoped semantic coverage report.

The global official coverage matrix intentionally contains the complete official
inventory.  This tool projects that inventory onto the concrete opponent pool
used by CUDA PPO.  It follows card -> attack/skill references and the linked
attack/skill references embedded in typed-IR effects, so a deck report is a
semantic dependency report rather than a list of card IDs only.

The input IR and coverage matrix may contain private provenance, therefore the
output contains IDs and status counts only and does not copy source text or
private rule payloads.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="private typed IR JSON")
    parser.add_argument("--matrix", type=Path, required=True, help="semantic coverage JSON")
    parser.add_argument("--pool-config", type=Path, required=True, help="BC opponent pool JSON")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            for raw in row:
                value = raw.strip()
                if not value:
                    continue
                try:
                    counts[int(value)] += 1
                except ValueError as exc:
                    raise ValueError(f"non-integer card ID {value!r} in {path}") from exc
    return counts


def walk_refs(value: Any) -> Iterable[tuple[str, int]]:
    """Yield typed-IR skill/attack references from nested effect records."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"skill_id", "linked_skill_id"} and isinstance(item, int) and item:
                yield "skills", item
            elif key in {"attack_id", "linked_attack_id"} and isinstance(item, int) and item:
                yield "attacks", item
            yield from walk_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_refs(item)


def matrix_index(matrix: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    domains = matrix.get("domains", {})
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for domain in ("cards", "skills", "attacks"):
        result[domain] = {
            int(row["id"]): row for row in domains.get(domain, []) if "id" in row
        }
    return result


def status_summary(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("status", "missing") for row in rows).items()))


def main() -> int:
    args = parse_args()
    ir = json.loads(args.input.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    pool = json.loads(args.pool_config.read_text(encoding="utf-8"))

    cards_by_id = {int(row["id"]): row for row in ir.get("cards", [])}
    skills_by_id = {int(row["id"]): row for row in ir.get("skills", [])}
    attacks_by_id = {int(row["id"]): row for row in ir.get("attacks", [])}
    indexes = matrix_index(matrix)

    # Pre-index all typed-IR entities by their owner card and collect all
    # references once.  The traversal is intentionally deterministic.
    skills_by_card: defaultdict[int, set[int]] = defaultdict(set)
    attacks_by_card: defaultdict[int, set[int]] = defaultdict(set)
    for entity_id, row in skills_by_id.items():
        skills_by_card[int(row.get("card_id", 0))].add(entity_id)
    for entity_id, row in attacks_by_id.items():
        attacks_by_card[int(row.get("card_id", 0))].add(entity_id)

    refs: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for domain, entities in (("skills", skills_by_id), ("attacks", attacks_by_id)):
        for entity_id, row in entities.items():
            for target_domain, target_id in walk_refs(row):
                if target_id in (skills_by_id if target_domain == "skills" else attacks_by_id):
                    refs[(domain, entity_id)].add((target_domain, target_id))

    def project(deck_path: Path) -> dict[str, Any]:
        card_counts = read_deck(deck_path)
        missing_cards = sorted(card_id for card_id in card_counts if card_id not in cards_by_id)
        selected_cards = set(card_counts).intersection(cards_by_id)
        selected_skills = {
            skill_id
            for card_id in selected_cards
            for skill_id in skills_by_card.get(card_id, set())
        }
        selected_attacks = {
            attack_id
            for card_id in selected_cards
            for attack_id in attacks_by_card.get(card_id, set())
        }

        # Follow links until the typed-IR dependency closure is stable.
        queue: deque[tuple[str, int]] = deque(
            [("skills", i) for i in sorted(selected_skills)]
            + [("attacks", i) for i in sorted(selected_attacks)]
        )
        seen = set(queue)
        while queue:
            ref = queue.popleft()
            for target in sorted(refs.get(ref, set())):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
                    if target[0] == "skills":
                        selected_skills.add(target[1])
                    else:
                        selected_attacks.add(target[1])

        branch_rows = [
            row
            for row in matrix.get("domains", {}).get("branches", [])
            if (
                (row.get("owner_kind") == "skill" and int(row.get("owner_id", -1)) in selected_skills)
                or (
                    row.get("owner_kind") == "attack"
                    and int(row.get("owner_id", -1)) in selected_attacks
                )
            )
        ]
        domain_rows = {
            "cards": [indexes["cards"].get(i, {"id": i, "status": "missing"}) for i in sorted(selected_cards)],
            "skills": [indexes["skills"].get(i, {"id": i, "status": "missing"}) for i in sorted(selected_skills)],
            "attacks": [indexes["attacks"].get(i, {"id": i, "status": "missing"}) for i in sorted(selected_attacks)],
            "branches": sorted(
                branch_rows,
                key=lambda row: (
                    str(row.get("owner_kind", "")),
                    int(row.get("owner_id", -1)),
                    str(row.get("branch_id", "")),
                ),
            ),
        }
        return {
            "deck": str(deck_path),
            "card_counts": {str(k): v for k, v in sorted(card_counts.items())},
            "card_count_total": sum(card_counts.values()),
            "unique_card_count": len(card_counts),
            "missing_card_ids": missing_cards,
            "entity_counts": {domain: len(rows) for domain, rows in domain_rows.items()},
            "status_counts": {domain: status_summary(rows) for domain, rows in domain_rows.items()},
            "missing_entity_ids": {
                domain: sorted(int(row["id"]) for row in rows if row.get("status") == "missing")
                for domain, rows in domain_rows.items()
                if domain != "branches"
            },
            "rows": {
                domain: [
                    {
                        key: row[key]
                        for key in ("id", "owner_id", "owner_kind", "branch_id", "status")
                        if key in row
                    }
                    for row in rows
                ]
                for domain, rows in domain_rows.items()
            },
        }

    reports: list[dict[str, Any]] = []
    for opponent in pool.get("opponents", []):
        directory = Path(opponent["directory"])
        deck_path = directory / "deck.csv"
        if not deck_path.is_file():
            raise FileNotFoundError(deck_path)
        report = project(deck_path)
        report.update(
            {
                "name": opponent.get("name"),
                "archetype": opponent.get("archetype"),
                "kind": opponent.get("kind"),
                "deck_sha256": sha256_file(deck_path),
            }
        )
        reports.append(report)

    all_statuses: Counter[str] = Counter()
    for report in reports:
        for domain, values in report["status_counts"].items():
            for status, count in values.items():
                all_statuses[f"{domain}:{status}"] += count
    output = {
        "schema_version": 1,
        "source_ir_sha256": sha256_file(args.input),
        "source_matrix_sha256": sha256_file(args.matrix),
        "pool_config_sha256": sha256_file(args.pool_config),
        "opponent_count": len(reports),
        "aggregate_status_counts": dict(sorted(all_statuses.items())),
        "opponents": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "opponent_count": len(reports),
        "aggregate_status_counts": output["aggregate_status_counts"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
