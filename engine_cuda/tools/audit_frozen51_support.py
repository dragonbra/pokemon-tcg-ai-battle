from __future__ import annotations

"""Build the fail-closed CUDA support report for the Frozen51 deck pool.

The official CPU engine remains the oracle. This tool does not execute private
card logic; it validates and aggregates evidence produced by the paired
CPU/POD/CUDA runner, then records the known static or interaction blockers.
"""

import argparse
from collections import Counter, deque
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable


CUDA_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CUDA_ROOT.parent
DEFAULT_FROZEN_ROOT = REPOSITORY_ROOT / "evaluation/arena/frozen"
DEFAULT_FIXTURE_ROOT = CUDA_ROOT / "fixtures/0022_deck40"
DEFAULT_SUPPORTED = REPOSITORY_ROOT / ".tmp/cuda_support_audit/supported38_decks.txt"
DEFAULT_EVIDENCE = (
    REPOSITORY_ROOT / ".tmp/cuda_support_audit/supported38_ordered_s1.jsonl"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "evaluation/arena/combat_mat/0031_latest_frozen_test/cuda_support.json"
)


BLOCKERS: dict[str, dict[str, Any]] = {
    "arboliva_ex_meganium_001": {
        "kind": "missing_selection_continuation",
        "cards": [{"id": 404, "name": "Arboliva ex"}],
        "effect_types": [{"id": 39, "name": "AttackDamageMulti"}],
        "reason": "AttackDamageMulti can require a selection continuation that the CUDA runtime does not implement.",
    },
    "cynthias_garchomp_ex_roserade_002": {
        "kind": "continual_state_mismatch",
        "cards": [{"id": 1249, "name": "Grand Tree"}],
        "reason": "The continual refresh/control flag state diverges from the official engine.",
    },
    "festival_lead_dipplin_001": {
        "kind": "missing_selection_continuation",
        "cards": [{"id": 93, "name": "Dipplin"}],
        "attacks": [115],
        "continuations": [{"id": 88, "name": "SelectedSecondAttack"}],
        "reason": "The selected second-attack continuation is not implemented.",
    },
    "festival_lead_dipplin_002": {"same_as": "festival_lead_dipplin_001"},
    "festival_lead_dipplin_003": {"same_as": "festival_lead_dipplin_001"},
    "hydrapple_ex_meganium_001": {"same_as": "festival_lead_dipplin_001"},
    "ns_zoroark_ex_001": {
        "kind": "unsupported_copy_attack",
        "cards": [{"id": 293, "name": "N's Zoroark ex"}],
        "attacks": [403],
        "reason": "The copy-attack continuation explicitly returns unsupported_continuation.",
    },
    "team_rockets_mewtwo_ex_spidops_001": {
        "kind": "unsupported_copy_attack",
        "cards": [{"id": 434, "name": "Team Rocket's Mimikyu"}],
        "attacks": [612],
        "reason": "The copy-attack path explicitly returns unsupported_continuation.",
    },
    "team_rockets_mewtwo_ex_spidops_002": {"same_as": "team_rockets_mewtwo_ex_spidops_001"},
    "team_rockets_mewtwo_ex_spidops_003": {"same_as": "team_rockets_mewtwo_ex_spidops_001"},
    "team_rockets_mewtwo_ex_spidops_004": {"same_as": "team_rockets_mewtwo_ex_spidops_001"},
    "dragapult_ex_dusknoir_002": {
        "kind": "cross_interaction_state_mismatch",
        "cards": [
            {"id": 44, "name": "Bloodmoon Ursaluna ex"},
            {"id": 1256, "name": "Team Rocket's Watchtower", "role": "opposing_card"},
        ],
        "failed_case": 333,
        "failed_matchup": ["dragapult_ex_crushing_hammer_001", "dragapult_ex_dusknoir_002"],
        "reason": "A continual-state byte differs when Bloodmoon Ursaluna ex interacts with Team Rocket's Watchtower.",
    },
    "team_rockets_mewtwo_ex_spidops_005": {
        "kind": "missing_selection_continuation",
        "cards": [{"id": 432, "name": "Team Rocket's Wobbuffet"}],
        "attacks": [609],
        "effect_types": [{"id": 42, "name": "RemoveDamageCounter"}],
        "reason": "RemoveDamageCounter lacks the required selection continuation when an opposing target is valid.",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    cards: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            cards.extend(int(value.strip()) for value in row if value.strip())
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError(f"deck must contain exactly 60 positive card IDs: {path}")
    return cards


def exact_deck_hash(cards: Iterable[int]) -> str:
    payload = ",".join(str(card) for card in sorted(cards)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_decks(root: Path) -> dict[str, list[int]]:
    paths = sorted(path for path in root.glob("*/deck.csv") if path.parent.name != "_policy")
    return {path.parent.name: read_deck(path) for path in paths}


def walk_refs(value: Any) -> Iterable[tuple[str, int]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"skill_id", "linked_skill_id"} and isinstance(item, int) and item > 0:
                yield "skills", item
            elif key in {"attack_id", "linked_attack_id"} and isinstance(item, int) and item > 0:
                yield "attacks", item
            yield from walk_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_refs(item)


def reachable_rules(rules: dict[str, Any], card_ids: Iterable[int]) -> dict[str, list[int]]:
    cards = {int(row["id"]): row for row in rules.get("cards", [])}
    skills = {int(row["id"]): row for row in rules.get("skills", [])}
    attacks = {int(row["id"]): row for row in rules.get("attacks", [])}
    queue: deque[tuple[str, int]] = deque()
    for card_id in sorted(set(card_ids)):
        card = cards.get(card_id)
        if card is None:
            continue
        for key in ("ability_id", "play_id", "delay_id"):
            value = int(card.get(key, 0))
            if value > 0:
                queue.append(("skills", value))
        for value in card.get("attack_ids", []):
            if int(value) > 0:
                queue.append(("attacks", int(value)))
    seen: set[tuple[str, int]] = set()
    while queue:
        domain, entity_id = queue.popleft()
        key = (domain, entity_id)
        if key in seen:
            continue
        seen.add(key)
        row = (skills if domain == "skills" else attacks).get(entity_id)
        if row is not None:
            queue.extend(ref for ref in walk_refs(row) if ref not in seen)
    return {
        "cards": sorted(set(card_ids)),
        "skills": sorted(entity_id for domain, entity_id in seen if domain == "skills"),
        "attacks": sorted(entity_id for domain, entity_id in seen if domain == "attacks"),
    }


def parse_case_evidence(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("CASE\t"):
            continue
        _, ordinal, payload = line.split("\t", 2)
        row = json.loads(payload)
        row["ordinal"] = int(ordinal)
        cases.append(row)
    return cases


def resolved_blocker(deck_id: str) -> dict[str, Any]:
    blocker = dict(BLOCKERS[deck_id])
    source = blocker.pop("same_as", None)
    if source:
        blocker = dict(BLOCKERS[source])
        blocker["shared_with"] = source
    return blocker


def build_report(
    frozen_root: Path,
    fixture_root: Path,
    supported_path: Path,
    evidence_path: Path,
    *,
    rules_path: Path | None = None,
) -> dict[str, Any]:
    frozen = load_decks(frozen_root)
    fixtures = load_decks(fixture_root)
    if len(frozen) != 51:
        raise ValueError(f"expected 51 Frozen decks, found {len(frozen)}")
    if len(fixtures) != 40:
        raise ValueError(f"expected 40 prior fixture decks, found {len(fixtures)}")
    supported_paths = [Path(line) for line in supported_path.read_text().splitlines() if line.strip()]
    supported = [path.parent.name for path in supported_paths]
    if len(supported) != 38 or len(set(supported)) != 38:
        raise ValueError("supported admission list must contain 38 unique decks")
    if set(supported) | set(BLOCKERS) != set(frozen) or set(supported) & set(BLOCKERS):
        raise ValueError("supported and blocked decks must partition Frozen51")

    cases = parse_case_evidence(evidence_path)
    if len(cases) != len(supported) ** 2:
        raise ValueError(f"expected {len(supported) ** 2} ordered cases, found {len(cases)}")
    if [row["ordinal"] for row in cases] != list(range(1, len(cases) + 1)):
        raise ValueError("ordered evidence has missing or duplicated ordinals")
    for row in cases:
        if (
            row.get("passed") is not True
            or int(row.get("state_mismatches", 0)) != 0
            or int(row.get("status_mismatches", 0)) != 0
            or int(row.get("outcome_mismatches", 0)) != 0
            or int(row.get("unfinished_battles", 0)) != 0
        ):
            raise ValueError(f"ordered case {row['ordinal']} is not parity-clean")

    fixture_hashes = {exact_deck_hash(deck) for deck in fixtures.values()}
    frozen_cards = set(card for deck in frozen.values() for card in deck)
    fixture_cards = set(card for deck in fixtures.values() for card in deck)
    overlap_count = sum(exact_deck_hash(deck) in fixture_hashes for deck in frozen.values())
    if overlap_count != 12:
        raise ValueError(f"expected 12 exact prior-fixture matches, found {overlap_count}")

    rule_inventory = None
    if rules_path is not None:
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        rule_inventory = reachable_rules(rules, frozen_cards)
    deck_rows = []
    for deck_id, deck in frozen.items():
        if deck_id in supported:
            status = "supported"
            detail: dict[str, Any] = {
                "evidence": "complete_supported38_ordered_matrix",
                "ordered_opponents": 38,
                "seeds_per_matchup": 1,
            }
            if deck_id == "mega_lopunny_ex_001":
                detail["rl_contract_note"] = (
                    "One mirror seed did not terminate within 2048 decisions without a parity mismatch; "
                    "RL must define a truncation horizon."
                )
        else:
            status = "unsupported"
            detail = resolved_blocker(deck_id)
            own_cards = set(deck)
            for card in detail.get("cards", []):
                if card.get("role") != "opposing_card" and int(card["id"]) not in own_cards:
                    raise ValueError(f"blocker card {card['id']} is absent from {deck_id}")
        deck_rows.append(
            {
                "deck_id": deck_id,
                "status": status,
                "deck_sha256": exact_deck_hash(deck),
                "prior_fixture_exact_match": exact_deck_hash(deck) in fixture_hashes,
                **detail,
            }
        )

    counters = Counter()
    offsets: dict[str, set[int]] = {
        field: set()
        for field in (
            "effect_offsets_reached",
            "effect_offsets_applied",
            "effect_offsets_condition_true",
            "effect_offsets_condition_false",
        )
    }
    for row in cases:
        for field in (
            "decisions_compared", "player0_wins", "player1_wins", "draws",
        ):
            counters[field] += int(row.get(field, 0))
        for field in offsets:
            offsets[field].update(int(value) for value in row.get(field, []))

    return {
        "schema_version": "cuda_frozen51_support_v1",
        "generated_from": {
            "frozen_root": str(frozen_root.relative_to(REPOSITORY_ROOT)),
            "prior_fixture_root": str(fixture_root.relative_to(REPOSITORY_ROOT)),
            "supported_list_sha256": sha256_file(supported_path),
            "ordered_evidence_sha256": sha256_file(evidence_path),
            "rules_sha256": sha256_file(rules_path) if rules_path else None,
        },
        "admission_contract": {
            "oracle": "unmodified official CPU engine",
            "candidate": "OfficialStatePod CUDA runtime",
            "policy": "coverage-first-legal",
            "seed_start": 1,
            "seeds_per_ordered_matchup": 1,
            "decision_limit": 2048,
            "fail_closed": True,
            "cpu_fallback": False,
        },
        "summary": {
            "frozen_decks": len(frozen),
            "supported": len(supported),
            "unsupported": len(BLOCKERS),
            "not_proven": 0,
            "unique_frozen_card_ids": len(frozen_cards),
            "prior_fixture_unique_card_ids": len(fixture_cards),
            "shared_card_ids": len(frozen_cards & fixture_cards),
            "frozen_only_card_ids": len(frozen_cards - fixture_cards),
            "prior_fixture_exact_deck_matches": overlap_count,
            "ordered_matchups": len(cases),
            **dict(counters),
            **{f"unique_{field}": len(values) for field, values in offsets.items()},
        },
        "frozen_only_card_ids": sorted(frozen_cards - fixture_cards),
        "reachable_rule_inventory": rule_inventory,
        "decks": deck_rows,
        "limitations": [
            "Finite deterministic trajectories do not prove every possible card interaction.",
            "This is engine-semantic admission evidence, not policy-strength evidence.",
            "The 0031 chronological observation/event contract is not currently projected from OfficialStatePod on GPU.",
        ],
    }


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = []
    for deck in report["decks"]:
        if deck["status"] == "supported":
            reason = "38x38 official CPU/POD/CUDA ordered parity passed"
        else:
            cards = ", ".join(
                f"{card['name']} ({card['id']})" for card in deck.get("cards", [])
            )
            reason = f"{cards}: {deck['reason']}"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(deck['deck_id'])}</code></td>"
            f"<td><span class=\"{deck['status']}\">{deck['status']}</span></td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{'yes' if deck['prior_fixture_exact_match'] else 'no'}</td>"
            "</tr>"
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen51 CUDA support audit</title>
<style>body{{font:14px/1.5 system-ui;margin:0;background:#f4f7f6;color:#17231f}}main{{max-width:1320px;margin:auto;padding:28px 20px}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:18px 0}}.metric{{background:#fff;border:1px solid #d9e3de;padding:12px}}.metric b{{display:block;font-size:24px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:9px;border:1px solid #d9e3de;text-align:left;vertical-align:top}}.supported{{color:#087443;font-weight:700}}.unsupported{{color:#a12a2a;font-weight:700}}code{{font-size:12px}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main>
<h1>Frozen51 CUDA engine support audit</h1>
<p>Oracle: unmodified official CPU engine. Admission is fail-closed and has no CPU fallback.</p>
<div class="metrics"><div class="metric"><b>{summary['supported']}/51</b>supported decks</div><div class="metric"><b>{summary['unsupported']}</b>unsupported decks</div><div class="metric"><b>{summary['ordered_matchups']:,}</b>ordered parity matchups</div><div class="metric"><b>{summary['decisions_compared']:,}</b>decisions compared</div></div>
<p>The old 40-deck evidence overlaps only {summary['shared_card_ids']} of {summary['unique_frozen_card_ids']} Frozen card IDs and only {summary['prior_fixture_exact_deck_matches']} exact deck multisets. The new 38x38 gate completed with zero state, status, or outcome mismatch.</p>
<p><strong>Important:</strong> this proves the admitted engine subset, not direct 0031 checkpoint compatibility. The 0031 causal observation/event tensors are not yet available from the resident GPU state.</p>
<table><thead><tr><th>Deck</th><th>Status</th><th>Evidence / blocker</th><th>Exact old-fixture deck</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</main></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--supported", type=Path, default=DEFAULT_SUPPORTED)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        args.frozen_root.resolve(), args.fixture_root.resolve(),
        args.supported.resolve(), args.evidence.resolve(),
        rules_path=args.rules.resolve() if args.rules else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
