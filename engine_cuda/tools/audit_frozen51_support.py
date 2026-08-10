from __future__ import annotations

"""Build the fail-closed CUDA support report for the Frozen51 deck pool.

The official CPU engine remains the oracle. This tool does not execute private
card logic; it validates and aggregates evidence produced by the paired
CPU/POD/CUDA runner. Every non-mirror ordered pair must be present and clean.
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
DEFAULT_EVIDENCE = (
    CUDA_ROOT / "artifacts/frozen51_all_ordered_s1.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs/evaluation/combat_mat/0031_latest_frozen_test/cuda_support.json"
)


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


def load_matrix_evidence(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("ordered matrix evidence must be a JSON object")
    return report


def validate_matrix_evidence(
    report: dict[str, Any], deck_ids: Iterable[str]
) -> list[dict[str, Any]]:
    ids = sorted(set(deck_ids))
    expected_pairs = {
        (left, right) for left in ids for right in ids if left != right
    }
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("ordered matrix evidence has no cases list")
    if (
        report.get("passed") is not True
        or report.get("contract")
            != "official_cpu_reference_cuda_ordered_battle_matrix_v1"
        or int(report.get("case_count", -1)) != len(expected_pairs)
        or int(report.get("completed_cases", -1)) != len(expected_pairs)
        or int(report.get("battles_compared", -1)) != len(expected_pairs)
        or report.get("first_failure") is not None
    ):
        raise ValueError("ordered matrix evidence is incomplete or failed")
    for field in (
        "state_mismatches",
        "status_mismatches",
        "outcome_mismatches",
        "unfinished_battles",
    ):
        if int(report.get(field, -1)) != 0:
            raise ValueError(f"ordered matrix evidence has nonzero {field}")

    observed_pairs: list[tuple[str, str]] = []
    for ordinal, row in enumerate(cases, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"ordered case {ordinal} is not an object")
        pair = (str(row.get("deck0_name", "")), str(row.get("deck1_name", "")))
        observed_pairs.append(pair)
        if (
            row.get("passed") is not True
            or int(row.get("state_mismatches", -1)) != 0
            or int(row.get("status_mismatches", -1)) != 0
            or int(row.get("outcome_mismatches", -1)) != 0
            or int(row.get("unfinished_battles", -1)) != 0
        ):
            raise ValueError(f"ordered case {ordinal} is not parity-clean")
    if (
        len(observed_pairs) != len(expected_pairs)
        or set(observed_pairs) != expected_pairs
    ):
        raise ValueError("ordered matrix has missing, duplicated, mirrored, or unknown pairs")
    return cases


def build_report(
    frozen_root: Path,
    fixture_root: Path,
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
    matrix = load_matrix_evidence(evidence_path)
    cases = validate_matrix_evidence(matrix, frozen)

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
        deck_rows.append(
            {
                "deck_id": deck_id,
                "status": "supported",
                "deck_sha256": exact_deck_hash(deck),
                "prior_fixture_exact_match": exact_deck_hash(deck) in fixture_hashes,
                "evidence": "complete_frozen51_non_mirror_ordered_matrix",
                "ordered_opponents": len(frozen) - 1,
                "seeds_per_matchup": int(matrix.get("seeds_per_case", 0)),
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
        "schema_version": "cuda_frozen51_support_v2",
        "generated_from": {
            "frozen_root": frozen_root.relative_to(REPOSITORY_ROOT).as_posix(),
            "prior_fixture_root": fixture_root.relative_to(REPOSITORY_ROOT).as_posix(),
            "ordered_evidence_sha256": sha256_file(evidence_path),
            "ordered_evidence_manifest_sha256": matrix.get("manifest_sha256"),
            "official_source_snapshot": matrix.get("official_source_snapshot"),
            "rules_sha256": sha256_file(rules_path) if rules_path else None,
        },
        "admission_contract": {
            "oracle": "unmodified official CPU engine",
            "candidate": "OfficialStatePod CUDA runtime",
            "policy": "coverage-first-legal",
            "seed_start": int(matrix.get("seed_start", 0)),
            "seeds_per_ordered_matchup": int(matrix.get("seeds_per_case", 0)),
            "decision_limit": 512,
            "pair_mode": "all_non_mirror_ordered",
            "fail_closed": True,
            "cpu_fallback": False,
        },
        "summary": {
            "frozen_decks": len(frozen),
            "supported": len(frozen),
            "unsupported": 0,
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
            "Mirror matchups and additional seeds remain separate regression gates.",
        ],
    }


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = []
    for deck in report["decks"]:
        reason = "50 non-mirror ordered official CPU/POD/CUDA matchups passed"
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
<p>The old 40-deck evidence overlaps {summary['shared_card_ids']} of {summary['unique_frozen_card_ids']} Frozen card IDs and {summary['prior_fixture_exact_deck_matches']} exact deck multisets. The new 51-deck non-mirror ordered gate completed with zero state, status, or outcome mismatch, including official draws.</p>
<p><strong>Important:</strong> this is finite deterministic engine-semantic evidence. It does not prove every seed, policy trajectory, mirror matchup, or model observation contract.</p>
<table><thead><tr><th>Deck</th><th>Status</th><th>Evidence</th><th>Exact old-fixture deck</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</main></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        args.frozen_root.resolve(), args.fixture_root.resolve(),
        args.evidence.resolve(),
        rules_path=args.rules.resolve() if args.rules else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
