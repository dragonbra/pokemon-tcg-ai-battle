"""Run exhaustive official-protocol Phantom Dive parity for Bench sizes 6..8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.runtime.seeded import build_seeded_runtime

from ..action_boundary.dragapult import StableTargetIdentity
from ..action_boundary.official_parity import (
    ParityScenario,
    exhaustive_scenario_parity,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = (
    ROOT
    / "train/0038_action_boundary_rl/tests/fixtures/"
    "phantom_area_zero_official_v1/scenarios.json"
)


def _deck(path: Path, expected_sha256: str) -> tuple[int, ...]:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"fixture deck hash mismatch for {path}: {actual} != {expected_sha256}"
        )
    cards = tuple(int(value) for value in payload.split())
    if len(cards) != 60:
        raise RuntimeError(f"fixture deck must contain 60 cards: {path}")
    return cards


def run(fixture_path: Path) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    focal = _deck(
        ROOT / fixture["focal_deck_path"], fixture["focal_deck_sha256"]
    )
    opponent = _deck(
        ROOT / fixture["opponent_deck_path"], fixture["opponent_deck_sha256"]
    )
    runtime = build_seeded_runtime()
    results = []
    for raw in fixture["scenarios"]:
        targets = tuple(sorted(
            StableTargetIdentity(**target) for target in raw["targets"]
        ))
        scenario = ParityScenario(
            battle_id=f"official-area-zero-n{raw['n']}",
            seed=int(raw["seed"]),
            search_seed=int(raw["search_seed"]),
            focal_deck=focal,
            opponent_deck=opponent,
            primitive_action_prefix=tuple(
                tuple(int(index) for index in action)
                for action in raw["primitive_action_prefix"]
            ),
            targets=targets,
        )
        results.append(exhaustive_scenario_parity(scenario, runtime.library_path))
    return {
        "schema_version": "0038_gate_b_area_zero_official_exhaustive_v1",
        "fixture_schema_version": fixture["schema_version"],
        "fixture_path": str(fixture_path.relative_to(ROOT)),
        "official_runtime_library": str(runtime.library_path),
        "expected_allocation_counts": {"6": 462, "7": 924, "8": 1716},
        "results": results,
        "total_allocations": sum(item["allocations"] for item in results),
        "parity_failures": sum(item["parity_failures"] for item in results),
        "status": "PASS" if all(item["parity_failures"] == 0 for item in results) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.fixture.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError(f"Gate B parity failures: {report['parity_failures']}")


if __name__ == "__main__":
    main()
