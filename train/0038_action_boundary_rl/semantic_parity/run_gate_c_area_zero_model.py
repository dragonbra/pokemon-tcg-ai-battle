"""Compare training and packaged intent on real n=6..8 Area Zero snapshots."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import threading

import torch
from evaluation.runtime.seeded import build_seeded_runtime, load_seeded_library

from engine_cuda.tools.run_official_semantic0031_v2_parity_scaffold import (
    compare_training_to_package_chunks,
    load_training_actor_critic,
)
from ..action_boundary.decision_gate import DecisionClass, DecisionGate
from ..action_boundary.dragapult import PHANTOM_DIVE_ATTACK_ID
from ..rollout.pool_worker import _PointerBattle
from ..rollout.worker_compiler import WorkerLocalCompiler


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = (
    ROOT
    / "train/0038_action_boundary_rl/tests/fixtures/"
    "phantom_area_zero_official_v1/scenarios.json"
)


def _cards(path: Path) -> tuple[int, ...]:
    cards = tuple(int(value) for value in path.read_text().split())
    if len(cards) != 60:
        raise ValueError(f"deck must have 60 cards: {path}")
    return cards


def _is_phantom_root(observation, action) -> bool:
    options = (observation.get("select") or {}).get("option") or []
    return any(
        0 <= index < len(options)
        and isinstance(options[index], dict)
        and options[index].get("attackId") == PHANTOM_DIVE_ATTACK_ID
        for index in action
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    focal = _cards(ROOT / fixture["focal_deck_path"])
    opponent = _cards(ROOT / fixture["opponent_deck_path"])

    package_root = args.package.resolve()
    sys.path.insert(0, str(package_root))
    compound = importlib.import_module("strategy.deployment.compound_inference")
    collate = importlib.import_module(
        "strategy.features.collate"
    ).collate_canonical_records
    device = torch.device(args.device)
    package_policy = compound.PortableCompoundSemanticPolicy.from_checkpoint(
        package_root / "strategy/model.bin", focal
    )
    package_policy.actor.to(device).eval()
    package_policy.value_head.to(device).eval()
    package_policy.allocation_head.to(device).eval()
    package_policy.meta_head.to(device).eval()
    package_policy.meta_conditioner.to(device).eval()
    training_model = load_training_actor_critic(
        args.checkpoint.resolve(), focal, device
    )

    library = load_seeded_library(build_seeded_runtime().library_path)
    lock = threading.Lock()
    gate = DecisionGate()
    records = []
    option_counts = []
    snapshots = []
    for scenario in fixture["scenarios"]:
        compiler = WorkerLocalCompiler(0, focal)
        pending_callbacks = 0
        battle = _PointerBattle(library, lock, [])
        try:
            observation = battle.start(
                focal,
                opponent,
                int(scenario["seed"]),
                int(scenario["search_seed"]),
            )
            for action in scenario["primitive_action_prefix"][:-1]:
                actor = int((observation.get("current") or {}).get("yourIndex", -1))
                if actor == 0:
                    if pending_callbacks:
                        compiler.observe_only(observation)
                        pending_callbacks -= 1
                    else:
                        decision = gate.classify(observation)
                        if decision.classification in {
                            DecisionClass.FORCED, DecisionClass.LEGAL_EMPTY_PASS,
                        }:
                            compiler.observe_only(observation)
                        elif decision.classification is DecisionClass.STRATEGIC:
                            compiler.compile(observation)
                        else:
                            raise RuntimeError(
                                f"unexpected focal gate while replaying fixture: {decision}"
                            )
                    if _is_phantom_root(observation, action):
                        pending_callbacks = 6
                observation = battle.select(list(action))
            if int(observation["current"]["yourIndex"]) != 0:
                raise RuntimeError("Area Zero fixture does not end at a focal root")
            record = compiler.compile(observation)
            records.append(record)
            option_counts.append(len(observation["select"]["option"]))
            snapshots.append({
                "n": int(scenario["n"]),
                "seed": int(scenario["seed"]),
                "turn": int(observation["current"]["turn"]),
                "root_index": int(scenario["root_index"]),
            })
        finally:
            battle.finish()

    comparison = compare_training_to_package_chunks(
        training_model,
        package_policy,
        collate,
        records,
        option_counts,
        chunk_size=1,
        logits_atol=5e-3,
        logits_rtol=5e-3,
        value_atol=2e-3,
    )
    report = {
        "schema_version": "0038_gate_c_area_zero_model_parity_v1",
        "snapshots": snapshots,
        "comparison": comparison,
        "status": "PASS" if comparison["passed"] else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("Area Zero training/package intent parity failed")


if __name__ == "__main__":
    main()
