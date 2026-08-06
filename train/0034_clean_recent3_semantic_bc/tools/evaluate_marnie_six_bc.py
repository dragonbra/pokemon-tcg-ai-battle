"""Evaluate a packaged 0034 policy on Marnie against the fixed six-BC pool."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_deck(path: Path) -> list[int]:
    return [
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _deck_manifest(deck: list[int]) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(deck)
    return {"counts": [[card, int(counts[card])] for card in sorted(counts)]}


def _runner_class(base_module, package_dir: Path):
    class Semantic0034Runner(base_module.Semantic0033Runner):
        def __init__(self, deck: list[int], *, device: str = "cpu") -> None:
            import torch

            self.torch = torch
            self.device = torch.device(device)
            contract = json.loads(
                (package_dir / "model_contract.json").read_text(encoding="utf-8")
            )
            source_root = package_dir / "model_source"
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            package = contract["package_name"]
            self.compile_canonical_row = importlib.import_module(
                f"{package}.features.compiler"
            ).compile_canonical_row
            self.collate_canonical_records = importlib.import_module(
                f"{package}.features.collate"
            ).collate_canonical_records
            self.CausalKnowledge = importlib.import_module(
                f"{package}.knowledge.state"
            ).CausalKnowledge
            prototypes_module = importlib.import_module(
                f"{package}.domain.prototypes"
            )
            loader = _load_module(
                package_dir / "load_model.py",
                "ptcg_0034_local_eval_loader",
            )
            self.policy, loaded_contract = loader.load_model(
                package_dir,
                device=self.device,
            )
            if loaded_contract != contract:
                raise ValueError("package loader contract drift")
            self.policy.eval().requires_grad_(False)
            self.prototypes = prototypes_module.PrototypeIndex.load(
                package_dir / "official_public_prototypes_v1.json",
                package_dir / "official_full_engine_prototypes_v2.json",
            )
            self.model_contract = {
                "architecture": contract["architecture"],
                "actor_schema": contract["actor_schema"],
                "parameter_count": contract["parameter_count"],
                "random_initialization": contract["random_initialization"],
                "model_config": contract["model_config"],
            }
            self.deck = list(deck)
            self.deck_manifest = _deck_manifest(deck)
            self.knowledge_by_actor: dict[int, Any] = {}
            self.errors: list[dict[str, Any]] = []
            self.decisions = 0
            self.fallback_decisions = 0

    return Semantic0034Runner


def run(args: argparse.Namespace) -> dict[str, Any]:
    engine_root = args.engine_root.resolve()
    package_dir = args.package.resolve()
    accepted_harness = engine_root / ".tmp" / "evaluate_0033_marnie_cpu.py"
    if not accepted_harness.is_file():
        raise FileNotFoundError(
            "accepted six-BC harness is missing; expected " + str(accepted_harness)
        )
    for name in (
        "model_contract.json",
        "load_model.py",
        "best_model.pt",
        "official_public_prototypes_v1.json",
        "official_full_engine_prototypes_v2.json",
    ):
        if not (package_dir / name).is_file():
            raise FileNotFoundError(f"model package is missing {name}")

    base = _load_module(accepted_harness, "ptcg_0034_accepted_six_bc_harness")
    base.ROOT = engine_root
    base.POOL_PATH = (
        engine_root
        / "cuda_engine"
        / "configs"
        / "strongest_bc_opponent_pool_v1.json"
    )
    base.MODEL_PATH = package_dir / "best_model.pt"
    base.Semantic0033Runner = _runner_class(base, package_dir)
    report = base.run(args)
    report.update(
        {
            "schema_version": "0034_marnie_six_bc_eval_v1",
            "candidate": "0034_clean_recent3_semantic_decision_v1_marnie_deck",
            "model_path": str(package_dir / "best_model.pt"),
            "model_package": str(package_dir),
            "engine": (
                "tools.evaluate_candidates CPU official sample runner with "
                "packaged 0034 SemanticPolicy inference on requested device"
            ),
            "evaluation_protocol": (
                "experiments/0034_clean_recent3_semantic_bc/"
                "evaluation_protocol.json"
            ),
        }
    )
    expected_games = args.games_per_opponent * len(report["by_opponent"])
    if report["totals"]["games"] != expected_games:
        raise ValueError("evaluation game-count mismatch")
    if report["totals"]["errors"]:
        raise RuntimeError("evaluation contains engine/runtime errors")
    if any(item["fallback_decisions"] for item in report["by_opponent"]):
        raise RuntimeError("0034 policy used fallback actions during evaluation")
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=Path(r"D:\Users\admin\Desktop\pokemon"),
    )
    parser.add_argument("--games-per-opponent", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=2026080501)
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opponent")
    parser.add_argument("--candidate-seat-offset", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fixed-candidate-seat", type=int, choices=(0, 1))
    parser.add_argument(
        "--candidate-start",
        choices=("model", "first", "second", "balanced"),
        default="model",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games_per_opponent <= 0 or args.max_steps <= 0:
        raise ValueError("positive game and step counts are required")
    args.output = args.output.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
