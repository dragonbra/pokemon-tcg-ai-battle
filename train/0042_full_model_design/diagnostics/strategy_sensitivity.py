"""Counterfactual V/Meta sensitivity for a trained 0042 Strategy Adapter."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path

import torch

from ..initialization import build_update0_model
from ..policy.strategy_adapters import StrategyContext
from ..training.storage_full_semantic import load_adapted_model_only


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY = ROOT / (
    "data/replays/0016_alakazam_multideck_bc/goonew/"
    "submission-54960905/episode-88310066-replay.json"
)
DEFAULT_DECK = ROOT / "train/0042_full_model_design/league/decks/007_dragapult_ex/deck.csv"


def _features(replay: Path, deck: tuple[int, ...]):
    compiler_module = importlib.import_module(
        "train.0042_full_model_design.rollout.worker_compiler"
    )
    collate_module = importlib.import_module(
        "train.0042_full_model_design.semantic_policy.features.collate"
    )
    raw = json.loads(replay.read_text(encoding="utf-8"))
    observation = next(
        agent["observation"]
        for step in raw["steps"]
        for agent in step
        if isinstance((agent.get("observation") or {}).get("current"), dict)
        and agent["observation"]["current"].get("yourIndex") == 0
        and (agent["observation"].get("select") or {}).get("option")
    )
    compiler = compiler_module.WorkerLocalCompiler(0, deck)
    return collate_module.collate_canonical_records([compiler.compile(observation)])


def _score(model, batch, state, options, context: StrategyContext) -> torch.Tensor:
    decoder_state = model.actor.action_decoder.initialize(batch, state.summary)
    return model.head.logits(batch, options, decoder_state, context).float()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference_log = reference.log_softmax(dim=-1)
    candidate_log = candidate.log_softmax(dim=-1)
    probability = reference_log.exp()
    kl = (probability * (reference_log - candidate_log)).sum(dim=-1)
    entropy = -(candidate_log.exp() * candidate_log).sum(dim=-1)
    width = min(5, candidate.shape[-1])
    return {
        "kl_from_real": float(kl[0]),
        "entropy": float(entropy[0]),
        "top1": int(candidate.argmax(dim=-1)[0]),
        "topk": [int(value) for value in candidate.topk(width, dim=-1).indices[0]],
    }


def run(checkpoint: Path, replay: Path = DEFAULT_REPLAY) -> dict[str, object]:
    deck = tuple(int(value) for value in DEFAULT_DECK.read_text().splitlines() if value)
    model, _ = build_update0_model(deck)
    load_adapted_model_only(model, checkpoint)
    model.eval()
    features = _features(replay, deck)
    with torch.inference_mode():
        batch, state, options, value, auxiliary, real = model.encode_with_strategy(features)
        reference = _score(model, batch, state, options, real)
        value_rows = {}
        for candidate in (-1.0, -0.5, 0.0, 0.5, 1.0):
            context = replace(real, value=torch.full_like(real.value, candidate))
            value_rows[f"{candidate:+.1f}"] = _comparison(
                reference, _score(model, batch, state, options, context)
            )
        uniform = torch.full_like(real.meta_probabilities, 1.0 / 15.0)
        alternative = torch.zeros_like(real.meta_probabilities)
        alternative[:, (real.meta_probabilities.argmax(dim=-1) + 1) % 15] = 1.0
        meta_cases = {
            "real": real,
            "uniform": replace(real, meta_probabilities=uniform),
            "alternative_onehot": replace(real, meta_probabilities=alternative),
            "zero_z_meta": replace(real, z_meta=torch.zeros_like(real.z_meta)),
        }
        meta_rows = {
            name: _comparison(reference, _score(model, batch, state, options, context))
            for name, context in meta_cases.items()
        }
    return {
        "schema_version": "0042_strategy_sensitivity_v1",
        "checkpoint": str(checkpoint),
        "replay": str(replay),
        "observed_value": float(value[0]),
        "value_sensitivity": value_rows,
        "meta_sensitivity": meta_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.checkpoint, args.replay)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
