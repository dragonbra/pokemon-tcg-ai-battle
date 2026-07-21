from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rl.core.checkpoint import CheckpointManager
from rl.core.losses import masked_cross_entropy
from rl.core.logging import TrainingLogger
from rl.core.model import CandidatePolicyValueNet, ModelConfig
from rl.demo.toy_env import ToyLineEnv


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "rl" / "runs" / "toy_behavior_cloning"


@dataclass(frozen=True)
class ToySample:
    position: int
    target: int
    legal_actions: tuple[int, ...]
    expert_action: int


def collect_samples(count: int, target: int, seed: int) -> list[ToySample]:
    rng = random.Random(seed)
    env = ToyLineEnv(target=target)
    samples = []
    for _ in range(count):
        env.reset(position=rng.randint(0, target))
        samples.append(
            ToySample(
                position=env.state.position,
                target=env.state.target,
                legal_actions=tuple(env.legal_actions()),
                expert_action=env.expert_action(),
            )
        )
    return samples


def _batch(samples: list[ToySample], config: ModelConfig) -> dict[str, Tensor]:
    state_numeric = []
    state_card_ids = []
    action_type_ids = []
    action_card_ids = []
    action_target_ids = []
    action_numeric = []
    action_mask = []
    targets = []

    for sample in samples:
        state_numeric.append([sample.position / sample.target, sample.target / 10.0])
        state_card_ids.append([0])
        candidate_types = []
        candidate_cards = []
        candidate_targets = []
        candidate_numeric = []
        candidate_mask = []
        action_to_index = {action: index for index, action in enumerate(sample.legal_actions)}
        for index in range(config.max_candidates):
            if index < len(sample.legal_actions):
                action = sample.legal_actions[index]
                candidate_types.append(action + 1)
                candidate_cards.append(0)
                candidate_targets.append(0)
                candidate_numeric.append([action / 2.0, 0.0])
                candidate_mask.append(True)
            else:
                candidate_types.append(0)
                candidate_cards.append(0)
                candidate_targets.append(0)
                candidate_numeric.append([0.0, 0.0])
                candidate_mask.append(False)
        action_type_ids.append(candidate_types)
        action_card_ids.append(candidate_cards)
        action_target_ids.append(candidate_targets)
        action_numeric.append(candidate_numeric)
        action_mask.append(candidate_mask)
        targets.append(action_to_index[sample.expert_action])

    return {
        "state_numeric": torch.tensor(state_numeric, dtype=torch.float32),
        "state_card_ids": torch.tensor(state_card_ids, dtype=torch.long),
        "action_type_ids": torch.tensor(action_type_ids, dtype=torch.long),
        "action_card_ids": torch.tensor(action_card_ids, dtype=torch.long),
        "action_target_ids": torch.tensor(action_target_ids, dtype=torch.long),
        "action_numeric": torch.tensor(action_numeric, dtype=torch.float32),
        "action_mask": torch.tensor(action_mask, dtype=torch.bool),
        "target": torch.tensor(targets, dtype=torch.long),
    }


def _model_input(batch: dict[str, Tensor]) -> tuple[Tensor, ...]:
    return tuple(
        batch[key]
        for key in (
            "state_numeric",
            "state_card_ids",
            "action_type_ids",
            "action_card_ids",
            "action_target_ids",
            "action_numeric",
            "action_mask",
        )
    )


def evaluate_model(
    model: CandidatePolicyValueNet,
    config: ModelConfig,
    target: int,
) -> dict[str, float]:
    """Evaluate the learned policy on every toy starting position."""
    model.eval()
    successes = 0
    illegal_actions = 0
    env = ToyLineEnv(target=target)
    with torch.no_grad():
        for position in range(target + 1):
            env.reset(position=position)
            done = False
            while not done:
                sample = ToySample(
                    position=env.state.position,
                    target=env.state.target,
                    legal_actions=tuple(env.legal_actions()),
                    expert_action=env.expert_action(),
                )
                batch = _batch([sample], config)
                _, logits = model(*_model_input(batch))
                candidate_index = int(logits.argmax(dim=-1).item())
                action = sample.legal_actions[candidate_index]
                if action not in env.legal_actions():
                    illegal_actions += 1
                    break
                _, _, done, _ = env.step(action)
            if env.state.position == target:
                successes += 1
    total = target + 1
    return {
        "toy_eval/success_rate": successes / total,
        "toy_eval/illegal_action_rate": illegal_actions / total,
    }


def train(
    *,
    output_dir: Path,
    epochs: int = 40,
    sample_count: int = 512,
    seed: int = 7,
) -> dict[str, float]:
    torch.manual_seed(seed)
    config = ModelConfig(
        state_numeric_dim=2,
        state_token_count=1,
        candidate_numeric_dim=2,
        max_candidates=3,
        card_vocab_size=4,
        action_type_vocab_size=3,
        d_model=32,
        hidden_dim=64,
        num_heads=2,
    )
    model = CandidatePolicyValueNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    manager = CheckpointManager(output_dir / "checkpoints")
    batch = _batch(collect_samples(sample_count, target=5, seed=seed), config)

    with TrainingLogger(output_dir / "metrics.jsonl", output_dir / "tensorboard") as logger:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            value, logits = model(*_model_input(batch))
            loss = masked_cross_entropy(logits, batch["target"], batch["action_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                prediction = logits.argmax(dim=-1)
                accuracy = (prediction == batch["target"]).float().mean().item()
                legal = batch["action_mask"].gather(1, prediction.unsqueeze(1)).squeeze(1)
                legal_rate = legal.float().mean().item()
            logger.log(
                epoch,
                {
                    "train/bc_loss": loss.item(),
                    "train/action_accuracy": accuracy,
                    "train/legal_action_rate": legal_rate,
                },
            )

    eval_metrics = evaluate_model(model, config, target=5)
    manager.save(
        "toy_behavior_clone",
        model,
        optimizer=optimizer,
        step=epochs,
        metadata={
            "model_config": config.to_dict(),
            "seed": seed,
            "task": "toy_behavior_clone",
            **eval_metrics,
        },
    )
    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "legal_action_rate": legal_rate,
        **eval_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train(
        output_dir=args.output,
        epochs=args.epochs,
        sample_count=args.samples,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
