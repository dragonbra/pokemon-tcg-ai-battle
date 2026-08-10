"""Rollout/backend scaling contract independent of trajectory schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

OptimizationMode = Literal["fixed_epochs", "fixed_optimizer_budget"]


@dataclass(frozen=True, slots=True)
class RolloutScalingConfig:
    engine_backend: str = "official"
    rollout_games_per_update: int = 512
    rollout_worker_count: int = 16
    envs_per_worker: int = 8
    inference_batch_size: int = 64
    max_inflight_requests: int = 256
    rollout_queue_depth: int = 512
    seed_shard_count: int = 8
    ppo_minibatch_size: int = 256
    ppo_gradient_accumulation: int = 1
    ppo_epochs: int = 4
    optimizer_steps_per_update: int = 32
    optimization_mode: OptimizationMode = "fixed_optimizer_budget"

    def validate(self) -> None:
        integer_fields = (
            "rollout_games_per_update", "rollout_worker_count", "envs_per_worker",
            "inference_batch_size", "max_inflight_requests", "rollout_queue_depth",
            "seed_shard_count", "ppo_minibatch_size", "ppo_gradient_accumulation",
            "ppo_epochs", "optimizer_steps_per_update",
        )
        if any(getattr(self, name) <= 0 for name in integer_fields):
            raise ValueError("rollout and PPO capacity fields must be positive")
        if self.optimization_mode not in {"fixed_epochs", "fixed_optimizer_budget"}:
            raise ValueError("invalid optimization mode")
        if self.engine_backend != "official" and not self.engine_backend.startswith("accelerated:"):
            raise ValueError("non-official backends must use accelerated:<version>")

    @property
    def effective_minibatch_size(self) -> int:
        return self.ppo_minibatch_size * self.ppo_gradient_accumulation

    def metadata(self, *, policy_transitions: int, optimizer_steps: int) -> dict[str, object]:
        self.validate()
        if policy_transitions < 0 or optimizer_steps < 0:
            raise ValueError("runtime counts cannot be negative")
        return {**asdict(self), "effective_minibatch_size": self.effective_minibatch_size,
                "policy_transitions": policy_transitions,
                "actual_optimizer_steps": optimizer_steps}


class EngineBackend:
    """Backend boundary; implementations receive the same official Agent contract."""

    name: str
    version: str

    def run_game(self, job, agent_factory):  # pragma: no cover - interface only
        raise NotImplementedError


def assert_backend_parity(report: dict[str, bool]) -> None:
    required = {
        "observation", "legal_action", "reward_terminal_winner", "rng_seed",
        "complete_game", "action_boundary_macro",
    }
    missing = required - report.keys()
    failed = {name for name in required if not report.get(name, False)}
    if missing or failed:
        raise RuntimeError(f"accelerated backend parity gate failed: missing={missing}, failed={failed}")


__all__ = ["EngineBackend", "OptimizationMode", "RolloutScalingConfig", "assert_backend_parity"]
