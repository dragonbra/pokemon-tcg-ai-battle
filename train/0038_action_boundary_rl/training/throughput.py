"""Per-update in-memory timing aggregation; no per-request disk logging."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGES = (
    "environment_engine", "observation_compile", "model_inference",
    "trajectory_assembly", "gae", "ppo_forward_backward", "logging_checkpoint",
)


@dataclass(slots=True)
class UpdateThroughput:
    games: int = 0
    official_selects: int = 0
    strategic_decisions: int = 0
    policy_transitions: int = 0
    ppo_samples: int = 0
    seconds: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in STAGES})

    def add_stage(self, name: str, elapsed: float) -> None:
        if name not in self.seconds or elapsed < 0:
            raise ValueError("invalid throughput stage")
        self.seconds[name] += elapsed

    def metrics(self) -> dict[str, float]:
        rollout = sum(self.seconds[name] for name in (
            "environment_engine", "observation_compile", "model_inference",
            "trajectory_assembly", "gae",
        ))
        total = rollout + self.seconds["ppo_forward_backward"] + self.seconds["logging_checkpoint"]
        denominator = max(rollout, 1e-12)
        return {
            **{f"throughput/seconds/{name}": value for name, value in self.seconds.items()},
            "throughput/seconds/total": total,
            "throughput/games_per_sec": self.games / denominator,
            "throughput/official_selects_per_sec": self.official_selects / denominator,
            "throughput/strategic_decisions_per_sec": self.strategic_decisions / denominator,
            "throughput/policy_transitions_per_sec": self.policy_transitions / denominator,
            "throughput/ppo_samples_per_sec": (
                self.ppo_samples / max(self.seconds["ppo_forward_backward"], 1e-12)
            ),
        }


__all__ = ["STAGES", "UpdateThroughput"]
