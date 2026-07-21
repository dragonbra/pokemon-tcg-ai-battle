from __future__ import annotations

from dataclasses import dataclass


LEFT = 0
STAY = 1
RIGHT = 2
ACTION_DELTAS = (-1, 0, 1)


@dataclass(frozen=True)
class ToyState:
    position: int
    target: int
    step: int


class ToyLineEnv:
    """A tiny legal-action environment for the first behavior-cloning demo."""

    def __init__(self, target: int = 5, max_steps: int = 12):
        self.target = target
        self.max_steps = max_steps
        self.state = ToyState(0, target, 0)

    def reset(self, position: int = 0) -> ToyState:
        if not 0 <= position <= self.target:
            raise ValueError("position must be within the toy line")
        self.state = ToyState(position, self.target, 0)
        return self.state

    def legal_actions(self) -> list[int]:
        legal = []
        for action, delta in enumerate(ACTION_DELTAS):
            next_position = self.state.position + delta
            if 0 <= next_position <= self.target:
                legal.append(action)
        return legal

    def expert_action(self) -> int:
        if self.state.position < self.target:
            return RIGHT
        return STAY

    def step(self, action: int) -> tuple[ToyState, float, bool, dict[str, int]]:
        if action not in self.legal_actions():
            raise ValueError(f"illegal toy action: {action}")
        next_position = self.state.position + ACTION_DELTAS[action]
        next_step = self.state.step + 1
        done = next_position == self.target or next_step >= self.max_steps
        reward = 1.0 if next_position == self.target else -0.01
        self.state = ToyState(next_position, self.target, next_step)
        return self.state, reward, done, {"position": next_position}
