from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

from ..rollout.protocol import Episode


def _wilson(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


class OutcomeTracker:
    def __init__(self, maximum_window: int = 2_000) -> None:
        self._outcomes: deque[float] = deque(maxlen=maximum_window)
        self._seats: deque[tuple[bool, float]] = deque(maxlen=maximum_window)
        self._opponents: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=maximum_window)
        )
        self._cumulative = Counter()

    def add(self, episode: Episode) -> None:
        if not episode.valid or episode.reward is None:
            self._cumulative["discarded"] += 1
            self._cumulative[f"status/{episode.status}"] += 1
            return
        outcome = float(episode.reward)
        self._outcomes.append(outcome)
        self._seats.append((episode.candidate_first, outcome))
        self._opponents[episode.opponent].append(outcome)
        self._cumulative["episodes"] += 1
        self._cumulative["wins" if outcome > 0 else "losses" if outcome < 0 else "draws"] += 1

    def metrics(self) -> dict[str, float]:
        result: dict[str, float] = {
            "rollout/cumulative/episodes": float(self._cumulative["episodes"]),
            "rollout/cumulative/wins": float(self._cumulative["wins"]),
            "rollout/cumulative/losses": float(self._cumulative["losses"]),
            "rollout/cumulative/draws": float(self._cumulative["draws"]),
            "rollout/discarded": float(self._cumulative["discarded"]),
        }
        for window in (100, 500, 2_000):
            values = list(self._outcomes)[-window:]
            wins = sum(value > 0 for value in values)
            lower, upper = _wilson(wins, len(values))
            prefix = f"rollout/rolling_{window}"
            result.update(
                {
                    f"{prefix}/games": float(len(values)),
                    f"{prefix}/wins": float(wins),
                    f"{prefix}/losses": float(sum(value < 0 for value in values)),
                    f"{prefix}/draws": float(sum(value == 0 for value in values)),
                    f"{prefix}/win_rate": wins / len(values) if values else 0.0,
                    f"{prefix}/reward_mean": sum(values) / len(values) if values else 0.0,
                    f"{prefix}/wilson_low": lower,
                    f"{prefix}/wilson_high": upper,
                }
            )
        for first in (True, False):
            values = [outcome for seat, outcome in self._seats if seat == first]
            wins = sum(value > 0 for value in values)
            name = "first" if first else "second"
            result[f"rollout/seat/{name}/games"] = float(len(values))
            result[f"rollout/seat/{name}/win_rate"] = wins / len(values) if values else 0.0
        for opponent, outcomes in sorted(self._opponents.items()):
            values = list(outcomes)[-500:]
            result[f"rollout/opponent/{opponent}/games"] = float(len(values))
            result[f"rollout/opponent/{opponent}/win_rate"] = (
                sum(value > 0 for value in values) / len(values) if values else 0.0
            )
        return result


def episode_metrics(episodes: list[Episode]) -> dict[str, float]:
    valid = [episode for episode in episodes if episode.valid and episode.reward is not None]
    attempts = len(episodes)
    decisions = sum(len(episode.decisions) for episode in valid)
    metrics = {
        "rollout/batch/attempts": float(attempts),
        "rollout/batch/valid_episodes": float(len(valid)),
        "rollout/batch/discarded": float(attempts - len(valid)),
        "rollout/batch/error_rate": (attempts - len(valid)) / attempts if attempts else 0.0,
        "rollout/batch/decisions": float(decisions),
        "rollout/batch/mean_decisions": decisions / len(valid) if valid else 0.0,
        "rollout/batch/mean_rounds": (
            sum(episode.complete_rounds or 0 for episode in valid) / len(valid)
            if valid
            else 0.0
        ),
    }
    latencies = sorted(
        decision.inference_ms
        for episode in valid
        for decision in episode.decisions
    )
    if latencies:
        metrics["system/rollout/inference_p50_ms"] = latencies[len(latencies) // 2]
        metrics["system/rollout/inference_p95_ms"] = latencies[
            min(len(latencies) - 1, int(0.95 * len(latencies)))
        ]
    diagnostic_keys = sorted(
        {key for episode in valid for key in episode.diagnostics}
    )
    for key in diagnostic_keys:
        metrics[f"rollout/dragapult/{key}"] = sum(
            episode.diagnostics.get(key, 0.0) for episode in valid
        ) / max(len(valid), 1)
    return metrics


__all__ = ["OutcomeTracker", "episode_metrics"]
