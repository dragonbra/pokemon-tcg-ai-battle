"""Versioned 2,048-game Frozen panel and paired statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable

from evaluation.frozen_0806_contract import (
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    evaluation_coin_winner,
)

PANEL_VERSION = "0038_frozen_2048_agent_first_player_v3"
SHARDS = 8
GAMES_PER_SHARD = 256


@dataclass(frozen=True, slots=True)
class FrozenGame:
    frozen_panel_version: str
    seed: int
    shard_id: int
    opponent_deck: str
    opponent_archetype: str
    focal_won_toss: bool
    first_player_contract: str
    engine_backend: str
    engine_version: str
    action_protocol: str
    greedy: bool = True


def build_panel(
    opponents: list[tuple[str, str]],
    *,
    seed_base: int = 38_204_800,
    focal_identity: str = "focal",
) -> list[FrozenGame]:
    if not opponents:
        raise ValueError("Frozen panel requires opponents")
    rows: list[FrozenGame] = []
    total = SHARDS * GAMES_PER_SHARD
    for index in range(total):
        shard = index // GAMES_PER_SHARD
        deck, archetype = opponents[index % len(opponents)]
        rows.append(FrozenGame(
            PANEL_VERSION, seed_base + index, shard, deck, archetype,
            focal_won_toss=evaluation_coin_winner(
                evaluation_seed=seed_base,
                focal_identity=focal_identity,
                opponent_identity=deck,
                slot=index % GAMES_PER_SHARD,
                replica=shard,
            ),
            first_player_contract=FROZEN_0806_FIRST_PLAYER_CONTRACT,
            engine_backend="official",
            engine_version="official_runtime_current", action_protocol="0038_macro_v1",
        ))
    validate_panel(rows)
    return rows


def validate_panel(rows: list[FrozenGame]) -> None:
    if len(rows) != SHARDS * GAMES_PER_SHARD:
        raise ValueError("Frozen panel must contain exactly 2,048 games")
    if len({row.seed for row in rows}) != len(rows):
        raise ValueError("Frozen panel seeds must be globally unique")
    shard_counts = Counter(row.shard_id for row in rows)
    if shard_counts != Counter({i: GAMES_PER_SHARD for i in range(SHARDS)}):
        raise ValueError("Frozen panel must contain eight complete 256-game shards")
    contracts = {(row.frozen_panel_version, row.engine_backend, row.engine_version,
                  row.action_protocol, row.greedy, row.first_player_contract) for row in rows}
    if len(contracts) != 1:
        raise ValueError("Frozen panel deterministic contract drift")


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("invalid Wilson counts")
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return centre - radius, centre + radius


def paired_summary(baseline: dict[int, int], checkpoint: dict[int, int]) -> dict[str, object]:
    if baseline.keys() != checkpoint.keys():
        raise ValueError("paired results require identical Frozen seeds")
    loss_to_win = sum(baseline[s] <= 0 and checkpoint[s] > 0 for s in baseline)
    win_to_loss = sum(baseline[s] > 0 and checkpoint[s] <= 0 for s in baseline)
    discordant = loss_to_win + win_to_loss
    chi_square = 0.0 if discordant == 0 else (abs(loss_to_win - win_to_loss) - 1) ** 2 / discordant
    wins = sum(value > 0 for value in checkpoint.values())
    return {"games": len(checkpoint), "wins": wins, "win_rate": wins / len(checkpoint),
            "wilson_95": wilson_interval(wins, len(checkpoint)),
            "baseline_loss_to_checkpoint_win": loss_to_win,
            "baseline_win_to_checkpoint_loss": win_to_loss,
            "mcnemar_continuity_corrected_chi2": chi_square}


__all__ = [
    "FrozenGame", "GAMES_PER_SHARD", "PANEL_VERSION", "SHARDS", "build_panel",
    "paired_summary", "validate_panel", "wilson_interval",
]
