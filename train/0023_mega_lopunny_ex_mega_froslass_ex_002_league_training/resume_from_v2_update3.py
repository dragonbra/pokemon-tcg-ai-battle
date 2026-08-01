"""Resume 0023 League PPO from V2 live update 3 checkpoints.

This module exists so multiprocessing workers can re-import a real main file.
Running the same logic from stdin makes Python's spawn start method fail when it
tries to reload the stdin pseudo-path in child processes.
"""

from __future__ import annotations

from pathlib import Path

from .foundation import verify_foundation
from .training.run import LeagueTrainingConfig
from .training import league_run


PROJECT = "0023_mega_lopunny_ex_mega_froslass_ex_002_league_training"
SOURCE_VERSION = "V2_mega_lopunny_ex_mega_froslass_ex_002_continuous_league"
SOURCE_UPDATE = 3
SOURCE_ROOT = Path("rl_runs") / PROJECT / "versions" / SOURCE_VERSION
SOURCE_LIVE = SOURCE_ROOT / "checkpoint" / "live"


def resolve_from_v2_update3(plugins, *, previous_root=None):
    identity = verify_foundation()
    checkpoints = {}
    audit = {}
    for plugin in plugins:
        checkpoint = SOURCE_LIVE / plugin.deck_id / f"update-{SOURCE_UPDATE:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"missing V2 live checkpoint for {plugin.deck_id}: {checkpoint}"
            )
        checkpoints[plugin.deck_id] = checkpoint
        audit[plugin.deck_id] = {
            "initialization": "inherited_live",
            "source_project": PROJECT,
            "source_version": SOURCE_VERSION,
            "source_update": SOURCE_UPDATE,
            "source_checkpoint": str(checkpoint),
            "deck_sha256": plugin.deck_sha256,
            "foundation_sha256": identity.weights_sha256,
        }
    return checkpoints, audit


def main() -> int:
    league_run.resolve_initial_checkpoint_map = resolve_from_v2_update3
    league_run.PREVIOUS_VERSION = SOURCE_VERSION
    league_run.PREVIOUS_COMPLETE_UPDATE = SOURCE_UPDATE
    config = LeagueTrainingConfig(
        version="V6_from_v2_update3_20h_gpu_resume",
        device="cuda:0",
        workers=128,
        coalesce_ms=5.0,
        games_per_update=512,
        frozen_eval_interval=5,
    )
    return league_run.run_league_training(config, max_updates=None)


if __name__ == "__main__":
    raise SystemExit(main())
