"""Multi-decoder League PPO runner for 0022 V3 and later versions."""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch

from rl_environment.logging import TrainingLogger
from rl_environment.runs import initialize_version, project_version_paths, write_version_status

from .. import PROJECT_ID
from ..decks import DeckPlugin, load_deck_plugins
from ..decoder import save_decoder_checkpoint
from ..foundation import verify_foundation
from ..league import DEFAULT_DECK_ROOT, initialize_league_version
from ..policy import LeaguePolicyPool
from ..rollout import LeaguePolicyView, LeagueRolloutCollector, RolloutJob
from ..storage import preflight_storage, runtime_storage
from .batch import prepare_episodes
from .ppo import PPOConfig, PPOTrainer, frozen_reference
from .run import LeagueTrainingConfig, _atomic_json, _episode_metrics, _runtime_root, schedule_jobs


V2_ROOT = Path("rl_runs/0022_league_training/versions/V2_dragapult_focal_20h_w128")
UPDATE39 = V2_ROOT / "checkpoint/selected/update-000039.pt"


def _checkpoint_map(plugins: tuple[DeckPlugin, ...]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for plugin in plugins:
        if plugin.focal:
            paths[plugin.deck_id] = str(UPDATE39)
        else:
            paths[plugin.deck_id] = str(V2_ROOT / "checkpoint/decks" / f"{plugin.deck_id}.pt")
    return paths


def _jobs_with_view(
    plugins: tuple[DeckPlugin, ...], *, view: LeaguePolicyView, update: int,
    config: LeagueTrainingConfig, runtime_root: Path,
) -> list:
    jobs = schedule_jobs(plugins, count=0, update=update, seed=config.seed, runtime_root=runtime_root, evaluation=True)
    return [replace(job, opponent_view=view) for job in jobs]


def _eval_view(
    collector: LeagueRolloutCollector, plugins: tuple[DeckPlugin, ...], view: LeaguePolicyView,
    update: int, config: LeagueTrainingConfig, runtime_root: Path,
) -> tuple[list, dict[str, float]]:
    episodes = collector.collect(_jobs_with_view(plugins, view=view, update=update, config=config, runtime_root=runtime_root))
    metrics = _episode_metrics(episodes, f"eval/{view}")
    by_opponent: dict[str, list] = {}
    for episode in episodes:
        by_opponent.setdefault(episode.job.opponent_deck_id, []).append(episode)
    for deck_id, deck_episodes in sorted(by_opponent.items()):
        metrics.update(_episode_metrics(deck_episodes, f"eval/{view}/opponent/{deck_id}"))
    expected = len(plugins) * 2
    if len(episodes) != expected or metrics[f"eval/{view}/errors"] != 0:
        raise RuntimeError(
            f"{view} greedy evaluation gate failed: expected {expected} finished games, "
            f"got {len(episodes)} with {metrics[f'eval/{view}/errors']} errors"
        )
    return episodes, metrics


def _matchup_jobs(
    focal: DeckPlugin, opponent: DeckPlugin, *, view: LeaguePolicyView,
    update: int, config: LeagueTrainingConfig, runtime_root: Path, tag: str,
) -> list[RolloutJob]:
    """Create a balanced two-game probe with each turn order represented once."""
    return [
        RolloutJob(
            game_id=f"probe-u{update:06d}-{tag}-{index}",
            focal_deck_id=focal.deck_id, opponent_deck_id=opponent.deck_id,
            opponent_view=view, focal_first=index == 0,
            seed=config.seed + update * 100_000 + 70_000 + index,
            source_policy_update=update, focal_deck=focal.deck,
            opponent_deck=opponent.deck, runtime_root=runtime_root,
        )
        for index in range(2)
    ]


def _probe_metrics(
    collector: LeagueRolloutCollector, plugins: tuple[DeckPlugin, ...],
    focal: DeckPlugin, update: int, config: LeagueTrainingConfig,
    runtime_root: Path,
) -> tuple[list, dict[str, float]]:
    """Record focal-vs-probe Frozen/Live and probe-Live-vs-focal-Live curves."""
    probe_ids = ("alakazam_dudunsparce_001", "marnies_grimmsnarl_ex_froslass_001")
    by_id = {plugin.deck_id: plugin for plugin in plugins}
    all_episodes: list = []
    metrics: dict[str, float] = {}
    for probe_id in probe_ids:
        probe = by_id[probe_id]
        for view in (LeaguePolicyView.FROZEN, LeaguePolicyView.LIVE):
            jobs = _matchup_jobs(
                focal, probe, view=view, update=update, config=config,
                runtime_root=runtime_root, tag=f"focal-vs-{probe_id}-{view}",
            )
            episodes = collector.collect(jobs)
            all_episodes.extend(episodes)
            metrics.update(_episode_metrics(episodes, f"eval/probe/{probe_id}/focal_vs_{view}"))
            if any(not episode.valid for episode in episodes):
                raise RuntimeError(f"probe evaluation failed for {probe_id} focal_vs_{view}")

        reverse = _matchup_jobs(
            probe, focal, view=LeaguePolicyView.LIVE, update=update,
            config=config, runtime_root=runtime_root, tag=f"{probe_id}-vs-focal-live",
        )
        episodes = collector.collect(reverse)
        all_episodes.extend(episodes)
        metrics.update(_episode_metrics(episodes, f"eval/probe/{probe_id}/live_vs_focal_live"))
        if any(not episode.valid for episode in episodes):
            raise RuntimeError(f"probe evaluation failed for {probe_id} live_vs_focal_live")
    return all_episodes, metrics


def _skip_reason(error: Exception) -> str:
    text = str(error)
    if text.startswith("no decisions for policy deck"):
        return "no_decisions"
    if "mixes source_policy_update" in text:
        return "mixed_source_update"
    if text.startswith("no valid terminal episodes"):
        return "no_valid_episode"
    return "other"


def _prune_live_checkpoints(
    root: Path, *, update: int, keep_latest: int, snapshot_interval: int,
) -> None:
    """Bound per-deck history while retaining every gate snapshot."""
    protected_updates = {update}
    protected_updates.update(
        value for value in range(1, update + 1)
        if value <= keep_latest or value % snapshot_interval == 0
    )
    for deck_dir in root.iterdir():
        if not deck_dir.is_dir():
            continue
        for checkpoint in deck_dir.glob("update-*.pt"):
            try:
                checkpoint_update = int(checkpoint.stem.split("-")[-1])
            except ValueError:
                continue
            if checkpoint_update in protected_updates:
                continue
            checkpoint.unlink(missing_ok=True)
            checkpoint.with_suffix(checkpoint.suffix + ".sha256").unlink(missing_ok=True)


def run_league_training(
    config: LeagueTrainingConfig, *, deck_root: Path = DEFAULT_DECK_ROOT,
    max_updates: int | None = None,
) -> int:
    plugins = load_deck_plugins(deck_root)
    if len(plugins) != 48:
        raise ValueError(f"V3 League requires exactly 48 deck plugins, got {len(plugins)}")
    identity = verify_foundation()
    if not UPDATE39.is_file():
        raise FileNotFoundError(f"protected update-39 checkpoint is missing: {UPDATE39}")
    prospective = project_version_paths(PROJECT_ID, config.version)
    preflight_storage(prospective.run_root.parent, minimum_free_gib=config.launch_minimum_free_gib)
    paths = initialize_league_version(config.version, deck_root=deck_root)
    runtime_root = _runtime_root()
    initialized = json.loads(paths.config.read_text(encoding="utf-8"))
    _atomic_json(paths.config, {
        **initialized,
        "schema_version": "0022_multidecoder_league_ppo_v2",
        "run": asdict(config),
        "focal_deck_id": "dragapult_ex_001",
        "opponent_views": ["frozen", "live"],
        "live_opponent_updates_enabled": True,
        "live_deck_update_contract": "all_48_isolated_decoder_value_optimizers",
        "probe_deck_ids": [
            "alakazam_dudunsparce_001",
            "marnies_grimmsnarl_ex_froslass_001",
        ],
        "runtime_root": str(runtime_root),
        "ppo": asdict(PPOConfig()),
    })
    device = torch.device(config.device)
    checkpoints = _checkpoint_map(plugins)
    pool = LeaguePolicyPool.from_foundation(
        plugins, device=device, decoder_checkpoints=checkpoints,
        foundation_sha256=identity.weights_sha256,
    )
    focal = next(plugin for plugin in plugins if plugin.focal)
    focal_model = pool.policy(focal.deck_id)
    trainers = {
        plugin.deck_id: PPOTrainer(
            pool.policy(plugin.deck_id),
            frozen_reference(pool.policy(plugin.deck_id), device),
            device=device,
            config=PPOConfig(),
        )
        for plugin in plugins
    }
    os.environ.update({
        "WANDB_MODE": "online", "WANDB_ENTITY": "dragon_bra",
        "WANDB_PROJECT": "pokemon-tcg-policy-learning", "WANDB_JOB_TYPE": "ppo_league",
        "WANDB_TAGS": "0022,league,multidecoder,update39_branch",
    })
    started = time.time()
    deadline = started + config.duration_hours * 3600
    update = 0
    stop_reason = "time_budget"
    write_version_status(paths, {
        "state": "running", "training_started": True,
        "foundation_verified": True, "branch_checkpoint": str(UPDATE39),
        "deck_count": len(plugins),
    })
    with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
        while time.time() < deadline and (max_updates is None or update < max_updates):
            disk = runtime_storage(paths.run_root, stop_free_gib=config.runtime_stop_free_gib, version_cap_gib=config.version_cap_gib)
            if disk.stop_requested:
                stop_reason = "disk_low_water"
                break
            collector = LeagueRolloutCollector(
                focal_model, device=device, workers=config.workers, mode="sample",
                coalesce_ms=config.coalesce_ms, policy_pool=pool,
            )
            jobs = schedule_jobs(plugins, count=config.games_per_update, update=update, seed=config.seed, runtime_root=runtime_root)
            episodes = collector.collect(jobs)
            if any(not episode.valid for episode in episodes):
                raise RuntimeError("V3 League rollout contains official-engine errors")
            deck_metrics: dict[str, float] = {}
            decisions_by_deck = Counter(
                decision.policy_deck_id
                for episode in episodes
                if episode.valid
                for decision in episode.decisions
            )
            missing_live = [
                plugin.deck_id for plugin in plugins
                if decisions_by_deck.get(plugin.deck_id, 0) == 0
            ]
            deck_metrics["rollout/live_decks_seen"] = float(len(plugins) - len(missing_live))
            deck_metrics["rollout/live_decks_missing"] = float(len(missing_live))
            if missing_live:
                raise RuntimeError(
                    f"League update {update} did not collect Live decisions for decks: "
                    f"{missing_live}"
                )
            for plugin in plugins:
                try:
                    batch = prepare_episodes(episodes, policy_deck_id=plugin.deck_id, gamma=1.0, gae_lambda=0.95)
                except ValueError as error:
                    deck_metrics[f"ppo/deck/{plugin.deck_id}/skipped"] = 1.0
                    deck_metrics[f"ppo/deck/{plugin.deck_id}/skip_reason/{_skip_reason(error)}"] = 1.0
                    continue
                metrics = trainers[plugin.deck_id].update(batch)
                for key, value in metrics.items():
                    deck_metrics[f"{key}/{plugin.deck_id}"] = value
                checkpoint = paths.checkpoints / "live" / plugin.deck_id / f"update-{update + 1:06d}.pt"
                save_decoder_checkpoint(
                    checkpoint, pool.policy(plugin.deck_id).actor, pool.policy(plugin.deck_id).value_head,
                    foundation_sha256=identity.weights_sha256, deck_id=plugin.deck_id,
                    deck_sha256=plugin.deck_sha256, policy_role="live", policy_version=config.version,
                    update=update + 1,
                )
            _prune_live_checkpoints(
                paths.checkpoints / "live", update=update + 1,
                keep_latest=config.checkpoint_keep_latest,
                snapshot_interval=config.frozen_eval_interval,
            )
            update += 1
            log_record = {
                "trainer/update": update,
                "rollout/source_policy_update": update - 1,
                "rollout/league_episodes": float(len(episodes)),
                "rollout/league_finished": float(sum(episode.valid for episode in episodes)),
                **_episode_metrics(episodes, "rollout/league"),
                **collector.metrics(), **deck_metrics, **disk.metrics(),
            }
            if device.type == "cuda":
                log_record.update({
                    "system/gpu/allocated_bytes": float(torch.cuda.memory_allocated(device)),
                    "system/gpu/reserved_bytes": float(torch.cuda.memory_reserved(device)),
                    "system/gpu/peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
                })
            if update % config.frozen_eval_interval == 0:
                eval_collector = LeagueRolloutCollector(
                    focal_model, device=device, workers=config.workers, mode="greedy",
                    coalesce_ms=config.coalesce_ms, policy_pool=pool,
                )
                _, frozen_metrics = _eval_view(eval_collector, plugins, LeaguePolicyView.FROZEN, update, config, runtime_root)
                _, live_metrics = _eval_view(eval_collector, plugins, LeaguePolicyView.LIVE, update, config, runtime_root)
                _, probe_metrics = _probe_metrics(
                    eval_collector, plugins, focal, update, config, runtime_root
                )
                # Keep all evaluation metrics in the same canonical record as the
                # update. This preserves the policy-update/evaluation association.
                log_record.update({
                    "trainer/update": update,
                    "eval/checkpoint_update": float(update),
                    "eval/frozen/gate_pass": 1.0,
                    "eval/live/gate_pass": 1.0,
                    **frozen_metrics, **live_metrics, **probe_metrics,
                })
            logger.log(update, log_record)
            write_version_status(paths, {
                "state": "running", "checkpoint_update": update,
                "rollout_source_policy_update": update - 1,
                "live_decks_seen": int(deck_metrics["rollout/live_decks_seen"]),
                "disk_free_gib": disk.free_bytes / (1024 ** 3),
            })
    summary = {
        "state": f"completed_{stop_reason}", "updates": update,
        "elapsed_hours": (time.time() - started) / 3600,
        "foundation_sha256": identity.weights_sha256,
        "branch_checkpoint": str(UPDATE39), "deck_count": len(plugins),
    }
    (paths.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_version_status(paths, summary)
    return 0


__all__ = ["run_league_training"]
