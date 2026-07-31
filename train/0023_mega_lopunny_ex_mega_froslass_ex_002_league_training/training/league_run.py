"""Continuous focal-primary multi-decoder League PPO runner for 0023."""
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
from rl_environment.runs import project_version_paths, write_version_status

from .. import FOCAL_DECK_ID, PROJECT_ID
from ..decks import DeckPlugin, load_deck_plugins
from ..decoder import load_decoder_checkpoint, save_decoder_checkpoint
from ..foundation import verify_foundation
from ..foundation.contract import EXPECTED_WEIGHTS_SHA256
from ..league import DEFAULT_DECK_ROOT, initialize_league_version
from ..policy import LeaguePolicyPool
from ..rollout import LeaguePolicyView, LeagueRolloutCollector, RolloutJob
from ..storage import preflight_storage, runtime_storage
from .batch import prepare_episodes
from .ppo import PPOConfig, PPOTrainer, frozen_reference
from .run import (
    LeagueTrainingConfig,
    StopRequest,
    _atomic_json,
    _episode_metrics,
    _runtime_root,
    schedule_jobs,
)


PREVIOUS_VERSION = "V11_multidecoder_league_20h"
PREVIOUS_ROOT = Path("rl_runs/0022_league_training/versions") / PREVIOUS_VERSION
PREVIOUS_COMPLETE_UPDATE = 81
ACCEPTED_0022_FOCAL_UPDATE = 75


def resolve_initial_checkpoint_map(
    plugins: tuple[DeckPlugin, ...], *, previous_root: Path = PREVIOUS_ROOT
) -> tuple[dict[str, Path | None], dict[str, dict[str, object]]]:
    """Resolve each prior Live deck to its last complete asset, else Foundation."""
    status_path = previous_root / "artifact/status.json"
    catalog_path = previous_root / "artifact/league_catalog.json"
    if not status_path.is_file() or not catalog_path.is_file():
        raise FileNotFoundError(f"previous League provenance is incomplete: {previous_root}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    last_complete = int(status.get("last_complete_update", -1))
    if last_complete != PREVIOUS_COMPLETE_UPDATE:
        raise ValueError(
            f"previous League last complete update changed: {last_complete} "
            f"!= {PREVIOUS_COMPLETE_UPDATE}"
        )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = catalog.get("decks")
    if not isinstance(records, list):
        raise ValueError("previous League catalog has no deck list")
    prior = {
        str(record["deck_id"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("deck_id"), str)
    }
    current = {plugin.deck_id: plugin for plugin in plugins}
    missing_current = sorted(set(prior) - set(current))
    if missing_current:
        raise ValueError(f"previous Live decks disappeared from 0023: {missing_current}")

    checkpoints: dict[str, Path | None] = {}
    audit: dict[str, dict[str, object]] = {}
    for plugin in plugins:
        prior_record = prior.get(plugin.deck_id)
        if prior_record is None:
            checkpoints[plugin.deck_id] = None
            audit[plugin.deck_id] = {
                "initialization": "foundation",
                "foundation_sha256": None,
                "reason": "deck_absent_from_0022_v11_catalog",
            }
            continue
        prior_deck_sha = str(prior_record.get("deck_sha256"))
        if prior_deck_sha != plugin.deck_sha256:
            raise ValueError(
                f"previous Live exact deck changed for {plugin.deck_id}: "
                f"{prior_deck_sha} != {plugin.deck_sha256}"
            )
        checkpoint = (
            previous_root / "checkpoint/live" / plugin.deck_id
            / f"update-{last_complete:06d}.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"previous catalog member lacks update-{last_complete}: {plugin.deck_id}"
            )
        checkpoint_audit = load_decoder_checkpoint(
            checkpoint,
            expected_foundation_sha256=str(
                status.get("foundation_sha256") or EXPECTED_WEIGHTS_SHA256
            ),
            expected_deck_id=plugin.deck_id,
            expected_deck_sha256=plugin.deck_sha256,
        )
        if checkpoint_audit.update != last_complete:
            raise ValueError(f"checkpoint update mismatch for {plugin.deck_id}")
        checkpoints[plugin.deck_id] = checkpoint
        audit[plugin.deck_id] = {
            "initialization": "inherited_live",
            "source_project": "0022_league_training",
            "source_version": PREVIOUS_VERSION,
            "source_update": checkpoint_audit.update,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_audit.checkpoint_sha256,
            "deck_sha256": checkpoint_audit.deck_sha256,
            "foundation_sha256": checkpoint_audit.foundation_sha256,
        }
    return checkpoints, audit


def _materialized_checkpoint_map(initialized: dict[str, object]) -> dict[str, str]:
    records = initialized.get("checkpoints")
    if not isinstance(records, dict):
        raise ValueError("initialized version has no checkpoint catalog")
    result: dict[str, str] = {}
    for deck_id, raw in records.items():
        if not isinstance(raw, dict) or not raw.get("materialized"):
            raise ValueError(f"Live initialization was not materialized: {deck_id}")
        result[str(deck_id)] = str(raw["decoder_ref"])
    return result


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
    root: Path,
    *,
    update: int,
    keep_latest: int,
    snapshot_interval: int,
    keep_gate_snapshots: int,
) -> None:
    """Bound every deck to latest checkpoints plus recent gate snapshots."""
    for deck_dir in root.iterdir():
        if not deck_dir.is_dir():
            continue
        checkpoints: list[tuple[int, Path]] = []
        for checkpoint in deck_dir.glob("update-*.pt"):
            try:
                checkpoint_update = int(checkpoint.stem.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append((checkpoint_update, checkpoint))
        checkpoints.sort(reverse=True)
        protected = {value for value, _ in checkpoints[:keep_latest]}
        gate_updates = [
            value for value, _ in checkpoints if value % snapshot_interval == 0
        ][:keep_gate_snapshots]
        protected.update(gate_updates)
        protected.add(update)
        for checkpoint_update, checkpoint in checkpoints:
            if checkpoint_update in protected:
                continue
            checkpoint.unlink(missing_ok=True)
            checkpoint.with_suffix(checkpoint.suffix + ".sha256").unlink(missing_ok=True)


def run_league_training(
    config: LeagueTrainingConfig, *, deck_root: Path = DEFAULT_DECK_ROOT,
    max_updates: int | None = None,
) -> int:
    if (
        config.games_per_update < 1
        or config.frozen_eval_interval < 1
        or config.workers < 1
        or config.coalesce_ms < 0
        or config.checkpoint_keep_latest < 1
        or config.checkpoint_keep_gate_snapshots < 1
    ):
        raise ValueError("League counts and checkpoint retention must be positive")
    plugins = load_deck_plugins(deck_root)
    if len(plugins) != 50:
        raise ValueError(f"League requires exactly 50 deck plugins, got {len(plugins)}")
    focal = next(plugin for plugin in plugins if plugin.focal)
    if focal.deck_id != FOCAL_DECK_ID:
        raise ValueError(f"0023 focal deck must be {FOCAL_DECK_ID}, got {focal.deck_id}")
    identity = verify_foundation()
    initial_sources, initialization_audit = resolve_initial_checkpoint_map(plugins)
    for record in initialization_audit.values():
        if record["initialization"] == "foundation":
            record["foundation_sha256"] = identity.weights_sha256
    prospective = project_version_paths(PROJECT_ID, config.version)
    preflight_storage(prospective.run_root.parent, minimum_free_gib=config.launch_minimum_free_gib)
    paths = initialize_league_version(
        config.version,
        deck_root=deck_root,
        initial_checkpoints=initial_sources,
    )
    runtime_root = _runtime_root()
    initialized = json.loads(paths.config.read_text(encoding="utf-8"))
    initialization_audit_path = paths.artifact / "initialization_audit.json"
    inherited_count = sum(
        record["initialization"] == "inherited_live"
        for record in initialization_audit.values()
    )
    foundation_count = len(initialization_audit) - inherited_count
    _atomic_json(initialization_audit_path, {
        "schema_version": "0023_live_initialization_audit_v1",
        "previous_project": "0022_league_training",
        "previous_version": PREVIOUS_VERSION,
        "previous_complete_update": PREVIOUS_COMPLETE_UPDATE,
        "accepted_0022_focal_update": ACCEPTED_0022_FOCAL_UPDATE,
        "focal_deck_id": FOCAL_DECK_ID,
        "inherited_live_count": inherited_count,
        "foundation_initialized_count": foundation_count,
        "decks": initialization_audit,
    })
    _atomic_json(paths.config, {
        **initialized,
        "schema_version": "0023_multidecoder_league_ppo_v1",
        "run": asdict(config),
        "continuous_until_explicit_stop": True,
        "focal_deck_id": FOCAL_DECK_ID,
        "opponent_views": ["frozen", "live"],
        "live_opponent_updates_enabled": True,
        "live_deck_update_contract": "all_50_isolated_decoder_value_optimizers",
        "reference_kl_contract": "fixed_to_each_deck_0023_initialization",
        "initialization_audit": str(initialization_audit_path),
        "initialization_summary": {
            "inherited_live_count": inherited_count,
            "foundation_initialized_count": foundation_count,
            "previous_complete_update": PREVIOUS_COMPLETE_UPDATE,
        },
        "probe_deck_ids": [
            "alakazam_dudunsparce_001",
            "marnies_grimmsnarl_ex_froslass_001",
        ],
        "runtime_root": str(runtime_root),
        "ppo": asdict(PPOConfig()),
    })
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA League training requested but CUDA is unavailable")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    checkpoints = _materialized_checkpoint_map(initialized)
    pool = LeaguePolicyPool.from_foundation(
        plugins, device=device, decoder_checkpoints=checkpoints,
        foundation_sha256=identity.weights_sha256,
    )
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
        "WANDB_TAGS": "0023,league,multidecoder,mega_lopunny,mega_froslass,continuous",
    })
    started = time.time()
    update = 0
    stop_reason = "explicit_stop"
    stop = StopRequest()
    stop.install()
    write_version_status(paths, {
        "state": "running", "training_started": True,
        "foundation_verified": True,
        "focal_deck_id": FOCAL_DECK_ID,
        "continuous_until_explicit_stop": True,
        "previous_league_version": PREVIOUS_VERSION,
        "previous_complete_update": PREVIOUS_COMPLETE_UPDATE,
        "inherited_live_count": inherited_count,
        "foundation_initialized_count": foundation_count,
        "deck_count": len(plugins),
    })
    try:
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            while not stop.requested and (max_updates is None or update < max_updates):
                disk = runtime_storage(
                    paths.run_root,
                    stop_free_gib=config.runtime_stop_free_gib,
                    version_cap_gib=config.version_cap_gib,
                )
                if disk.stop_requested:
                    stop_reason = "disk_guard"
                    break
                collector = LeagueRolloutCollector(
                    focal_model, device=device, workers=config.workers, mode="sample",
                    coalesce_ms=config.coalesce_ms, policy_pool=pool,
                )
                jobs = schedule_jobs(
                    plugins, count=config.games_per_update, update=update,
                    seed=config.seed, runtime_root=runtime_root,
                )
                episodes = collector.collect(jobs)
                if any(not episode.valid for episode in episodes):
                    raise RuntimeError("0023 League rollout contains official-engine errors")
                deck_metrics: dict[str, float] = {}
                decisions_by_deck = Counter(
                    decision.policy_deck_id
                    for episode in episodes if episode.valid
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
                        batch = prepare_episodes(
                            episodes, policy_deck_id=plugin.deck_id,
                            gamma=1.0, gae_lambda=0.95,
                        )
                    except ValueError as error:
                        deck_metrics[f"ppo/deck/{plugin.deck_id}/skipped"] = 1.0
                        deck_metrics[
                            f"ppo/deck/{plugin.deck_id}/skip_reason/{_skip_reason(error)}"
                        ] = 1.0
                        continue
                    metrics = trainers[plugin.deck_id].update(batch)
                    deck_metrics[f"rollout/deck/{plugin.deck_id}/decisions"] = float(
                        decisions_by_deck[plugin.deck_id]
                    )
                    for key, value in metrics.items():
                        deck_metrics[f"{key}/{plugin.deck_id}"] = value
                    checkpoint = (
                        paths.checkpoints / "live" / plugin.deck_id
                        / f"update-{update + 1:06d}.pt"
                    )
                    save_decoder_checkpoint(
                        checkpoint,
                        pool.policy(plugin.deck_id).actor,
                        pool.policy(plugin.deck_id).value_head,
                        foundation_sha256=identity.weights_sha256,
                        deck_id=plugin.deck_id,
                        deck_sha256=plugin.deck_sha256,
                        policy_role="live",
                        policy_version=config.version,
                        update=update + 1,
                    )
                _prune_live_checkpoints(
                    paths.checkpoints / "live",
                    update=update + 1,
                    keep_latest=config.checkpoint_keep_latest,
                    snapshot_interval=config.frozen_eval_interval,
                    keep_gate_snapshots=config.checkpoint_keep_gate_snapshots,
                )
                update += 1
                log_record = {
                    "trainer/update": update,
                    "rollout/source_policy_update": update - 1,
                    "checkpoint/update": update,
                    "rollout/league_episodes": float(len(episodes)),
                    "rollout/league_finished": float(sum(episode.valid for episode in episodes)),
                    "run/elapsed_hours": (time.time() - started) / 3600,
                    **_episode_metrics(episodes, "rollout/league"),
                    **_episode_metrics(episodes, "rollout/focal"),
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
                    _, frozen_metrics = _eval_view(
                        eval_collector, plugins, LeaguePolicyView.FROZEN,
                        update, config, runtime_root,
                    )
                    _, live_metrics = _eval_view(
                        eval_collector, plugins, LeaguePolicyView.LIVE,
                        update, config, runtime_root,
                    )
                    _, probe_metrics = _probe_metrics(
                        eval_collector, plugins, focal, update, config, runtime_root
                    )
                    log_record.update({
                        "eval/checkpoint_update": float(update),
                        "eval/frozen/gate_pass": 1.0,
                        "eval/live/gate_pass": 1.0,
                        **frozen_metrics, **live_metrics, **probe_metrics,
                    })
                logger.log(update, log_record)
                write_version_status(paths, {
                    "state": "running",
                    "checkpoint_update": update,
                    "rollout_source_policy_update": update - 1,
                    "live_decks_seen": int(deck_metrics["rollout/live_decks_seen"]),
                    "elapsed_hours": (time.time() - started) / 3600,
                    "disk_free_gib": disk.free_bytes / (1024 ** 3),
                })
        if stop.requested:
            stop_reason = "stop_requested"
        elif max_updates is not None and update >= max_updates:
            stop_reason = "max_updates"
        summary = {
            "state": f"completed_{stop_reason}",
            "updates": update,
            "elapsed_hours": (time.time() - started) / 3600,
            "stop_signal": stop.signal_number,
            "foundation_sha256": identity.weights_sha256,
            "focal_deck_id": FOCAL_DECK_ID,
            "previous_league_version": PREVIOUS_VERSION,
            "previous_complete_update": PREVIOUS_COMPLETE_UPDATE,
            "deck_count": len(plugins),
        }
        _atomic_json(paths.summary, summary)
        write_version_status(paths, summary)
        return 0
    except BaseException as error:
        write_version_status(paths, {
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
            "checkpoint_update": update,
            "elapsed_hours": (time.time() - started) / 3600,
        })
        raise
    finally:
        stop.restore()


__all__ = ["resolve_initial_checkpoint_map", "run_league_training"]
