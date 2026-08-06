"""Formal full-0031 decoder PPO over official CPU engine Frozen51 games."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any

import torch

from rl_environment.logging import TrainingLogger

from ..league import load_frozen_catalog
from ..parity import assert_pt0805_runtime_parity, collect_official_observations
from ..policy import load_actor_critic
from ..rollout import FullSemanticRolloutCollector, RolloutJob
from .batch_full_semantic import prepare_episodes
from .ppo_full_semantic import PPOConfig, PPOTrainer
from .storage_full_semantic import save_model_only


ROOT = Path(__file__).resolve().parents[3]
PROJECT = "0033_dragapult_third_ptcg_club_rl"
WANDB_DISPLAY_PREFIX = "0033 · dragapult_third_ptcg_club_rl"
FORMAL_VERSION = "V6_turn_clock_lambda097_20u"
SOURCE_CHECKPOINT = ROOT / "archive/pretrained/0031_friend_pt0805_epoch13_best_validation_loss/model.pt"
CANDIDATE_ROOT = ROOT / "evaluation/arena/candidates/0033_dragapult_third_ptcg_club_zero_shot"
FOCAL_DECK_PATH = ROOT / "train" / PROJECT / "league/decks/dragapult_third_ptcg_club/deck.csv"


@dataclass(frozen=True, slots=True)
class RunConfig:
    version: str = FORMAL_VERSION
    updates: int = 200
    workers: int = 24
    coalesce_ms: float = 5.0
    device: str = "cuda:0"
    seed: int = 330031001
    games_per_update: int = 204
    eval_every: int = 10
    wandb_mode: str = "online"
    ppo: PPOConfig = PPOConfig()

    def validate(self) -> None:
        if re.fullmatch(r"V[1-9]\d*_[a-z0-9]+(?:_[a-z0-9]+)*", self.version) is None:
            raise ValueError("version must match V<n>_<ascii_snake_case>")
        if min(self.updates, self.workers, self.eval_every) < 1 or self.coalesce_ms < 0:
            raise ValueError("updates, workers, and eval_every must be positive")
        if self.games_per_update not in {204, 408}:
            raise ValueError("games_per_update must be 204 or 408 balanced Frozen51 games")
        if self.wandb_mode not in {"online", "offline"}:
            raise ValueError("invalid W&B mode")
        self.ppo.validate()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _merge_status(path: Path, payload: dict[str, Any]) -> None:
    existing = json.loads(path.read_text()) if path.is_file() else {}
    _atomic_json(path, {**existing, **payload})


def _paths(version: str) -> dict[str, Path]:
    root = ROOT / "rl_runs" / PROJECT / "versions" / version
    return {
        "root": root,
        "artifact": root / "artifact",
        "checkpoint": root / "checkpoint",
        "tensorboard": root / "tensorboard",
        "wandb": root / "wandb",
    }


def assert_fresh_version(version: str) -> dict[str, Path]:
    paths = _paths(version)
    occupied = [str(paths[name]) for name in ("artifact", "checkpoint", "tensorboard", "wandb") if paths[name].exists()]
    if occupied:
        raise FileExistsError("full-semantic version already used: " + ", ".join(occupied))
    target_number = int(version.split("_", 1)[0][1:])
    existing_numbers = [
        int(path.name.split("_", 1)[0][1:])
        for path in paths["root"].parent.glob("V[1-9]*_*")
        if re.fullmatch(r"V[1-9]\d*_[a-z0-9]+(?:_[a-z0-9]+)*", path.name)
    ]
    if existing_numbers and target_number <= max(existing_numbers):
        raise ValueError(
            f"version number must be greater than existing V{max(existing_numbers)}"
        )
    return paths


def focal_deck() -> tuple[int, ...]:
    cards = tuple(int(line) for line in FOCAL_DECK_PATH.read_text().splitlines())
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError("0033 focal deck is not exact 60")
    return cards


def runtime_root() -> Path:
    roots = sorted((ROOT / "evaluation/arena/opponents").glob("*/cg/game.py"))
    if not roots:
        raise FileNotFoundError("official CPU engine runtime is unavailable")
    return roots[0].parents[1].resolve()


def build_jobs(
    *,
    source_policy_update: int,
    seed: int,
    count: int = 102,
    greedy: bool = False,
) -> list[RolloutJob]:
    catalog = load_frozen_catalog()
    if count < 2 or count % 2:
        raise ValueError("job count must be positive and seat-balanced")
    full_repeats = count // 102 if count >= 102 and count % 102 == 0 else 0
    selected = catalog if full_repeats else catalog[: count // 2]
    jobs: list[RolloutJob] = []
    deck = focal_deck()
    root = runtime_root()
    for repeat in range(max(1, full_repeats)):
        for opponent in selected:
            for focal_first in (True, False):
                index = len(jobs)
                jobs.append(
                    RolloutJob(
                    game_id=("eval" if greedy else "rollout")
                    + f"-u{source_policy_update:04d}-{index:04d}",
                    opponent_id=opponent.deck_id,
                    focal_first=focal_first,
                    seed=seed + source_policy_update * 1_000_003 + repeat * 100_003 + index,
                    source_policy_update=source_policy_update,
                    focal_deck=deck,
                    opponent_deck=opponent.deck,
                    runtime_root=root,
                    )
                )
    if full_repeats and len({job.opponent_id for job in jobs}) != 51:
        raise RuntimeError("formal schedule omitted a Frozen opponent")
    if len(jobs) != count:
        raise RuntimeError(f"schedule produced {len(jobs)} jobs instead of {count}")
    return jobs


def _episode_metrics(episodes: list[Any], prefix: str) -> dict[str, float]:
    invalid = [episode for episode in episodes if not episode.valid]
    if invalid:
        raise RuntimeError(
            f"official CPU engine errors ({len(invalid)}): "
            + repr([episode.error for episode in invalid[:5]])
        )
    games = len(episodes)
    wins = sum(episode.reward == 1.0 for episode in episodes)
    losses = sum(episode.reward == -1.0 for episode in episodes)
    draws = sum(episode.reward == 0.0 for episode in episodes)
    first = [episode for episode in episodes if episode.job.focal_first]
    second = [episode for episode in episodes if not episode.job.focal_first]
    return {
        f"{prefix}/episodes": float(games),
        f"{prefix}/wins": float(wins),
        f"{prefix}/losses": float(losses),
        f"{prefix}/draws": float(draws),
        f"{prefix}/win_rate": wins / games,
        f"{prefix}/first_win_rate": sum(ep.reward == 1.0 for ep in first) / len(first),
        f"{prefix}/second_win_rate": sum(ep.reward == 1.0 for ep in second) / len(second),
        f"{prefix}/decisions": float(sum(len(ep.decisions) for ep in episodes)),
        f"{prefix}/engine_selections": float(
            sum(int(ep.diagnostics.get("engine_selections", 0)) for ep in episodes)
        ),
        f"{prefix}/mean_engine_turn": sum(ep.turns for ep in episodes) / games,
        f"{prefix}/error_games": 0.0,
    }


def _opponent_snapshot() -> dict[str, Any]:
    catalog = load_frozen_catalog()
    return {
        "schema": "0033_full_frozen0019_cpu_snapshot_v1",
        "opponent_count": len(catalog),
        "seat_contract": "every_opponent_twice_each_seat_per_update",
        "policy": "0019_epoch13_immutable_foundation",
        "policy_sha256": "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb",
        "opponents": [
            {"deck_id": item.deck_id, "deck_sha256": item.deck_sha256}
            for item in catalog
        ],
    }


def _run_parity(model, output: Path) -> dict[str, Any]:
    catalog = load_frozen_catalog()
    observations = collect_official_observations(
        runtime_root(),
        focal_deck(),
        catalog[0].deck,
        focal_index=0,
        decisions=8,
    )
    return assert_pt0805_runtime_parity(
        observations=observations,
        actor_index=0,
        deck=focal_deck(),
        checkpoint=SOURCE_CHECKPOINT,
        candidate_root=CANDIDATE_ROOT,
        model=model,
        output=output,
    )


def run_gate(
    *,
    output: Path,
    games: int,
    workers: int,
    run_ppo: bool,
    device_name: str = "cuda:0",
    mode: str = "sample",
    coalesce_ms: float = 5.0,
    gae_lambda: float = 1.0,
    credit_clock: str = "selection",
    loss_weighting: str = "episode_equal_decisions",
) -> dict[str, Any]:
    device = torch.device(device_name)
    model, identity = load_actor_critic(SOURCE_CHECKPOINT, focal_deck(), device)
    parity = _run_parity(model, output.parent / "pt0805_runtime_parity.json")
    representation = model.representation_sha256()
    decoder_before = model.decoder_sha256()
    collector = FullSemanticRolloutCollector(
        model, device=device, workers=workers, mode=mode, coalesce_ms=coalesce_ms
    )
    started = time.perf_counter()
    episodes = collector.collect(
        build_jobs(source_policy_update=0, seed=330033001, count=games)
    )
    rollout_seconds = time.perf_counter() - started
    metrics = _episode_metrics(episodes, "rollout")
    metrics.update(collector.metrics())
    ppo_metrics: dict[str, float] = {}
    ppo_seconds = 0.0
    if run_ppo:
        ppo_config = PPOConfig(
            gae_lambda=gae_lambda,
            credit_clock=credit_clock,
            loss_weighting=loss_weighting,
        )
        batch = prepare_episodes(
            episodes,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            credit_clock=ppo_config.credit_clock,
            loss_weighting=ppo_config.loss_weighting,
        )
        trainer = PPOTrainer(model, device=device, config=ppo_config)
        ppo_started = time.perf_counter()
        ppo_metrics = trainer.update(batch)
        ppo_seconds = time.perf_counter() - ppo_started
    report = {
        "schema": "0033_full_semantic_official_cpu_gate_v1",
        "passed": True,
        "games": games,
        "all_51_both_seats": games >= 102 and games % 102 == 0,
        "ppo_update": run_ppo,
        "credit_clock": credit_clock,
        "gae_lambda": gae_lambda,
        "loss_weighting": loss_weighting,
        "rollout_seconds": rollout_seconds,
        "ppo_seconds": ppo_seconds,
        "source_identity": asdict(identity),
        "parity": parity,
        "representation_sha256": representation,
        "representation_unchanged": model.representation_sha256() == representation,
        "decoder_sha256_before": decoder_before,
        "decoder_sha256_after": model.decoder_sha256(),
        "decoder_changed": model.decoder_sha256() != decoder_before,
        "metrics": {**metrics, **ppo_metrics},
        "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if run_ppo and not report["decoder_changed"]:
        raise RuntimeError("PPO gate did not change the original PT0805 decoder")
    if not report["representation_unchanged"]:
        raise RuntimeError("PPO gate changed the original PT0805 representation")
    _atomic_json(output, report)
    return report


def run(config: RunConfig) -> dict[str, Any]:
    config.validate()
    paths = assert_fresh_version(config.version)
    for name in ("artifact", "checkpoint", "tensorboard", "wandb"):
        paths[name].mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference/training requested but unavailable")
    run_id = "0033-" + config.version.lower().replace("_", "-")
    wandb_url = f"https://wandb.ai/dragon_bra/pokemon-tcg-policy-learning/runs/{run_id}"
    os.environ.update(
        {
            "WANDB_MODE": config.wandb_mode,
            "WANDB_ENTITY": "dragon_bra",
            "WANDB_PROJECT": "pokemon-tcg-policy-learning",
            "WANDB_JOB_TYPE": "ppo_decoder_only",
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": f"{WANDB_DISPLAY_PREFIX} · {config.version}",
            "WANDB_RUN_GROUP": PROJECT,
            "WANDB_TAGS": (
                "0033,full_semantic,official_cpu,ppo,dragapult_third_ptcg_club,"
                + config.ppo.credit_clock
                + "_clock"
            ),
            "WANDB_DIR": str(paths["wandb"]),
        }
    )
    if config.wandb_mode == "online":
        import wandb
        if not getattr(wandb.Api(), "api_key", None):
            raise RuntimeError("W&B online authentication is unavailable")
    config_payload = {
        "schema": "0033_full_semantic_cpu_ppo_config_v1",
        "project_id": PROJECT,
        **asdict(config),
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": _sha256(SOURCE_CHECKPOINT),
        "actor_schema": "0031_rule_faithful_semantic_decision_v2",
        "actor": "exact_0031_semantic_policy_no_reduction",
        "opponent_count": 51,
        "official_engine": "cpu_cg_runtime",
        "trainable_contract": ["actor.action_decoder.*", "value_head.*"],
        "checkpoint_retention": "all",
        "reward": "terminal_only_minus_one_zero_plus_one",
        "temporal_credit": {
            "gamma": config.ppo.gamma,
            "gae_lambda": config.ppo.gae_lambda,
            "clock": config.ppo.credit_clock,
            "same_turn_lambda": 1.0 if config.ppo.credit_clock == "turn" else config.ppo.gae_lambda,
            "boundary_lambda": config.ppo.gae_lambda,
        },
        "episode_weighting": config.ppo.loss_weighting,
    }
    _atomic_json(paths["artifact"] / "training_config.json", config_payload)
    _atomic_json(paths["artifact"] / "opponent_snapshot.json", _opponent_snapshot())
    _atomic_json(
        paths["artifact"] / "status.json",
        {
            "state": "initializing",
            "version": config.version,
            "wandb_run_id": run_id,
            "wandb_url": wandb_url,
            "wandb_sync_status": "initializing",
        },
    )
    started = time.time()
    update = 0
    cumulative_episodes = 0
    cumulative_decisions = 0
    try:
        model, identity = load_actor_critic(SOURCE_CHECKPOINT, focal_deck(), device)
        parity = _run_parity(model, paths["artifact"] / "pt0805_runtime_parity.json")
        if not parity["passed"]:
            raise RuntimeError("PT0805 runtime parity did not pass")
        trainer = PPOTrainer(model, device=device, config=config.ppo)
        representation = model.representation_sha256()
        initial_decoder = model.decoder_sha256()
        save_model_only(
            model,
            paths["checkpoint"] / "update-000000.pt",
            update=0,
            metadata={
                "project": PROJECT,
                "version": config.version,
                "source_checkpoint_sha256": identity.checkpoint_sha256,
                "representation_sha256": representation,
            },
        )
        _merge_status(
            paths["artifact"] / "status.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "checkpoint_update": 0,
                "rollout_source_policy_update": 0,
                "representation_sha256": representation,
                "initial_decoder_sha256": initial_decoder,
                "full_schema_parity": True,
                "wandb_sync_status": "online_active" if config.wandb_mode == "online" else "offline_staging",
            },
        )
        with TrainingLogger(
            paths["artifact"] / "training_metrics.jsonl", paths["tensorboard"]
        ) as logger:
            for update in range(1, config.updates + 1):
                source_update = update - 1
                collector = FullSemanticRolloutCollector(
                    model,
                    device=device,
                    workers=config.workers,
                    mode="sample",
                    coalesce_ms=config.coalesce_ms,
                )
                rollout_started = time.perf_counter()
                episodes = collector.collect(
                    build_jobs(
                        source_policy_update=source_update,
                        seed=config.seed,
                        count=config.games_per_update,
                    )
                )
                rollout_seconds = time.perf_counter() - rollout_started
                rollout_metrics = _episode_metrics(episodes, "rollout")
                rollout_metrics.update(collector.metrics())
                rollout_metrics["rollout/wall_seconds"] = rollout_seconds
                cumulative_episodes += len(episodes)
                cumulative_decisions += int(rollout_metrics["rollout/decisions"])
                batch = prepare_episodes(
                    episodes,
                    gamma=config.ppo.gamma,
                    gae_lambda=config.ppo.gae_lambda,
                    credit_clock=config.ppo.credit_clock,
                    loss_weighting=config.ppo.loss_weighting,
                )
                ppo_started = time.perf_counter()
                ppo_metrics = trainer.update(batch)
                ppo_seconds = time.perf_counter() - ppo_started
                checkpoint = paths["checkpoint"] / f"update-{update:06d}.pt"
                digest = save_model_only(
                    model,
                    checkpoint,
                    update=update,
                    metadata={
                        "project": PROJECT,
                        "version": config.version,
                        "source_policy_update": source_update,
                        "source_checkpoint_sha256": identity.checkpoint_sha256,
                        "representation_sha256": representation,
                        "opponent_snapshot": "artifact/opponent_snapshot.json",
                        "parity_report": "artifact/pt0805_runtime_parity.json",
                    },
                )
                metrics: dict[str, Any] = {
                    "trainer/update": update,
                    "rollout/source_policy_update": source_update,
                    "checkpoint/update": update,
                    "env/episodes": cumulative_episodes,
                    "env/decisions": cumulative_decisions,
                    "checkpoint/sha256_present": float(bool(digest)),
                    "representation/sha256_unchanged": float(
                        model.representation_sha256() == representation
                    ),
                    "ppo/wall_seconds": ppo_seconds,
                    "system/end_to_end/wall_seconds": rollout_seconds + ppo_seconds,
                    "system/disk/free_gib": shutil.disk_usage(paths["root"]).free / 1024**3,
                    "system/gpu/max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "system/gpu/max_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    **rollout_metrics,
                    **ppo_metrics,
                }
                if update % config.eval_every == 0:
                    evaluator = FullSemanticRolloutCollector(
                        model,
                        device=device,
                        workers=config.workers,
                        mode="greedy",
                        coalesce_ms=config.coalesce_ms,
                    )
                    eval_started = time.perf_counter()
                    evaluation = evaluator.collect(
                        build_jobs(
                            source_policy_update=update,
                            seed=config.seed + 70_000_000,
                            count=102,
                            greedy=True,
                        )
                    )
                    metrics.update(_episode_metrics(evaluation, "eval"))
                    metrics["eval/checkpoint_update"] = update
                    metrics["eval/wall_seconds"] = time.perf_counter() - eval_started
                logger.log(update, metrics)
                _merge_status(
                    paths["artifact"] / "status.json",
                    {
                        "state": "running",
                        "checkpoint_update": update,
                        "rollout_source_policy_update": source_update,
                        "episodes": cumulative_episodes,
                        "decisions": cumulative_decisions,
                        "representation_sha256_unchanged": True,
                        "decoder_sha256": model.decoder_sha256(),
                        "elapsed_seconds": time.time() - started,
                    },
                )
        summary = {
            "state": "complete",
            "version": config.version,
            "updates": update,
            "episodes": cumulative_episodes,
            "decisions": cumulative_decisions,
            "elapsed_seconds": time.time() - started,
            "representation_sha256": representation,
            "representation_unchanged": model.representation_sha256() == representation,
            "initial_decoder_sha256": initial_decoder,
            "final_decoder_sha256": model.decoder_sha256(),
            "checkpoint_retention": "all",
            "wandb_url": wandb_url,
        }
        _atomic_json(paths["artifact"] / "training_summary.json", summary)
        _merge_status(paths["artifact"] / "status.json", summary)
        return summary
    except BaseException as error:
        _merge_status(
            paths["artifact"] / "status.json",
            {
                "state": "failed",
                "checkpoint_update": update,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.time() - started,
                "wandb_sync_status": "failed_or_interrupted",
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--gate-games", type=int, default=4)
    parser.add_argument("--gate-workers", type=int, default=2)
    parser.add_argument("--gate-coalesce-ms", type=float, default=5.0)
    parser.add_argument("--gate-ppo", action="store_true")
    parser.add_argument("--version", default=FORMAL_VERSION)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--coalesce-ms", type=float, default=5.0)
    parser.add_argument("--games-per-update", type=int, choices=(204, 408), default=204)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--credit-clock", choices=("selection", "turn"), default="selection")
    parser.add_argument(
        "--loss-weighting",
        choices=("episode_equal_decisions", "episode_equal_turns"),
        default="episode_equal_decisions",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if args.gate_output is not None:
        report = run_gate(
            output=args.gate_output,
            games=args.gate_games,
            workers=args.gate_workers,
            run_ppo=args.gate_ppo,
            coalesce_ms=args.gate_coalesce_ms,
            gae_lambda=args.gae_lambda,
            credit_clock=args.credit_clock,
            loss_weighting=args.loss_weighting,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    run(
        RunConfig(
            version=args.version,
            updates=args.updates,
            workers=args.workers,
            coalesce_ms=args.coalesce_ms,
            games_per_update=args.games_per_update,
            eval_every=args.eval_every,
            wandb_mode=args.wandb_mode,
            ppo=PPOConfig(
                gae_lambda=args.gae_lambda,
                credit_clock=args.credit_clock,
                loss_weighting=args.loss_weighting,
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
