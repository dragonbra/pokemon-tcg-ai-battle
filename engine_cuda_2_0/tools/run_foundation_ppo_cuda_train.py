"""Run a wall-clock PPO job for the shared 0020 foundation CUDA policy.

The script extends the existing smoke entrypoint into a repeatable training
entrypoint: rollout, GAE, actor/critic update, and optimizer stay on CUDA,
while only scalar telemetry/checkpoints are synchronized to host.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.cuda_ppo import (  # noqa: E402
    CudaRolloutBuffer,
    ppo_update_device,
)
from ptcg_cuda_engine.legacy_codecs import policy_codec_v1_to_idonly_codec_v1  # noqa: E402
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_adapters import foundation_r15_static_fields_for_decks  # noqa: E402
from run_official_seeded_multi_policy_loop import (  # noqa: E402
    ERROR,
    load_0020_runtime_package,
    load_foundation_0020_checkpoint,
    load_pool_rows,
    read_deck,
)


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="GPU-resident PPO training for the shared 0020 foundation model."
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--pool",
        type=Path,
        default=WORKSPACE_ROOT / ".tmp" / "cuda_zero_shot_six_decks" / "pool.json",
    )
    parser.add_argument(
        "--learner-deck",
        type=Path,
        default=None,
        help=(
            "Optional player0 deck.csv. The six pool entries remain frozen player1 "
            "opponents and continue to supply the foundation checkpoint/ontology."
        ),
    )
    parser.add_argument(
        "--learner-deck-name",
        default=None,
        help="Telemetry name for --learner-deck; defaults to the CSV filename stem.",
    )
    parser.add_argument("--batch", type=int, default=7680)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=2026080101)
    parser.add_argument("--torch-seed", type=int, default=2026080101)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--train-scope",
        choices=("full", "decoder_only"),
        default="decoder_only",
        help="Train all shared foundation parameters or only decoder/value head.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--value-lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--target-games-per-deck",
        type=int,
        default=0,
        help="If positive, collect until every deck slot reaches this many terminal games before PPO update.",
    )
    parser.add_argument(
        "--max-steps-per-update",
        type=int,
        default=128,
        help="Safety cap for game-targeted rollout collection.",
    )
    parser.add_argument("--duration-hours", type=float, default=10.0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Stop after this many iterations in the current process; 0 uses only duration-hours.",
    )
    parser.add_argument("--checkpoint-every-minutes", type=float, default=10.0)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, iteration, elapsed time, and history tail from a saved PPO checkpoint.",
    )
    parser.add_argument(
        "--resume-lr-source",
        choices=("cli", "checkpoint"),
        default="cli",
        help=(
            "After restoring AdamW state, use the CLI actor/value learning rates "
            "or retain the checkpoint learning rates."
        ),
    )
    parser.add_argument(
        "--best-checkpoint-count",
        type=int,
        default=5,
        help="Keep this many best checkpoints by the selected win-rate metric. Use 0 to disable.",
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        default="main_win_rate_roll5",
        help="Metric used for best checkpoint ranking; roll5 falls back to current win rate for the first 4 generations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "foundation_ppo_10h_tf32_b7680_5090",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ptcg-cuda-ppo"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    for name in (
        "rules",
        "pool",
        "learner_deck",
        "output_dir",
        "resume_checkpoint",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(str(value).replace("\r", "")))
    for name in (
        "learner_deck_name",
        "wandb_project",
        "wandb_run_name",
        "wandb_mode",
    ):
        value = getattr(args, name)
        if isinstance(value, str):
            setattr(args, name, value.replace("\r", ""))
    return args


def _move_masks(batch: dict[str, Any]) -> dict[str, Any]:
    batch = dict(batch)
    batch["entity_mask"] = batch["entity_mask"].bool()
    batch["option_mask"] = batch["option_mask"].bool()
    return batch


def _json_default(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().item()
            return f"Tensor(shape={tuple(value.shape)}, device={value.device}, dtype={value.dtype})"
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _module_state_hash(module: Any) -> str:
    """Hash a frozen decoder state for the run manifest and smoke gate."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _set_actor_value_learning_rates(
    optimizer: Any, *, actor_lr: float, value_lr: float
) -> None:
    if len(optimizer.param_groups) != 2:
        raise RuntimeError(
            "expected AdamW actor/value parameter groups, got "
            f"{len(optimizer.param_groups)}"
        )
    optimizer.param_groups[0]["lr"] = float(actor_lr)
    optimizer.param_groups[1]["lr"] = float(value_lr)


def _balanced_lane_game_quotas(
    lane_deck_ids: list[int], *, deck_count: int, target_games_per_deck: int
) -> list[int]:
    """Assign exact per-lane quotas whose sum is identical for every deck."""
    if deck_count <= 0 or target_games_per_deck < 0:
        raise ValueError("deck_count must be positive and target must be non-negative")
    groups: list[list[int]] = [[] for _ in range(deck_count)]
    for lane, deck_id in enumerate(lane_deck_ids):
        if not 0 <= int(deck_id) < deck_count:
            raise ValueError(f"lane {lane} has invalid deck id {deck_id}")
        groups[int(deck_id)].append(lane)
    if any(not lanes for lanes in groups):
        raise ValueError("every opponent deck needs at least one CUDA lane")
    quotas = [0] * len(lane_deck_ids)
    for lanes in groups:
        base, remainder = divmod(int(target_games_per_deck), len(lanes))
        for rank, lane in enumerate(lanes):
            quotas[lane] = base + int(rank < remainder)
    if any(
        sum(quotas[lane] for lane in lanes) != int(target_games_per_deck)
        for lanes in groups
    ):
        raise AssertionError("balanced lane quota construction failed")
    return quotas


def _init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any | None:
    try:
        import wandb
    except Exception as exc:
        print(f"WANDB_UNAVAILABLE {exc!r}", flush=True)
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=config,
        )
    except Exception as exc:
        print(f"WANDB_INIT_FAILED {exc!r}", flush=True)
        return None


def _install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, _raise_keyboard_interrupt)
        except Exception:
            pass


def _save_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    optimizer: Any,
    iteration: int,
    elapsed_seconds: float,
    config: dict[str, Any],
    history_tail: list[dict[str, Any]],
    final: bool = False,
    name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    import torch

    checkpoint = {
        "schema_version": 2,
        "iteration": int(iteration),
        "elapsed_seconds": float(elapsed_seconds),
        "config": config,
        "history_tail": history_tail[-20:],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    if name is None:
        name = "checkpoint_final.pt" if final else "checkpoint_last.pt"
    path = output_dir / name
    torch.save(checkpoint, path)
    return path


def _metric_value(
    row: dict[str, Any],
    metric: str,
    history_length: int,
) -> tuple[float | None, str]:
    metric_used = metric
    value = row.get(metric)
    if metric == "main_win_rate_roll5" and history_length < 5:
        metric_used = "main_win_rate"
        value = row.get(metric_used)
    if value is None:
        metric_used = "main_win_rate"
        value = row.get(metric_used)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None, metric_used
    if not math.isfinite(score):
        return None, metric_used
    return score, metric_used


def _load_best_checkpoint_entries(output_dir: Path) -> list[dict[str, Any]]:
    manifest_path = output_dir / "top_checkpoints.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"BEST_CHECKPOINT_MANIFEST_IGNORED {exc!r}", flush=True)
        return []
    raw_entries = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("file")
        if not isinstance(filename, str):
            continue
        if (output_dir / filename).exists():
            entries.append(dict(entry))
    entries.sort(key=lambda item: (float(item.get("score", -1.0)), int(item.get("iteration", -1))), reverse=True)
    return entries


def _write_best_checkpoint_manifest(
    *,
    output_dir: Path,
    entries: list[dict[str, Any]],
    metric: str,
    max_count: int,
) -> None:
    manifest = {
        "schema_version": 1,
        "metric": metric,
        "max_count": int(max_count),
        "checkpoints": entries,
    }
    (output_dir / "top_checkpoints.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_best_rank_aliases(output_dir: Path, entries: list[dict[str, Any]], max_count: int) -> None:
    for rank in range(1, max_count + 1):
        alias = output_dir / f"checkpoint_best_rank{rank}.pt"
        if alias.exists():
            alias.unlink()
    for rank, entry in enumerate(entries, start=1):
        target = output_dir / str(entry["file"])
        alias = output_dir / f"checkpoint_best_rank{rank}.pt"
        try:
            os.link(target, alias)
        except OSError:
            shutil.copy2(target, alias)


def _maybe_save_best_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    optimizer: Any,
    iteration: int,
    elapsed_seconds: float,
    config: dict[str, Any],
    history_tail: list[dict[str, Any]],
    row: dict[str, Any],
    entries: list[dict[str, Any]],
    metric: str,
    max_count: int,
    source: str,
) -> dict[str, Any] | None:
    if max_count <= 0:
        return None
    score, metric_used = _metric_value(row, metric, len(history_tail))
    if score is None:
        return None
    if len(entries) >= max_count:
        worst_score = min(float(entry.get("score", -1.0)) for entry in entries)
        if score <= worst_score:
            return None

    safe_metric = metric_used.replace("/", "_")
    filename = f"checkpoint_best_iter_{iteration:06d}_{safe_metric}_{score:.6f}.pt"
    path = _save_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        elapsed_seconds=elapsed_seconds,
        config=config,
        history_tail=history_tail,
        name=filename,
        extra={
            "best_checkpoint": True,
            "best_checkpoint_source": source,
            "best_metric": metric,
            "best_metric_used": metric_used,
            "best_metric_value": score,
        },
    )
    entry = {
        "file": path.name,
        "iteration": int(iteration),
        "score": score,
        "metric": metric,
        "metric_used": metric_used,
        "source": source,
        "elapsed_seconds": float(elapsed_seconds),
        "main_win_rate": row.get("main_win_rate"),
        "main_win_rate_roll5": row.get("main_win_rate_roll5"),
        "main_win_rate_roll10": row.get("main_win_rate_roll10"),
    }
    entries[:] = [
        item for item in entries if item.get("file") != entry["file"]
    ] + [entry]
    entries.sort(key=lambda item: (float(item.get("score", -1.0)), int(item.get("iteration", -1))), reverse=True)
    pruned = entries[max_count:]
    entries[:] = entries[:max_count]
    for rank, item in enumerate(entries, start=1):
        item["rank"] = rank
    _refresh_best_rank_aliases(output_dir, entries, max_count)
    for item in pruned:
        stale = output_dir / str(item.get("file", ""))
        if stale.exists():
            stale.unlink()
    _write_best_checkpoint_manifest(
        output_dir=output_dir,
        entries=entries,
        metric=metric,
        max_count=max_count,
    )
    return entry


def main() -> int:
    args = parse_args()
    _install_signal_handlers()
    if args.batch <= 0 or args.steps <= 0 or not 1 <= args.max_select <= 80:
        raise ValueError("batch/steps must be positive and max-select must be in [1, 80]")
    if args.duration_hours <= 0:
        raise ValueError("duration-hours must be positive")
    if args.max_iterations < 0:
        raise ValueError("max-iterations must be non-negative")
    if args.best_checkpoint_count < 0:
        raise ValueError("best-checkpoint-count must be non-negative")
    if args.train_scope != "decoder_only":
        raise ValueError(
            "the corrected matchup PPO keeps the shared foundation encoder frozen; "
            "use --train-scope decoder_only"
        )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    summary_path = output_dir / "summary.json"

    rules_path = args.rules.resolve()
    pool_path = args.pool.resolve()
    rows, unsupported = load_pool_rows(pool_path, [])
    if len(rows) != 6:
        raise ValueError(f"expected six promoted deck rows, got {len(rows)}")
    checkpoint = (
        WORKSPACE_ROOT / str(rows[0]["directory"]) / str(rows[0]["checkpoint"])
    ).resolve()
    ontology = (
        WORKSPACE_ROOT / str(rows[0]["ontology"])
        if rows[0].get("ontology")
        else checkpoint.parent.parent / "artifact" / "card_ontology.json"
    ).resolve()
    opponent_deck_paths = [
        (WORKSPACE_ROOT / str(row["directory"])).resolve() / "deck.csv"
        for row in rows
    ]
    opponent_decks = [read_deck(path) for path in opponent_deck_paths]
    learner_deck_path = (
        args.learner_deck.resolve()
        if args.learner_deck is not None
        else opponent_deck_paths[0]
    )
    if not learner_deck_path.is_file():
        raise FileNotFoundError(f"learner deck not found: {learner_deck_path}")
    learner_deck = read_deck(learner_deck_path)
    learner_deck_name = (
        args.learner_deck_name
        or (
            str(rows[0]["name"])
            if args.learner_deck is None
            else learner_deck_path.stem
        )
    )
    lane_opponent_ids = torch.tensor(
        [index % len(opponent_decks) for index in range(args.batch)],
        dtype=torch.long,
        device=device,
    )
    lane_opponent_ids_host = [
        index % len(opponent_decks) for index in range(args.batch)
    ]
    lane_game_quotas = torch.tensor(
        _balanced_lane_game_quotas(
            lane_opponent_ids_host,
            deck_count=len(opponent_decks),
            target_games_per_deck=args.target_games_per_deck,
        ),
        dtype=torch.int64,
        device=device,
    )
    decks = torch.tensor(
        [
            [learner_deck, opponent_decks[opponent_id]]
            for opponent_id in lane_opponent_ids_host
        ],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.arange(
        args.seed_start,
        args.seed_start + args.batch,
        dtype=torch.int64,
        device=device,
    )
    # One catalog serves both seats.  At each decision we gather row 0 for
    # player0 and row opponent_id+1 for player1, so deck/static fields remain
    # correct without a host round-trip or a second encoder forward.
    static_field_catalog = foundation_r15_static_fields_for_decks(
        [learner_deck, *opponent_decks], device
    )

    runtime_package = load_0020_runtime_package()
    actor_critic_module = importlib.import_module(f"{runtime_package}.actor_critic")
    policy, _config = load_foundation_0020_checkpoint(checkpoint, ontology, device)
    model = actor_critic_module.SourceR15ActorCritic(policy).to(device)
    scope = model.configure_decoder_only()
    frozen_opponent_head = actor_critic_module.FrozenDecoderHead.from_policy(
        model.policy
    ).to(device)
    frozen_opponent_head_hash = _module_state_hash(frozen_opponent_head)
    actor_lr = float(args.actor_lr if args.actor_lr is not None else args.lr)
    value_lr = float(args.value_lr if args.value_lr is not None else args.lr)
    value_param_ids = {id(parameter) for parameter in model.value_head.parameters()}
    actor_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in value_param_ids
    ]
    value_params = [
        parameter for parameter in model.value_head.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": actor_params, "lr": actor_lr},
            {"params": value_params, "lr": value_lr},
        ]
    )

    resume_path = args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None
    resume_iteration = 0
    resume_elapsed_seconds = 0.0
    resume_history: list[dict[str, Any]] = []
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if args.resume_lr_source == "cli":
            _set_actor_value_learning_rates(
                optimizer, actor_lr=actor_lr, value_lr=value_lr
            )
        resume_iteration = int(resume_state.get("iteration", 0))
        resume_elapsed_seconds = float(resume_state.get("elapsed_seconds", 0.0))
        raw_history = resume_state.get("history_tail", [])
        if isinstance(raw_history, list):
            resume_history = [dict(row) for row in raw_history if isinstance(row, dict)]
        print(
            f"RESUMED_CHECKPOINT {resume_path} "
            f"iteration={resume_iteration} elapsed_seconds={resume_elapsed_seconds:.3f}",
            flush=True,
        )

    engine = create_official_engine(
        rules_path.read_bytes(), batch_size=args.batch, device_index=args.device_index
    )
    engine.reset_seeded_interactive(decks, seeds)

    config = {
        "schema_version": 3,
        "batch": args.batch,
        "steps": args.steps,
        "max_select": args.max_select,
        "torch_seed": args.torch_seed,
        "train_scope": args.train_scope,
        "scope": scope,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "lr": args.lr,
        "actor_lr": actor_lr,
        "value_lr": value_lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "target_games_per_deck": args.target_games_per_deck,
        "max_steps_per_update": args.max_steps_per_update,
        "terminal_reward": "main_player0_win_plus_one_loss_minus_one_draw_zero",
        "matchup_contract": "player0_configured_learner_deck_vs_player1_six_frozen_opponents",
        "decision_routing_source": "official_policy_codec_global_cat_3_before_idonly_conversion",
        "ppo_train_mask": "ready_player0_decisions_only",
        "gae_contract": "skip_player1_steps_and_assign_terminal_reward_to_prior_player0_transition",
        "deck_target_behavior": "exact_per_lane_game_quotas_sum_to_target_for_each_opponent",
        "last_value_bootstrap": "zero_when_target_games_per_deck_positive",
        "duration_hours": args.duration_hours,
        "max_iterations": args.max_iterations,
        "checkpoint_every_minutes": args.checkpoint_every_minutes,
        "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        "resume_lr_source": args.resume_lr_source,
        "resume_iteration": resume_iteration,
        "resume_elapsed_seconds": resume_elapsed_seconds,
        "best_checkpoint_count": args.best_checkpoint_count,
        "best_checkpoint_metric": args.best_checkpoint_metric,
        "learner_deck_profile": learner_deck_name,
        "learner_deck_path": str(learner_deck_path),
        "learner_deck_file_sha256": hashlib.sha256(
            learner_deck_path.read_bytes()
        ).hexdigest(),
        "opponent_deck_profiles": [str(row["name"]) for row in rows],
        "deck_profiles": [str(row["name"]) for row in rows],
        "frozen_opponent_decoder_hash": frozen_opponent_head_hash,
        "shared_encoder": True,
        "shared_encoder_contract": "learner_policy.encoder_and_source_modules_used_for_both_seats; decoder_head_only_varies",
        "unsupported_pool_entries": unsupported,
        "checkpoint": str(checkpoint),
        "ontology": str(ontology),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32": {
            "float32_matmul_precision": "high",
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        },
    }
    summary_path.write_text(json.dumps(config, indent=2, default=_json_default) + "\n", encoding="utf-8")
    run = _init_wandb(args, config)

    history: list[dict[str, Any]] = list(resume_history)
    completed_games_total = (
        int(history[-1].get("completed_games_total", 0) or 0) if history else 0
    )
    decisions_total = int(history[-1].get("decisions_total", 0) or 0) if history else 0
    main_wins_history: list[float] = []
    for row in history:
        if "main_win_rate" in row:
            try:
                main_wins_history.append(float(row["main_win_rate"]))
            except (TypeError, ValueError):
                pass
    iteration = resume_iteration
    started = time.monotonic() - resume_elapsed_seconds
    deadline = started + args.duration_hours * 3600.0
    checkpoint_interval = max(args.checkpoint_every_minutes * 60.0, 1.0)
    next_checkpoint_at = time.monotonic() + checkpoint_interval
    status_error_count = torch.zeros((), dtype=torch.int64, device=device)
    best_checkpoint_entries = _load_best_checkpoint_entries(output_dir)
    iterations_this_run = 0
    print("TRAINING_STARTED " + json.dumps(config, default=_json_default), flush=True)
    if resume_path is not None and history and not best_checkpoint_entries:
        seed_entry = _maybe_save_best_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            elapsed_seconds=resume_elapsed_seconds,
            config=config,
            history_tail=history,
            row=history[-1],
            entries=best_checkpoint_entries,
            metric=args.best_checkpoint_metric,
            max_count=args.best_checkpoint_count,
            source="resume_checkpoint",
        )
        if seed_entry is not None:
            print(
                "BEST_CHECKPOINT "
                + json.dumps(seed_entry, sort_keys=True, default=_json_default),
                flush=True,
            )

    try:
        while (
            time.monotonic() < deadline
            and (
                args.max_iterations == 0
                or iterations_this_run < args.max_iterations
            )
        ):
            iteration += 1
            iterations_this_run += 1
            rollout_started = time.monotonic()
            model.eval()
            rollout_step_budget = (
                int(args.max_steps_per_update)
                if args.target_games_per_deck > 0
                else int(args.steps)
            )
            buffer = CudaRolloutBuffer(max_steps=rollout_step_budget, batch_size=args.batch)
            last_batch: dict[str, Any] | None = None
            iteration_completed_games = 0
            iteration_errors = 0
            iteration_player0_decisions = 0
            iteration_player1_decisions = 0
            completed_by_lane = torch.zeros(args.batch, dtype=torch.int64, device=device)
            completed_by_deck = torch.zeros(
                len(opponent_decks), dtype=torch.int64, device=device
            )
            wins_by_deck = torch.zeros(
                len(opponent_decks), dtype=torch.int64, device=device
            )
            draws_by_deck = torch.zeros(
                len(opponent_decks), dtype=torch.int64, device=device
            )
            for _step in range(rollout_step_budget):
                collecting_lane = (
                    completed_by_lane.lt(lane_game_quotas)
                    if args.target_games_per_deck > 0
                    else torch.ones(
                        args.batch, dtype=torch.bool, device=device
                    )
                )
                statuses = engine.statuses()
                active_errors = statuses.eq(ERROR) & collecting_lane
                reset_mask = (statuses.eq(2) | statuses.eq(ERROR)) & collecting_lane
                iteration_errors += int(active_errors.sum().item())
                seeds.add_(reset_mask.to(torch.int64) * args.batch)
                engine.reset_seeded_interactive_masked(decks, seeds, reset_mask)
                status_error_count.add_(active_errors.long().sum())
                engine.advance_to_decision()
                encoded = _move_masks(dict(engine.encode_policy_v1()))
                idonly = policy_codec_v1_to_idonly_codec_v1(
                    encoded,
                    max_card_id=2048,
                    max_action_steps=args.max_select,
                    target_entity_capacity=128,
                    target_option_capacity=80,
                )
                foundation_batch = dict(idonly)
                ready = engine.statuses().eq(1)
                acting_player = (
                    encoded["global_cat"][:, 3].long() - 1
                ).clamp(min=0, max=1)
                static_deck_ids = torch.where(
                    acting_player.eq(0),
                    torch.zeros_like(lane_opponent_ids),
                    lane_opponent_ids + 1,
                )
                for name, value in static_field_catalog.items():
                    foundation_batch[name] = value.index_select(0, static_deck_ids)
                collecting_ready = ready & collecting_lane
                learner_route = collecting_ready & acting_player.eq(0)
                opponent_route = collecting_ready & acting_player.eq(1)
                iteration_player0_decisions += int(learner_route.long().sum().item())
                iteration_player1_decisions += int(opponent_route.long().sum().item())
                with torch.no_grad():
                    shared_encoded = model.encode(foundation_batch)
                    learner_sample = model.sample_actions_device(
                        foundation_batch,
                        max_select=args.max_select,
                        temperature=1.0,
                        encoded=shared_encoded,
                        route_mask=learner_route,
                    )
                    opponent_sample = model.sample_actions_device(
                        foundation_batch,
                        max_select=args.max_select,
                        temperature=1.0,
                        encoded=shared_encoded,
                        decoder_head=frozen_opponent_head,
                        route_mask=opponent_route,
                    )
                    sample_actions = torch.where(
                        learner_route.view(-1, 1),
                        learner_sample["actions"],
                        opponent_sample["actions"],
                    )
                    sample_lengths = torch.where(
                        learner_route,
                        learner_sample["lengths"],
                        opponent_sample["lengths"],
                    )
                    sample_targets = torch.where(
                        learner_route.view(-1, 1),
                        learner_sample["targets"],
                        opponent_sample["targets"],
                    )
                    sample_logprob = torch.where(
                        learner_route,
                        learner_sample["logprob"],
                        opponent_sample["logprob"],
                    )
                    sample_value = model.value_from_encoding(shared_encoded)
                foundation_batch["_route_mask"] = collecting_ready
                sample = {
                    "actions": sample_actions,
                    "lengths": sample_lengths,
                    "targets": sample_targets,
                    "logprob": sample_logprob,
                    "value": sample_value,
                }
                record_batch = {
                    key: value
                    for key, value in foundation_batch.items()
                    if key != "_route_mask"
                }
                record_batch["targets"] = sample_targets
                record_batch["train_mask"] = learner_route
                engine.pack_actions(sample_actions, sample_lengths)
                engine.apply_packed_actions()
                after_statuses = engine.statuses()
                terminal = after_statuses.eq(2)
                error_done = after_statuses.eq(ERROR)
                done = terminal | error_done
                game_results = engine.game_results().long()
                winner = game_results - 1
                decided = terminal & (winner.eq(0) | winner.eq(1))
                main_won = decided & winner.eq(0)
                main_lost = decided & winner.eq(1)
                draw = terminal & game_results.eq(3)
                reward = torch.zeros(args.batch, dtype=torch.float32, device=device)
                reward = torch.where(main_won, torch.ones_like(reward), reward)
                reward = torch.where(main_lost | error_done, -torch.ones_like(reward), reward)
                counted_terminal = terminal & collecting_lane
                counted_main_won = main_won & collecting_lane
                counted_draw = draw & collecting_lane
                if counted_terminal.any():
                    completed_by_lane.add_(counted_terminal.long())
                    terminal_indices = counted_terminal.nonzero(as_tuple=False).flatten()
                    deck_counts = torch.bincount(
                        lane_opponent_ids.index_select(0, terminal_indices),
                        minlength=len(opponent_decks),
                    )
                    completed_by_deck.add_(deck_counts)
                    win_counts = torch.bincount(
                        lane_opponent_ids.index_select(
                            0, counted_main_won.nonzero(as_tuple=False).flatten()
                        ),
                        minlength=len(opponent_decks),
                    )
                    draw_counts = torch.bincount(
                        lane_opponent_ids.index_select(
                            0, counted_draw.nonzero(as_tuple=False).flatten()
                        ),
                        minlength=len(opponent_decks),
                    )
                    wins_by_deck.add_(win_counts)
                    draws_by_deck.add_(draw_counts)
                    iteration_completed_games += int(counted_terminal.long().sum().item())
                buffer.append(
                    record_batch,
                    old_logprob=sample_logprob,
                    old_value=sample_value,
                    reward=reward,
                    done=done,
                )
                last_batch = record_batch
                if (
                    args.target_games_per_deck > 0
                    and bool(completed_by_deck.ge(args.target_games_per_deck).all().item())
                ):
                    break

            if last_batch is None:
                raise RuntimeError("no rollout batch was produced")
            with torch.no_grad():
                last_value_estimate = model.value(last_batch)
                last_value_estimate = torch.where(
                    last_batch["train_mask"].bool(),
                    last_value_estimate,
                    torch.zeros_like(last_value_estimate),
                )
                last_value = (
                    torch.zeros_like(last_value_estimate)
                    if args.target_games_per_deck > 0
                    else last_value_estimate
                )
            torch.cuda.synchronize(device)
            rollout_seconds = time.monotonic() - rollout_started
            rollout_steps = len(buffer)
            update_started = time.monotonic()
            stats = ppo_update_device(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                last_value=last_value,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                target_kl=args.target_kl,
            )
            torch.cuda.synchronize(device)
            update_seconds = time.monotonic() - update_started
            elapsed_seconds = time.monotonic() - started
            decisions = args.batch * rollout_steps
            decisions_total += decisions
            completed_games_total += iteration_completed_games
            completed_by_deck_host = [int(value) for value in completed_by_deck.detach().cpu().tolist()]
            wins_by_deck_host = [int(value) for value in wins_by_deck.detach().cpu().tolist()]
            draws_by_deck_host = [int(value) for value in draws_by_deck.detach().cpu().tolist()]
            losses_by_deck_host = [
                max(c - w - d, 0)
                for c, w, d in zip(completed_by_deck_host, wins_by_deck_host, draws_by_deck_host)
            ]
            win_rate_by_deck = [
                (w / c) if c else 0.0 for w, c in zip(wins_by_deck_host, completed_by_deck_host)
            ]
            generation_win_rate = (
                sum(wins_by_deck_host) / max(sum(completed_by_deck_host), 1)
            )
            main_wins_history.append(generation_win_rate)
            rolling_5_win_rate = sum(main_wins_history[-5:]) / len(main_wins_history[-5:])
            rolling_10_win_rate = sum(main_wins_history[-10:]) / len(main_wins_history[-10:])
            deck_metric_scalars: dict[str, float | int] = {}
            for deck_name, games, wins, losses, draws, win_rate in zip(
                [str(row["name"]) for row in rows],
                completed_by_deck_host,
                wins_by_deck_host,
                losses_by_deck_host,
                draws_by_deck_host,
                win_rate_by_deck,
            ):
                safe_name = deck_name.replace("/", "_")
                deck_metric_scalars[f"deck/{safe_name}/games"] = games
                deck_metric_scalars[f"deck/{safe_name}/wins"] = wins
                deck_metric_scalars[f"deck/{safe_name}/losses"] = losses
                deck_metric_scalars[f"deck/{safe_name}/draws"] = draws
                deck_metric_scalars[f"deck/{safe_name}/win_rate"] = win_rate
            stats_host = stats.as_host()
            row = {
                "iteration": iteration,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_hours": elapsed_seconds / 3600.0,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "iteration_seconds": rollout_seconds + update_seconds,
                "decisions": decisions,
                "player0_decisions": iteration_player0_decisions,
                "player1_decisions": iteration_player1_decisions,
                "decisions_total": decisions_total,
                "rollout_steps": rollout_steps,
                "decisions_per_sec": decisions / max(rollout_seconds + update_seconds, 1.0e-9),
                "rollout_decisions_per_sec": decisions / max(rollout_seconds, 1.0e-9),
                "completed_games": iteration_completed_games,
                "completed_games_total": completed_games_total,
                "completed_games_by_deck": completed_by_deck_host,
                "main_wins_by_deck": wins_by_deck_host,
                "main_losses_by_deck": losses_by_deck_host,
                "draws_by_deck": draws_by_deck_host,
                "main_win_rate": generation_win_rate,
                "main_win_rate_by_deck": win_rate_by_deck,
                "main_win_rate_roll5": rolling_5_win_rate,
                "main_win_rate_roll10": rolling_10_win_rate,
                "metrics/main_win_rate": generation_win_rate,
                "metrics/main_win_rate_roll5": rolling_5_win_rate,
                "metrics/main_win_rate_roll10": rolling_10_win_rate,
                "target_games_per_deck_reached": (
                    min(completed_by_deck_host) >= int(args.target_games_per_deck)
                    if args.target_games_per_deck > 0
                    else None
                ),
                "completed_games_per_sec": iteration_completed_games / max(rollout_seconds, 1.0e-9),
                "status_errors": iteration_errors,
                "status_error_total": int(status_error_count.item()),
                "gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                "gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                **deck_metric_scalars,
                **{f"ppo/{name}": value for name, value in stats_host.items()},
            }
            history.append(row)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print("ITERATION " + json.dumps(row, sort_keys=True), flush=True)
            if run is not None:
                run.log(row, step=iteration)
            best_entry = _maybe_save_best_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                elapsed_seconds=elapsed_seconds,
                config=config,
                history_tail=history,
                row=row,
                entries=best_checkpoint_entries,
                metric=args.best_checkpoint_metric,
                max_count=args.best_checkpoint_count,
                source="iteration",
            )
            if best_entry is not None:
                print(
                    "BEST_CHECKPOINT "
                    + json.dumps(best_entry, sort_keys=True, default=_json_default),
                    flush=True,
                )
            if time.monotonic() >= next_checkpoint_at:
                checkpoint_path = _save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    elapsed_seconds=elapsed_seconds,
                    config=config,
                    history_tail=history,
                )
                print(f"CHECKPOINT {checkpoint_path}", flush=True)
                next_checkpoint_at = time.monotonic() + checkpoint_interval
            del buffer
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("TRAINING_INTERRUPTED", flush=True)
    finally:
        elapsed_seconds = time.monotonic() - started
        checkpoint_path = _save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            elapsed_seconds=elapsed_seconds,
            config=config,
            history_tail=history,
            final=True,
        )
        final_summary = {
            **config,
            "iterations": iteration,
            "iterations_this_run": iterations_this_run,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hours": elapsed_seconds / 3600.0,
            "decisions_total": decisions_total,
            "completed_games_total": completed_games_total,
            "status_error_total": int(status_error_count.item()),
            "checkpoint_final": str(checkpoint_path),
            "best_checkpoint_manifest": str(output_dir / "top_checkpoints.json"),
            "best_checkpoints": best_checkpoint_entries,
        }
        summary_path.write_text(
            json.dumps(final_summary, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        if run is not None:
            run.summary.update(final_summary)
            run.finish()
        print("TRAINING_FINISHED " + json.dumps(final_summary, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
