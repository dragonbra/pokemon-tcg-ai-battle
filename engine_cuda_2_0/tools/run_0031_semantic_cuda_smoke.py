from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CUDA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ROOT.parent
sys.path.insert(0, str(CUDA_ROOT / "python"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_pool import extend_normalized_actions_device  # noqa: E402
from ptcg_cuda_engine.semantic0031_bridge import (  # noqa: E402
    KNOWN_0031_ARCHIVE_SHA256,
    Semantic0031DeviceAdapter,
    load_semantic0031_package,
    policy_codec_v1_to_semantic0031_v2,
)


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


def parse_args() -> argparse.Namespace:
    private = CUDA_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="Run the compact 0031 semantic model in real official CUDA battles."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "0031_mega_lopunny_ex_mega_froslass_pt0805_compact_fp16.tar.gz"
        ),
    )
    parser.add_argument("--archive-sha256", default=KNOWN_0031_ARCHIVE_SHA256)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional shared FP32 checkpoint overriding the archive checkpoint.",
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--opponent-deck",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "evaluation"
            / "arena"
            / "frozen"
            / "alakazam_dudunsparce_001"
            / "deck.csv"
        ),
    )
    parser.add_argument("--extension-dir", type=Path, default=CUDA_ROOT / "build" / "torch_0031")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=2026080601)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--verify-option-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ROOT / "artifacts" / "semantic0031_cuda_smoke.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [int(value) for value in path.read_text(encoding="ascii").splitlines() if value.strip()]
    if len(deck) != 60 or any(card <= 0 for card in deck):
        raise ValueError(f"deck must contain exactly 60 positive IDs: {path}")
    return deck


def codec_batch(engine: Any) -> dict[str, Any]:
    batch = dict(engine.encode_policy_v1())
    batch["entity_mask"] = batch["entity_mask"].bool()
    batch["option_mask"] = batch["option_mask"].bool()
    if hasattr(engine, "semantic_history_raw"):
        history = engine.semantic_history_raw()
        batch.update(
            {
                "semantic0031_history_total_count": history["total_count"],
                "semantic0031_history_write_index": history["write_index"],
                "semantic0031_history_log_type": history["log_type"],
                "semantic0031_history_param_count": history["param_count"],
                "semantic0031_history_params": history["params"],
            }
        )
    return batch


def first_legal_actions(batch: dict[str, Any], *, max_select: int) -> tuple[Any, Any]:
    import torch

    mask = batch["option_mask"].bool()
    batch_size, option_count = mask.shape
    minimum = batch["min_count"].long().view(batch_size).clamp(min=0, max=max_select)
    rank = mask.long().cumsum(dim=1) - 1
    take = mask & rank.lt(minimum.unsqueeze(1))
    destination = torch.where(take, rank, torch.full_like(rank, max_select))
    option_ids = torch.arange(option_count, dtype=torch.long, device=mask.device).view(1, -1)
    values = torch.where(
        take,
        option_ids.expand(batch_size, -1),
        torch.full_like(destination, -1),
    )
    output = torch.full((batch_size, max_select + 1), -1, dtype=torch.long, device=mask.device)
    output.scatter_(1, destination, values)
    return output[:, :max_select], take.long().sum(dim=1)


def materialize_prototype_relations(semantic: dict[str, Any], model: Any) -> dict[str, Any]:
    """Construct the package's ragged relation contract for an equivalence check."""

    import torch

    output = dict(semantic)
    option_cat = semantic["option_cat"]
    option_mask = semantic["option_mask"]
    batch_size, option_count = option_mask.shape
    prototype = model.prototype_encoder
    card_ids = torch.stack((option_cat[..., 5], option_cat[..., 11], option_cat[..., 12]), dim=2)
    skill_ids = prototype.card_skill_table[card_ids]
    role_slots = torch.arange(1, 4, dtype=torch.long, device=option_mask.device)
    relation_offsets = torch.arange(3, dtype=torch.long, device=option_mask.device) * 3
    skill_roles = relation_offsets.view(1, 1, 3, 1) + role_slots.view(1, 1, 1, 3)
    skill_roles = skill_roles.expand(batch_size, option_count, -1, -1)
    skill_parents = (
        torch.arange(1, option_count + 1, dtype=torch.long, device=option_mask.device)
        .view(1, option_count, 1, 1)
        .expand_as(skill_ids)
    )
    skill_mask = skill_ids.gt(0) & option_mask.view(batch_size, option_count, 1, 1)
    output["option_skill_id"] = skill_ids.flatten(1)
    output["option_skill_role"] = skill_roles.flatten(1)
    output["option_skill_parent"] = skill_parents.flatten(1)
    output["option_skill_mask"] = skill_mask.flatten(1)

    skill_effect_ids = prototype.skill_effect_table[skill_ids]
    skill_effect_mask = skill_effect_ids.gt(0) & skill_mask.unsqueeze(-1)
    attack_effect_ids = prototype.attack_effect_table[option_cat[..., 7]]
    attack_effect_mask = attack_effect_ids.gt(0) & option_mask.unsqueeze(-1)
    effect_ids = torch.cat((skill_effect_ids.flatten(2), attack_effect_ids), dim=2)
    effect_roles = torch.cat(
        (
            torch.ones_like(skill_effect_ids).flatten(2),
            torch.full_like(attack_effect_ids, 2),
        ),
        dim=2,
    )
    effect_mask = torch.cat((skill_effect_mask.flatten(2), attack_effect_mask), dim=2)
    effect_parents = (
        torch.arange(1, option_count + 1, dtype=torch.long, device=option_mask.device)
        .view(1, option_count, 1)
        .expand_as(effect_ids)
    )
    output["option_effect_id"] = effect_ids.flatten(1)
    output["option_effect_role"] = effect_roles.flatten(1)
    output["option_effect_parent"] = effect_parents.flatten(1)
    output["option_effect_mask"] = effect_mask.flatten(1)
    return output


def verify_option_cache(adapter: Semantic0031DeviceAdapter, batch: dict[str, Any]) -> float:
    import torch

    augmented = adapter._inject_static(batch)
    semantic = policy_codec_v1_to_semantic0031_v2(augmented, max_action_steps=adapter.max_select)
    base = adapter.model.validate_batch(semantic)
    materialized = adapter.model.validate_batch(
        materialize_prototype_relations(semantic, adapter.model)
    )
    with torch.inference_mode():
        state = adapter.model.state_encoder(base, adapter.prototype_memory)
        optimized = adapter._encode_options(base, state)
        reference = adapter.model.option_encoder(materialized, state, adapter.prototype_memory)
    return float((optimized - reference).abs().max().item())


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("batch/steps must be positive and warmup non-negative")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    archive = args.archive.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    rules = args.rules.resolve()
    opponent_deck_path = args.opponent_deck.resolve()
    required_paths = [archive, rules, opponent_deck_path]
    if checkpoint is not None:
        required_paths.append(checkpoint)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    package = load_semantic0031_package(
        archive,
        device,
        expected_archive_sha256=args.archive_sha256,
        checkpoint_override=checkpoint,
    )
    adapter = Semantic0031DeviceAdapter(package.model, package.deck, max_select=64)
    opponent_deck = read_deck(opponent_deck_path)
    decks = torch.tensor(
        [[package.deck, opponent_deck] for _ in range(args.batch)],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.arange(
        args.seed_start,
        args.seed_start + args.batch,
        dtype=torch.int64,
        device=device,
    )
    engine = create_official_engine(
        rules.read_bytes(), batch_size=args.batch, device_index=args.device_index
    )
    engine.reset_seeded_interactive_semantic(decks, seeds)
    learner_decisions = torch.zeros((), dtype=torch.int64, device=device)
    baseline_decisions = torch.zeros((), dtype=torch.int64, device=device)
    finished_games = torch.zeros((), dtype=torch.int64, device=device)
    error_count = torch.zeros((), dtype=torch.int64, device=device)
    selection_overflow = torch.zeros((), dtype=torch.int64, device=device)
    option_cache_max_abs = None

    def one_step(*, count_metrics: bool) -> None:
        nonlocal option_cache_max_abs
        status = engine.statuses()
        terminal = status.eq(TERMINAL)
        failed = status.eq(ERROR)
        reset = terminal | failed
        seeds.add_(reset.long() * args.batch)
        engine.reset_seeded_interactive_masked(decks, seeds, reset)
        if count_metrics:
            finished_games.add_(terminal.long().sum())
            error_count.add_(failed.long().sum())
        engine.advance_to_decision()
        batch = codec_batch(engine)
        ready = engine.statuses().eq(NEEDS_ACTION)
        actor = batch["global_cat"][:, 3].long() - 1
        learner_ready = ready & actor.eq(0)
        baseline_ready = ready & actor.eq(1)
        batch["_route_mask"] = learner_ready
        if args.verify_option_cache and option_cache_max_abs is None:
            option_cache_max_abs = verify_option_cache(adapter, batch)
        proposed, proposed_lengths = adapter.act_device(batch)
        model_actions, model_lengths = extend_normalized_actions_device(
            proposed,
            proposed_lengths,
            batch["option_mask"],
            batch["min_count"],
            batch["max_count"],
            max_select=80,
        )
        baseline_actions, baseline_lengths = first_legal_actions(batch, max_select=80)
        actions = torch.where(learner_ready.unsqueeze(1), model_actions, baseline_actions)
        lengths = torch.where(learner_ready, model_lengths, baseline_lengths)
        engine.pack_actions(actions, lengths)
        engine.apply_packed_actions()
        if count_metrics:
            learner_decisions.add_(learner_ready.long().sum())
            baseline_decisions.add_(baseline_ready.long().sum())
            selection_overflow.add_((learner_ready & batch["min_count"].gt(64)).long().sum())

    if args.verify_option_cache:
        one_step(count_metrics=False)
    for _ in range(args.warmup):
        one_step(count_metrics=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        one_step(count_metrics=True)
    end.record()
    end.synchronize()
    elapsed_sec = start.elapsed_time(end) / 1000.0

    final_status = engine.statuses()
    finished_games.add_(final_status.eq(TERMINAL).long().sum())
    error_count.add_(final_status.eq(ERROR).long().sum())
    learner_count = int(learner_decisions.item())
    baseline_count = int(baseline_decisions.item())
    finished_count = int(finished_games.item())
    errors = int(error_count.item())
    overflow = int(selection_overflow.item())
    status_counts = torch.bincount(final_status.long(), minlength=4).cpu().tolist()
    cache_error = option_cache_max_abs if option_cache_max_abs is not None else 0.0
    passed = (
        learner_count > 0
        and finished_count > 0
        and errors == 0
        and overflow == 0
        and cache_error <= 1.0e-4
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "scope": "0031 semantic learner vs device first-legal opponent in official CUDA battles",
        "batch": args.batch,
        "warmup": args.warmup,
        "steps": args.steps,
        "seed_start": args.seed_start,
        "learner_model_decisions": learner_count,
        "baseline_decisions": baseline_count,
        "inference_fallbacks": 0,
        "finished_games": finished_count,
        "engine_errors": errors,
        "minimum_selection_overflow": overflow,
        "elapsed_sec": elapsed_sec,
        "learner_decisions_per_sec": learner_count / elapsed_sec,
        "all_decisions_per_sec": (learner_count + baseline_count) / elapsed_sec,
        "finished_games_per_sec": finished_count / elapsed_sec,
        "option_cache_reference_max_abs": option_cache_max_abs,
        "status_counts": {
            "idle": int(status_counts[0]),
            "needs_action": int(status_counts[1]),
            "terminal": int(status_counts[2]),
            "error": int(status_counts[3]),
        },
        "cuda_resident": {
            "engine": True,
            "policy_codec": True,
            "semantic_projection": True,
            "prototype_memory": True,
            "ordered_decoder": True,
            "actions": True,
            "hot_path_h2d_copies": 0,
            "hot_path_d2h_copies": 0,
            "hot_path_host_scalar_reads": 0,
        },
        "math": {
            "parameter_dtype": str(next(package.model.parameters()).dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "memory": {
            "official_arena_bytes": int(engine.allocated_bytes),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "identity": {
            "archive_sha256": package.archive_sha256,
            "checkpoint_sha256": (
                sha256_file(checkpoint)
                if checkpoint is not None
                else package.manifest["compact_checkpoint_sha256"]
            ),
            "checkpoint_override": str(checkpoint) if checkpoint is not None else None,
            "registered_deck_sha256": package.manifest["deck_sha256"],
            "opponent_deck_file_sha256": sha256_file(opponent_deck_path),
            "rule_pack_sha256": sha256_file(rules),
            "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "output": str(output)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
