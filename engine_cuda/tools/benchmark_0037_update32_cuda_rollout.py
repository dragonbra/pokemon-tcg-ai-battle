"""Benchmark complete CUDA-engine games with the exported 0037 Update32 actor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CUDA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ROOT.parent
DEFAULT_VERSION = "V5_last_option_qv_lora_r4_eval5_50u"
DEFAULT_PACKAGE = (
    WORKSPACE_ROOT
    / "archive/submission/0037_dragapult_ex_007_update32_last_option_qv_lora"
)
DEFAULT_POOL = (
    WORKSPACE_ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
)
DEFAULT_OPPONENT_MODEL = (
    WORKSPACE_ROOT
    / "rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt"
)
DEFAULT_ACTOR_CHECKPOINT = (
    WORKSPACE_ROOT
    / "rl_runs/0037_dragapult_value_initialized_rl/versions"
    / DEFAULT_VERSION
    / "checkpoint/update-000032.pt"
)
DEFAULT_SCHEDULE = (
    WORKSPACE_ROOT
    / "rl_runs/0037_dragapult_value_initialized_rl/versions"
    / DEFAULT_VERSION
    / "artifact/schedules/eval_fixed_seeded512.json"
)
DEFAULT_RULES = WORKSPACE_ROOT / ".tmp/cuda_0032_rules/official_rules.bin"
DEFAULT_EXTENSION = WORKSPACE_ROOT / ".tmp/engine_cuda_benchmark/build_sm120_staged"

NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3
_PROTOTYPE_ALIASES = ("state_encoder.prototypes.", "option_encoder.prototypes.")
_SEMANTIC_PREFIX_FAMILIES = {
    "card": ("card_cat", "card_num", "card_state", "card_mask", "card_parent"),
    "resource": ("resource_cat", "resource_num", "resource_state", "resource_mask"),
    "event": (
        "event_cat", "event_num", "event_state", "event_mask", "event_source",
        "event_target", "event_before", "event_after",
    ),
    "option": (
        "option_cat", "option_num", "option_state", "option_mask", "option_source",
        "option_target", "option_context", "option_effect_card",
    ),
    "option_skill": (
        "option_skill_id", "option_skill_role", "option_skill_parent", "option_skill_mask",
    ),
    "option_effect": (
        "option_effect_id", "option_effect_role", "option_effect_parent", "option_effect_mask",
    ),
}
_SEMANTIC_PREFIX_MASKS = {
    "card": "card_mask",
    "resource": "resource_mask",
    "event": "event_mask",
    "option": "option_mask",
    "option_skill": "option_skill_mask",
    "option_effect": "option_effect_mask",
}


@dataclass(frozen=True)
class ScheduleBatch:
    engine_seeds: tuple[int, ...]
    focal_players: tuple[int, ...]
    deck_rows: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    opponent_ids: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(
        int(value)
        for value in path.read_text(encoding="ascii").splitlines()
        if value.strip()
    )
    if len(deck) != 60 or any(card <= 0 for card in deck):
        raise ValueError(f"deck must contain exactly 60 positive IDs: {path}")
    return deck


def load_schedule(
    schedule_path: Path,
    deck_root: Path,
    focal_deck: Sequence[int],
) -> ScheduleBatch:
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("0037 fixed schedule contains no jobs")
    focal = tuple(int(card) for card in focal_deck)
    if len(focal) != 60:
        raise ValueError("focal deck must contain exactly 60 cards")
    engine_seeds: list[int] = []
    focal_players: list[int] = []
    deck_rows: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    opponent_ids: list[str] = []
    deck_cache: dict[str, tuple[int, ...]] = {}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"schedule job {index} is not an object")
        opponent_id = job.get("opponent_id")
        engine_seed = job.get("engine_seed")
        focal_first = job.get("focal_first")
        if (
            not isinstance(opponent_id, str)
            or not opponent_id
            or isinstance(engine_seed, bool)
            or not isinstance(engine_seed, int)
            or not isinstance(focal_first, bool)
        ):
            raise ValueError(f"schedule job {index} has an invalid identity/seed/seat")
        if opponent_id not in deck_cache:
            deck_cache[opponent_id] = read_deck(deck_root / opponent_id / "deck.csv")
        opponent = deck_cache[opponent_id]
        engine_seeds.append(engine_seed)
        focal_players.append(0 if focal_first else 1)
        deck_rows.append((focal, opponent) if focal_first else (opponent, focal))
        opponent_ids.append(opponent_id)
    return ScheduleBatch(
        engine_seeds=tuple(engine_seeds),
        focal_players=tuple(focal_players),
        deck_rows=tuple(deck_rows),
        opponent_ids=tuple(opponent_ids),
    )


def expanded_portable_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    if any(name.startswith(_PROTOTYPE_ALIASES) for name in state):
        raise ValueError("portable checkpoint contains redundant prototype aliases")
    canonical = {
        name.removeprefix("prototype_encoder."): value
        for name, value in state.items()
        if name.startswith("prototype_encoder.")
    }
    if not canonical:
        raise ValueError("portable checkpoint has no canonical prototype encoder")
    expanded = dict(state)
    for prefix in _PROTOTYPE_ALIASES:
        expanded.update(
            {f"{prefix}{suffix}": value for suffix, value in canonical.items()}
        )
    return expanded


def merge_qv_weight(
    base: Any,
    *,
    q_a: Any,
    q_b: Any,
    v_a: Any,
    v_b: Any,
    alpha: float,
    rank: int,
) -> Any:
    width = base.shape[0] // 3
    merged = base.clone()
    scale = alpha / rank
    merged[:width].add_(q_b.float() @ q_a.float(), alpha=scale)
    merged[2 * width :].add_(v_b.float() @ v_a.float(), alpha=scale)
    return merged


def merged_update32_state(base_state: Mapping[str, Any], rl_payload: Mapping[str, Any]) -> dict[str, Any]:
    state = rl_payload.get("state_dict")
    adaptation = rl_payload.get("adaptation")
    metadata = rl_payload.get("metadata")
    if (
        rl_payload.get("schema_version") != "0037_value_initialized_adapted_model_only_v2"
        or rl_payload.get("update") != 32
        or not isinstance(metadata, dict)
        or metadata.get("version") != DEFAULT_VERSION
        or not isinstance(state, dict)
        or adaptation
        != {
            "lora": True,
            "layernorm_tuning": False,
            "rank": 4,
            "alpha": 8.0,
            "option_block": 1,
        }
    ):
        raise ValueError("actor checkpoint is not 0037 Update32 Last Option Q/V LoRA")
    merged = {name: value.clone() for name, value in base_state.items()}
    for name in tuple(merged):
        if name.startswith("action_decoder."):
            merged[name] = state[f"actor.{name}"].clone()
    for module_name in ("self_attn", "multihead_attn"):
        base_name = (
            "option_encoder.cross_attention_transformer.layers.1."
            f"{module_name}.in_proj_weight"
        )
        prefix = (
            "actor.option_encoder.cross_attention_transformer.layers.1."
            f"{module_name}.parametrizations.in_proj_weight.0."
        )
        merged[base_name] = merge_qv_weight(
            merged[base_name],
            q_a=state[prefix + "q_a"],
            q_b=state[prefix + "q_b"],
            v_a=state[prefix + "v_a"],
            v_b=state[prefix + "v_b"],
            alpha=8.0,
            rank=4,
        )
    return merged


def validate_completion(*, total: int, completed: int, errors: int) -> None:
    if completed != total or errors != 0:
        raise RuntimeError(
            "CUDA rollout did not finish cleanly: "
            f"completed={completed}/{total} errors={errors}"
        )


def compact_semantic_batch(
    batch: Mapping[str, Any], indices: Any
) -> dict[str, Any]:
    """Select routed lanes without copying observations through the host."""

    if indices.ndim != 1:
        raise ValueError("semantic cohort indices must be rank one")
    if not batch:
        raise ValueError("semantic batch is empty")
    batch_size = next(iter(batch.values())).shape[0]
    if any(value.shape[0] != batch_size for value in batch.values()):
        raise ValueError("semantic tensors disagree on batch size")
    return {name: value.index_select(0, indices) for name, value in batch.items()}


def compact_semantic_prefixes(
    batch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Trim right padding excluded by each semantic sequence mask."""

    import torch

    families = [
        family
        for family, mask in _SEMANTIC_PREFIX_MASKS.items()
        if mask in batch
    ]
    maxima = torch.stack(
        [batch[_SEMANTIC_PREFIX_MASKS[family]].long().sum(dim=1).amax()
         for family in families]
    ).tolist()
    widths = {
        family: max(1, int(maximum))
        for family, maximum in zip(families, maxima, strict=True)
    }
    output = dict(batch)
    for family in families:
        width = widths[family]
        for name in _SEMANTIC_PREFIX_FAMILIES[family]:
            if name in batch:
                output[name] = batch[name][:, :width]
    return output, widths


def routing_row_metrics(
    *, routing_mode: str, total: int, decisions: int, routed_rows: int
) -> dict[str, int]:
    dense_rows = 2 * total * decisions
    actual_rows = dense_rows if routing_mode == "dense_masked" else routed_rows
    return {
        "routed_ready_rows": routed_rows,
        "theoretical_dense_unused_decoder_rows": dense_rows - routed_rows,
        "actual_decoder_rows": actual_rows,
        "actual_dense_decoder_rows_avoided": dense_rows - actual_rows,
    }


def assert_modules_equal(left: Any, right: Any, *, label: str) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise RuntimeError(f"{label} module keys differ")
    mismatched = [
        name for name in left_state if not left_state[name].equal(right_state[name])
    ]
    if mismatched:
        raise RuntimeError(f"{label} tensors differ: {mismatched[:8]}")


def assert_last_option_qv_only(left: Any, right: Any) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise RuntimeError("option encoder module keys differ")
    mismatched = {
        name for name in left_state if not left_state[name].equal(right_state[name])
    }
    expected = {
        "cross_attention_transformer.layers.1.self_attn.in_proj_weight",
        "cross_attention_transformer.layers.1.multihead_attn.in_proj_weight",
    }
    if mismatched != expected:
        raise RuntimeError(
            "Update32 option encoder differs outside final Q/V weights: "
            f"{sorted(mismatched ^ expected)[:8]}"
        )


def branched_option_outputs(
    actor_adapter: Any,
    opponent_adapter: Any,
    batch: Any,
    state: Any,
) -> tuple[Any, Any]:
    actor_decoder = actor_adapter.model.option_encoder.cross_attention_transformer
    opponent_decoder = opponent_adapter.model.option_encoder.cross_attention_transformer
    if len(actor_decoder.layers) != 2 or len(opponent_decoder.layers) != 2:
        raise RuntimeError("0037 shared Option path requires exactly two decoder blocks")
    options = actor_adapter._encode_option_inputs(batch, state)
    masks = {
        "tgt_key_padding_mask": ~batch.option_mask,
        "memory_key_padding_mask": ~state.mask,
    }
    shared = actor_decoder.layers[0](options, state.tokens, **masks)
    actor = actor_decoder.layers[1](shared, state.tokens, **masks)
    opponent = opponent_decoder.layers[1](shared, state.tokens, **masks)
    if actor_decoder.norm is not None:
        actor = actor_decoder.norm(actor)
    if opponent_decoder.norm is not None:
        opponent = opponent_decoder.norm(opponent)
    mask = batch.option_mask.unsqueeze(-1)
    return actor * mask, opponent * mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark complete CUDA games for 0037 deck-007 Update32."
    )
    parser.add_argument("--actor-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument(
        "--opponent-model",
        type=Path,
        default=DEFAULT_OPPONENT_MODEL,
    )
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_POOL / "decks")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--extension-dir", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--game-limit", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=2048)
    parser.add_argument("--check-interval", type=int, default=16)
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument(
        "--routing-mode",
        choices=("dense_masked", "compact"),
        default="dense_masked",
        help="Run both policies over all lanes with masks, or compact each routed cohort.",
    )
    parser.add_argument(
        "--compact-semantic-prefixes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim mask-excluded semantic right padding before shared encoding.",
    )
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_models(
    actor_package: Path,
    actor_checkpoint_path: Path,
    opponent_checkpoint: Path,
    focal_deck: tuple[int, ...],
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    sys.path.insert(0, str(actor_package))
    portable = importlib.import_module("strategy.deployment.inference")
    portable_checkpoint = actor_package / "strategy/model.bin"
    actor_policy = portable.PortableSemanticPolicy.from_checkpoint(
        portable_checkpoint, focal_deck
    )
    opponent_payload = torch.load(
        opponent_checkpoint, map_location="cpu", weights_only=True
    )
    state = opponent_payload.get("state_dict") if isinstance(opponent_payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("0806 opponent checkpoint has no state_dict")
    opponent_model = copy.deepcopy(actor_policy.model).cpu()
    expected_names = set(opponent_model.state_dict())
    normalized_state = (
        state if set(state) == expected_names else expanded_portable_state(state)
    )
    opponent_model.load_state_dict(normalized_state, strict=True)
    actor_payload = torch.load(
        actor_checkpoint_path, map_location="cpu", weights_only=True
    )
    actor_model = copy.deepcopy(opponent_model).cpu()
    actor_model.load_state_dict(
        merged_update32_state(normalized_state, actor_payload), strict=True
    )
    actor_model = actor_model.requires_grad_(False).to(
        device=device, dtype=torch.float32
    ).eval()
    opponent_model = opponent_model.requires_grad_(False).to(
        device=device, dtype=torch.float32
    ).eval()
    return actor_model, opponent_model, {
        "actor_checkpoint": actor_checkpoint_path,
        "actor_checkpoint_sha256": sha256_file(actor_checkpoint_path),
        "actor_portable_architecture": portable_checkpoint,
        "actor_portable_architecture_sha256": sha256_file(portable_checkpoint),
        "opponent_checkpoint": opponent_checkpoint,
        "opponent_checkpoint_sha256": sha256_file(opponent_checkpoint),
    }


def main() -> int:
    args = parse_args()
    if min(args.game_limit, args.max_decisions, args.check_interval, args.max_select) <= 0:
        raise ValueError("game-limit, max-decisions, check-interval and max-select must be positive")
    if args.max_select > 64:
        raise ValueError("max-select must not exceed 64")
    actor_package = args.actor_package.resolve()
    actor_checkpoint = args.actor_checkpoint.resolve()
    actor_portable_checkpoint = actor_package / "strategy/model.bin"
    focal_deck_path = actor_package / "deck.csv"
    opponent_checkpoint = args.opponent_model.resolve()
    schedule_path = args.schedule.resolve()
    deck_root = args.deck_root.resolve()
    rules_path = args.rules.resolve()
    extension_dir = args.extension_dir.resolve()
    for path in (
        actor_checkpoint,
        actor_portable_checkpoint,
        focal_deck_path,
        opponent_checkpoint,
        schedule_path,
        rules_path,
        extension_dir / "_ptcg_cuda.so",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(extension_dir))
    sys.path.insert(0, str(CUDA_ROOT / "python"))
    import torch
    import _ptcg_cuda
    from ptcg_cuda_engine.native import create_official_engine
    from ptcg_cuda_engine.semantic0031_bridge import (
        Semantic0031DeviceAdapter,
        semantic0031_greedy_decode_device,
        semantic0031_v2_ready_batch,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("CUDA extension lacks semantic0031 v2 observation support")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    focal_deck = read_deck(focal_deck_path)
    full_schedule = load_schedule(schedule_path, deck_root, focal_deck)
    total = min(args.game_limit, len(full_schedule.engine_seeds))
    schedule = ScheduleBatch(
        engine_seeds=full_schedule.engine_seeds[:total],
        focal_players=full_schedule.focal_players[:total],
        deck_rows=full_schedule.deck_rows[:total],
        opponent_ids=full_schedule.opponent_ids[:total],
    )
    actor_model, opponent_model, model_provenance = _load_models(
        actor_package, actor_checkpoint, opponent_checkpoint, focal_deck, device
    )
    actor_adapter = Semantic0031DeviceAdapter(
        actor_model, focal_deck, max_select=args.max_select
    )
    opponent_adapter = Semantic0031DeviceAdapter(
        opponent_model, focal_deck, max_select=args.max_select
    )
    assert_modules_equal(
        actor_model.state_encoder,
        opponent_model.state_encoder,
        label="shared state encoder",
    )
    assert_last_option_qv_only(actor_model.option_encoder, opponent_model.option_encoder)
    decks = torch.tensor(schedule.deck_rows, dtype=torch.int32, device=device)
    seeds = torch.tensor(schedule.engine_seeds, dtype=torch.int64, device=device)
    focal_players = torch.tensor(
        schedule.focal_players, dtype=torch.long, device=device
    )
    lanes = torch.arange(total, dtype=torch.int32, device=device)
    engine = create_official_engine(
        rules_path.read_bytes(), batch_size=total, device_index=args.device_index
    )

    def reset() -> None:
        engine.reset_seeded_interactive_semantic(decks, seeds)
        engine.advance_to_decision()

    reset()
    with torch.inference_mode():
        validation_lanes = lanes[: min(total, 64)]
        validation_semantic = semantic0031_v2_ready_batch(
            engine.encode_semantic0031_v2_lanes(validation_lanes),
            max_action_steps=args.max_select,
        )
        if args.compact_semantic_prefixes:
            validation_semantic, _ = compact_semantic_prefixes(validation_semantic)
        validation_batch = actor_model.validate_batch(validation_semantic)
        validation_state = actor_model.state_encoder(
            validation_batch, actor_adapter.prototype_memory
        )
        branch_actor, branch_opponent = branched_option_outputs(
            actor_adapter, opponent_adapter, validation_batch, validation_state
        )
        full_actor = actor_adapter._encode_options(validation_batch, validation_state)
        full_opponent = opponent_adapter._encode_options(
            validation_batch, validation_state
        )
        if not branch_actor.equal(full_actor) or not branch_opponent.equal(full_opponent):
            raise RuntimeError("shared Option-prefix execution changed encoded options")
        semantic0031_greedy_decode_device(
            actor_model.action_decoder,
            validation_batch,
            branch_actor,
            validation_state.summary,
            max_select=args.max_select,
        )
    torch.cuda.synchronize(device)
    reset()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    started.record()
    decisions = 0
    status_checks = 0
    completed = 0
    errors = 0
    routed_rows = 0
    component_events: list[tuple[str, Any, Any]] = []

    def component_start() -> Any | None:
        if not args.profile_components:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def component_end(name: str, begin: Any | None) -> None:
        if begin is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        component_events.append((name, begin, event))

    with torch.inference_mode():
        for step in range(args.max_decisions):
            component = component_start()
            statuses = engine.statuses()
            ready = statuses.eq(NEEDS_ACTION)
            raw_semantic = engine.encode_semantic0031_v2_lanes(lanes)
            semantic = semantic0031_v2_ready_batch(
                raw_semantic, max_action_steps=args.max_select
            )
            if args.compact_semantic_prefixes:
                semantic, _prefix_widths = compact_semantic_prefixes(semantic)
            component_end("engine_observation", component)
            component = component_start()
            validated = actor_model.validate_batch(semantic)
            state = actor_model.state_encoder(
                validated, actor_adapter.prototype_memory
            )
            actor_options, opponent_options = branched_option_outputs(
                actor_adapter, opponent_adapter, validated, state
            )
            component_end("shared_and_branched_encoder", component)
            actor = semantic["global_cat"][:, 3].long() - 1
            learner_turn = ready & actor.eq(focal_players)
            opponent_turn = ready & ~learner_turn
            learner_indices = learner_turn.nonzero(as_tuple=False).flatten()
            opponent_indices = opponent_turn.nonzero(as_tuple=False).flatten()
            action_width = max(
                1,
                min(
                    args.max_select,
                    int(torch.where(ready, validated.max_count, 0).amax().item()),
                ),
            )
            if args.routing_mode == "dense_masked":
                component = component_start()
                learner_actions, learner_lengths = semantic0031_greedy_decode_device(
                    actor_model.action_decoder,
                    validated,
                    actor_options,
                    state.summary,
                    max_select=action_width,
                    route_mask=learner_turn,
                )
                opponent_actions, opponent_lengths = semantic0031_greedy_decode_device(
                    opponent_model.action_decoder,
                    validated,
                    opponent_options,
                    state.summary,
                    max_select=action_width,
                    route_mask=opponent_turn,
                )
                component_end("two_decoders", component)
                actions = torch.where(
                    learner_turn[:, None], learner_actions, opponent_actions
                ).contiguous()
                lengths = torch.where(
                    learner_turn, learner_lengths, opponent_lengths
                ).contiguous()
            else:
                component = component_start()
                actions = torch.full(
                    (total, action_width), -1, dtype=torch.long, device=device
                )
                lengths = torch.zeros(total, dtype=torch.long, device=device)
                if learner_indices.shape[0]:
                    learner_batch = compact_semantic_batch(semantic, learner_indices)
                    learner_validated = actor_model.validate_batch(learner_batch)
                    learner_actions, learner_lengths = semantic0031_greedy_decode_device(
                        actor_model.action_decoder,
                        learner_validated,
                        actor_options.index_select(0, learner_indices),
                        state.summary.index_select(0, learner_indices),
                        max_select=action_width,
                    )
                    actions.index_copy_(0, learner_indices, learner_actions)
                    lengths.index_copy_(0, learner_indices, learner_lengths)
                if opponent_indices.shape[0]:
                    opponent_batch = compact_semantic_batch(semantic, opponent_indices)
                    opponent_validated = opponent_model.validate_batch(opponent_batch)
                    opponent_actions, opponent_lengths = semantic0031_greedy_decode_device(
                        opponent_model.action_decoder,
                        opponent_validated,
                        opponent_options.index_select(0, opponent_indices),
                        state.summary.index_select(0, opponent_indices),
                        max_select=action_width,
                    )
                    actions.index_copy_(0, opponent_indices, opponent_actions)
                    lengths.index_copy_(0, opponent_indices, opponent_lengths)
                component_end("two_decoders", component)
            routed_rows += int(learner_indices.shape[0] + opponent_indices.shape[0])
            component = component_start()
            engine.pack_actions(actions, lengths)
            engine.apply_packed_actions()
            engine.advance_to_decision()
            component_end("action_apply_advance", component)
            decisions = step + 1
            if decisions % args.check_interval == 0 or decisions == args.max_decisions:
                torch.cuda.synchronize(device)
                checked = engine.statuses()
                status_checks += 1
                completed = int(checked.eq(TERMINAL).sum().item())
                errors = int(checked.eq(ERROR).sum().item())
                if errors or completed == total:
                    break
    ended.record()
    ended.synchronize()
    gpu_seconds = started.elapsed_time(ended) / 1000.0
    wall_seconds = time.perf_counter() - wall_started
    component_ms: dict[str, float] = {}
    for name, begin, finish in component_events:
        component_ms[name] = component_ms.get(name, 0.0) + float(
            begin.elapsed_time(finish)
        )
    statuses = engine.statuses()
    completed = int(statuses.eq(TERMINAL).sum().item())
    errors = int(statuses.eq(ERROR).sum().item())
    validate_completion(total=total, completed=completed, errors=errors)

    output = {
        "schema_version": "0037_update32_cuda_rollout_benchmark_v1",
        "passed": True,
        "not_policy_strength_evidence": True,
        "official_cpu_engine_remains_oracle": True,
        "seed_contract_difference": (
            "CUDA reset consumes engine_seed only; the 0037 official seeded ABI also records "
            "a separate search_seed, so outcomes are not compared as exact EGRL parity."
        ),
        "scope": (
            "complete CUDA-resident engine + semantic0031-v2 observation + "
            "Update32 greedy actor + immutable 0806 greedy opponent"
        ),
        "collector": {
            "games": total,
            "completed_games": completed,
            "errors": errors,
            "decisions": decisions,
            "status_checks": status_checks,
            **routing_row_metrics(
                routing_mode=args.routing_mode,
                total=total,
                decisions=decisions,
                routed_rows=routed_rows,
            ),
            "routing_mode": args.routing_mode,
            "compact_semantic_prefixes": args.compact_semantic_prefixes,
            "shared_state_encoder": True,
            "shared_option_prefix_blocks": 1,
            "gpu_seconds": gpu_seconds,
            "wall_seconds": wall_seconds,
            "component_gpu_seconds": {
                name: milliseconds / 1000.0
                for name, milliseconds in component_ms.items()
            },
            "games_per_second_gpu": total / gpu_seconds,
            "games_per_second_wall": total / wall_seconds,
            "host_action_or_observation_copies": 0,
            "host_scalar_decode_bound_syncs": decisions,
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
            "source_jobs": len(full_schedule.engine_seeds),
            "used_jobs": total,
            "opponent_count": len(set(schedule.opponent_ids)),
        },
        "models": {
            **{key: str(value) for key, value in model_provenance.items()},
            "actor": "0037 V5 Update32 merged Last Option Q/V LoRA",
            "opponent": "immutable friend-0806 epoch11",
            "runtime_dtype": "float32",
            "math_mode": "TF32 matmul allowed",
        },
        "engine": {
            "rules": str(rules_path),
            "rules_sha256": sha256_file(rules_path),
            "extension": str(Path(_ptcg_cuda.__file__).resolve()),
            "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
            "allocated_bytes": int(engine.allocated_bytes),
            "semantic0031_v2": True,
        },
        "memory": {
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
