from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))
sys.path.insert(0, str(WORKSPACE_ROOT / "tools"))

from probe_actual_policy_residency import (  # noqa: E402
    checkpoint_state,
    load_module,
)
from ptcg_cuda_engine.legacy_codecs import (  # noqa: E402
    marnie_prize_control_v4_static_fields,
    policy_codec_v1_to_idonly_codec_v1,
)
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    AutocastDeviceAdapter,
    CompiledDeviceAdapter,
    EntityPointerPolicyV1DeviceAdapter,
    FoundationR15DeviceAdapter,
    IDOnlyPointerPolicyDeviceAdapter,
    StaticBatchFieldsDeviceAdapter,
    foundation_r15_static_fields,
)
from ptcg_cuda_engine.policy_pool import (  # noqa: E402
    GPUResidentPolicyPool,
    MAX_POLICIES,
    PolicyPoolManifest,
)
from pure_policy_model_v1 import load_policy_checkpoint  # noqa: E402
from run_official_seeded_reset_paired import read_deck  # noqa: E402


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3
ERROR_NAMES = {
    0: "none",
    1: "invalid_player",
    2: "invalid_card_ref",
    3: "invalid_area",
    4: "invalid_area_index",
    5: "zone_overflow",
    6: "option_overflow",
    7: "selection_overflow",
    8: "effect_stack_overflow",
    9: "continuation_stack_overflow",
    10: "trigger_stack_overflow",
    11: "turn_record_overflow",
    12: "effect_scratch_overflow",
    13: "rule_pack_bounds",
    14: "unsupported_effect",
    15: "interpreter_budget",
    16: "invalid_action",
    17: "deck_out",
    18: "unsupported_target",
    19: "unsupported_condition",
    20: "unsupported_continuation",
    6601207: "known_divergence_660_1207",
}
SUPPORTED_CODECS = {
    "foundation_0020_codec_v1",
    "policy_codec_v1",
    "idonly_codec_v1",
    "marnie_prize_codec_v4",
}
SUPPORTED_MATH_MODES = {"fp32", "tf32", "bf16", "fp16"}
MIXED_PRECISION_PRESETS: dict[str, dict[str, str]] = {
    "none": {},
    # Validated locally on RTX 3060 Laptop GPU:
    # - 24k same-state TF32-vs-candidate semantic gate: equiv_mismatches=0.
    # - b480/s120 resident rollout: 926.2 decisions/s vs 747.7 TF32 baseline.
    "safe_fp16_v1": {
        "pure_lucario_bc512_e4": "fp16",
        "kangaskhan_crustle_meanpool_epochmix_v2": "fp16",
        "marnie_prize_control_v4": "fp16",
    },
}


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Official CUDA state/codec/action loop with real "
            "heterogeneous frozen BC checkpoints and device-only seat routing."
        )
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--pool",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "configs"
        / "strongest_bc_opponent_pool_v1.json",
    )
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-select", type=int, default=80)
    parser.add_argument(
        "--route-capacity",
        type=int,
        default=0,
        help=(
            "Fixed per-policy route cohort capacity. The default 0 keeps the "
            "lossless full-seat capacity. Smaller values are allowed for "
            "performance experiments and remain fail-closed through the "
            "route_overflow_decisions gate."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=2026080101)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--math-mode",
        choices=tuple(sorted(SUPPORTED_MATH_MODES)),
        default="fp32",
        help=(
            "Frozen policy math contract. fp32 is the strict baseline; tf32 "
            "bf16 and fp16 require separate battle A/B promotion."
        ),
    )
    parser.add_argument(
        "--policy-math-override",
        action="append",
        default=[],
        metavar="POLICY=MODE",
        help=(
            "Override one frozen policy's inference dtype while keeping the "
            "global math mode for the rest, e.g. pure_lucario_bc512_e4=fp16. "
            "Repeat for multiple policies."
        ),
    )
    parser.add_argument(
        "--mixed-precision-preset",
        choices=tuple(sorted(MIXED_PRECISION_PRESETS)),
        default="none",
        help=(
            "Apply a validated policy-level math-mode preset. Explicit "
            "--policy-math-override entries take precedence."
        ),
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help=(
            "Policy name to include; repeat for several. By default all pool "
            "entries whose resident device codec is implemented are included."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "artifacts"
        / "official_seeded_multi_policy_loop.json",
    )
    parser.add_argument(
        "--debug-error-lanes",
        action="store_true",
        help=(
            "Synchronize after each apply, stop at the first error, and dump "
            "the failing lane's packed action and codec rows."
        ),
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help=(
            "Synchronize around each rollout phase and report diagnostic wall "
            "time. This is intentionally not a hot-path benchmark."
        ),
    )
    parser.add_argument(
        "--profile-policy-adapters",
        action="store_true",
        help=(
            "When profiling phases, also synchronize inside policy dispatch and "
            "report per-policy adapter/postprocess/scatter wall time. This is "
            "diagnostic only and is not a hot-path benchmark."
        ),
    )
    parser.add_argument(
        "--compile-policy-adapters",
        action="store_true",
        help=(
            "Experimentally wrap each resident policy adapter with torch.compile. "
            "Compilation is warmed up before timed rollout."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="torch.compile mode used by --compile-policy-adapters.",
    )
    return parser.parse_args()


def parse_policy_math_overrides(
    items: list[str],
    *,
    preset: str = "none",
) -> dict[str, str]:
    if preset not in MIXED_PRECISION_PRESETS:
        raise ValueError(f"unknown mixed precision preset: {preset}")
    overrides: dict[str, str] = dict(MIXED_PRECISION_PRESETS[preset])
    for item in items:
        if "=" not in item:
            raise ValueError(f"bad --policy-math-override {item!r}; expected POLICY=MODE")
        name, mode = item.split("=", 1)
        name = name.strip()
        mode = mode.strip().lower()
        if not name:
            raise ValueError("policy override name must not be empty")
        if mode not in SUPPORTED_MATH_MODES:
            raise ValueError(f"unsupported policy math mode override: {mode}")
        overrides[name] = mode
    return overrides


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool_rows(path: Path, selected_names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("opponents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("strongest BC pool has no opponents")
    requested = set(selected_names)
    known = {str(row.get("name")) for row in rows}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown requested policies: {sorted(unknown)}")
    candidates = [row for row in rows if not requested or row.get("name") in requested]
    supported = [row for row in candidates if row.get("codec") in SUPPORTED_CODECS]
    unsupported = [row for row in candidates if row.get("codec") not in SUPPORTED_CODECS]
    if not supported:
        raise RuntimeError("no requested policy has a resident device codec")
    return supported, unsupported


def registered_deck_fields(deck: list[int], device: Any) -> dict[str, Any]:
    import torch

    if len(deck) != 60 or any(card <= 0 for card in deck):
        raise ValueError("registered deck must contain 60 positive card IDs")
    counts = Counter(deck)
    return {
        "deck_ids": torch.tensor(deck, dtype=torch.long, device=device).view(1, 60),
        "deck_counts": torch.tensor(
            [counts[card] for card in deck], dtype=torch.long, device=device
        ).view(1, 60),
    }


def load_0020_runtime_package() -> str:
    package_name = "_ptcg_0020_foundation_runtime"
    if package_name in sys.modules:
        return package_name
    package_root = WORKSPACE_ROOT / "train" / "0020_pluggable_deck_rl"
    init_path = package_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load 0020 runtime package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return package_name


def load_foundation_0020_checkpoint(
    checkpoint: Path,
    ontology_path: Path,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    package_name = load_0020_runtime_package()
    inference = importlib.import_module(f"{package_name}.inference")
    source_r15 = importlib.import_module(f"{package_name}.source_r15_model")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"model", "epoch", "global_step", "metadata"}:
        raise ValueError("0020 foundation checkpoint is not the audited payload")
    config, source_config = inference._model_config(payload["metadata"])
    model = source_r15.SourceConditionedR15Policy(
        config,
        source_config,
        ontology_path=ontology_path,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device)
    return model, config


def load_adapters(
    rows: list[dict[str, Any]],
    device: Any,
    max_select: int,
    math_mode: str,
    *,
    compile_policy_adapters: bool = False,
    compile_mode: str = "reduce-overhead",
    policy_math_overrides: Mapping[str, str] | None = None,
) -> tuple[
    PolicyPoolManifest,
    dict[int, Any],
    list[list[int]],
    list[dict[str, Any]],
    dict[str, int],
]:
    import torch

    overrides = dict(policy_math_overrides or {})
    row_names = {str(row["name"]) for row in rows}
    unknown_overrides = sorted(set(overrides) - row_names)
    if unknown_overrides:
        raise ValueError(f"unknown policy math override names: {unknown_overrides}")
    specs: list[dict[str, Any]] = []
    adapters: dict[int, Any] = {}
    decks: list[list[int]] = []
    report: list[dict[str, Any]] = []
    model_cache: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
    loader_stats = {
        "model_instances_loaded": 0,
        "model_cache_hits": 0,
    }

    def prepare_model(model: Any, mode: str) -> Any:
        model.eval()
        model.requires_grad_(False)
        if mode in {"bf16", "fp16"}:
            dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            model.to(dtype=dtype)
        return model
    for policy_id, row in enumerate(rows):
        policy_name = str(row["name"])
        effective_math_mode = overrides.get(policy_name, math_mode)
        directory = (WORKSPACE_ROOT / str(row["directory"])).resolve()
        checkpoint = (directory / str(row["checkpoint"])).resolve()
        deck_path = directory / "deck.csv"
        for required in (checkpoint, deck_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        deck = read_deck(deck_path)
        codec = str(row["codec"])
        adapter_name = str(row["adapter"])
        if codec == "policy_codec_v1":
            cache_key = (codec, adapter_name, str(checkpoint), effective_math_mode)
            cached = model_cache.get(cache_key)
            if cached is None:
                model, _payload = load_policy_checkpoint(checkpoint, device)
                config = model.config
                prepare_model(model, effective_math_mode)
                model_cache[cache_key] = (model, config)
                loader_stats["model_instances_loaded"] += 1
            else:
                model, config = cached
                loader_stats["model_cache_hits"] += 1
            adapter: Any = EntityPointerPolicyV1DeviceAdapter(
                model, max_select=max_select
            )
        elif codec == "foundation_0020_codec_v1":
            ontology = (
                WORKSPACE_ROOT / str(row["ontology"])
                if row.get("ontology")
                else checkpoint.parent.parent / "artifact" / "card_ontology.json"
            ).resolve()
            if not ontology.is_file():
                raise FileNotFoundError(ontology)
            cache_key = (
                codec,
                adapter_name,
                str(checkpoint),
                str(ontology),
                effective_math_mode,
            )
            cached = model_cache.get(cache_key)
            if cached is None:
                model, config = load_foundation_0020_checkpoint(
                    checkpoint, ontology, device
                )
                prepare_model(model, effective_math_mode)
                model_cache[cache_key] = (model, config)
                loader_stats["model_instances_loaded"] += 1
            else:
                model, config = cached
                loader_stats["model_cache_hits"] += 1
            adapter = FoundationR15DeviceAdapter(
                model,
                max_select=min(int(config.ac.base.max_action_steps), max_select),
            )
            adapter = StaticBatchFieldsDeviceAdapter(
                adapter,
                foundation_r15_static_fields(deck, device),
            )
        elif codec in {"idonly_codec_v1", "marnie_prize_codec_v4"}:
            source = directory / "idonly_policy.py"
            cache_key = (
                codec,
                adapter_name,
                str(source),
                str(checkpoint),
                effective_math_mode,
            )
            cached = model_cache.get(cache_key)
            if cached is None:
                module = load_module(
                    source, f"_resident_pool_{policy_id}_{directory.name}"
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                config = module.ModelConfig(**payload["model_config"])
                model = module.IDOnlyPointerPolicy(config)
                model.load_state_dict(checkpoint_state(payload), strict=True)
                model.to(device=device)
                prepare_model(model, effective_math_mode)
                model_cache[cache_key] = (model, config)
                loader_stats["model_instances_loaded"] += 1
            else:
                model, config = cached
                loader_stats["model_cache_hits"] += 1
            adapter = IDOnlyPointerPolicyDeviceAdapter(
                model, max_select=int(config.max_action_steps)
            )
            static_fields: dict[str, Any] = {}
            if hasattr(model, "deck_count"):
                static_fields.update(registered_deck_fields(deck, device))
            if codec == "marnie_prize_codec_v4":
                prize_mode = str(getattr(config, "prize_feature_mode", ""))
                if prize_mode != "control":
                    raise RuntimeError(
                        "a belief-mode Marnie checkpoint requires the stateful "
                        "resident PrizeLedger codec"
                    )
                static_fields.update(
                    marnie_prize_control_v4_static_fields(deck, device)
                )
            if static_fields:
                adapter = StaticBatchFieldsDeviceAdapter(
                    adapter, static_fields
                )
        else:
            raise AssertionError(codec)
        if effective_math_mode in {"bf16", "fp16"}:
            dtype = torch.bfloat16 if effective_math_mode == "bf16" else torch.float16
            adapter = AutocastDeviceAdapter(adapter, dtype)
        if compile_policy_adapters:
            adapter = CompiledDeviceAdapter(adapter, mode=compile_mode)
        specs.append(
            {
                "policy_id": policy_id,
                "name": policy_name,
                "deck": str(deck_path),
                "checkpoint": str(checkpoint),
                "adapter": adapter_name,
                "codec": codec,
                "frozen": True,
                "dtype": (
                    effective_math_mode
                    if effective_math_mode in {"bf16", "fp16"}
                    else "fp32"
                ),
            }
        )
        adapters[policy_id] = adapter
        decks.append(deck)
        report.append(
            {
                "policy_id": policy_id,
                "name": policy_name,
                "codec": codec,
                "adapter": adapter_name,
                "math_mode": effective_math_mode,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "max_entities": int(getattr(config, "max_entities", 128)),
                "max_options": int(getattr(config, "max_options", 80)),
                "max_action_steps": int(
                    getattr(config, "max_action_steps", max_select)
                ),
                "prize_contract": (
                    "static_registered_control_v4"
                    if codec == "marnie_prize_codec_v4"
                    else None
                ),
                "checkpoint_sha256": sha256_file(checkpoint),
                "deck_sha256": sha256_file(deck_path),
            }
        )
    manifest = PolicyPoolManifest.from_dict(
        {
            "name": "official_seeded_strongest_bc_resident_pool",
            "max_policies": MAX_POLICIES,
            "routing": {
                "mode": "balanced_fixed_padded_device_cohorts",
                "read_counts_on_host": False,
            },
            "policies": specs,
        }
    )
    return manifest, adapters, decks, report, loader_stats


def balanced_seat_schedule(batch: int, policy_count: int) -> list[tuple[int, int]]:
    if policy_count == 1:
        return [(0, 0)] * batch
    return [
        (
            env % policy_count,
            (env % policy_count + 1 + env // policy_count) % policy_count,
        )
        for env in range(batch)
    ]


def codec_batch(engine: Any) -> dict[str, Any]:
    encoded = dict(engine.encode_policy_v1())
    encoded["entity_mask"] = encoded["entity_mask"].bool()
    encoded["option_mask"] = encoded["option_mask"].bool()
    return encoded


def error_diagnostics(engine: Any, schedule: list[tuple[int, int]], policies: tuple[Any, ...]) -> dict[str, Any]:
    states = engine.state_bytes().cpu().numpy()
    statuses = engine.statuses().cpu().numpy().tolist()
    rows: list[dict[str, Any]] = []
    histogram: Counter[str] = Counter()
    for lane, status in enumerate(statuses):
        if int(status) != ERROR:
            continue
        error = struct.unpack_from("<i", states[lane].tobytes(), 4)[0]
        detail = struct.unpack_from("<i", states[lane].tobytes(), 8)[0]
        name = ERROR_NAMES.get(error, f"unknown_{error}")
        histogram[name] += 1
        left_id, right_id = schedule[lane]
        rows.append(
            {
                "lane": lane,
                "status": int(status),
                "error": error,
                "error_name": name,
                "error_detail": detail,
                "seat_policy_ids": [left_id, right_id],
                "seat_policy_names": [
                    policies[left_id].name,
                    policies[right_id].name,
                ],
            }
        )
    return {
        "error_histogram": dict(histogram),
        "first_error_lanes": rows[:16],
    }


def debug_error_lanes(
    engine: Any,
    lanes: list[int],
    *,
    step: int,
    ready: Any,
    acting_policy_ids: Any,
    actions: Any,
    encoded_by_codec: dict[str, dict[str, Any]],
    policies: tuple[Any, ...],
) -> dict[str, Any]:
    packed_rows = engine.action_bytes().detach().cpu()
    rows: list[dict[str, Any]] = []
    for lane in lanes[:16]:
        policy_id = int(acting_policy_ids[lane].item())
        policy = policies[policy_id]
        codec = encoded_by_codec[policy.codec]
        option_mask = codec["option_mask"][lane].bool()
        valid_options = option_mask.nonzero(as_tuple=False).flatten()
        action_length = int(actions.lengths[lane].item())
        normalized_indices = [
            int(value)
            for value in actions.indices[lane, :action_length].detach().cpu().tolist()
        ]
        packed = bytes(packed_rows[lane].tolist())
        packed_count = struct.unpack_from("<H", packed, 256)[0]
        packed_indices = list(struct.unpack_from("<128H", packed, 0))[:packed_count]
        option_rows = []
        for option_index in valid_options.detach().cpu().tolist():
            option_row = {
                "index": int(option_index),
                "cat": codec["option_cat"][lane, option_index].detach().cpu().tolist(),
            }
            if "option_num" in codec:
                option_row["num"] = (
                    codec["option_num"][lane, option_index].detach().cpu().tolist()
                )
            if "option_equiv" in codec:
                option_row["equiv"] = int(
                    codec["option_equiv"][lane, option_index].item()
                )
            option_rows.append(option_row)
        rows.append(
            {
                "lane": lane,
                "step": step,
                "ready_before_apply": bool(ready[lane].item()),
                "policy_id": policy_id,
                "policy_name": policy.name,
                "codec": policy.codec,
                "min_count": int(codec["min_count"][lane].item()),
                "max_count": int(codec["max_count"][lane].item()),
                "option_count": int(valid_options.numel()),
                "normalized_action_length": action_length,
                "normalized_action_indices": normalized_indices,
                "packed_action_count": packed_count,
                "packed_action_indices": packed_indices,
                "global_cat": codec["global_cat"][lane].detach().cpu().tolist(),
                "global_num": codec["global_num"][lane].detach().cpu().tolist(),
                "options": option_rows,
            }
        )
    return {"first_error_step": step, "lanes": rows}


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("batch/steps must be positive and warmup non-negative")
    if not 1 <= args.max_select <= 80:
        raise ValueError("max-select must be in [1, 80]")

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    if args.math_mode == "tf32":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rules = args.rules.resolve()
    pool_path = args.pool.resolve()
    for required in (rules, pool_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rows, unsupported_rows = load_pool_rows(pool_path, args.policy)
    policy_math_overrides = parse_policy_math_overrides(
        args.policy_math_override,
        preset=args.mixed_precision_preset,
    )
    manifest, adapters, registered_decks, policy_report, loader_stats = load_adapters(
        rows,
        device,
        args.max_select,
        args.math_mode,
        compile_policy_adapters=args.compile_policy_adapters,
        compile_mode=args.compile_mode,
        policy_math_overrides=policy_math_overrides,
    )
    policy_count = len(manifest.policies)
    schedule = balanced_seat_schedule(args.batch, policy_count)
    seat_counts = Counter(policy for pair in schedule for policy in pair)
    full_route_capacity = max(seat_counts.values())
    if args.route_capacity < 0:
        raise ValueError("--route-capacity must be non-negative")
    route_capacity = args.route_capacity or full_route_capacity
    if route_capacity > args.batch:
        raise ValueError("--route-capacity cannot exceed --batch")
    seat_policy_ids = torch.tensor(schedule, dtype=torch.long, device=device)
    decks = torch.tensor(
        [[registered_decks[left], registered_decks[right]] for left, right in schedule],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.arange(
        args.seed_start,
        args.seed_start + args.batch,
        dtype=torch.int64,
        device=device,
    )
    lane_mask = torch.ones(args.batch, dtype=torch.bool, device=device)

    engine = create_official_engine(
        rules.read_bytes(), batch_size=args.batch, device_index=args.device_index
    )
    policy_pool = GPUResidentPolicyPool(
        manifest,
        adapters,
        capacity=route_capacity,
        max_select=args.max_select,
    )
    engine.reset_seeded_interactive_masked(decks, seeds, lane_mask)
    decision_counter = torch.zeros((), dtype=torch.int64, device=device)
    episode_counter = torch.zeros((), dtype=torch.int64, device=device)
    overflow_counter = torch.zeros((), dtype=torch.int64, device=device)
    action_bound_overflow = torch.zeros((), dtype=torch.int64, device=device)
    route_totals = torch.zeros(policy_count, dtype=torch.int64, device=device)
    phase_wall_ms: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    policy_profile: dict[str, Any] | None = {} if args.profile_policy_adapters else None

    needs_idonly = bool(
        {
            "foundation_0020_codec_v1",
            "idonly_codec_v1",
            "marnie_prize_codec_v4",
        }
        & set(manifest.codecs)
    )
    idonly_entities = max(
        (
            row["max_entities"]
            for row in policy_report
            if row["codec"] in {"idonly_codec_v1", "marnie_prize_codec_v4"}
            or row["codec"] == "foundation_0020_codec_v1"
        ),
        default=128,
    )
    idonly_options = max(
        (
            row["max_options"]
            for row in policy_report
            if row["codec"] in {"idonly_codec_v1", "marnie_prize_codec_v4"}
            or row["codec"] == "foundation_0020_codec_v1"
        ),
        default=80,
    )

    error_debug: dict[str, Any] | None = None

    def phase(name: str, fn: Any) -> Any:
        if not args.profile_phases:
            return fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize(device)
        phase_wall_ms[name] += (time.perf_counter() - start) * 1000.0
        phase_counts[name] += 1
        return value

    def one_resident_step(*, count_metrics: bool, step: int) -> bool:
        nonlocal error_debug
        def reset_status_phase() -> Any:
            terminal_local = engine.statuses().eq(TERMINAL)
            seeds.add_(terminal_local.to(torch.int64) * args.batch)
            engine.reset_seeded_interactive_masked(decks, seeds, terminal_local)
            return terminal_local

        terminal = phase("reset_status", reset_status_phase)
        if count_metrics:
            episode_counter.add_(terminal.long().sum())
        phase("advance_to_decision", engine.advance_to_decision)

        def encode_policy_phase() -> tuple[dict[str, Any], Any, Any]:
            policy_codec_local = codec_batch(engine)
            ready_local = engine.statuses().eq(NEEDS_ACTION)
            actor_local = (
                policy_codec_local["global_cat"][:, 3].long() - 1
            ).clamp(min=0, max=1)
            acting_policy_ids_local = seat_policy_ids.gather(
                1, actor_local.view(-1, 1)
            ).view(-1)
            return policy_codec_local, ready_local, acting_policy_ids_local

        policy_codec, ready, acting_policy_ids = phase(
            "encode_policy_v1", encode_policy_phase
        )
        encoded_by_codec: dict[str, dict[str, Any]] = {
            "policy_codec_v1": policy_codec
        }
        if needs_idonly:
            idonly_codec = phase(
                "idonly_conversion",
                lambda: policy_codec_v1_to_idonly_codec_v1(
                    policy_codec,
                    max_card_id=2048,
                    max_action_steps=args.max_select,
                    target_entity_capacity=idonly_entities,
                    target_option_capacity=idonly_options,
                ),
            )
            encoded_by_codec["idonly_codec_v1"] = idonly_codec
            encoded_by_codec["marnie_prize_codec_v4"] = idonly_codec
            encoded_by_codec["foundation_0020_codec_v1"] = idonly_codec
        actions = phase(
            "policy_pool_act",
            lambda: policy_pool.act(
                encoded_by_codec,
                acting_policy_ids,
                ready,
                policy_profile=policy_profile,
            ),
        )

        def pack_apply_phase() -> None:
            engine.pack_actions(actions.indices, actions.lengths)
            engine.apply_packed_actions()

        phase("pack_apply", pack_apply_phase)
        if count_metrics:
            decision_counter.add_(ready.long().sum())
            route_totals.add_(actions.route_counts)
            overflow_counter.add_(
                (actions.route_counts - route_capacity).clamp_min(0).sum()
            )
            action_bound_overflow.add_(
                (ready & policy_codec["max_count"].long().gt(args.max_select))
                .long()
                .sum()
            )
        if args.debug_error_lanes:
            error_lanes = (
                engine.statuses()
                .eq(ERROR)
                .nonzero(as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            if error_lanes:
                error_debug = debug_error_lanes(
                    engine,
                    [int(lane) for lane in error_lanes],
                    step=step,
                    ready=ready,
                    acting_policy_ids=acting_policy_ids,
                    actions=actions,
                    encoded_by_codec=encoded_by_codec,
                    policies=manifest.policies,
                )
                return True
        return False

    with torch.inference_mode():
        step = 0
        stopped_on_error = False
        for _ in range(args.warmup):
            stopped_on_error = one_resident_step(count_metrics=False, step=step)
            step += 1
            if stopped_on_error:
                break
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nvtx_range = "official_seeded_multi_bc_resident_loop"
        torch.cuda.nvtx.range_push(nvtx_range)
        start.record()
        if not stopped_on_error:
            for _ in range(args.steps):
                stopped_on_error = one_resident_step(count_metrics=True, step=step)
                step += 1
                if stopped_on_error:
                    break
        end.record()
        torch.cuda.nvtx.range_pop()
        end.synchronize()
    elapsed_sec = start.elapsed_time(end) / 1000.0

    decisions = int(decision_counter.item())
    episodes = int(episode_counter.item())
    route_overflows = int(overflow_counter.item())
    bound_overflows = int(action_bound_overflow.item())
    route_counts = route_totals.cpu().tolist()
    status_counts = torch.bincount(engine.statuses().long(), minlength=4).cpu().tolist()
    diagnostics = error_diagnostics(engine, schedule, manifest.policies)
    phase_profile = None
    if args.profile_phases:
        total_ms = sum(float(value) for value in phase_wall_ms.values())
        phase_profile = {
            "mode": "synchronized_diagnostic_not_hot_path",
            "total_wall_ms": total_ms,
            "phases": {
                name: {
                    "calls": int(phase_counts[name]),
                    "wall_ms": float(phase_wall_ms[name]),
                    "avg_wall_ms": (
                        float(phase_wall_ms[name]) / int(phase_counts[name])
                        if phase_counts[name]
                        else 0.0
                    ),
                    "share": (
                        float(phase_wall_ms[name]) / total_ms
                        if total_ms
                        else 0.0
                    ),
                }
                for name in sorted(phase_wall_ms)
            },
        }
    policy_adapter_profile = None
    if policy_profile is not None:
        adapter_ms = dict(policy_profile.get("adapter_wall_ms", {}))
        postprocess_ms = dict(policy_profile.get("postprocess_wall_ms", {}))
        scatter_ms = dict(policy_profile.get("scatter_wall_ms", {}))
        calls = dict(policy_profile.get("calls", {}))
        policy_names = {policy.name for policy in manifest.policies}
        shared_names = sorted(
            (set(adapter_ms) | set(postprocess_ms) | set(scatter_ms) | set(calls))
            - policy_names
        )
        policy_adapter_profile = {
            "mode": "synchronized_diagnostic_not_hot_path",
            "policies": {
                policy.name: {
                    "calls": int(calls.get(policy.name, 0)),
                    "adapter_wall_ms": float(adapter_ms.get(policy.name, 0.0)),
                    "adapter_avg_wall_ms": (
                        float(adapter_ms.get(policy.name, 0.0))
                        / int(calls.get(policy.name, 1))
                    ),
                    "postprocess_wall_ms": float(postprocess_ms.get(policy.name, 0.0)),
                    "postprocess_avg_wall_ms": (
                        float(postprocess_ms.get(policy.name, 0.0))
                        / int(calls.get(policy.name, 1))
                    ),
                    "scatter_wall_ms": float(scatter_ms.get(policy.name, 0.0)),
                    "scatter_avg_wall_ms": (
                        float(scatter_ms.get(policy.name, 0.0))
                        / int(calls.get(policy.name, 1))
                    ),
                }
                for policy in manifest.policies
            },
            "shared_groups": {
                name: {
                    "calls": int(calls.get(name, 0)),
                    "adapter_wall_ms": float(adapter_ms.get(name, 0.0)),
                    "adapter_avg_wall_ms": (
                        float(adapter_ms.get(name, 0.0))
                        / int(calls.get(name, 1))
                    ),
                    "postprocess_wall_ms": float(postprocess_ms.get(name, 0.0)),
                    "postprocess_avg_wall_ms": (
                        float(postprocess_ms.get(name, 0.0))
                        / int(calls.get(name, 1))
                    ),
                    "scatter_wall_ms": float(scatter_ms.get(name, 0.0)),
                    "scatter_avg_wall_ms": (
                        float(scatter_ms.get(name, 0.0))
                        / int(calls.get(name, 1))
                    ),
                }
                for name in shared_names
            },
        }
    result = {
        "schema_version": 1,
        "passed": (
            decisions > 0
            and route_overflows == 0
            and bound_overflows == 0
            and status_counts[ERROR] == 0
        ),
        "scope": (
            "Official seeded CUDA setup/Main/Battle with heterogeneous real "
            "frozen BCs, device codec conversion/routing/action/reset"
        ),
        "batch": args.batch,
        "warmup": args.warmup,
        "steps": args.steps,
        "policies_loaded": policy_count,
        "route_capacity": route_capacity,
        "full_route_capacity": full_route_capacity,
        "max_select": args.max_select,
        "policy_math_mode": args.math_mode,
        "mixed_precision_preset": args.mixed_precision_preset,
        "policy_math_overrides": policy_math_overrides,
        "compile_policy_adapters": bool(args.compile_policy_adapters),
        "compile_mode": args.compile_mode if args.compile_policy_adapters else None,
        "decisions": decisions,
        "episodes_reset_on_device": episodes,
        "elapsed_sec": elapsed_sec,
        "decisions_per_sec": decisions / elapsed_sec if elapsed_sec else 0.0,
        "completed_games_per_sec": episodes / elapsed_sec if elapsed_sec else 0.0,
        "route_overflow_decisions": route_overflows,
        "action_bound_overflow_decisions": bound_overflows,
        "route_counts": {
            policy.name: int(route_counts[policy.policy_id])
            for policy in manifest.policies
        },
        "status_counts": {
            "idle": int(status_counts[0]),
            "needs_action": int(status_counts[1]),
            "terminal": int(status_counts[2]),
            "error": int(status_counts[3]),
        },
        "diagnostics": diagnostics,
        "error_debug": error_debug,
        "phase_profile": phase_profile,
        "policy_adapter_profile": policy_adapter_profile,
        "policies": policy_report,
        "loader_stats": loader_stats,
        "unsupported_pool_entries": [
            {
                "name": str(row.get("name")),
                "codec": str(row.get("codec")),
                "reason": "resident device codec is not implemented yet",
            }
            for row in unsupported_rows
        ],
        "hot_path": {
            "cpu_engine_calls": 0,
            "cpu_policy_calls": 0,
            "h2d_copies": 0,
            "d2h_copies": 0,
            "host_synchronizations": 0,
            "final_measurement_synchronizations": 1,
            "device_seeded_masked_reset": True,
            "device_policy_codec_v1": True,
            "device_idonly_codec_v1": needs_idonly,
            "device_policy_routing": True,
            "device_action_apply": True,
        },
        "memory": {
            "official_arena_bytes": int(engine.allocated_bytes),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rule_pack_sha256": sha256_file(rules),
        "pool_sha256": sha256_file(pool_path),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "nvtx_hot_range": nvtx_range,
        "limitations": [
            "The promoted Marnie v4 checkpoint is exact through its frozen control-mode contract; any future belief-mode checkpoint still requires a stateful resident PrizeLedger.",
            "Official resident codec capacity is still 128 entities/80 options; padding to legacy 192/128 does not recover overflowed source rows.",
            "Use the default official max_select=80 for full safety; smaller horizons are diagnostic only.",
            "Frozen-policy rollout is validated here; CUDA PPO learner integration is covered by run_foundation_ppo_cuda_smoke.py.",
            "5090 throughput and Nsight transfer/synchronization claims require server validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
