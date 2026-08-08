"""Measure complete CUDA-native 0035 rollouts over the frozen opponent pool.

This benchmark intentionally keeps all game state, semantic-v2 materialization,
learner/opponent decoding and action application on CUDA.  It uses a static
concurrent batch for one frozen schedule, so it measures a real complete-game
collector throughput without claiming that it is the final 512-game PPO
collector (which also needs streaming lane replacement and model-only formal
checkpointing).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.semantic0031_bridge import semantic0031_v2_ready_batch  # noqa: E402
from run_0035_cuda_ppo_collector_probe import (  # noqa: E402
    CudaSemantic0031ActorCritic,
    DEFAULT_PROJECT,
    DEFAULT_VERSION,
    ERROR,
    EXPECTED_CHECKPOINT_SHA256,
    NEEDS_ACTION,
    TERMINAL,
    _decode_device,
    _compact_semantic_prefixes,
    _SEMANTIC_PREFIX_FAMILIES,
    _load_first50_snapshot,
    _read_deck,
)


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="Benchmark complete CUDA-native semantic0031 games over the frozen 0035 pool."
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "rl_runs"
            / DEFAULT_PROJECT
            / "versions"
            / DEFAULT_VERSION
            / "artifact/opponent_snapshot.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "bc_models/semantic0031_0806_shared_prototype_fp32.pt",
    )
    parser.add_argument(
        "--focal-deck",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "train/0034_dragapult_third_large_model_rl/league/decks"
            / "mega_lucario_ex_solrock_77a53ffc32f8/deck.csv"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="50 runs each snapshot deck once; 512 uses the audited weight scaling.",
    )
    parser.add_argument(
        "--opponent-limit",
        type=int,
        default=50,
        help="Use the first N audited opponents, preserving their snapshot weights.",
    )
    parser.add_argument("--max-decisions", type=int, default=1024)
    parser.add_argument("--check-interval", type=int, default=32)
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--seed", type=int, default=350031001)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--profile-components",
        action="store_true",
        help="Record CUDA event timings for the major resident-loop components.",
    )
    parser.add_argument(
        "--encoder-execution",
        choices=("eager", "cuda_graph"),
        default="eager",
        help="Execute the frozen shared semantic encoder eagerly or by CUDA Graph replay.",
    )
    parser.add_argument(
        "--compact-semantic-prefixes",
        action="store_true",
        help="Trim mask-excluded right padding before frozen semantic inference.",
    )
    parser.add_argument(
        "--capture-error-trace",
        action="store_true",
        help="Retain device action tensors and serialize only failing lanes for replay.",
    )
    parser.add_argument(
        "--capture-trace-path",
        type=Path,
        help="Write the complete diagnostic action trace for engine-only replay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_schedule(catalog: list[Any], total: int) -> list[Any]:
    if total == len(catalog):
        return list(catalog)
    source_total = sum(int(item.games) for item in catalog)
    raw = [total * int(item.games) / source_total for item in catalog]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(catalog)),
        key=lambda index: (
            -(raw[index] - counts[index]),
            int(catalog[index].best_rank),
            str(catalog[index].deck_id),
        ),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if any(count < 1 for count in counts):
        raise ValueError("weighted schedule dropped an audited opponent")
    schedule = [item for item, count in zip(catalog, counts, strict=True) for _ in range(count)]
    if len(schedule) != total:
        raise AssertionError("weighted schedule has the wrong length")
    return schedule


def _make_deck_tensor(schedule: list[Any], focal_deck: tuple[int, ...], device: Any) -> tuple[Any, Any]:
    import torch

    focal_player = torch.arange(len(schedule), device=device, dtype=torch.long).remainder(2)
    rows: list[list[tuple[int, ...]]] = []
    for player, opponent in zip(focal_player.tolist(), schedule, strict=True):
        opponent_deck = tuple(int(card) for card in opponent.deck)
        rows.append(
            [focal_deck, opponent_deck]
            if player == 0
            else [opponent_deck, focal_deck]
        )
    return torch.tensor(rows, dtype=torch.int32, device=device), focal_player


def _error_diagnostics(engine: Any, schedule: list[Any]) -> list[dict[str, Any]]:
    states = engine.state_bytes().cpu().numpy()
    statuses = engine.statuses().cpu().tolist()
    names = {
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
    rows: list[dict[str, Any]] = []
    for lane, status in enumerate(statuses):
        if int(status) != ERROR:
            continue
        state = states[lane].tobytes()
        error = struct.unpack_from("<i", state, 4)[0]
        detail = struct.unpack_from("<i", state, 8)[0]
        rows.append(
            {
                "lane": lane,
                "deck_id": str(schedule[lane].deck_id),
                "deck_sha256": str(schedule[lane].deck_sha256),
                "error": error,
                "error_name": names.get(error, f"unknown_{error}"),
                "error_detail": detail,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not 1 <= args.opponent_limit <= 50:
        raise ValueError("opponent-limit must be in [1, 50]")
    if args.games < args.opponent_limit:
        raise ValueError("games must be at least opponent-limit")
    if args.compact_semantic_prefixes and args.encoder_execution == "cuda_graph":
        raise ValueError("dynamic semantic prefix compaction is incompatible with one fixed CUDA Graph")
    if min(args.max_decisions, args.check_interval, args.max_select) <= 0:
        raise ValueError("max-decisions, check-interval and max-select must be positive")
    if not 1 <= args.max_select <= 64:
        raise ValueError("max-select must be in [1, 64]")
    for path in (args.rules, args.snapshot, args.checkpoint, args.focal_deck):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows, catalog = _load_first50_snapshot(args.snapshot.resolve())
    selected_catalog = catalog[: args.opponent_limit]
    schedule = _weighted_schedule(selected_catalog, args.games)
    focal_deck = _read_deck(args.focal_deck.resolve())
    loader = importlib.import_module(
        "train.0034_dragapult_third_large_model_rl.policy.compact_actor_critic"
    )
    loaded, source, storage = loader.load_compact_actor_critic(
        args.checkpoint.resolve(),
        focal_deck,
        device=device,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    model = CudaSemantic0031ActorCritic.build(
        loaded.actor, max_select=args.max_select
    ).to(device).eval()
    model.policy.eval()
    model.value_head.eval()
    opponent_decoder = copy.deepcopy(model.policy.action_decoder).to(device).eval()
    opponent_decoder.requires_grad_(False)
    decks, focal_player = _make_deck_tensor(schedule, focal_deck, device)
    batch_size = len(schedule)
    seeds = torch.arange(args.seed, args.seed + batch_size, dtype=torch.int64, device=device)
    lanes = torch.arange(batch_size, dtype=torch.int32, device=device)
    engine = create_official_engine(
        args.rules.resolve().read_bytes(),
        batch_size=batch_size,
        device_index=args.device_index,
    )

    def reset() -> None:
        engine.reset_seeded_interactive_semantic(decks, seeds)
        engine.advance_to_decision()

    # Materialize model kernels and all engine paths before measurement.
    reset()
    with torch.inference_mode():
        raw_semantic = engine.encode_semantic0031_v2_lanes(lanes)
        full_semantic = semantic0031_v2_ready_batch(
            raw_semantic,
            max_action_steps=args.max_select,
        )
        # Parity validation does not need to materialize the frozen 56M model
        # over every rollout lane.  Bound this preflight batch so large resident
        # collectors spend memory on the measured compact path instead.
        validation_rows = min(batch_size, 64)
        full_semantic = {
            name: value[:validation_rows] for name, value in full_semantic.items()
        }
        semantic, warmup_prefix_widths = (
            _compact_semantic_prefixes(full_semantic)
            if args.compact_semantic_prefixes
            else (full_semantic, {})
        )
        validated, summary, options = model.encode(semantic)
        _uncached_validated, uncached_summary, uncached_options = model.encode_uncached(
            semantic
        )
        cache_summary_max_abs = float((summary - uncached_summary).abs().max().item())
        cache_options_max_abs = float((options - uncached_options).abs().max().item())
        cache_exact = torch.equal(summary, uncached_summary) and torch.equal(
            options, uncached_options
        )
        if not cache_exact:
            raise RuntimeError(
                "cached prototype encoding changed semantic outputs: "
                f"summary={cache_summary_max_abs} options={cache_options_max_abs}"
            )
        compaction_summary_max_abs = 0.0
        compaction_options_max_abs = 0.0
        compaction_logprob_max_abs = 0.0
        compaction_greedy_exact = True
        if args.compact_semantic_prefixes:
            full_validated, full_summary, full_options = model.encode(full_semantic)
            compaction_summary_max_abs = float(
                (summary - full_summary).abs().max().item()
            )
            compaction_options_max_abs = float(
                (options - full_options[:, : options.shape[1]]).abs().max().item()
            )
            compact_greedy = _decode_device(
                model.policy.action_decoder,
                validated,
                summary,
                options,
                max_select=args.max_select,
                greedy=True,
            )
            full_greedy = _decode_device(
                model.policy.action_decoder,
                full_validated,
                full_summary,
                full_options,
                max_select=args.max_select,
                greedy=True,
            )
            compaction_greedy_exact = torch.equal(
                compact_greedy["actions"], full_greedy["actions"]
            ) and torch.equal(compact_greedy["lengths"], full_greedy["lengths"])
            compact_record = dict(semantic)
            compact_record.update(
                {
                    "targets": compact_greedy["actions"],
                    "sequence_lengths": compact_greedy["lengths"],
                    "sequence_stopped": compact_greedy["stopped"],
                }
            )
            full_record = dict(full_semantic)
            full_record.update(
                {
                    "targets": full_greedy["actions"],
                    "sequence_lengths": full_greedy["lengths"],
                    "sequence_stopped": full_greedy["stopped"],
                }
            )
            compact_logprob, _compact_value = model.evaluate_targets_from_encoding(
                compact_record, (validated, summary, options)
            )
            full_logprob, _full_value = model.evaluate_targets_from_encoding(
                full_record, (full_validated, full_summary, full_options)
            )
            compaction_logprob_max_abs = float(
                (compact_logprob - full_logprob).abs().max().item()
            )
            if (
                # Trimming masked padding changes the GEMM reduction shape, so
                # fp32 round-off is not bitwise stable across pool widths.  The
                # action-space contract below is the hard boundary; keep this
                # diagnostic tolerance tight enough to catch semantic drift.
                compaction_summary_max_abs > 2.0e-5
                or compaction_logprob_max_abs > 2.0e-4
                or not compaction_greedy_exact
            ):
                raise RuntimeError(
                    "semantic prefix compaction exceeded the parity boundary: "
                    f"summary={compaction_summary_max_abs} "
                    f"options={compaction_options_max_abs} "
                    f"logprob={compaction_logprob_max_abs} "
                    f"greedy={compaction_greedy_exact}"
                )
        _decode_device(
            model.policy.action_decoder,
            validated,
            summary,
            options,
            max_select=args.max_select,
            greedy=False,
        )
    torch.cuda.synchronize(device)

    encoder_graph = None
    graph_static_semantic = None
    graph_copy_groups: list[tuple[list[Any], list[str]]] = []
    graph_static_input_bytes = 0
    graph_semantic = None
    graph_encoding = None
    graph_summary_max_abs = 0.0
    graph_options_max_abs = 0.0
    graph_exact = True
    if args.encoder_execution == "cuda_graph":
        # Extension outputs are allocation-backed and may change address.  Copy
        # the current resident observation into fixed CUDA buffers grouped by
        # dtype, then replay the frozen encoder against those stable addresses.
        raw_semantic = engine.encode_semantic0031_v2_lanes(lanes)
        semantic = semantic0031_v2_ready_batch(
            raw_semantic,
            max_action_steps=args.max_select,
        )
        graph_static_semantic = {
            name: value.clone() for name, value in semantic.items()
        }
        graph_static_input_bytes = sum(
            int(value.numel() * value.element_size())
            for value in graph_static_semantic.values()
        )
        for dtype in sorted(
            {value.dtype for value in graph_static_semantic.values()}, key=str
        ):
            names = [
                name
                for name, value in graph_static_semantic.items()
                if value.dtype == dtype
            ]
            graph_copy_groups.append(
                ([graph_static_semantic[name] for name in names], names)
            )

        def graph_encoder() -> Any:
            assert graph_static_semantic is not None
            return model.encode(graph_static_semantic)

        with torch.inference_mode():
            graph_encoder()
            torch.cuda.synchronize(device)
            encoder_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(encoder_graph):
                graph_encoding = graph_encoder()
            encoder_graph.replay()
            eager_graph_validation = model.encode(graph_static_semantic)
        torch.cuda.synchronize(device)
        assert graph_encoding is not None
        graph_summary_max_abs = float(
            (graph_encoding[1] - eager_graph_validation[1]).abs().max().item()
        )
        graph_options_max_abs = float(
            (graph_encoding[2] - eager_graph_validation[2]).abs().max().item()
        )
        graph_exact = torch.equal(
            graph_encoding[1], eager_graph_validation[1]
        ) and torch.equal(graph_encoding[2], eager_graph_validation[2])
        if not graph_exact:
            raise RuntimeError(
                "CUDA Graph encoder changed semantic outputs: "
                f"summary={graph_summary_max_abs} options={graph_options_max_abs}"
            )

    reset()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    started.record()
    completed = 0
    errors = 0
    executed = 0
    status_checks = 0
    prefix_width_totals = {name: 0 for name in _SEMANTIC_PREFIX_FAMILIES}
    prefix_width_maxima = {name: 0 for name in _SEMANTIC_PREFIX_FAMILIES}
    component_events: list[tuple[str, Any, Any]] = []
    action_trace: list[tuple[Any, Any, Any]] = []

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
            if encoder_graph is None:
                semantic = semantic0031_v2_ready_batch(
                    raw_semantic,
                    max_action_steps=args.max_select,
                )
                if args.compact_semantic_prefixes:
                    semantic, prefix_widths = _compact_semantic_prefixes(semantic)
                    for name, width in prefix_widths.items():
                        prefix_width_totals[name] += width
                        prefix_width_maxima[name] = max(prefix_width_maxima[name], width)
                component_end("engine_observation", component)
                if args.profile_components:
                    component = component_start()
                    validated = model.policy.validate_batch(semantic)
                    prototype_memory = model._frozen_prototype_memory()
                    state = model.policy.state_encoder(validated, prototype_memory)
                    summary = state.summary.detach()
                    component_end("semantic_state_encoder", component)
                    component = component_start()
                    options = model.policy.option_encoder(
                        validated, state, prototype_memory
                    ).detach()
                    component_end("semantic_option_encoder", component)
                    component = None
                else:
                    component = component_start()
                    validated, summary, options = model.encode(semantic)
            else:
                current_semantic = semantic0031_v2_ready_batch(
                    raw_semantic,
                    max_action_steps=args.max_select,
                )
                for destinations, names in graph_copy_groups:
                    torch._foreach_copy_(
                        destinations,
                        [current_semantic[name] for name in names],
                    )
                component_end("engine_observation", component)
                component = component_start()
                encoder_graph.replay()
                assert graph_static_semantic is not None and graph_encoding is not None
                semantic = graph_static_semantic
                validated, summary, options = graph_encoding
            component_end("shared_semantic_encoder", component)
            actor = (semantic["global_cat"][:, 3] - 1).clamp(min=0, max=1)
            learner_turn = ready & actor.eq(focal_player)
            opponent_turn = ready & ~actor.eq(focal_player)
            component = component_start()
            learner = _decode_device(
                model.policy.action_decoder,
                validated,
                summary,
                options,
                max_select=args.max_select,
                greedy=False,
                route_mask=learner_turn,
            )
            component_end("learner_decoder", component)
            component = component_start()
            opponent = _decode_device(
                opponent_decoder,
                validated,
                summary,
                options,
                max_select=args.max_select,
                greedy=True,
                route_mask=opponent_turn,
            )
            component_end("opponent_decoder", component)
            component = component_start()
            actions = torch.where(
                learner_turn[:, None], learner["actions"], opponent["actions"]
            ).contiguous()
            lengths = torch.where(learner_turn, learner["lengths"], opponent["lengths"])
            if args.capture_error_trace or args.capture_trace_path is not None:
                action_trace.append((statuses.clone(), actions.clone(), lengths.clone()))
            engine.pack_actions(actions, lengths)
            engine.apply_packed_actions()
            engine.advance_to_decision()
            component_end("action_apply_advance", component)
            executed = step + 1
            if executed % args.check_interval == 0 or executed == args.max_decisions:
                torch.cuda.synchronize(device)
                checked = engine.statuses()
                status_checks += 1
                errors = int(checked.eq(ERROR).sum().item())
                completed = int(checked.eq(TERMINAL).sum().item())
                if errors or completed == batch_size:
                    break
    ended.record()
    ended.synchronize()
    gpu_seconds = started.elapsed_time(ended) / 1000.0
    wall_seconds = time.perf_counter() - wall_start
    component_ms: dict[str, float] = {}
    for name, begin, finish in component_events:
        component_ms[name] = component_ms.get(name, 0.0) + float(
            begin.elapsed_time(finish)
        )
    final_statuses = engine.statuses()
    errors = int(final_statuses.eq(ERROR).sum().item())
    completed = int(final_statuses.eq(TERMINAL).sum().item())
    error_diagnostics = _error_diagnostics(engine, schedule) if errors else []
    if error_diagnostics and args.capture_error_trace:
        for diagnostic in error_diagnostics:
            lane = int(diagnostic["lane"])
            diagnostic["action_trace"] = [
                {
                    "status": int(statuses[lane].item()),
                    "length": int(lengths[lane].item()),
                    "actions": actions[lane, : int(lengths[lane].item())].cpu().tolist(),
                }
                for statuses, actions, lengths in action_trace
            ]
    if args.capture_trace_path is not None:
        trace_path = args.capture_trace_path.resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "decks": decks.cpu(),
                "seeds": seeds.cpu(),
                "statuses": torch.stack([row[0] for row in action_trace]).cpu(),
                "actions": torch.stack([row[1] for row in action_trace]).cpu(),
                "lengths": torch.stack([row[2] for row in action_trace]).cpu(),
            },
            trace_path,
        )
    passed = errors == 0 and completed == batch_size
    output = {
        "schema_version": 1,
        "passed": passed,
        "scope": "complete_official_cuda_games_with_0031_learner_and_frozen_opponent",
        "not_policy_strength_evidence": True,
        "snapshot": {
            "path": str(args.snapshot.resolve()),
            "sha256": _sha256(args.snapshot.resolve()),
            "source_selected_count": len(catalog),
            "benchmark_opponent_count": len(selected_catalog),
            "schedule_games": batch_size,
            "schedule_mode": (
                "one_each"
                if args.games == len(selected_catalog)
                else "selected_opponents_weighted"
            ),
            "first_ids": [str(row["deck_id"]) for row in rows[:3]],
        },
        "checkpoint": {
            "sha256": source.checkpoint_sha256,
            "storage_schema": storage.schema_version,
        },
        "collector": {
            "completed_games": completed,
            "errors": errors,
            "error_diagnostics": error_diagnostics,
            "decisions": executed,
            "max_decisions": args.max_decisions,
            "status_checks": status_checks,
            "gpu_seconds": gpu_seconds,
            "wall_seconds": wall_seconds,
            "games_per_second_gpu": completed / gpu_seconds if gpu_seconds else 0.0,
            "games_per_second_wall": completed / wall_seconds if wall_seconds else 0.0,
            "actions": "learner_stochastic_opponent_frozen_greedy",
            "profile_components": bool(args.profile_components),
            "encoder_execution": args.encoder_execution,
            "compact_semantic_prefixes": bool(args.compact_semantic_prefixes),
            "semantic_prefix_width_mean": {
                name: total / executed if executed else 0.0
                for name, total in prefix_width_totals.items()
            },
            "semantic_prefix_width_max": prefix_width_maxima,
            "component_gpu_seconds": {
                name: milliseconds / 1000.0
                for name, milliseconds in sorted(component_ms.items())
            },
        },
        "cuda_resident": {
            "official_engine": True,
            "semantic0031_v2_observation": True,
            "learner_decoder_sampling": True,
            "frozen_opponent_decoder": True,
            "host_status_checks": status_checks,
            "host_action_or_observation_copies": (
                len(action_trace)
                if args.capture_error_trace or args.capture_trace_path is not None
                else 0
            ),
            "frozen_prototype_cache": True,
            "prototype_cache_builds": int(model.prototype_cache_builds),
            "prototype_cache_bytes": int(model.prototype_cache_bytes),
            "prototype_cache_exact": bool(cache_exact),
            "prototype_cache_summary_max_abs": cache_summary_max_abs,
            "prototype_cache_options_max_abs": cache_options_max_abs,
            "encoder_cuda_graph": args.encoder_execution == "cuda_graph",
            "encoder_graph_static_input_bytes": graph_static_input_bytes,
            "encoder_graph_copy_groups": len(graph_copy_groups),
            "encoder_graph_exact": bool(graph_exact),
            "encoder_graph_summary_max_abs": graph_summary_max_abs,
            "encoder_graph_options_max_abs": graph_options_max_abs,
            "semantic_prefix_warmup_widths": warmup_prefix_widths,
            "semantic_prefix_summary_max_abs": compaction_summary_max_abs,
            "semantic_prefix_options_max_abs": compaction_options_max_abs,
            "semantic_prefix_logprob_max_abs": compaction_logprob_max_abs,
            "semantic_prefix_greedy_exact": bool(compaction_greedy_exact),
        },
        "memory": {
            "official_arena_bytes": int(engine.allocated_bytes),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "extension": importlib.import_module("_ptcg_cuda").__file__,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
