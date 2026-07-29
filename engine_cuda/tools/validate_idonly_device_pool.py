from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CUDA_ENGINE_ROOT.parent
for module_root in (REPO_ROOT / "tools", CUDA_ENGINE_ROOT / "python", CUDA_ENGINE_ROOT / "tools"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from evaluate_candidates import AgentRunner, normalize_action  # noqa: E402
from probe_actual_policy_residency import (  # noqa: E402
    checkpoint_state,
    load_module,
)
from ptcg_cuda_engine.policy_adapters import IDOnlyPointerPolicyDeviceAdapter  # noqa: E402
from ptcg_cuda_engine.policy_pool import PolicyPoolManifest  # noqa: E402
from seeded_cpp_shim import SeededCppLib  # noqa: E402
from train_seeded_ppo_smoke import resolve_agent_path  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "pure_lucario_v1.json"
DEFAULT_LIB = REPO_ROOT / "tmp" / "seeded_cpp_shim_policy_codec_v1" / "libcg_seeded.so"
DEFAULT_MANIFEST = CUDA_ENGINE_ROOT / "configs" / "policy_pool.example.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_observations(
    *,
    config: dict[str, Any],
    lib: Path,
    games: int,
    limit: int,
    seed: int,
    engine_threads: int,
) -> list[tuple[int, dict[str, Any], list[int]]]:
    shim = SeededCppLib(lib)
    shim.set_batch_threads(engine_threads)
    teacher_path = resolve_agent_path(config["teacher"])
    teacher = AgentRunner(teacher_path.name, teacher_path, seed=seed + 1)
    opponent_names = [str(item["name"]) for item in config["opponents"]]
    opponents = {
        name: AgentRunner(name, resolve_agent_path(name), seed=seed + 100 + index)
        for index, name in enumerate(opponent_names)
    }
    deck_pairs: list[tuple[list[int], list[int]]] = []
    player_agents: list[tuple[AgentRunner, AgentRunner]] = []
    seeds: list[int] = []
    for game_index in range(games):
        opponent = opponents[opponent_names[game_index % len(opponent_names)]]
        pair = (teacher, opponent) if game_index % 2 == 0 else (opponent, teacher)
        player_agents.append(pair)
        deck_pairs.append((pair[0].deck, pair[1].deck))
        seeds.append(seed + game_index * 104729)

    battles = shim.start_battles(deck_pairs, seeds)
    active = list(range(games))
    samples: list[tuple[int, dict[str, Any], list[int]]] = []
    try:
        for _step in range(700):
            if not active or len(samples) >= limit:
                break
            active_battles = [battles[index] for index in active]
            metas = shim.select_meta_many(active_battles)
            live_positions = [index for index, meta in enumerate(metas) if not meta.game_result]
            active = [active[index] for index in live_positions]
            if not active:
                break
            active_battles = [battles[index] for index in active]
            metas = [metas[index] for index in live_positions]
            observations = shim.observation_many(active_battles)
            actions: list[list[int]] = []
            for game_index, meta, observation in zip(active, metas, observations):
                runner = player_agents[game_index][meta.select_player]
                actions.append(normalize_action(observation, runner(observation)))
            errors = shim.select_many(active_battles, actions)
            if any(error != 0 for error in errors):
                raise RuntimeError(f"official engine select error: {errors[:8]}")
            remaining = limit - len(samples)
            samples.extend(list(zip(active, observations, actions))[:remaining])
    finally:
        shim.finish_many(battles)
    if len(samples) < limit:
        raise RuntimeError(f"collected only {len(samples)} of {limit} observations")
    return samples


def registered_deck(checkpoint: Path) -> tuple[list[int], list[int]]:
    deck_path = checkpoint.parent / "deck.csv"
    deck = [
        int(line)
        for line in deck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"{deck_path} contains {len(deck)} cards")
    counts = Counter(deck)
    return deck, [counts[card] for card in deck]


def canonical_actions(actions: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    values = actions.detach().cpu().numpy()
    sizes = lengths.detach().cpu().numpy()
    return [
        [int(value) for value in values[index, : int(sizes[index])]]
        for index in range(len(sizes))
    ]


def pad_axis_one(value: torch.Tensor, capacity: int) -> torch.Tensor:
    if value.ndim < 2 or value.shape[1] > capacity:
        raise ValueError(f"cannot pad tensor shape {tuple(value.shape)} to {capacity}")
    if value.shape[1] == capacity:
        return value
    shape = list(value.shape)
    shape[1] = capacity
    padded = torch.zeros(shape, dtype=value.dtype, device=value.device)
    padded[:, : value.shape[1]] = value
    return padded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate real ID-only BC checkpoints through the device adapter."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--policy-ids", default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--observation-decisions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--parity-rows",
        type=int,
        default=0,
        help="Rows per policy checked on CPU; 0 checks the full GPU cohort.",
    )
    parser.add_argument("--seed", type=int, default=2026072901)
    parser.add_argument("--engine-threads", type=int, default=15)
    parser.add_argument("--cpu-threads", type=int, default=15)
    parser.add_argument("--cpu-repeats", type=int, default=2)
    parser.add_argument("--gpu-warmup", type=int, default=2)
    parser.add_argument("--gpu-repeats", type=int, default=50)
    parser.add_argument("--profile-policy-ranges", action="store_true")
    parser.add_argument(
        "--fixed-capacity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad every cohort to the checkpoint's max entity and option capacities.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if min(
        args.games,
        args.observation_decisions,
        args.batch_size,
        args.engine_threads,
        args.cpu_threads,
        args.cpu_repeats,
        args.gpu_warmup,
        args.gpu_repeats,
    ) <= 0:
        raise ValueError("count arguments must be positive")
    parity_rows = args.batch_size if args.parity_rows == 0 else args.parity_rows
    if not 1 <= parity_rows <= args.batch_size:
        raise ValueError("parity rows must be in [1, batch size]")
    policy_ids = [int(value) for value in args.policy_ids.split(",") if value.strip()]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("policy IDs must be unique")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    samples = collect_observations(
        config=load_json(args.config.resolve()),
        lib=args.lib.resolve(),
        games=args.games,
        limit=args.observation_decisions,
        seed=args.seed,
        engine_threads=args.engine_threads,
    )
    manifest = PolicyPoolManifest.load(args.manifest.resolve())
    policies = {policy.policy_id: policy for policy in manifest.policies}
    resident: list[tuple[int, IDOnlyPointerPolicyDeviceAdapter, dict[str, torch.Tensor]]] = []
    rows_report: list[dict[str, Any]] = []
    total_cpu_sec = 0.0
    total_cpu_workload = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for policy_id in policy_ids:
        policy = policies[policy_id]
        checkpoint = Path(policy.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        source = checkpoint.parent / "idonly_policy.py"
        if not source.is_file():
            raise FileNotFoundError(source)
        module = load_module(source, f"_cuda_device_pool_{policy_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = module.ModelConfig(**payload["model_config"])
        cpu_model = module.IDOnlyPointerPolicy(config)
        cpu_model.load_state_dict(checkpoint_state(payload), strict=True)
        cpu_model.requires_grad_(False).eval()
        gpu_model = copy.deepcopy(cpu_model).to(device).eval()
        deck_ids, deck_counts = registered_deck(checkpoint)
        encoded_rows: list[dict[str, Any]] = []
        codec = module.IDOnlyCodec(config)
        prize_ledgers: dict[int, Any] = {}
        needs_prize_state = policy.adapter in {
            "marnie_prize_pointer_v4",
            "marnie_compact_v5",
        }
        for game_index, observation, action in samples:
            codec_observation = observation
            if policy.adapter == "marnie_compact_v5":
                codec_observation, _projection = module.project_public_information(
                    observation
                )
            row = codec.encode(codec_observation, action)
            if row is None:
                continue
            row.update(
                deck_ids=deck_ids,
                deck_counts=deck_counts,
                sample_weight=1.0,
                deck_hash_code=0,
                prize_source_code=0,
            )
            if needs_prize_state:
                ledger = prize_ledgers.setdefault(
                    game_index,
                    module.PrizeLedger(deck_ids),
                )
                prize = ledger.observe(codec_observation)
                if policy.adapter == "marnie_compact_v5":
                    compact = module.compact_probability_observation(
                        prize,
                        draw_horizon=3,
                    )
                    row.update(
                        resource_card=list(compact.card_ids),
                        resource_registered_count=list(compact.registered_counts),
                        resource_visible_count=list(compact.visible_counts),
                        resource_probability=[list(values) for values in compact.features],
                        resource_source_code=compact.source_code,
                    )
                else:
                    row.update(
                        prize_card=list(prize.card_ids),
                        prize_num=[list(values) for values in prize.features],
                        prize_source_code=prize.source_code,
                    )
            encoded_rows.append(row)
            if len(encoded_rows) >= args.batch_size:
                break
        if len(encoded_rows) < args.batch_size:
            raise RuntimeError(
                f"policy {policy_id} encoded only {len(encoded_rows)} rows"
            )
        cpu_batch = module.collate_examples(encoded_rows)
        cpu_batch["max_count"] = torch.tensor(
            [int(row["max_count"]) for row in encoded_rows],
            dtype=torch.long,
        )
        cpu_batch["_route_mask"] = torch.ones(args.batch_size, dtype=torch.bool)
        if args.fixed_capacity:
            for name in ("entity_cat", "entity_num", "entity_mask"):
                cpu_batch[name] = pad_axis_one(cpu_batch[name], int(config.max_entities))
            for name in ("option_cat", "option_mask"):
                cpu_batch[name] = pad_axis_one(cpu_batch[name], int(config.max_options))
        cpu_parity_batch = {
            name: (
                value[:parity_rows]
                if value.ndim > 0 and value.shape[0] == args.batch_size
                else value
            )
            for name, value in cpu_batch.items()
        }
        gpu_batch = {name: value.to(device) for name, value in cpu_batch.items()}
        gpu_parity_batch = {
            name: (
                value[:parity_rows]
                if value.ndim > 0 and value.shape[0] == args.batch_size
                else value
            )
            for name, value in gpu_batch.items()
        }
        cpu_adapter = IDOnlyPointerPolicyDeviceAdapter(
            cpu_model,
            max_select=int(config.max_action_steps),
        )
        gpu_adapter = IDOnlyPointerPolicyDeviceAdapter(
            gpu_model,
            max_select=int(config.max_action_steps),
        )
        with torch.inference_mode():
            cpu_actions, cpu_lengths = cpu_adapter.act_device(cpu_parity_batch)
            gpu_actions, gpu_lengths = gpu_adapter.act_device(gpu_parity_batch)
        torch.cuda.synchronize(device)
        expected = canonical_actions(cpu_actions, cpu_lengths)
        actual = canonical_actions(gpu_actions, gpu_lengths)
        mismatches = sum(left != right for left, right in zip(expected, actual))

        with torch.inference_mode():
            started = time.perf_counter()
            for _ in range(args.cpu_repeats):
                cpu_adapter.act_device(cpu_parity_batch)
            cpu_sec = time.perf_counter() - started
        total_cpu_sec += cpu_sec
        total_cpu_workload += parity_rows * args.cpu_repeats
        rows_report.append(
            {
                "policy_id": policy_id,
                "name": policy.name,
                "adapter": policy.adapter,
                "checkpoint": str(checkpoint),
                "parameters": sum(parameter.numel() for parameter in gpu_model.parameters()),
                "encoded_candidates": len(encoded_rows),
                "parity_decisions": parity_rows,
                "action_mismatches": mismatches,
                "cpu_sec": round(cpu_sec, 6),
                "entity_capacity": int(cpu_batch["entity_mask"].shape[1]),
                "option_capacity": int(cpu_batch["option_mask"].shape[1]),
            }
        )
        resident.append((policy_id, gpu_adapter, gpu_batch))
        del cpu_adapter, cpu_model, cpu_batch, cpu_parity_batch, gpu_parity_batch, payload

    with torch.inference_mode():
        for _ in range(args.gpu_warmup):
            for _policy_id, adapter, batch in resident:
                adapter.act_device(batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push("idonly_policy_pool_hot")
        start_event.record()
        for _ in range(args.gpu_repeats):
            for policy_id, adapter, batch in resident:
                if args.profile_policy_ranges:
                    torch.cuda.nvtx.range_push(f"idonly_policy_{policy_id}_hot")
                try:
                    adapter.act_device(batch)
                finally:
                    if args.profile_policy_ranges:
                        torch.cuda.nvtx.range_pop()
        end_event.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        gpu_sec = start_event.elapsed_time(end_event) / 1000.0

    gpu_workload = len(resident) * args.batch_size * args.gpu_repeats
    cpu_rate = total_cpu_workload / max(total_cpu_sec, 1.0e-9)
    gpu_rate = gpu_workload / max(gpu_sec, 1.0e-9)
    mismatches = sum(int(row["action_mismatches"]) for row in rows_report)
    report = {
        "status": "pass" if mismatches == 0 else "fail",
        "scope": f"{len(resident)} ID-only policy checkpoints; excludes independent Alakazam",
        "device": torch.cuda.get_device_name(device),
        "policy_count": len(resident),
        "frozen_policy_count": sum(policies[policy_id].frozen for policy_id in policy_ids),
        "learner_policy_count": sum(not policies[policy_id].frozen for policy_id in policy_ids),
        "batch_per_policy": args.batch_size,
        "parity_rows_per_policy": parity_rows,
        "fixed_capacity": args.fixed_capacity,
        "resident_decision_capacity": len(resident) * args.batch_size,
        "observation_decisions": len(samples),
        "total_action_mismatches": mismatches,
        "cpu_threads": args.cpu_threads,
        "cpu_decisions_per_sec": round(cpu_rate, 3),
        "gpu_decisions_per_sec": round(gpu_rate, 3),
        "gpu_over_cpu_speedup": round(gpu_rate / max(cpu_rate, 1.0e-9), 3),
        "gpu_hot_sec": round(gpu_sec, 6),
        "gpu_hot_workload_decisions": gpu_workload,
        "gpu_memory": {
            "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 3),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 3),
        },
        "policies": rows_report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out is not None:
        output = args.out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
