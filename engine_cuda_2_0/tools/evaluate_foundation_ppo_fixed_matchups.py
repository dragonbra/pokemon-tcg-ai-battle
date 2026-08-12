"""Evaluate a Lucario PPO learner against six frozen foundation opponents."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.legacy_codecs import policy_codec_v1_to_idonly_codec_v1  # noqa: E402
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    FoundationR15DeviceAdapter,
    StaticBatchFieldsDeviceAdapter,
    foundation_r15_static_fields,
)
from ptcg_cuda_engine.policy_pool import GPUResidentPolicyPool, PolicyPoolManifest  # noqa: E402
from run_official_seeded_multi_policy_loop import (  # noqa: E402
    load_foundation_0020_checkpoint,
    load_pool_rows,
    read_deck,
)


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--pool",
        type=Path,
        default=WORKSPACE_ROOT / ".tmp" / "cuda_zero_shot_six_decks" / "pool.json",
    )
    parser.add_argument(
        "--learner-deck",
        type=Path,
        help="Optional 60-card deck.csv for player0; defaults to pool row 0.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-count", type=int, default=5)
    parser.add_argument(
        "--games-per-opponent",
        "--games-per-deck",
        dest="games_per_opponent",
        type=int,
        default=200,
    )
    parser.add_argument("--seed-start", type=int, default=2026090101)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument(
        "--timeout-as-draw",
        action="store_true",
        help="Count lanes still active at max-steps as draws and report them separately.",
    )
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidate_policy(
    checkpoint: Path,
    foundation_checkpoint: Path,
    ontology: Path,
    device: Any,
) -> tuple[Any, int]:
    import torch

    model, _config = load_foundation_0020_checkpoint(
        foundation_checkpoint, ontology, device
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError(f"{checkpoint} is missing model_state_dict")
    policy_state = {
        name.removeprefix("policy."): value
        for name, value in model_state.items()
        if name.startswith("policy.")
    }
    model.load_state_dict(policy_state, strict=True)
    model.eval().requires_grad_(False)
    return model, int(payload.get("iteration", -1))


def fixed_schedule(
    learner_deck: list[int],
    opponent_decks: list[list[int]],
    games_per_opponent: int,
    seed_start: int,
) -> tuple[list[list[list[int]]], list[list[int]], list[int], list[int]]:
    if games_per_opponent <= 0:
        raise ValueError("games-per-opponent must be positive")
    battle_decks: list[list[list[int]]] = []
    seat_policy_ids: list[list[int]] = []
    seeds: list[int] = []
    lane_opponent_ids: list[int] = []
    for opponent_id, opponent_deck in enumerate(opponent_decks):
        opponent_policy_id = opponent_id + 1
        for game_index in range(games_per_opponent):
            seed = seed_start + opponent_id * 100_003 + game_index
            battle_decks.append([learner_deck, opponent_deck])
            seat_policy_ids.append([0, opponent_policy_id])
            seeds.append(seed)
            lane_opponent_ids.append(opponent_id)
    return battle_decks, seat_policy_ids, seeds, lane_opponent_ids


def build_policy_pool(
    candidate_model: Any,
    initial_model: Any,
    learner_deck: list[int],
    opponent_decks: list[list[int]],
    device: Any,
    capacity: int,
    max_select: int,
) -> GPUResidentPolicyPool:
    policies: list[dict[str, Any]] = [
        {
            "policy_id": 0,
            "name": "candidate_lucario_player0",
            "deck": "lucario",
            "checkpoint": "candidate",
            "adapter": "foundation_r15_zero_history_v1",
            "codec": "foundation_0020_codec_v1",
            "frozen": True,
            "dtype": "fp32",
        }
    ]
    adapters: dict[int, Any] = {
        0: StaticBatchFieldsDeviceAdapter(
            FoundationR15DeviceAdapter(candidate_model, max_select=max_select),
            foundation_r15_static_fields(learner_deck, device),
        )
    }
    for opponent_id, opponent_deck in enumerate(opponent_decks):
        policy_id = opponent_id + 1
        policies.append(
            {
                "policy_id": policy_id,
                "name": f"frozen_opponent_{opponent_id}",
                "deck": f"opponent_{opponent_id}",
                "checkpoint": "initial_foundation",
                "adapter": "foundation_r15_zero_history_v1",
                "codec": "foundation_0020_codec_v1",
                "frozen": True,
                "dtype": "fp32",
            }
        )
        adapters[policy_id] = StaticBatchFieldsDeviceAdapter(
            FoundationR15DeviceAdapter(initial_model, max_select=max_select),
            foundation_r15_static_fields(opponent_deck, device),
        )
    manifest = PolicyPoolManifest.from_dict(
        {
            "name": "fixed_lucario_candidate_vs_six_frozen_opponents",
            "max_policies": len(policies),
            "routing": {"mode": "fixed_player0_learner_player1_opponent"},
            "policies": policies,
        }
    )
    return GPUResidentPolicyPool(
        manifest,
        adapters,
        capacity=capacity,
        max_select=max_select,
    )


def summarize_counts(
    wins: int, losses: int, draws: int, timeouts: int = 0
) -> dict[str, float | int]:
    games = wins + losses + draws
    win_rate = wins / games if games else 0.0
    effective_win_rate = (wins + 0.5 * draws) / games if games else 0.0
    standard_error = math.sqrt(
        effective_win_rate * (1.0 - effective_win_rate) / games
    ) if games else 0.0
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "effective_win_rate": effective_win_rate,
        "effective_win_rate_ci95_low": max(0.0, effective_win_rate - 1.96 * standard_error),
        "effective_win_rate_ci95_high": min(1.0, effective_win_rate + 1.96 * standard_error),
    }


def evaluate_one(
    *,
    label: str,
    checkpoint: Path | None,
    initial_model: Any,
    foundation_checkpoint: Path,
    ontology: Path,
    rules: bytes,
    learner_deck: list[int],
    opponent_decks: list[list[int]],
    opponent_names: list[str],
    games_per_opponent: int,
    seed_start: int,
    max_steps: int,
    chunk_size: int,
    timeout_as_draw: bool,
    max_select: int,
    device: Any,
    device_index: int,
) -> dict[str, Any]:
    import torch

    if checkpoint is None:
        candidate_model = initial_model
        iteration = 0
    else:
        candidate_model, iteration = load_candidate_policy(
            checkpoint, foundation_checkpoint, ontology, device
        )
    policy_pool = build_policy_pool(
        candidate_model,
        initial_model,
        learner_deck,
        opponent_decks,
        device,
        chunk_size,
        max_select,
    )
    battle_decks, seat_policy_ids, seeds, lane_opponent_ids = fixed_schedule(
        learner_deck,
        opponent_decks,
        games_per_opponent,
        seed_start,
    )
    started = time.perf_counter()
    decisions = 0
    max_chunk_steps = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_timeouts = 0
    per_opponent_counts = [[0, 0, 0, 0] for _ in opponent_names]
    chunk_rows: list[dict[str, Any]] = []
    for chunk_start in range(0, len(seeds), chunk_size):
        chunk_stop = min(chunk_start + chunk_size, len(seeds))
        chunk_started = time.perf_counter()
        chunk_seeds = seeds[chunk_start:chunk_stop]
        engine = create_official_engine(
            rules, batch_size=len(chunk_seeds), device_index=device_index
        )
        decks_tensor = torch.tensor(
            battle_decks[chunk_start:chunk_stop], dtype=torch.int32, device=device
        )
        policy_tensor = torch.tensor(
            seat_policy_ids[chunk_start:chunk_stop], dtype=torch.long, device=device
        )
        seed_tensor = torch.tensor(chunk_seeds, dtype=torch.int64, device=device)
        opponent_id_tensor = torch.tensor(
            lane_opponent_ids[chunk_start:chunk_stop], dtype=torch.long, device=device
        )
        engine.reset_seeded_interactive(decks_tensor, seed_tensor)

        chunk_decisions = 0
        chunk_steps = 0
        hit_step_limit = False
        with torch.inference_mode():
            for chunk_steps in range(1, max_steps + 1):
                engine.advance_to_decision()
                statuses = engine.statuses()
                if bool(statuses.eq(ERROR).any().item()):
                    error_indices = statuses.eq(ERROR).nonzero(as_tuple=False).flatten()
                    error_seeds = seed_tensor.index_select(0, error_indices).cpu().tolist()
                    raise RuntimeError(
                        f"{label}: engine error status for seeds {error_seeds[:16]}"
                    )
                ready = statuses.eq(NEEDS_ACTION)
                if not bool(ready.any().item()):
                    break
                policy_codec = dict(engine.encode_policy_v1())
                policy_codec["entity_mask"] = policy_codec["entity_mask"].bool()
                policy_codec["option_mask"] = policy_codec["option_mask"].bool()
                acting_player = (
                    policy_codec["global_cat"][:, 3].long() - 1
                ).clamp(min=0, max=1)
                acting_policy = policy_tensor.gather(
                    1, acting_player.view(-1, 1)
                ).view(-1)
                idonly = policy_codec_v1_to_idonly_codec_v1(
                    policy_codec,
                    max_card_id=2048,
                    max_action_steps=max_select,
                    target_entity_capacity=128,
                    target_option_capacity=80,
                )
                actions = policy_pool.act(
                    {"foundation_0020_codec_v1": idonly},
                    acting_policy,
                    ready,
                )
                engine.pack_actions(actions.indices, actions.lengths)
                engine.apply_packed_actions()
                chunk_decisions += int(ready.long().sum().item())
            else:
                hit_step_limit = True

        statuses = engine.statuses()
        if bool(statuses.eq(ERROR).any().item()):
            error_indices = statuses.eq(ERROR).nonzero(as_tuple=False).flatten()
            error_seeds = seed_tensor.index_select(0, error_indices).cpu().tolist()
            raise RuntimeError(
                f"{label}: engine error status for seeds {error_seeds[:16]}"
            )
        timed_out = ~statuses.eq(TERMINAL)
        timeout_seeds = seed_tensor.masked_select(timed_out).cpu().tolist()
        if bool(timed_out.any().item()) and not (
            hit_step_limit and timeout_as_draw
        ):
            counts = torch.bincount(statuses.long(), minlength=4).cpu().tolist()
            raise RuntimeError(
                f"{label}: non-terminal lanes remain after {max_steps} steps: "
                f"statuses={counts}, seeds={timeout_seeds[:16]}"
            )
        results = engine.game_results().long()
        winners = results - 1
        draws = results.eq(3) | timed_out
        wins = ~draws & winners.eq(0)
        losses = ~draws & ~wins
        chunk_win_count = int(wins.long().sum().item())
        chunk_loss_count = int(losses.long().sum().item())
        chunk_draw_count = int(draws.long().sum().item())
        chunk_timeout_count = int(timed_out.long().sum().item())
        total_wins += chunk_win_count
        total_losses += chunk_loss_count
        total_draws += chunk_draw_count
        total_timeouts += chunk_timeout_count
        for opponent_id in range(len(opponent_names)):
            mask = opponent_id_tensor.eq(opponent_id)
            per_opponent_counts[opponent_id][0] += int(
                (wins & mask).long().sum().item()
            )
            per_opponent_counts[opponent_id][1] += int(
                (losses & mask).long().sum().item()
            )
            per_opponent_counts[opponent_id][2] += int(
                (draws & mask).long().sum().item()
            )
            per_opponent_counts[opponent_id][3] += int(
                (timed_out & mask).long().sum().item()
            )
        chunk_elapsed = time.perf_counter() - chunk_started
        chunk_row = {
            "chunk_index": len(chunk_rows),
            "games": len(chunk_seeds),
            "steps": chunk_steps,
            "decisions": chunk_decisions,
            "wall_seconds": chunk_elapsed,
            "timeouts": chunk_timeout_count,
            "timeout_seeds": timeout_seeds,
        }
        chunk_rows.append(chunk_row)
        decisions += chunk_decisions
        max_chunk_steps = max(max_chunk_steps, chunk_steps)
        print(
            "EVAL_CHUNK "
            + json.dumps(
                {"label": label, **chunk_row}, sort_keys=True
            ),
            flush=True,
        )
        del engine, decks_tensor, policy_tensor, seed_tensor
        del opponent_id_tensor

    elapsed = time.perf_counter() - started

    per_opponent: dict[str, Any] = {}
    for opponent_id, opponent_name in enumerate(opponent_names):
        per_opponent[opponent_name] = summarize_counts(
            *per_opponent_counts[opponent_id]
        )
    overall = summarize_counts(
        total_wins, total_losses, total_draws, total_timeouts
    )
    result = {
        "label": label,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "iteration": iteration,
        "overall": overall,
        "per_opponent": per_opponent,
        "learner_seat": 0,
        "steps": max_chunk_steps,
        "decisions": decisions,
        "wall_seconds": elapsed,
        "games_per_second": len(seeds) / max(elapsed, 1.0e-9),
        "chunk_size": chunk_size,
        "chunks": chunk_rows,
    }
    del policy_pool
    if checkpoint is not None:
        del candidate_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.top_count <= 0:
        raise ValueError("top-count must be positive")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "top_checkpoints.json").read_text(encoding="utf-8"))
    entries = list(manifest.get("checkpoints") or [])[: args.top_count]
    if not entries:
        raise RuntimeError("top checkpoint manifest is empty")
    rows, unsupported = load_pool_rows(args.pool.resolve(), [])
    if unsupported or len(rows) != 6:
        raise ValueError("fixed evaluation requires exactly six supported deck rows")
    all_decks = [
        read_deck((WORKSPACE_ROOT / str(row["directory"])).resolve() / "deck.csv")
        for row in rows
    ]
    learner_deck_path = (
        args.learner_deck.resolve()
        if args.learner_deck is not None
        else (WORKSPACE_ROOT / str(rows[0]["directory"])).resolve() / "deck.csv"
    )
    learner_deck = read_deck(learner_deck_path)
    opponent_decks = all_decks
    opponent_names = [str(row["name"]) for row in rows]
    foundation_checkpoint = (
        WORKSPACE_ROOT / str(rows[0]["directory"]) / str(rows[0]["checkpoint"])
    ).resolve()
    ontology = (WORKSPACE_ROOT / str(rows[0]["ontology"])).resolve()
    initial_model, _config = load_foundation_0020_checkpoint(
        foundation_checkpoint, ontology, device
    )
    initial_model.eval().requires_grad_(False)
    rules = args.rules.resolve().read_bytes()

    report: dict[str, Any] = {
        "schema_version": "lucario_ppo_fixed_opponents_v2",
        "contract": {
            "learner": "lucario_candidate_checkpoint",
            "learner_player": 0,
            "opponents": "six_decks_with_frozen_initial_foundation",
            "opponent_player": 1,
            "decode": "greedy_argmax",
            "seat_assignment": "fixed_player0_learner_player1_opponent",
            "games_per_opponent": args.games_per_opponent,
            "total_games_per_checkpoint": (
                args.games_per_opponent * len(opponent_decks)
            ),
            "seed_start": args.seed_start,
            "max_steps_per_chunk": args.max_steps,
            "chunk_size": args.chunk_size,
            "timeout_handling": (
                "count_as_draw_and_report_separately"
                if args.timeout_as_draw
                else "error"
            ),
            "learner_deck_name": learner_deck_path.parent.name,
            "learner_deck_path": str(learner_deck_path),
            "opponent_names": opponent_names,
            "math": "tf32",
        },
        "locked_top_checkpoints": entries,
        "results": [],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    evaluation_rows: list[tuple[str, Path | None]] = [("initial", None)] + [
        (f"iter_{int(entry['iteration']):06d}", run_dir / str(entry["file"]))
        for entry in entries
    ]
    for label, checkpoint in evaluation_rows:
        result = evaluate_one(
            label=label,
            checkpoint=checkpoint,
            initial_model=initial_model,
            foundation_checkpoint=foundation_checkpoint,
            ontology=ontology,
            rules=rules,
            learner_deck=learner_deck,
            opponent_decks=opponent_decks,
            opponent_names=opponent_names,
            games_per_opponent=args.games_per_opponent,
            seed_start=args.seed_start,
            max_steps=args.max_steps,
            chunk_size=args.chunk_size,
            timeout_as_draw=args.timeout_as_draw,
            max_select=args.max_select,
            device=device,
            device_index=args.device_index,
        )
        report["results"].append(result)
        baseline = report["results"][0]
        result["effective_win_rate_delta_vs_initial"] = (
            float(result["overall"]["effective_win_rate"])
            - float(baseline["overall"]["effective_win_rate"])
        )
        result["per_opponent_effective_delta_vs_initial"] = {
            name: (
                float(result["per_opponent"][name]["effective_win_rate"])
                - float(baseline["per_opponent"][name]["effective_win_rate"])
            )
            for name in opponent_names
        }
        args.output.resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("EVAL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    report["ranking"] = sorted(
        (
            {
                "label": row["label"],
                "iteration": row["iteration"],
                "effective_win_rate": row["overall"]["effective_win_rate"],
                "delta_vs_initial": row["effective_win_rate_delta_vs_initial"],
            }
            for row in report["results"]
            if row["label"] != "initial"
        ),
        key=lambda row: (row["effective_win_rate"], row["iteration"]),
        reverse=True,
    )
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("EVAL_COMPLETE " + json.dumps(report["ranking"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
