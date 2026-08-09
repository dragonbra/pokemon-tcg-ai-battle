"""Benchmark 0037/0806 Semantic0031 policies with refillable CUDA lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = ROOT / "engine_cuda"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CUDA_ROOT / "python"))

from engine_cuda.tools import benchmark_0037_update32_cuda_rollout as legacy


def json_ready(value):
    """Return a JSON-serializable copy of nested benchmark metadata."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-mode", choices=("update32", "0806"), default="update32")
    parser.add_argument("--actor-package", type=Path, default=legacy.DEFAULT_PACKAGE)
    parser.add_argument("--actor-checkpoint", type=Path, default=legacy.DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--opponent-model", type=Path, default=legacy.DEFAULT_OPPONENT_MODEL)
    parser.add_argument("--focal-deck", type=Path, required=True)
    parser.add_argument("--deck-root", type=Path, default=legacy.DEFAULT_POOL / "decks")
    parser.add_argument("--schedule", type=Path, default=legacy.DEFAULT_SCHEDULE)
    parser.add_argument("--rules", type=Path, default=legacy.DEFAULT_RULES)
    parser.add_argument("--extension-dir", type=Path, default=legacy.DEFAULT_EXTENSION)
    parser.add_argument("--game-limit", type=int, default=512)
    parser.add_argument("--lane-count", type=int, default=152)
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--max-decisions", type=int, default=8192)
    parser.add_argument("--check-interval", type=int, default=8)
    parser.add_argument("--ability-repeat-limit", type=int, default=20)
    parser.add_argument("--engine-turn-draw-limit", type=int, default=100)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-states-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extension = args.extension_dir.resolve() / "_ptcg_cuda.so"
    if not extension.is_file():
        raise FileNotFoundError(extension)
    sys.path.insert(0, str(args.extension_dir.resolve()))
    import torch
    import _ptcg_cuda
    from ptcg_cuda_engine.semantic0031_bridge import Semantic0031DeviceAdapter
    from ptcg_cuda_engine.semantic0031_resident import (
        ResidentJob,
        run_resident_greedy_jobs,
    )
    from ptcg_cuda_engine.semantic0031_router import Semantic0031ResidentRouter

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    focal_deck = legacy.read_deck(args.focal_deck.resolve())
    schedule = legacy.load_schedule(
        args.schedule.resolve(), args.deck_root.resolve(), focal_deck
    )
    count = min(args.game_limit, len(schedule.engine_seeds))
    actor, opponent, provenance = legacy._load_models(
        args.actor_package.resolve(),
        args.actor_checkpoint.resolve(),
        args.opponent_model.resolve(),
        focal_deck,
        device,
        actor_mode=args.actor_mode,
    )
    actor_adapter = Semantic0031DeviceAdapter(actor, focal_deck, max_select=args.max_select)
    opponent_transformer = opponent.option_encoder.cross_attention_transformer
    router = Semantic0031ResidentRouter(
        focal_adapter=actor_adapter,
        opponent_last_option_layer=opponent_transformer.layers[1],
        opponent_option_norm=opponent_transformer.norm,
        opponent_decoder=opponent.action_decoder,
        same_policy=args.actor_mode == "0806",
    )
    jobs = tuple(
        ResidentJob(
            schedule_index=index,
            decks=schedule.deck_rows[index],
            engine_seed=schedule.engine_seeds[index],
            focal_player=schedule.focal_players[index],
            opponent_id=schedule.opponent_ids[index],
        )
        for index in range(count)
    )
    result = run_resident_greedy_jobs(
        jobs=jobs,
        rules=args.rules.resolve().read_bytes(),
        router=router,
        lane_count=min(args.lane_count, count),
        device=device,
        device_index=args.device_index,
        max_select=args.max_select,
        max_decisions=args.max_decisions,
        check_interval=args.check_interval,
        ability_repeat_limit=args.ability_repeat_limit,
        engine_turn_draw_limit=args.engine_turn_draw_limit,
    )
    focal_wins = sum(
        (job.focal_player == 0 and game_result == 1)
        or (job.focal_player == 1 and game_result == 2)
        for job, game_result in zip(jobs, result.game_results, strict=True)
    )
    focal_losses = sum(
        (job.focal_player == 0 and game_result == 2)
        or (job.focal_player == 1 and game_result == 1)
        for job, game_result in zip(jobs, result.game_results, strict=True)
    )
    game_bytes = json.dumps(result.game_results, separators=(",", ":")).encode("ascii")
    payload = {
        "schema_version": "cuda_semantic0031_resident_refill_strict_fp32_v2",
        "passed": len(result.game_results) == count,
        "collector": {
            "games": count,
            "completed_games": len(result.game_results),
            "errors": 0,
            "decisions": result.decisions,
            "routed_ready_rows": result.routed_ready_rows,
            "refill_events": result.refill_events,
            "lane_count": min(args.lane_count, count),
            "wall_seconds": result.wall_seconds,
            "gpu_seconds": result.gpu_seconds,
            "games_per_second_wall": count / result.wall_seconds,
            "games_per_second_gpu": count / result.gpu_seconds,
        },
        "determinism": {
            "game_results": list(result.game_results),
            "game_results_sha256": hashlib.sha256(game_bytes).hexdigest(),
            "terminal_state_sha256": hashlib.sha256(result.terminal_state_bytes).hexdigest(),
            "focal_wins": focal_wins,
            "focal_losses": focal_losses,
            "draws": count - focal_wins - focal_losses,
        },
        # These arrays are already collected by the resident scheduler before a
        # terminal lane is reset. Serializing them adds no model forward and no
        # device-side hot-path instrumentation. Strategic/action-family counts
        # are deliberately not inferred from official selection callbacks.
        "per_game_diagnostics": {
            "schema": "cuda_resident_terminal_diagnostics_v1",
            "terminal_turns": list(result.terminal_turns),
            "engine_selections": list(result.engine_selections),
            "terminal_prize_counts": [
                list(row) for row in result.terminal_prize_counts
            ],
        },
        "progress_guard": {
            "ability_repeat_limit": args.ability_repeat_limit,
            "forfeit_count": len(result.forfeit_schedule_indices),
            "forfeit_schedule_indices": list(result.forfeit_schedule_indices),
            "kind": "same_actor_identical_leading_option_repeat_forfeit",
            "engine_turn_draw_limit": args.engine_turn_draw_limit,
            "full_round_draw_limit": (
                args.engine_turn_draw_limit // 2
                if args.engine_turn_draw_limit > 0 else 0
            ),
            "turn_limit_draw_count": len(
                result.turn_limit_draw_schedule_indices
            ),
            "turn_limit_draw_schedule_indices": list(
                result.turn_limit_draw_schedule_indices
            ),
        },
        "models": json_ready(provenance),
        "schedule": {
            "path": str(args.schedule.resolve()),
            "sha256": legacy.sha256_file(args.schedule.resolve()),
            "used_jobs": count,
        },
        "engine": {
            "extension": str(Path(_ptcg_cuda.__file__).resolve()),
            "extension_sha256": legacy.sha256_file(Path(_ptcg_cuda.__file__)),
            "rules_sha256": legacy.sha256_file(args.rules.resolve()),
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "memory": {
            "torch_peak_allocated_bytes": result.peak_allocated_bytes,
            "torch_peak_reserved_bytes": result.peak_reserved_bytes,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.terminal_states_output is not None:
        args.terminal_states_output.parent.mkdir(parents=True, exist_ok=True)
        args.terminal_states_output.write_bytes(result.terminal_state_bytes)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
