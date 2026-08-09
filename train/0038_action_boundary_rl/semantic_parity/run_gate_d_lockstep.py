"""Lockstep official-CPU package vs CUDA-training runtime semantic smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys
import threading
from typing import Any

import torch
from evaluation.runtime.seeded import build_seeded_runtime, load_seeded_library

from engine_cuda.tools.run_official_semantic0031_v2_parity_scaffold import (
    FAMILY_KEYS,
    GLOBAL_KEYS,
    compare_compiled_to_cuda,
    load_training_actor_critic,
)
from ..action_boundary.decision_gate import DecisionClass, DecisionGate
from ..action_boundary.macro_protocol import MacroProtocolError
from ..rollout.cuda_action_boundary import CudaActionBoundaryAdapter
from ..rollout.pool_worker import _PointerBattle
from ..rollout.worker_compiler import WorkerLocalCompiler


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOCAL = (
    ROOT / "train/0038_action_boundary_rl/league/decks/"
    "dragapult_ex_07bedfffbfad/deck.csv"
)
DEFAULT_OPPONENT = (
    ROOT / "train/0038_action_boundary_rl/league/decks/"
    "dragapult_ex_dusknoir_b85bae9f3a21/deck.csv"
)


def _deck(path: Path) -> tuple[int, ...]:
    cards = tuple(int(value) for value in path.read_text().split())
    if len(cards) != 60:
        raise ValueError(f"deck must have 60 cards: {path}")
    return cards


def _legal_hash(observation: dict[str, Any]) -> str:
    select = observation.get("select") or {}
    payload = {
        key: select.get(key)
        for key in (
            "type", "context", "minCount", "maxCount",
            "remainDamageCounter", "remainEnergyCost", "option",
        )
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _feature_hash(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in GLOBAL_KEYS:
        tensor = mapping[name][0].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    for family, names in FAMILY_KEYS.items():
        mask = mapping[f"{family}_mask"][0].bool()
        live = int(mask.sum())
        digest.update(f"{family}:{live}".encode())
        for name in names:
            tensor = mapping[name][0, :live].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _decode_opponent(model, mapping):
    validated, state, options = model.encode(mapping)
    decoded = model.action_decoder.greedy(
        validated, options, state.summary
    )
    length = int(decoded.lengths[0])
    return decoded.sequences[0, :length].tolist()


def _pending_macro(policy) -> dict[str, Any] | None:
    pending = policy.pending
    if pending is None:
        return None
    allocation = pending.allocation
    return {
        "targets": [target.serial for target in allocation.target_ids],
        "counters": list(allocation.counters),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-deck", type=Path, default=DEFAULT_OPPONENT)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if str(args.extension_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    from ptcg_cuda_engine.native import create_official_engine
    from ptcg_cuda_engine.semantic0031_bridge import (
        Semantic0031DeviceAdapter,
        semantic0031_v2_ready_batch,
    )
    from ptcg_cuda_engine.semantic0031_resident import (
        ResidentJob,
        compact_semantic_prefixes,
    )
    from ptcg_cuda_engine.semantic0031_router import Semantic0031ResidentRouter

    package_root = args.package.resolve()
    sys.path.insert(0, str(package_root))
    namespace = runpy.run_path(str(package_root / "main.py"))
    package_policy = namespace["POLICY"]
    collate = importlib.import_module(
        "strategy.features.collate"
    ).collate_canonical_records
    focal = _deck(DEFAULT_FOCAL)
    opponent = _deck(args.opponent_deck.resolve())
    device = torch.device("cuda:0")
    training_model = load_training_actor_critic(
        args.checkpoint.resolve(), focal, device
    )
    run_module = importlib.import_module(
        "train.0038_action_boundary_rl.training.run_full_semantic"
    )
    frozen_opponent = run_module.load_frozen_opponent(device)
    adapter = Semantic0031DeviceAdapter(training_model.actor, focal, max_select=64)
    router = Semantic0031ResidentRouter(
        focal_adapter=adapter,
        opponent_last_option_layer=(
            frozen_opponent.option_encoder.cross_attention_transformer.layers[1]
        ),
        opponent_option_norm=(
            frozen_opponent.option_encoder.cross_attention_transformer.norm
        ),
        opponent_decoder=frozen_opponent.action_decoder,
        same_policy=False,
        focal_summary_fn=training_model.actor_summary,
    )
    cpu_library = load_seeded_library(build_seeded_runtime().library_path)
    cpu_lock = threading.Lock()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    games = []
    total_strategic = total_forced = total_macro = 0
    total_value_forwards = total_focal_forwards = 0
    first_divergence = None

    for seed in seeds:
        for focal_first in (True, False):
            decks = (focal, opponent) if focal_first else (opponent, focal)
            focal_player = 0 if focal_first else 1
            resident_job = ResidentJob(
                schedule_index=0,
                decks=decks,
                engine_seed=seed,
                focal_player=focal_player,
                opponent_id=args.opponent_deck.stem,
                policy_seed=seed,
            )
            boundary = CudaActionBoundaryAdapter(
                training_model, [resident_job], greedy=True, max_select=64
            )
            cuda = create_official_engine(
                args.rules.read_bytes(), batch_size=1, device_index=0
            )
            cuda_decks = torch.tensor([decks], dtype=torch.int32, device=device)
            cuda.reset_seeded_interactive_semantic(
                cuda_decks, torch.tensor([seed], dtype=torch.int64, device=device)
            )
            cuda.advance_to_decision()
            lane = torch.tensor([0], dtype=torch.int32, device=device)
            lane_job = torch.tensor([0], dtype=torch.long, device=device)
            cpu = _PointerBattle(cpu_library, cpu_lock, [])
            observation = cpu.start(
                decks[0], decks[1], seed,
                ((seed + 900_000_007) & 0x7FFFFFFF) or 1,
            )
            compilers = {
                actor: WorkerLocalCompiler(actor, decks[actor])
                for actor in (0, 1)
            }
            gate = DecisionGate()
            package_policy.reset()
            game = {
                "seed": seed,
                "focal_first": focal_first,
                "focal_player": focal_player,
                "decisions": 0,
                "strategic_forwards": 0,
                "focal_forwards": 0,
                "value_forwards": 0,
                "forced_shortcuts": 0,
                "macro_actions": 0,
                "macro_callbacks": 0,
                "status": "running",
                "first_divergence": None,
            }
            try:
                for decision_id in range(args.max_decisions):
                    cpu_current = observation.get("current") or {}
                    cpu_result = cpu_current.get("result")
                    cuda_status = int(cuda.statuses()[0].item())
                    if isinstance(cpu_result, int) and cpu_result >= 0:
                        cuda_result = int(cuda.game_results()[0].item())
                        expected_cuda_result = cpu_result + 1 if cpu_result in (0, 1) else 0
                        if cuda_status != 2 or cuda_result != expected_cuda_result:
                            game["first_divergence"] = {
                                "category": "terminal/winner",
                                "decision_id": decision_id,
                                "cpu_result": cpu_result,
                                "cuda_status": cuda_status,
                                "cuda_result": cuda_result,
                            }
                        else:
                            game["status"] = "PASS"
                            game["winner"] = cpu_result
                        break
                    if cuda_status != 1:
                        game["first_divergence"] = {
                            "category": "cuda_status",
                            "decision_id": decision_id,
                            "cuda_status": cuda_status,
                        }
                        break
                    actor = int(cpu_current.get("yourIndex", -1))
                    cuda_actor = int(cuda.decision_actors()[0].item())
                    if actor != cuda_actor:
                        game["first_divergence"] = {
                            "category": "actor/priority",
                            "decision_id": decision_id,
                            "cpu_actor": actor,
                            "cuda_actor": cuda_actor,
                        }
                        break
                    semantic = compact_semantic_prefixes(
                        semantic0031_v2_ready_batch(
                            cuda.encode_semantic0031_v2_lanes(lane),
                            max_action_steps=64,
                        )
                    )
                    context = (observation.get("select") or {}).get("context")
                    cpu_mapping = None
                    mismatch_counts: Counter[str] = Counter()
                    mismatch_details: list[dict[str, Any]] = []
                    if context != 41:
                        options = (observation.get("select") or {}).get("option") or []
                        if not options:
                            compilers[actor].observe_only(observation)
                        else:
                            record = compilers[actor].compile(observation)
                            cpu_mapping = {
                                name: value.to(device)
                                for name, value in collate([record]).items()
                            }
                        if cpu_mapping is None:
                            # The official empty-pass state has no policy tensor
                            # row by contract; DecisionGate parity is checked by
                            # the identical zero-option/min/max contract below.
                            if not (
                                int(semantic["option_mask"].sum()) == 0
                                and int(semantic["min_count"][0]) == 0
                                and int(semantic["max_count"][0]) == 0
                            ):
                                game["first_divergence"] = {
                                    "category": "legal_empty_pass",
                                    "decision_id": decision_id,
                                }
                                break
                        else:
                            compare_compiled_to_cuda(
                                cpu_mapping,
                                semantic,
                                seed=seed,
                                decision=decision_id,
                                actor=actor,
                                mismatch_counts=mismatch_counts,
                                mismatch_details=mismatch_details,
                                max_details=8,
                            )
                        if mismatch_counts:
                            select_payload = observation.get("select") or {}
                            players_payload = cpu_current.get("players") or [{}, {}]
                            own_payload = (
                                players_payload[actor]
                                if len(players_payload) == 2 else {}
                            )
                            game["first_divergence"] = {
                                "category": "observation/legal/mask",
                                "decision_id": decision_id,
                                "legal_set_hash": _legal_hash(observation),
                                "cpu_feature_hash": _feature_hash(cpu_mapping),
                                "cuda_feature_hash": _feature_hash(semantic),
                                "mismatch_counts": dict(mismatch_counts),
                                "details": mismatch_details,
                                "observation_excerpt": {
                                    "turn": cpu_current.get("turn"),
                                    "actor": actor,
                                    "select_type": select_payload.get("type"),
                                    "select_context": select_payload.get("context"),
                                    "legal_options": select_payload.get("option"),
                                    "cuda_live_option_cat": semantic["option_cat"][
                                        0, semantic["option_mask"][0].bool()
                                    ].detach().cpu().tolist(),
                                    "select_deck_count": len(select_payload.get("deck") or []),
                                    "select_deck_ids": [
                                        item.get("id") if isinstance(item, dict) else None
                                        for item in (select_payload.get("deck") or [])
                                    ],
                                    "own_deck_count": own_payload.get("deckCount"),
                                    "own_prize_count": len(own_payload.get("prize") or []),
                                    "own_visible": {
                                        zone: own_payload.get(zone)
                                        for zone in ("active", "bench", "hand", "discard")
                                    },
                                    "stadium": cpu_current.get("stadium"),
                                    "looking": cpu_current.get("looking"),
                                    "context_card": select_payload.get("contextCard"),
                                    "effect": select_payload.get("effect"),
                                    "logs": observation.get("logs"),
                                },
                            }
                            break

                    focal_route = torch.tensor(
                        [actor == focal_player], dtype=torch.bool, device=device
                    )
                    opponent_route = ~focal_route
                    try:
                        bypass = boundary.pre_route(
                            semantic=semantic,
                            ready=torch.ones(1, dtype=torch.bool, device=device),
                            focal_route=focal_route,
                            opponent_route=opponent_route,
                            lane_job=lane_job,
                            turns=cuda.turns(),
                            selections=torch.tensor(
                                [decision_id], dtype=torch.int32, device=device
                            ),
                            max_select=64,
                        )
                    except MacroProtocolError as error:
                        game["first_divergence"] = {
                            "category": "macro_transaction",
                            "decision_id": decision_id,
                            "engine_call_id": decision_id,
                            "turn": int(cpu_current.get("turn", -1)),
                            "actor": actor,
                            "detail": str(error),
                            "cpu_pending": _pending_macro(package_policy),
                            "cuda_invalid_jobs": dict(boundary.invalid_jobs),
                        }
                        break
                    routed = None
                    cuda_macro = None
                    if bool(bypass.bypass_mask[0]):
                        length = int(bypass.lengths[0])
                        cuda_action = bypass.actions[0, :length].tolist()
                    else:
                        routed = router.route(
                            semantic,
                            focal_route=focal_route,
                            opponent_route=opponent_route,
                            max_select=max(1, int(semantic["max_count"][0])),
                            focal_greedy=True,
                            compute_stats=False,
                        )
                        metadata = boundary.post_route(
                            semantic=semantic,
                            routed=routed,
                            focal_route=focal_route,
                            opponent_route=opponent_route,
                            lane_indices=lane,
                            lane_job=lane_job,
                            turns=cuda.turns(),
                        )[0]
                        length = int(routed.lengths[0])
                        cuda_action = routed.actions[0, :length].tolist()
                        cuda_macro = (
                            None if metadata is None else metadata.get("macro_action")
                        )
                        game["strategic_forwards"] += 1
                        if actor == focal_player:
                            game["focal_forwards"] += 1
                            game["value_forwards"] += 1

                    if context == 41:
                        cpu_action = [0]
                        cpu_gate = "FIRST_PLAYER_HARNESS"
                    elif actor == focal_player:
                        cpu_pending_before = package_policy.pending is not None
                        cpu_action = package_policy.select(observation)
                        cpu_gate = (
                            "MACRO_CALLBACK" if cpu_pending_before
                            else gate.classify(observation).classification.value
                        )
                    else:
                        cpu_gate_result = gate.classify(observation)
                        cpu_gate = cpu_gate_result.classification.value
                        if cpu_gate_result.classification in {
                            DecisionClass.FORCED, DecisionClass.LEGAL_EMPTY_PASS,
                        }:
                            cpu_action = list(cpu_gate_result.forced_action or ())
                        elif cpu_gate_result.classification is DecisionClass.STRATEGIC:
                            if cpu_mapping is None:
                                raise RuntimeError("opponent strategic mapping is missing")
                            cpu_action = _decode_opponent(frozen_opponent, cpu_mapping)
                        else:
                            raise RuntimeError(f"unsupported opponent gate: {cpu_gate_result}")
                    cpu_macro = _pending_macro(package_policy)
                    if cpu_macro is not None and cuda_macro is not None:
                        cuda_macro_canonical = {
                            "targets": cuda_macro["targets"],
                            "counters": cuda_macro["counters"],
                        }
                        if cpu_macro != cuda_macro_canonical:
                            game["first_divergence"] = {
                                "category": "canonical_macro",
                                "decision_id": decision_id,
                                "cpu_macro": cpu_macro,
                                "cuda_macro": cuda_macro_canonical,
                            }
                            break
                        game["macro_actions"] += 1
                    if cpu_action != cuda_action:
                        detail = {
                            "category": "greedy_action",
                            "decision_id": decision_id,
                            "engine_call_id": decision_id,
                            "turn": int(cpu_current.get("turn", -1)),
                            "actor": actor,
                            "decision_gate": cpu_gate,
                            "legal_set_hash": _legal_hash(observation),
                            "cpu_action": cpu_action,
                            "cuda_action": cuda_action,
                            "cpu_macro": cpu_macro,
                            "cuda_macro": cuda_macro,
                        }
                        game["first_divergence"] = detail
                        break
                    if bool(bypass.forced_mask[0]):
                        game["forced_shortcuts"] += 1
                    if bool(bypass.macro_mask[0]):
                        game["macro_callbacks"] += 1
                    width = max(1, len(cuda_action))
                    packed = torch.zeros((1, width), dtype=torch.long, device=device)
                    if cuda_action:
                        packed[0, :len(cuda_action)] = torch.tensor(
                            cuda_action, dtype=torch.long, device=device
                        )
                    cuda.pack_actions(
                        packed,
                        torch.tensor([len(cuda_action)], dtype=torch.long, device=device),
                    )
                    observation = cpu.select(cpu_action)
                    cuda.apply_packed_actions()
                    cuda.advance_to_decision()
                    game["decisions"] = decision_id + 1
                else:
                    game["first_divergence"] = {
                        "category": "decision_limit",
                        "limit": args.max_decisions,
                    }
            finally:
                cpu.finish()
            if game["first_divergence"] is not None:
                game["status"] = "FAIL"
                if first_divergence is None:
                    first_divergence = {
                        "game": len(games), **game["first_divergence"]
                    }
            total_strategic += game["strategic_forwards"]
            total_focal_forwards += game["focal_forwards"]
            total_value_forwards += game["value_forwards"]
            total_forced += game["forced_shortcuts"]
            total_macro += game["macro_actions"]
            games.append(game)
            if first_divergence is not None:
                break
        if first_divergence is not None:
            break

    passed = first_divergence is None and len(games) == len(seeds) * 2
    report = {
        "schema_version": "0038_gate_d_cpu_package_cuda_lockstep_v1",
        "gate": "D",
        "status": "PASS" if passed else "FAIL",
        "games_requested": len(seeds) * 2,
        "games_completed": len(games),
        "first_divergence": first_divergence,
        "totals": {
            "strategic_forwards": total_strategic,
            "focal_forwards": total_focal_forwards,
            "value_forwards": total_value_forwards,
            "forced_shortcuts": total_forced,
            "macro_actions": total_macro,
            "fallbacks": 0,
            "timeouts": 0,
        },
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"Gate D first divergence: {first_divergence}")


if __name__ == "__main__":
    main()
