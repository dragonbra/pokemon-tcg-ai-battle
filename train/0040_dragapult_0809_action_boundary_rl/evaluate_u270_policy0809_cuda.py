"""Evaluate immutable 0040 U270 on four exact decks against Policy-0809."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable
import uuid

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine_cuda.tools.evaluate_policy_0809_cuda import (
    _detail_payload,
    _publish as publish_combat_mat,
    _render_deck,
)
from evaluation.frozen_0806_contract import (
    FROZEN_0806_CONTRACT_ID,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    evaluation_coin_winner,
    evaluation_game_seed,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.runtime.seeded import build_seeded_runtime

from .candidate_deployment import (
    load_kaggle_evaluation_candidate,
    require_kaggle_candidate_deployment,
)
from .export_full_semantic_candidate import export_candidate
from .rollout.cuda_collector import CudaFullSemanticRolloutCollector
from .rollout.protocol import DEFAULT_FULL_ROUND_DRAW_LIMIT, RolloutJob
from .training.run_full_semantic import CUDA_EXTENSION, CUDA_RULES, runtime_root


CHECKPOINT = ROOT / (
    "rl_runs/0040_dragapult_0809_action_boundary_rl/versions/"
    "V2_snapshot_loader_fix_long_run/checkpoint/update-000270.pt"
)
CHECKPOINT_SHA256 = "c87bcf82ca2b22eed65876da7d0ec0648a85d72d40dd13dd614290e7de002142"
SOURCE = ROOT / (
    "archive/submission/0031_zero_shot_0809_007_dragapult_ex_"
    "fp16_storage_fp32_runtime"
)
POOL_ROOT = ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
FOCAL_POLICY_ID = "0040-U270"
OPPONENT_POLICY_ID = "Policy-0809"
OPPONENT_CHECKPOINT_SHA256 = (
    "926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f"
)
OPPONENT_EFFECTIVE_SHA256 = (
    "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
)
REQUESTED_NUMBERS = ("007", "002", "009", "003")
EXPECTED_GAMES = 2048
TEMP_ROOT = ROOT / ".tmp/evaluation/0040_u270_policy0809_cuda2048"
OUTPUT_ROOT = ROOT / (
    "docs/evaluation/combat_mat/policy_0809/"
    "u270_cross_deck_002_003_007_009_vs_frozen_0809_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1"
)
VERSION = "V4_u270_cross_deck_policy0809_cuda2048"
FORMAL_REPORT = (
    ROOT / "experiments/0040_dragapult_0809_action_boundary_rl/evaluation"
    / f"{VERSION}.html"
)
VERSION_ROOT = (
    ROOT / "rl_runs/0040_dragapult_0809_action_boundary_rl/versions" / VERSION
)
TRAINING_PROVENANCE_WARNING = (
    "U270 was trained with a heterogeneous opponent-deck CUDA batch whose opponent "
    "adapter used the first lane's registered-deck static fields. This corrected "
    "evaluation is valid strength evidence for the materialized candidate, but it "
    "does not validate the declared V2 opponent-pool training semantics."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _catalog() -> tuple[Any, tuple[Any, ...]]:
    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    candidates = tuple(sorted(
        catalog.candidates,
        key=lambda item: int(item.package_manifest["frozen_deck_number"]),
    ))
    if len(candidates) != 55 or sum(item.games for item in catalog.pool.schedule) != 256:
        raise RuntimeError("FATAL: immutable 55-deck/256-slot catalog mismatch")
    return catalog, candidates


def build_schedule(*, focal_deck_id: str, entries: Iterable[Any]) -> dict[str, Any]:
    jobs = []
    for replica in range(FROZEN_0806_EVALUATION_UNITS):
        for entry in entries:
            opponent_id = str(entry.deck_id)
            focal_identity = f"{FOCAL_POLICY_ID}:{focal_deck_id}"
            opponent_identity = f"{OPPONENT_POLICY_ID}:{opponent_id}"
            for slot in range(int(entry.games)):
                jobs.append({
                    "game_id": f"r{replica + 1:02d}-{opponent_id}-{slot + 1:03d}",
                    "opponent_id": opponent_id,
                    "replica": replica,
                    "slot": slot,
                    "engine_seed": evaluation_game_seed(
                        evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                        focal_identity=focal_identity,
                        opponent_identity=opponent_identity,
                        slot=slot,
                        replica=replica,
                    ),
                    "search_seed": evaluation_game_seed(
                        evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                        focal_identity=focal_identity,
                        opponent_identity=opponent_identity,
                        slot=slot,
                        replica=replica,
                        namespace="search",
                    ),
                    "focal_won_toss": evaluation_coin_winner(
                        evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                        focal_identity=focal_identity,
                        opponent_identity=opponent_identity,
                        slot=slot,
                        replica=replica,
                    ),
                    "focal_policy_id": FOCAL_POLICY_ID,
                    "opponent_policy_id": OPPONENT_POLICY_ID,
                })
    payload = {
        "schema": "0040_u270_policy0809_cuda2048_schedule_v1",
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
        "focal_policy_id": FOCAL_POLICY_ID,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "focal_deck_id": focal_deck_id,
        "jobs": jobs,
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        len(jobs) != EXPECTED_GAMES
        or len({row["engine_seed"] for row in jobs}) != EXPECTED_GAMES
        or len({row["search_seed"] for row in jobs}) != EXPECTED_GAMES
        or Counter(row["replica"] for row in jobs)
        != Counter({replica: 256 for replica in range(8)})
    ):
        raise RuntimeError("FATAL: U270 schedule is not eight unique 256-game units")
    return payload


def _deck_path(candidate: Any) -> Path:
    number = str(candidate.package_manifest["frozen_deck_number"])
    matches = tuple((POOL_ROOT / "decks").glob(f"{number}_*/deck.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"FATAL: deck {number} did not resolve uniquely")
    return matches[0]


def _package(candidate: Any) -> tuple[Path, dict[str, Any]]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    output = TEMP_ROOT / "packages" / number
    if not output.exists():
        export_candidate(
            source=SOURCE,
            checkpoint=CHECKPOINT,
            output=output,
            require_frozen_selection=False,
            deck_path=_deck_path(candidate),
            deck_id=candidate.name,
            deck_display_name=f"{number} · {candidate.display_name}",
            expected_deck_sha256=str(
                candidate.package_manifest["exact_deck_sha256"]
            ),
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("checkpoint_update") != 270
        or manifest.get("rl_checkpoint_sha256") != CHECKPOINT_SHA256
        or manifest.get("deck_id") != candidate.name
        or manifest.get("deck_sha256")
        != candidate.package_manifest["exact_deck_sha256"]
        or manifest.get("storage_dtype") != "fp16"
        or manifest.get("runtime_dtype") != "fp32"
        or manifest.get("portable_checkpoint_sha256")
        != _sha256(output / "strategy/model.bin")
    ):
        raise RuntimeError(f"FATAL: U270 package {number} identity mismatch")
    return output, manifest


def _opponent(device: torch.device, deck: tuple[int, ...]) -> Any:
    identity = importlib.import_module(
        "train.0040_dragapult_0809_action_boundary_rl.policy_identity"
    )
    resolved = identity.materialize_policy(
        OPPONENT_POLICY_ID,
        deck,
        device,
        purpose="0040_u270_cross_deck_policy0809_cuda2048",
    )
    if (
        resolved.policy_id != OPPONENT_POLICY_ID
        or resolved.audit.status != "PASS"
        or resolved.audit.requested_policy_id != OPPONENT_POLICY_ID
    ):
        raise RuntimeError("FATAL: independent Policy-0809 opponent audit failed")
    model = resolved.model.eval().requires_grad_(False)
    model._policy_id = resolved.policy_id
    model._policy_identity_audit = resolved.audit
    return model


def _rollout_jobs(
    candidate: Any, schedule: dict[str, Any], candidates: tuple[Any, ...]
) -> list[RolloutJob]:
    opponent_by_id = {item.name: item for item in candidates}
    seeded_runtime = build_seeded_runtime()
    root = runtime_root()
    focal_deck = tuple(int(value) for value in candidate.deck)
    return [
        RolloutJob(
            game_id=str(row["game_id"]),
            opponent_id=str(row["opponent_id"]),
            focal_first=False,
            seed=int(row["engine_seed"]),
            source_policy_update=270,
            focal_deck=focal_deck,
            opponent_deck=tuple(
                int(value) for value in opponent_by_id[str(row["opponent_id"])].deck
            ),
            runtime_root=root,
            opponent_policy_id=OPPONENT_POLICY_ID,
            policy_seed=int(row["search_seed"]),
            search_seed=int(row["search_seed"]),
            engine_library=seeded_runtime.library_path,
            full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
            ability_repeat_limit=20,
            action_boundary_mode="enabled",
            focal_won_toss=bool(row["focal_won_toss"]),
        )
        for row in schedule["jobs"]
    ]


def _group_jobs_by_opponent_deck(
    jobs: Iterable[RolloutJob],
) -> list[list[RolloutJob]]:
    groups: dict[tuple[int, ...], list[RolloutJob]] = {}
    for job in jobs:
        groups.setdefault(job.opponent_deck, []).append(job)
    output = list(groups.values())
    if any(
        len({job.opponent_deck for job in group}) != 1
        or len({job.opponent_id for job in group}) != 1
        for group in output
    ):
        raise RuntimeError("FATAL: opponent exact-deck grouping is not homogeneous")
    return output


def _collect_grouped_jobs(
    model: Any,
    opponent: Any,
    jobs: list[RolloutJob],
    device: torch.device,
) -> tuple[list[Any], list[dict[str, Any]]]:
    episodes_by_id: dict[str, Any] = {}
    group_metrics: list[dict[str, Any]] = []
    for group in _group_jobs_by_opponent_deck(jobs):
        collector = CudaFullSemanticRolloutCollector(
            model,
            opponent,
            device=device,
            rules_path=CUDA_RULES,
            extension_dir=CUDA_EXTENSION,
            lane_count=min(256, len(group)),
            mode="greedy",
            check_interval=8,
            ability_repeat_limit=20,
            record_trajectory=False,
            agent_selects_first_player=True,
            opponent_policy_id=OPPONENT_POLICY_ID,
            opponent_identity_audit=opponent._policy_identity_audit,
        )
        collected = collector.collect(group)
        if len(collected) != len(group):
            raise RuntimeError("FATAL: CUDA opponent group returned an incomplete batch")
        for episode in collected:
            if episode.job.game_id in episodes_by_id:
                raise RuntimeError("FATAL: duplicate CUDA game result")
            episodes_by_id[episode.job.game_id] = episode
        group_metrics.append({
            "opponent_id": group[0].opponent_id,
            "opponent_deck_sha256": hashlib.sha256(
                ",".join(str(card) for card in sorted(group[0].opponent_deck)).encode(
                    "ascii"
                )
            ).hexdigest(),
            "games": len(group),
            "metrics": collector.metrics(),
        })
    if set(episodes_by_id) != {job.game_id for job in jobs}:
        raise RuntimeError("FATAL: grouped CUDA results do not conserve the schedule")
    return [episodes_by_id[job.game_id] for job in jobs], group_metrics


def _game_records(episodes: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        choice = episode.diagnostics.get("first_player_choice")
        fallback = bool(episode.diagnostics.get("macro_fallback"))
        reason = episode.diagnostics.get("macro_fallback_reason")
        chance_boundary = fallback and reason == "chance_boundary_before_allocation"
        semantic_fallback = fallback and not chance_boundary
        outcome = (
            "win" if episode.reward == 1.0
            else "loss" if episode.reward == -1.0
            else "draw"
        )
        rows.append({
            "game_id": episode.job.game_id,
            "opponent_id": episode.job.opponent_id,
            "focal_policy_id": FOCAL_POLICY_ID,
            "opponent_policy_id": OPPONENT_POLICY_ID,
            "replica": int(episode.job.game_id[1:3]) - 1,
            "engine_seed": episode.job.seed,
            "search_seed": episode.job.search_seed,
            "policy_seed": episode.job.policy_seed,
            "focal_won_toss": episode.job.focal_won_toss,
            "first_player_choice": choice,
            "focal_first": choice.get("focal_first") if isinstance(choice, dict) else None,
            "focal_outcome": outcome,
            "terminal_turn": episode.turns,
            "game_length": math.ceil(episode.turns / 2),
            "valid": episode.valid,
            "error": episode.error,
            "termination_status": episode.diagnostics.get("termination_status"),
            "engine_selections": episode.diagnostics.get("engine_selections"),
            "chance_boundary": chance_boundary,
            "semantic_fallback": semantic_fallback,
            "fallback_reason": reason,
        })
    return rows


def _valid_result(
    result: dict[str, Any], *, candidate: Any, package: Path, schedule: dict[str, Any]
) -> bool:
    games = result.get("games")
    candidate_audit = result.get("candidate_deployment_identity_audit", {})
    opponent_audit = result.get("opponent_policy_identity_audit", {})
    collector = result.get("collector", {})
    engine = result.get("engine", {})
    package_manifest = json.loads(
        (package / "manifest.json").read_text(encoding="utf-8")
    )
    return bool(
        result.get("passed") is True
        and result.get("schema") == "0040_u270_policy0809_cuda2048_result_v1"
        and result.get("schedule_sha256") == schedule["schedule_sha256"]
        and result.get("deck_id") == candidate.name
        and result.get("exact_deck_sha256")
        == candidate.package_manifest["exact_deck_sha256"]
        and isinstance(games, list)
        and len(games) == EXPECTED_GAMES
        and all(
            row.get("game_id") == expected["game_id"]
            and row.get("opponent_id") == expected["opponent_id"]
            and row.get("replica") == expected["replica"]
            and row.get("engine_seed") == expected["engine_seed"]
            and row.get("search_seed") == expected["search_seed"]
            and row.get("focal_won_toss") == expected["focal_won_toss"]
            for row, expected in zip(games, schedule["jobs"], strict=True)
        )
        and all(
            row.get("valid") is True
            and row.get("error") in (None, "")
            and row.get("semantic_fallback") is False
            and isinstance(row.get("first_player_choice"), dict)
            and type(row.get("focal_first")) is bool
            and isinstance(row.get("terminal_turn"), int)
            for row in games
        )
        and candidate_audit.get("status") == "PASS"
        and candidate_audit.get("checkpoint_update") == 270
        and candidate_audit.get("source_checkpoint_sha256") == CHECKPOINT_SHA256
        and candidate_audit.get("contract_id")
        == "kaggle_fp16_storage_fp32_runtime_v1"
        and candidate_audit.get("storage_dtype") == "fp16"
        and candidate_audit.get("runtime_dtype") == "fp32"
        and candidate_audit.get("portable_checkpoint_sha256")
        == _sha256(package / "strategy/model.bin")
        and candidate_audit.get("effective_candidate_sha256")
        == package_manifest.get("deployment_effective_sha256")
        and result.get("package_manifest") == package_manifest
        and opponent_audit.get("status") == "PASS"
        and opponent_audit.get("requested_policy_id") == OPPONENT_POLICY_ID
        and opponent_audit.get("checkpoint_sha256") == OPPONENT_CHECKPOINT_SHA256
        and opponent_audit.get("effective_policy_sha256")
        == OPPONENT_EFFECTIVE_SHA256
        and collector.get("opponent_deck_groups") == 55
        and isinstance(collector.get("group_metrics"), list)
        and len(collector["group_metrics"]) == 55
        and sum(group.get("games", -EXPECTED_GAMES) for group in collector["group_metrics"])
        == EXPECTED_GAMES
        and engine.get("extension_sha256")
        == _sha256(CUDA_EXTENSION / "_ptcg_cuda.so")
        and engine.get("rules_sha256") == _sha256(CUDA_RULES)
    )


def _run_one(
    candidate: Any,
    candidates: tuple[Any, ...],
    catalog: Any,
    opponent: Any,
    device: torch.device,
) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    package, package_manifest = _package(candidate)
    schedule = build_schedule(focal_deck_id=candidate.name, entries=catalog.pool.schedule)
    _atomic_json(TEMP_ROOT / "schedules" / f"{number}.json", schedule)
    result_path = TEMP_ROOT / "results" / f"{number}.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if _valid_result(
            cached, candidate=candidate, package=package, schedule=schedule
        ):
            return cached
        raise RuntimeError(f"FATAL: cached deck {number} result identity mismatch")

    model, candidate_audit = load_kaggle_evaluation_candidate(
        package=package,
        checkpoint=CHECKPOINT,
        deck=tuple(int(value) for value in candidate.deck),
        device=device,
    )
    require_kaggle_candidate_deployment(model)
    jobs = _rollout_jobs(candidate, schedule, candidates)
    started = time.perf_counter()
    episodes, group_metrics = _collect_grouped_jobs(
        model, opponent, jobs, device
    )
    wall_seconds = time.perf_counter() - started
    games = _game_records(episodes)
    payload = {
        "schema": "0040_u270_policy0809_cuda2048_result_v1",
        "passed": True,
        "deck_number": number,
        "deck_id": candidate.name,
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "checkpoint_update": 270,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "package_manifest": package_manifest,
        "candidate_deployment_identity_audit": candidate_audit.to_manifest(),
        "opponent_policy_identity_audit": (
            opponent._policy_identity_audit.to_manifest()
        ),
        "collector": {
            "wall_seconds": wall_seconds,
            "opponent_deck_groups": len(group_metrics),
            "group_metrics": group_metrics,
        },
        "engine": {
            "extension": str((CUDA_EXTENSION / "_ptcg_cuda.so").relative_to(ROOT)),
            "extension_sha256": _sha256(CUDA_EXTENSION / "_ptcg_cuda.so"),
            "rules_sha256": _sha256(CUDA_RULES),
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "games": games,
    }
    payload["passed"] = _valid_result(
        payload, candidate=candidate, package=package, schedule=schedule
    )
    if not payload["passed"]:
        raise RuntimeError(f"FATAL: deck {number} result failed formal validation")
    _atomic_json(result_path, payload)
    del model
    torch.cuda.empty_cache()
    return payload


def _run_smoke(
    candidate: Any,
    candidates: tuple[Any, ...],
    catalog: Any,
    opponent: Any,
    device: torch.device,
    games: int,
) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    package, package_manifest = _package(candidate)
    schedule = build_schedule(
        focal_deck_id=candidate.name, entries=catalog.pool.schedule
    )
    model, candidate_audit = load_kaggle_evaluation_candidate(
        package=package,
        checkpoint=CHECKPOINT,
        deck=tuple(int(value) for value in candidate.deck),
        device=device,
    )
    require_kaggle_candidate_deployment(model)
    jobs = _rollout_jobs(candidate, schedule, candidates)[:games]
    started = time.perf_counter()
    episodes, group_metrics = _collect_grouped_jobs(
        model, opponent, jobs, device
    )
    records = _game_records(episodes)
    passed = bool(
        len(records) == games
        and candidate_audit.status == "PASS"
        and opponent._policy_identity_audit.status == "PASS"
        and all(
            row["valid"] is True
            and row["error"] in (None, "")
            and row["semantic_fallback"] is False
            and isinstance(row["first_player_choice"], dict)
            and type(row["focal_first"]) is bool
            for row in records
        )
    )
    payload = {
        "schema": "0040_u270_policy0809_cuda_smoke_v1",
        "passed": passed,
        "deck_number": number,
        "games_requested": games,
        "schedule_sha256": schedule["schedule_sha256"],
        "package_manifest": package_manifest,
        "candidate_deployment_identity_audit": candidate_audit.to_manifest(),
        "opponent_policy_identity_audit": (
            opponent._policy_identity_audit.to_manifest()
        ),
        "collector": {
            "wall_seconds": time.perf_counter() - started,
            "opponent_deck_groups": len(group_metrics),
            "group_metrics": group_metrics,
        },
        "games": records,
    }
    _atomic_json(TEMP_ROOT / "smoke" / f"{number}-{games}.json", payload)
    del model
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError("FATAL: U270 Policy-0809 CUDA smoke failed")
    return payload


def _summary(catalog: Any, candidate: Any, result: dict[str, Any]) -> dict[str, Any]:
    games = result["games"]
    wins = sum(row["focal_outcome"] == "win" for row in games)
    losses = sum(row["focal_outcome"] == "loss" for row in games)
    draws = len(games) - wins - losses
    first = [row for row in games if row["focal_first"]]
    second = [row for row in games if not row["focal_first"]]
    schedule = next(item for item in catalog.pool.schedule if item.deck_id == candidate.name)
    number = str(candidate.package_manifest["frozen_deck_number"])
    return {
        "deck_number": number,
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "representative_cards": [dict(item) for item in candidate.representative_cards],
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_games": int(schedule.games),
        "best_rank": int(schedule.best_rank),
        "observed_players": int(schedule.observed_players),
        "segment": str(schedule.segment),
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(games),
        "actual_first_games": len(first),
        "actual_first_wins": sum(row["focal_outcome"] == "win" for row in first),
        "actual_second_games": len(second),
        "actual_second_wins": sum(row["focal_outcome"] == "win" for row in second),
        "wall_seconds": result["collector"]["wall_seconds"],
        "schedule_sha256": result["schedule_sha256"],
        "candidate_deployment_identity_audit": result[
            "candidate_deployment_identity_audit"
        ],
        "opponent_policy_identity_audit": result["opponent_policy_identity_audit"],
        "engine": result["engine"],
        "device": result["device"],
        "training_provenance_warning": TRAINING_PROVENANCE_WARNING,
        "report": f"reports/{number}.html",
        "games_file": f"games/{number}.json",
    }


def _publish_formal(rows: list[dict[str, Any]], opponent_audit: dict[str, Any]) -> None:
    if FORMAL_REPORT.exists() or VERSION_ROOT.exists():
        raise FileExistsError("formal V4 destination already exists")
    total_games = sum(row["games"] for row in rows)
    total_wins = sum(row["wins"] for row in rows)
    total_losses = sum(row["losses"] for row in rows)
    total_draws = sum(row["draws"] for row in rows)
    table = "".join(
        f'<tr><td><a href="../../../docs/evaluation/combat_mat/policy_0809/'
        f'u270_cross_deck_002_003_007_009_vs_frozen_0809_cuda_seeded_2048_agent_choice_v3_'
        f'kaggle_fp16_storage_fp32_runtime_v1/{row["report"]}">'
        f'{html.escape(row["deck_number"])} · {html.escape(row["display_name"])}</a></td>'
        f'<td>{row["wins"]}-{row["losses"]}-{row["draws"]}</td>'
        f'<td>{row["win_rate"]:.2%}</td><td><code>{html.escape(row["exact_deck_sha256"][:12])}</code></td>'
        f'<td><code>{html.escape(row["candidate_deployment_identity_audit"]["portable_checkpoint_sha256"][:12])}</code></td>'
        f'<td><code>{html.escape(row["candidate_deployment_identity_audit"]["effective_candidate_sha256"][:12])}</code></td>'
        f'<td>{row["wall_seconds"]:.1f}s</td></tr>'
        for row in rows
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>0040 U270 Cross-Deck Policy-0809 CUDA2048</title><style>body{{font:14px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#172b25}}header{{padding:24px;background:#14563d;color:white}}table{{width:100%;border-collapse:collapse;margin-top:18px}}th,td{{padding:10px;border-bottom:1px solid #dce7e2;text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#217a58}}.warning{{padding:14px;border-left:4px solid #ad6b14;background:#fff4df}}</style></head><body><header><h1>0040 U270 Cross-Deck · Policy-0809 CUDA-2048</h1><p>Immutable update 270 weights · exact decks 007 / 002 / 009 / 003 · no automatic promotion</p></header><p>总计 {total_games:,} 局：{total_wins}-{total_losses}-{total_draws}，胜率 {total_wins/total_games:.2%}。Candidate contract <code>kaggle_fp16_storage_fp32_runtime_v1</code>；source checkpoint <code>{CHECKPOINT_SHA256}</code>；opponent identity <code>{html.escape(str(opponent_audit.get("effective_policy_sha256")))}</code>.</p><p class="warning"><strong>Training provenance boundary:</strong> {html.escape(TRAINING_PROVENANCE_WARNING)}</p><table><thead><tr><th>Exact deck</th><th>W-L-D</th><th>胜率</th><th>Deck hash</th><th>Portable FP16</th><th>Deployment effective</th><th>耗时</th></tr></thead><tbody>{table}</tbody></table><p>本页只记录同合同评测证据，不作 PROMOTE / HOLD / REJECT 决策。</p></body></html>"""
    _atomic_text(FORMAL_REPORT, page)
    artifact = VERSION_ROOT / "artifact"
    _atomic_json(artifact / "evaluation.json", {
        "version": VERSION,
        "report": str(FORMAL_REPORT.relative_to(ROOT)),
        "combat_mat": str((OUTPUT_ROOT / "index.html").relative_to(ROOT)),
        "games": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "training_provenance_warning": TRAINING_PROVENANCE_WARNING,
    })
    _atomic_json(artifact / "status.json", {
        "version": VERSION,
        "state": "completed",
        "phase": "evaluation_only",
        "checkpoint_update": 270,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "candidate_contract": "kaggle_fp16_storage_fp32_runtime_v1",
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_policy_identity_audit": opponent_audit,
        "games": total_games,
        "errors": 0,
        "unfinished": 0,
        "semantic_fallbacks": 0,
        "promotion_decision": None,
        "training_provenance_status": "INVALID_AS_DECLARED_OPPONENT_DECK_ROUTING",
        "training_provenance_warning": TRAINING_PROVENANCE_WARNING,
    })
    index_path = FORMAL_REPORT.parent / "index.html"
    if index_path.is_file():
        source = index_path.read_text(encoding="utf-8")
        if VERSION not in source:
            row = (
                f'<tr><td><a href="{VERSION}.html">{VERSION}</a></td>'
                '<td>U270 exact decks 007 / 002 / 009 / 003 vs Policy-0809</td>'
                f'<td>{total_games}</td><td>{total_wins}-{total_losses}-{total_draws}</td>'
                f'<td>{total_wins/total_games:.2%}</td><td>0</td><td>100%</td></tr>'
            )
            source = source.replace("</tbody>", row + "</tbody>")
            _atomic_text(index_path, source)


def publish(
    catalog: Any,
    candidates: tuple[Any, ...],
    selected: list[Any],
    results: list[dict[str, Any]],
) -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    candidate_audits = [
        result["candidate_deployment_identity_audit"] for result in results
    ]
    portable_hashes = {
        audit["portable_checkpoint_sha256"] for audit in candidate_audits
    }
    effective_hashes = {
        audit["effective_candidate_sha256"] for audit in candidate_audits
    }
    opponent_audits = {
        json.dumps(
            result["opponent_policy_identity_audit"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for result in results
    }
    if (
        len(results) != len(REQUESTED_NUMBERS)
        or len(portable_hashes) != 1
        or len(effective_hashes) != 1
        or len(opponent_audits) != 1
    ):
        raise RuntimeError(
            "FATAL: cross-deck candidate/opponent effective identity mismatch"
        )
    rows = [_summary(catalog, candidate, result) for candidate, result in zip(selected, results, strict=True)]
    completed = set(REQUESTED_NUMBERS)
    for candidate, result, summary in zip(selected, results, rows, strict=True):
        _atomic_json(OUTPUT_ROOT / summary["games_file"], {
            "schema": "0040_u270_policy0809_cuda2048_games_v1",
            "summary": summary,
            "games": result["games"],
        })
        detail = _detail_payload(candidate, result["games"], candidates)
        _atomic_text(
            OUTPUT_ROOT / summary["report"],
            _render_deck(
                summary,
                result["games"],
                catalog=catalog,
                candidates=candidates,
                detail=detail,
                completed_numbers=completed,
                focal_policy_label="0040-U270",
                run_label="0040-u270-policy0809-cuda2048",
                pool_id="0809_kaggle_top100_plus_v1",
            ),
        )
    opponent_audit = results[0]["opponent_policy_identity_audit"]
    identity = {
        "schema_version": "0040_u270_policy0809_formal_identity_v1",
        "status": "PASS",
        "focal": {
            "policy_id": FOCAL_POLICY_ID,
            "checkpoint_update": 270,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "deployment_effective_sha256": results[0][
                "candidate_deployment_identity_audit"
            ]["effective_candidate_sha256"],
        },
        "opponent": opponent_audit,
        "training_provenance_warning": TRAINING_PROVENANCE_WARNING,
    }
    publish_combat_mat(
        identity,
        rows,
        catalog,
        candidates,
        output_root=OUTPUT_ROOT,
        focal_policy_id=FOCAL_POLICY_ID,
        opponent_policy_id=OPPONENT_POLICY_ID,
        focal_policy_sha256=results[0][
            "candidate_deployment_identity_audit"
        ]["effective_candidate_sha256"],
        requested_numbers=REQUESTED_NUMBERS,
        page_title="0040 U270 Cross-Deck Report",
        evidence_note=TRAINING_PROVENANCE_WARNING,
    )
    _publish_formal(rows, opponent_audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-games", type=int)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args(argv)
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("FATAL: U270 checkpoint SHA-256 mismatch")
    sidecar = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != CHECKPOINT_SHA256:
        raise RuntimeError("FATAL: U270 checkpoint sidecar mismatch")
    catalog, candidates = _catalog()
    by_number = {
        str(item.package_manifest["frozen_deck_number"]): item for item in candidates
    }
    selected = [by_number[number] for number in REQUESTED_NUMBERS]
    if args.publish_only:
        results = []
        for number, candidate in zip(REQUESTED_NUMBERS, selected, strict=True):
            package, _ = _package(candidate)
            schedule = build_schedule(
                focal_deck_id=candidate.name,
                entries=catalog.pool.schedule,
            )
            result = json.loads(
                (TEMP_ROOT / "results" / f"{number}.json").read_text(
                    encoding="utf-8"
                )
            )
            if not _valid_result(
                result,
                candidate=candidate,
                package=package,
                schedule=schedule,
            ):
                raise RuntimeError(
                    f"FATAL: cached deck {number} failed publish-time validation"
                )
            results.append(result)
        publish(catalog, candidates, selected, results)
        return 0
    device = torch.device("cuda:0")
    opponent = _opponent(device, tuple(int(value) for value in selected[0].deck))
    if args.smoke_games is not None:
        if not 1 <= args.smoke_games < EXPECTED_GAMES:
            parser.error("--smoke-games must be between 1 and 2047")
        result = _run_smoke(
            selected[0], candidates, catalog, opponent, device, args.smoke_games
        )
        print(json.dumps({
            "status": "SMOKE_PASS",
            "games": args.smoke_games,
            "candidate_audit": result[
                "candidate_deployment_identity_audit"
            ]["status"],
            "opponent_audit": result["opponent_policy_identity_audit"]["status"],
            "opponent_deck_groups": result["collector"][
                "opponent_deck_groups"
            ],
        }, sort_keys=True))
        return 0
    results = []
    for candidate in selected:
        result = _run_one(candidate, candidates, catalog, opponent, device)
        results.append(result)
        wins = sum(row["focal_outcome"] == "win" for row in result["games"])
        losses = sum(row["focal_outcome"] == "loss" for row in result["games"])
        draws = EXPECTED_GAMES - wins - losses
        print(
            f'U270_POLICY0809_COMPLETE deck={result["deck_number"]} '
            f'record={wins}-{losses}-{draws} '
            f'wall={result["collector"]["wall_seconds"]:.3f}s',
            flush=True,
        )
    publish(catalog, candidates, selected, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
