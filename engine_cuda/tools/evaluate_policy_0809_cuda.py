"""Protocol-compliant Frozen-0809 CUDA-2048 evaluation for selected decks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.frozen_0806_contract import (
    FROZEN_0806_CONTRACT_ID,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    evaluation_coin_winner,
    evaluation_game_seed,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.cards import card_image_url, load_card_catalog
from evaluation.combat_mat_contract import (
    validate_combat_mat_detail,
    validate_combat_mat_index,
)
from evaluation.reporting.html import render_html
from evaluation.reporting.models import ReportData
from engine_cuda.tools.evaluate_policy_0806_cuda import (
    build_cuda_game_records,
    resolve_candidate_deck_path,
)
export_candidate = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.export_candidate"
).export_candidate
audit_checkpoint = importlib.import_module(
    "train.0040_dragapult_0809_action_boundary_rl.policy_identity"
).audit_checkpoint


POLICY_ID = "Policy-0809"
POLICY_CHECKPOINT = (
    ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt"
)
POLICY_SHA256 = "926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f"
POOL_ROOT = ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
BENCHMARK = ROOT / "engine_cuda/tools/benchmark_0037_cuda_resident_refill.py"
TEMP_ROOT = ROOT / (
    ".tmp/evaluation/policy_0809_cuda_seeded2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1"
)
OUTPUT_ROOT = ROOT / (
    "docs/evaluation/combat_mat/policy_0809/"
    "0809_kaggle_top100_plus_v1_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1"
)
EXPECTED_GAMES = 2048
LANE_COUNT = 256
TOP_TEN = tuple(f"{number:03d}" for number in range(1, 11))
RESULT_SCHEMA = "cuda_semantic0031_resident_refill_strict_fp32_v3_agent_first_player"
DIAGNOSTIC_SCHEMA = "cuda_resident_terminal_diagnostics_v2_agent_first_player"
CANDIDATE_CONTRACT = "kaggle_fp16_storage_fp32_runtime_v1"
META_ARCHETYPE_PATH = ROOT / "evaluation/meta_archetypes_v1.json"
CARD_CATALOG_PATH = ROOT / "data/official/EN_Card_Data.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(
        path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _catalog() -> tuple[Any, tuple[Any, ...]]:
    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    candidates = tuple(sorted(
        catalog.candidates,
        key=lambda item: int(item.package_manifest["frozen_deck_number"]),
    ))
    if len(candidates) != 55 or sum(item.games for item in catalog.pool.schedule) != 256:
        raise RuntimeError("FATAL: Frozen catalog is not the immutable 55-deck/256-slot pool")
    return catalog, candidates


def _meta_contract() -> dict[str, Any]:
    payload = json.loads(META_ARCHETYPE_PATH.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if (
        payload.get("schema_version") != "evaluation_meta_archetypes_14_axis_v1"
        or not isinstance(classes, list)
        or [item.get("class_id") for item in classes] != list(range(14))
    ):
        raise RuntimeError("FATAL: shared 14-axis meta archetype contract is invalid")
    return payload


def _classify_meta(deck: Iterable[int]) -> dict[str, Any]:
    cards = set(int(value) for value in deck)
    contract = _meta_contract()
    for item in contract["classes"]:
        if cards.intersection(int(value) for value in item["trigger_card_ids"]):
            return dict(item)
    return dict(contract["fallback"])


def build_cuda_schedule(*, focal_deck_id: str, entries: Iterable[Any]) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for replica in range(FROZEN_0806_EVALUATION_UNITS):
        for entry in entries:
            opponent_id = str(entry.deck_id)
            focal_identity = f"{POLICY_ID}:{focal_deck_id}"
            opponent_identity = f"{POLICY_ID}:{opponent_id}"
            for slot in range(int(entry.games)):
                jobs.append({
                    "game_id": f"r{replica + 1:02d}-{opponent_id}-{slot + 1:03d}",
                    "opponent_id": opponent_id,
                    "focal_policy_id": POLICY_ID,
                    "opponent_policy_id": POLICY_ID,
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
                })
    payload = {
        "schema": "policy_0809_cuda_seeded2048_agent_choice_v3",
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
        "focal_policy_id": POLICY_ID,
        "opponent_policy_id": POLICY_ID,
        "focal_deck_id": focal_deck_id,
        "jobs": jobs,
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if len(jobs) != EXPECTED_GAMES or len({row["engine_seed"] for row in jobs}) != EXPECTED_GAMES:
        raise RuntimeError("FATAL: Frozen-0809 schedule is not 8 x 256 unique games")
    return payload


def preflight_identity() -> dict[str, Any]:
    if _sha256(POLICY_CHECKPOINT) != POLICY_SHA256:
        raise RuntimeError("FATAL: Policy-0809 source checkpoint SHA-256 mismatch")
    focal = audit_checkpoint(
        POLICY_ID, purpose="formal_frozen_cuda2048_focal_source_preflight"
    ).to_manifest()
    opponent = audit_checkpoint(
        POLICY_ID, purpose="formal_frozen_cuda2048_opponent_preflight"
    ).to_manifest()
    if any(
        audit.get("status") != "PASS"
        or audit.get("requested_policy_id") != POLICY_ID
        or audit.get("checkpoint_sha256") != POLICY_SHA256
        for audit in (focal, opponent)
    ):
        raise RuntimeError("FATAL: Policy-0809 source/opponent identity audit failed")
    return {
        "schema_version": "rl_formal_evaluation_policy_identity_audit_v1",
        "status": "PASS",
        "focal_source": focal,
        "opponent": opponent,
    }


def _package(candidate: Any) -> Path:
    number = str(candidate.package_manifest["frozen_deck_number"])
    package = TEMP_ROOT / "packages" / number
    deck_path = resolve_candidate_deck_path(candidate)
    if not package.exists():
        export_candidate(
            checkpoint=POLICY_CHECKPOINT,
            deck_path=deck_path,
            cg_source=POOL_ROOT / "policies/policy_0806/cg",
            output=package,
            deck_id=candidate.name,
            storage_dtype="fp16",
            runtime_dtype="fp32",
        )
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("checkpoint_sha256") != POLICY_SHA256
        or manifest.get("deck_id") != candidate.name
        or manifest.get("storage_dtype") != "fp16"
        or manifest.get("runtime_dtype") != "fp32"
        or manifest.get("portable_checkpoint_sha256") != _sha256(package / "strategy/model.bin")
    ):
        raise RuntimeError(f"FATAL: deck {number} portable candidate package mismatch")
    return package


def _valid_result(result: dict[str, Any], *, schedule_path: Path, package: Path, games: int) -> bool:
    diagnostics = result.get("per_game_diagnostics", {})
    candidate_audit = result.get("candidate_deployment_identity_audit", {})
    opponent_audit = result.get("opponent_policy_identity_audit", {})
    return bool(
        result.get("passed") is True
        and result.get("schema_version") == RESULT_SCHEMA
        and result.get("collector", {}).get("completed_games") == games
        and result.get("collector", {}).get("errors") == 0
        and len(result.get("determinism", {}).get("game_results", [])) == games
        and result.get("schedule", {}).get("sha256") == _sha256(schedule_path)
        and result.get("models", {}).get("actor_checkpoint_sha256")
        == _sha256(package / "strategy/model.bin")
        and result.get("models", {}).get("opponent_checkpoint_sha256") == POLICY_SHA256
        and candidate_audit.get("status") == "PASS"
        and candidate_audit.get("contract_id") == CANDIDATE_CONTRACT
        and candidate_audit.get("source_checkpoint_sha256") == POLICY_SHA256
        and candidate_audit.get("portable_checkpoint_sha256")
        == _sha256(package / "strategy/model.bin")
        and len(str(candidate_audit.get("effective_candidate_sha256", ""))) == 64
        and candidate_audit.get("storage_dtype") == "fp16"
        and candidate_audit.get("runtime_dtype") == "fp32"
        and opponent_audit.get("status") == "PASS"
        and opponent_audit.get("requested_policy_id") == POLICY_ID
        and opponent_audit.get("checkpoint_sha256") == POLICY_SHA256
        and len(str(opponent_audit.get("effective_policy_sha256", ""))) == 64
        and result.get("device", {}).get("float32_matmul_precision") == "highest"
        and result.get("device", {}).get("matmul_allow_tf32") is False
        and result.get("device", {}).get("cudnn_allow_tf32") is False
        and diagnostics.get("schema") == DIAGNOSTIC_SCHEMA
        and all(len(diagnostics.get(field, [])) == games for field in (
            "terminal_turns", "engine_selections", "terminal_prize_counts",
            "first_player_choosers", "first_player_actions", "actual_first_players",
        ))
    )


def _run_one(catalog: Any, candidate: Any, *, games: int = EXPECTED_GAMES) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    package = _package(candidate)
    schedule = build_cuda_schedule(
        focal_deck_id=candidate.name, entries=catalog.pool.schedule
    )
    schedule_path = TEMP_ROOT / "schedules" / f"{number}.json"
    result_path = TEMP_ROOT / "results" / f"{number}.json"
    _atomic_json(schedule_path, schedule)
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if not _valid_result(cached, schedule_path=schedule_path, package=package, games=games):
            result_path.unlink()
    if not result_path.is_file():
        log_path = TEMP_ROOT / "logs" / f"{number}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(BENCHMARK),
            "--actor-mode", "portable",
            "--actor-package", str(package),
            "--actor-checkpoint", str(POLICY_CHECKPOINT),
            "--opponent-model", str(POLICY_CHECKPOINT),
            "--opponent-policy-id", POLICY_ID,
            "--focal-deck", str(package / "deck.csv"),
            "--deck-root", str(POOL_ROOT / "decks"),
            "--schedule", str(schedule_path),
            "--game-limit", str(games),
            "--lane-count", str(min(LANE_COUNT, games)),
            "--check-interval", "8",
            "--ability-repeat-limit", "20",
            "--engine-turn-draw-limit", "100",
            "--output", str(result_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode != 0:
            result_path.unlink(missing_ok=True)
            raise RuntimeError(f"Frozen-0809 CUDA failed for deck {number}; see {log_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not _valid_result(result, schedule_path=schedule_path, package=package, games=games):
        raise RuntimeError(f"FATAL: deck {number} result failed the Frozen-0809 contract")
    return result


def _summary(
    catalog: Any, candidate: Any, result: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    schedule = json.loads((TEMP_ROOT / "schedules" / f"{number}.json").read_text())
    games = build_cuda_game_records(schedule, result)
    wins = sum(row["focal_outcome"] == "win" for row in games)
    losses = sum(row["focal_outcome"] == "loss" for row in games)
    draws = len(games) - wins - losses
    first = [row for row in games if row["focal_first"]]
    second = [row for row in games if not row["focal_first"]]
    source = next(item for item in catalog.pool.schedule if item.deck_id == candidate.name)
    meta = _classify_meta(candidate.deck)
    summary = {
        "deck_number": number,
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "representative_cards": [dict(item) for item in candidate.representative_cards],
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_games": int(source.games),
        "best_rank": int(source.best_rank),
        "observed_players": int(source.observed_players),
        "segment": str(source.segment),
        "meta_archetype_id": int(meta["class_id"]),
        "meta_archetype": str(meta["name"]),
        "meta_archetype_display_name": str(meta["display_name"]),
        "games": len(games), "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / len(games),
        "actual_first_games": len(first),
        "actual_first_wins": sum(row["focal_outcome"] == "win" for row in first),
        "actual_second_games": len(second),
        "actual_second_wins": sum(row["focal_outcome"] == "win" for row in second),
        "wall_seconds": result["collector"]["wall_seconds"],
        "schedule_sha256": schedule["schedule_sha256"],
        "candidate_deployment_identity_audit": result["candidate_deployment_identity_audit"],
        "opponent_policy_identity_audit": result["opponent_policy_identity_audit"],
        "engine": result["engine"], "device": result["device"],
        "report": f"reports/{number}.html", "games_file": f"games/{number}.json",
    }
    return summary, games


def _deck_card_records(candidate: Any) -> list[dict[str, Any]]:
    catalog = load_card_catalog(CARD_CATALOG_PATH)
    records = []
    for card_id, count in sorted(Counter(int(value) for value in candidate.deck).items()):
        metadata = catalog[card_id]
        stage = str(metadata["stage_or_type"])
        category = (
            "pokemon" if stage.endswith("Pokémon")
            else "energy" if "Energy" in stage
            else "trainer"
        )
        image_url = card_image_url(
            str(metadata["expansion"]), str(metadata["collection_number"])
        )
        if not image_url:
            raise RuntimeError(f"FATAL: card {card_id} has no Combat Mat art URL")
        records.append({
            "card_id": card_id,
            "name": str(metadata["name"]),
            "count": count,
            "category": category,
            "expansion": str(metadata["expansion"]),
            "collection_number": str(metadata["collection_number"]),
            "image_url": image_url,
        })
    if sum(item["count"] for item in records) != 60:
        raise RuntimeError("FATAL: Combat Mat detail deck is not exact 60 cards")
    return records


def _aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    wins = sum(row["focal_outcome"] == "win" for row in rows)
    losses = sum(row["focal_outcome"] == "loss" for row in rows)
    draws = len(rows) - wins - losses
    first = [row for row in rows if row.get("focal_first") is True]
    second = [row for row in rows if row.get("focal_first") is False]
    return {
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(rows) if rows else None,
        "first_games": len(first),
        "first_wins": sum(row["focal_outcome"] == "win" for row in first),
        "second_games": len(second),
        "second_wins": sum(row["focal_outcome"] == "win" for row in second),
    }


def _opponent_aggregates(
    games: Iterable[dict[str, Any]], candidates: Iterable[Any]
) -> dict[str, dict[str, Any]]:
    game_rows = list(games)
    result = {}
    for opponent in candidates:
        result[opponent.name] = _aggregate_records(
            row for row in game_rows if row.get("opponent_id") == opponent.name
        )
    if sum(item["games"] for item in result.values()) != len(game_rows):
        raise RuntimeError("FATAL: opponent aggregation lost CUDA games")
    return result


def _meta_representative_cards(archetype: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = load_card_catalog(CARD_CATALOG_PATH)
    cards = []
    for card_id in archetype.get("trigger_card_ids", ())[:2]:
        metadata = catalog[int(card_id)]
        image_url = card_image_url(
            str(metadata["expansion"]), str(metadata["collection_number"])
        )
        if image_url:
            cards.append({
                "card_id": int(card_id),
                "name": str(metadata["name"]),
                "image_url": image_url,
            })
    return cards


def _opponent_meta_aggregates(
    by_opponent: dict[str, dict[str, Any]], candidates: Iterable[Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = tuple(candidates)
    classes = {candidate.name: _classify_meta(candidate.deck) for candidate in candidates}
    contract = _meta_contract()
    rows = []
    for archetype in contract["classes"]:
        members = [
            candidate.name for candidate in candidates
            if classes[candidate.name]["class_id"] == archetype["class_id"]
        ]
        totals = {
            key: sum(int(by_opponent[name][key]) for name in members)
            for key in ("games", "wins", "losses", "draws")
        }
        rows.append({
            **dict(archetype),
            **totals,
            "win_rate": totals["wins"] / totals["games"] if totals["games"] else None,
            "deck_ids": members,
            "representative_cards": _meta_representative_cards(dict(archetype)),
        })
    other_members = [
        candidate.name for candidate in candidates
        if classes[candidate.name]["class_id"] == 14
    ]
    other_totals = {
        key: sum(int(by_opponent[name][key]) for name in other_members)
        for key in ("games", "wins", "losses", "draws")
    }
    other = {
        **dict(contract["fallback"]),
        **other_totals,
        "win_rate": (
            other_totals["wins"] / other_totals["games"]
            if other_totals["games"] else None
        ),
        "deck_ids": other_members,
        "representative_cards": [],
    }
    if sum(item["games"] for item in rows) + other["games"] != sum(
        item["games"] for item in by_opponent.values()
    ):
        raise RuntimeError("FATAL: opponent meta aggregation lost CUDA games")
    return rows, other


def _detail_payload(
    candidate: Any, games: list[dict[str, Any]], candidates: tuple[Any, ...]
) -> dict[str, Any]:
    totals = _aggregate_records(games)
    by_opponent = _opponent_aggregates(games, candidates)
    by_meta, other = _opponent_meta_aggregates(by_opponent, candidates)
    detail = {
        "deck_total": len(candidate.deck),
        "deck_cards": _deck_card_records(candidate),
        "by_opponent": by_opponent,
        "by_meta_archetype": by_meta,
        "meta_archetype_other": other,
        **{key: totals[key] for key in ("games", "wins", "losses", "draws")},
    }
    validate_combat_mat_detail(
        detail, expected_opponent_ids={item.name for item in candidates}
    )
    return detail


def _render_deck(
    summary: dict[str, Any],
    games: list[dict[str, Any]],
    *,
    catalog: Any,
    candidates: tuple[Any, ...],
    detail: dict[str, Any],
    completed_numbers: set[str],
) -> str:
    del catalog
    candidate = next(
        item for item in candidates
        if str(item.package_manifest["frozen_deck_number"]) == summary["deck_number"]
    )
    candidate_manifest = dict(candidate.package_manifest)
    candidate_manifest.update({
        "frozen_policy_label": "0809",
        "frozen_report_href": f'{summary["deck_number"]}.html',
    })
    opponents = []
    for opponent in candidates:
        package_manifest = dict(opponent.package_manifest)
        number = str(package_manifest["frozen_deck_number"])
        package_manifest["frozen_policy_label"] = "0809"
        if number in completed_numbers:
            package_manifest["frozen_report_href"] = f"{number}.html"
        else:
            package_manifest.pop("frozen_report_href", None)
        opponents.append({
            "name": opponent.name,
            "display_name": opponent.display_name,
            "representative_cards": list(opponent.representative_cards),
            "package_manifest": package_manifest,
        })
    manifest = {
        "run_id": (
            f'policy-0809-frozen-0809-cuda2048-{summary["deck_number"]}-'
            f'{str(summary.get("schedule_sha256", "contract"))[:12]}'
        ),
        "candidate": {
            "name": candidate.name,
            "display_name": candidate.display_name,
            "representative_cards": list(candidate.representative_cards),
            "deck_cards": detail["deck_cards"],
            "deck_total": detail["deck_total"],
            "package_manifest": candidate_manifest,
        },
        "opponents": opponents,
        "opponent_pool": {
            "pool_id": "0809_kaggle_top100_plus_v1",
            "policy_label": "0809",
        },
        "candidate_deployment_identity_audit": summary.get(
            "candidate_deployment_identity_audit"
        ),
        "opponent_policy_identity_audit": summary.get(
            "opponent_policy_identity_audit"
        ),
    }
    report_summary = {
        "total_games": detail["games"],
        "wins": detail["wins"],
        "losses": detail["losses"],
        "draws": detail["draws"],
        "completion_rate": 1.0,
        "win_rate": detail["wins"] / detail["games"],
        "errors": 0,
        "by_opponent": detail["by_opponent"],
        "by_meta_archetype": detail["by_meta_archetype"],
        "meta_archetype_other": detail["meta_archetype_other"],
    }
    return render_html(ReportData(
        manifest=manifest,
        summary=report_summary,
        games=tuple(games),
        metrics={},
        cases=(),
        metric_profile={
            "id": "frozen_cuda_2048",
            "revision": 1,
            "metric_ids": ("outcome", "by_opponent", "by_meta_archetype"),
        },
    ))


def _meta_aggregates(
    candidates: Iterable[Any], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completed = {row["deck_number"]: row for row in rows}
    candidate_classes = {
        str(item.package_manifest["frozen_deck_number"]): _classify_meta(item.deck)
        for item in candidates
    }
    contract = _meta_contract()
    aggregates: list[dict[str, Any]] = []
    for archetype in contract["classes"]:
        class_id = int(archetype["class_id"])
        all_numbers = sorted(
            number for number, item in candidate_classes.items()
            if int(item["class_id"]) == class_id
        )
        tested = [completed[number] for number in all_numbers if number in completed]
        wins = sum(row["wins"] for row in tested)
        losses = sum(row["losses"] for row in tested)
        draws = sum(row["draws"] for row in tested)
        games = wins + losses + draws
        aggregates.append({
            **dict(archetype),
            "deck_numbers": all_numbers,
            "total_decks": len(all_numbers),
            "tested_decks": len(tested),
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / games if games else None,
        })
    other_numbers = sorted(
        number for number, item in candidate_classes.items()
        if int(item["class_id"]) == 14
    )
    tested_other = [completed[number] for number in other_numbers if number in completed]
    other_wins = sum(row["wins"] for row in tested_other)
    other_losses = sum(row["losses"] for row in tested_other)
    other_draws = sum(row["draws"] for row in tested_other)
    other_games = other_wins + other_losses + other_draws
    other = {
        **dict(contract["fallback"]),
        "deck_numbers": other_numbers,
        "total_decks": len(other_numbers),
        "tested_decks": len(tested_other),
        "games": other_games,
        "wins": other_wins,
        "losses": other_losses,
        "draws": other_draws,
        "win_rate": other_wins / other_games if other_games else None,
    }
    if sum(item["games"] for item in aggregates) + other_games != sum(
        row["games"] for row in rows
    ):
        raise RuntimeError("FATAL: 14-axis meta aggregation lost completed games")
    return aggregates, other


def _publish(
    identity: dict[str, Any],
    rows: list[dict[str, Any]],
    catalog: Any,
    candidates: tuple[Any, ...],
) -> None:
    total_games = sum(row["games"] for row in rows)
    meta_rows, other_meta = _meta_aggregates(candidates, rows)
    completed = {row["deck_number"]: row for row in rows}
    catalog_rows = [
        {
            "deck_number": str(candidate.package_manifest["frozen_deck_number"]),
            "deck_id": candidate.name,
            "display_name": candidate.display_name,
            "representative_cards": [dict(card) for card in candidate.representative_cards],
            "status": (
                "tested"
                if str(candidate.package_manifest["frozen_deck_number"]) in completed
                else "pending"
            ),
        }
        for candidate in candidates
    ]
    manifest = {
        "schema": "policy_0809_cuda_seeded2048_index_v1",
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "candidate_contract_id": CANDIDATE_CONTRACT,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
        "focal_policy_id": POLICY_ID,
        "opponent_policy_id": POLICY_ID,
        "policy_identity_audit": identity,
        "published_decks": len(rows),
        "published_games": total_games,
        "complete_requested_scope": len(rows) == 10 and total_games == 20_480,
        "catalog_decks": 55,
        "catalog": catalog_rows,
        "meta_archetype_contract": str(META_ARCHETYPE_PATH.relative_to(ROOT)),
        "meta_archetype_aggregates": meta_rows,
        "meta_archetype_other": other_meta,
        "reports": rows,
    }
    validate_combat_mat_index(manifest)
    _atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    source = {item.deck_id: item for item in catalog.pool.schedule}
    table_rows = []
    for candidate in candidates:
        number = str(candidate.package_manifest["frozen_deck_number"])
        row = completed.get(number)
        schedule_entry = source[candidate.name]
        images = "".join(
            f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
            f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
            for card in candidate.representative_cards
        )
        if row is None:
            title = html.escape(candidate.display_name)
            status = '<span class="pending">未测试</span>'
            record = rate = first = second = wall = "-"
            data_rate = "-1"
        else:
            title = (
                f'<a href="{html.escape(row["report"], quote=True)}">'
                f'{html.escape(candidate.display_name)}</a>'
            )
            status = '<span class="done">已完成</span>'
            record = f'{row["wins"]}-{row["losses"]}-{row["draws"]}'
            rate = f'{row["win_rate"]:.2%}'
            first_losses = row["actual_first_games"] - row["actual_first_wins"]
            second_losses = row["actual_second_games"] - row["actual_second_wins"]
            first = (
                f'{row["actual_first_wins"] / row["actual_first_games"]:.2%}'
                f'<small>{row["actual_first_wins"]}-{first_losses}</small>'
            )
            second = (
                f'{row["actual_second_wins"] / row["actual_second_games"]:.2%}'
                f'<small>{row["actual_second_wins"]}-{second_losses}</small>'
            )
            wall = f'{row["wall_seconds"]:.1f}s'
            data_rate = f'{row["win_rate"]:.12f}'
        table_rows.append(
            f'<tr data-rate="{data_rate}" data-rank="{int(schedule_entry.best_rank)}">'
            f'<td class="number">{number}</td><td><div class="deck">'
            f'<span class="art">{images}</span><span>{title}<small>{status} · '
            f'频率 {int(schedule_entry.games)}/256</small></span></div></td>'
            f'<td>{record}</td><td class="strong">{rate}</td><td>{first}</td>'
            f'<td>{second}</td><td>{int(schedule_entry.best_rank)}</td>'
            f'<td>{int(schedule_entry.observed_players)}</td>'
            f'<td>{html.escape(str(schedule_entry.segment))}</td><td>{wall}</td></tr>'
        )
    meta_table = []
    for item in meta_rows:
        if item["games"]:
            record = f'{item["wins"]}-{item["losses"]}-{item["draws"]}'
            rate = f'{item["win_rate"]:.2%}'
        else:
            record = rate = "-"
        meta_table.append(
            f'<tr><td class="number">{int(item["class_id"]) + 1:02d}</td>'
            f'<td>{html.escape(str(item["display_name"]))}'
            f'<small>{html.escape(str(item["name"]))}</small></td>'
            f'<td>{item["tested_decks"]}/{item["total_decks"]}</td>'
            f'<td>{item["games"]:,}</td><td>{record}</td>'
            f'<td class="strong">{rate}</td>'
            f'<td><code>{html.escape(", ".join(item["deck_numbers"]))}</code></td></tr>'
        )
    other_note = (
        '第 15 行 Other 是未命中 14 类 trigger-card taxonomy 的显式聚合桶。'
    )
    other_record = (
        f'{other_meta["wins"]}-{other_meta["losses"]}-{other_meta["draws"]}'
        if other_meta["games"] else "-"
    )
    other_rate = f'{other_meta["win_rate"]:.2%}' if other_meta["games"] else "-"
    meta_table.append(
        '<tr><td class="number">15</td><td>Other<small>other</small></td>'
        f'<td>{other_meta["tested_decks"]}/{other_meta["total_decks"]}</td>'
        f'<td>{other_meta["games"]:,}</td><td>{other_record}</td>'
        f'<td class="strong">{other_rate}</td>'
        f'<td><code>{html.escape(", ".join(other_meta["deck_numbers"]))}</code></td></tr>'
    )
    embedded = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Policy-0809 Report · Frozen-0809 卡组强度</title><style>
:root{{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d;--red:#a54343}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}header{{padding:28px max(20px,calc((100vw - 1500px)/2));background:#18382d;color:#fff}}h1{{margin:0;font-size:30px;letter-spacing:0}}header p{{max-width:1050px;margin:7px 0 0;color:#cfe1da}}main{{max-width:1500px;margin:auto;padding:20px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--paper)}}.stat{{padding:15px 18px;border-right:1px solid var(--line)}}.stat:last-child{{border:0}}.stat b{{display:block;font-size:23px}}.stat span,small{{display:block;color:var(--muted)}}.tools{{display:flex;gap:10px;margin:18px 0}}input{{width:min(420px,100%);padding:9px 11px;border:1px solid #b9c9c1;border-radius:4px;background:#fff}}.table{{overflow:auto;border:1px solid var(--line);background:#fff}}table{{width:100%;min-width:1180px;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#e9f0ec;color:#486158;font-size:12px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f8fbf9}}a{{color:var(--green);font-weight:700;text-decoration:none}}.deck{{display:flex;align-items:center;gap:10px}}.art{{display:flex;width:68px}}.art img{{width:38px;height:53px;margin-right:-8px;border:1px solid #c9d5cf;border-radius:3px;object-fit:cover;background:#e4ebe7}}.strong{{font-size:16px;font-weight:750}}code{{font-size:11px}}.number{{font-size:16px;font-weight:850;color:var(--green);font-variant-numeric:tabular-nums}}.done{{color:var(--green)}}.pending{{color:#8a6b28}}section{{margin-top:20px}}section h2{{margin:0 0 6px}}.section-note{{margin:0 0 12px;color:var(--muted)}}.contract{{margin-top:16px;padding:14px 16px;border-left:4px solid var(--green);background:#fff;color:var(--muted)}}@media(max-width:760px){{.stats{{grid-template-columns:1fr 1fr}}.stat:nth-child(2){{border-right:0}}header{{padding:22px 16px}}main{{padding:14px}}}}
</style></head><body><header><h1>Policy-0809 Report</h1><p>001–055 全量 exact deck 目录；本次正式范围为 001–010。已测试卡组使用 Policy-0809 的 Kaggle FP16-storage/FP32-runtime candidate，对手为独立完整 Frozen-0809；8 个 256 局 replica 均由 seeded toss winner Agent 自主选择先后手。</p></header><main><div class="stats"><div class="stat"><b>{len(rows)}/55</b><span>完成卡组</span></div><div class="stat"><b>{total_games:,}</b><span>正式对局</span></div><div class="stat"><b>{sum(row['wins'] for row in rows):,}</b><span>Policy-0809 胜局</span></div><div class="stat"><b>{sum(row['wall_seconds'] for row in rows)/3600:.2f}h</b><span>累计 wall time</span></div></div><div class="tools"><input id="search" type="search" placeholder="筛选编号或牌型"></div><div class="table" id="deck-catalog"><table id="results"><thead><tr><th>编号</th><th>卡组 / 2048 局报告</th><th>W-L-D</th><th>胜率</th><th>实际先攻</th><th>实际后攻</th><th>最佳名次</th><th>观察人数</th><th>来源</th><th>耗时</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div><section id="meta-archetype-summary"><h2>14 种 Meta Archetype 聚合胜率</h2><p class="section-note">按 focal exact deck 的 priority-ordered trigger-card taxonomy 分类；胜率按已完成对局加权，不对未测试构筑填 0。{html.escape(other_note)}</p><div class="table"><table><thead><tr><th>Meta</th><th>Archetype</th><th>已测/目录构筑</th><th>对局</th><th>W-L-D</th><th>加权胜率</th><th>构筑编号</th></tr></thead><tbody>{''.join(meta_table)}</tbody></table></div></section><div class="contract">Contract <code>{FROZEN_0806_CONTRACT_ID}</code> · Candidate <code>{CANDIDATE_CONTRACT}</code> · Policy-0809 <code>{POLICY_SHA256}</code> · 14-axis taxonomy <code>{html.escape(str(META_ARCHETYPE_PATH.relative_to(ROOT)))}</code></div><script>const q=document.querySelector('#search'),body=document.querySelector('#results tbody');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();for(const r of body.rows)r.hidden=!r.innerText.toLowerCase().includes(s)}});</script><script id="report-data" type="application/json">{embedded}</script></main></body></html>"""
    _atomic_text(OUTPUT_ROOT / "index.html", page)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-number", action="append", default=[])
    parser.add_argument("--top-ten", action="store_true")
    parser.add_argument("--smoke-games", type=int)
    args = parser.parse_args(argv)
    requested_numbers = TOP_TEN if args.top_ten else tuple(args.deck_number)
    if not requested_numbers or any(number not in TOP_TEN for number in requested_numbers):
        parser.error("choose --top-ten or deck numbers 001 through 010")
    identity = preflight_identity()
    catalog, candidates = _catalog()
    selected = [
        item for item in candidates
        if item.package_manifest["frozen_deck_number"] in set(requested_numbers)
    ]
    if len(selected) != len(set(requested_numbers)):
        raise RuntimeError("requested Frozen deck number did not resolve uniquely")
    if args.smoke_games is not None:
        if len(selected) != 1 or not 1 <= args.smoke_games < EXPECTED_GAMES:
            parser.error("--smoke-games requires exactly one deck and a value below 2048")
        result = _run_one(catalog, selected[0], games=args.smoke_games)
        print(json.dumps({
            "status": "SMOKE_PASS",
            "games": args.smoke_games,
            "candidate_audit": result["candidate_deployment_identity_audit"]["status"],
            "opponent_audit": result["opponent_policy_identity_audit"]["status"],
        }, sort_keys=True))
        return 0
    rows: list[dict[str, Any]] = []
    for candidate in selected:
        result = _run_one(catalog, candidate)
        summary, games = _summary(catalog, candidate, result)
        _atomic_json(OUTPUT_ROOT / summary["games_file"], {
            "schema": "policy_0809_cuda_seeded2048_games_v1",
            "summary": summary,
            "games": games,
        })
        detail = _detail_payload(candidate, games, candidates)
        _atomic_text(
            OUTPUT_ROOT / summary["report"],
            _render_deck(
                summary,
                games,
                catalog=catalog,
                candidates=candidates,
                detail=detail,
                completed_numbers=set(requested_numbers),
            ),
        )
        rows.append(summary)
        _publish(identity, rows, catalog, candidates)
        print(
            f"CUDA_FROZEN_0809_COMPLETE deck={summary['deck_number']} "
            f"record={summary['wins']}-{summary['losses']}-{summary['draws']} "
            f"wall={summary['wall_seconds']:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
