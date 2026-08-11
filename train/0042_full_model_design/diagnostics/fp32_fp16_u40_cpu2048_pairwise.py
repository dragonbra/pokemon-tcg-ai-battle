"""Paired raw-FP32 versus FP16-round-trip U40 CPU-2048 diagnostic.

This runner deliberately reuses the completed FP16 diagnostic schedule identity
so every official game has the same game/engine/search/policy seed.  The raw-FP32
result is diagnostic-only and is never valid Kaggle/Frozen/Promote evidence.

For action sensitivity, the raw-FP32 model drives the game while the FP16-stored,
FP32-runtime model receives the exact same focal observation and legal options as
a shadow.  This measures same-state greedy action flips without confusing later
trajectory divergence with additional quantization flips.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..action_boundary.dragapult import (
    PHANTOM_DIVE_ATTACK_ID,
    PHANTOM_DIVE_MAX_TARGETS,
    StableTargetIdentity,
)
from ..action_boundary.macro_planner import MacroPlanner
from ..action_boundary.public_card_features import with_public_prize
from ..candidate_deployment import (
    CONTRACT_ID as FP16_CONTRACT_ID,
    audit_candidate_deployment,
)
from ..evaluation.frozen_jobs import build_frozen_jobs
from ..policy.action_distribution import sample_actions_with_encoding
from ..policy.batching import move_batch
from ..rollout.collector import FullSemanticRolloutCollector, phantom_macro_eligible
from ..rollout.deck_routing import exact_deck_sha256
from ..semantic_policy.features.collate import collate_canonical_records
from ..training import run_full_semantic as runner
from .fp32_cpu2048_policy0809 import (
    CONTRACT_ID as RAW_FP32_CONTRACT_ID,
    _assert_cpu_chunk_health,
    _load_package,
    _sha256,
    export_fp32_diagnostic,
)


SCHEMA = "0042_u40_fp32_fp16_cpu2048_pairwise_diagnostic_v1"
EXPECTED_UPDATE = 40
EXPECTED_GAMES = 2048
NUMERIC_ACTION_FIELDS = (
    "strategic_decisions_examined",
    "engine_root_action_flips",
    "decoder_action_flips",
    "effective_action_flips",
    "first_token_flips",
    "stop_flips",
    "root_token_slots_examined",
    "root_token_slots_flipped",
    "macro_presence_flips",
    "allocation_decisions_examined",
    "allocation_flips",
    "primitive_slots_examined",
    "primitive_slots_flipped",
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _index_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    indexed: dict[str, int] = {}
    for row in rows:
        game_id = row.get("game_id")
        outcome = row.get("outcome")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("outcome row has no game_id")
        if game_id in indexed:
            raise ValueError(f"duplicate game_id in outcome rows: {game_id}")
        if outcome not in (-1, 0, 1):
            raise ValueError(f"invalid outcome for {game_id}: {outcome!r}")
        indexed[game_id] = int(outcome)
    return indexed


def compare_outcomes(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strictly pair outcomes by immutable game ID."""

    reference = _index_outcomes(reference_rows)
    candidate = _index_outcomes(candidate_rows)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())[:8]
        extra = sorted(candidate.keys() - reference.keys())[:8]
        raise ValueError(
            f"reference/candidate game-ID sets differ: missing={missing} extra={extra}"
        )
    labels = {-1: "L", 0: "D", 1: "W"}
    matrix = {source: {target: 0 for target in ("W", "L", "D")}
              for source in ("W", "L", "D")}
    changed: list[str] = []
    for game_id in sorted(reference):
        source, target = labels[reference[game_id]], labels[candidate[game_id]]
        matrix[source][target] += 1
        if source != target:
            changed.append(game_id)
    games = len(reference)
    result: dict[str, Any] = {
        "games": games,
        "unchanged_games": games - len(changed),
        "changed_games": len(changed),
        "changed_fraction": len(changed) / games if games else 0.0,
        "changed_game_ids": changed,
        "transition_matrix": matrix,
        "reference_wins": sum(value == 1 for value in reference.values()),
        "candidate_wins": sum(value == 1 for value in candidate.values()),
    }
    for source_name, source_label in (("win", "W"), ("loss", "L"), ("draw", "D")):
        for target_name, target_label in (("win", "W"), ("loss", "L"), ("draw", "D")):
            if source_label != target_label:
                result[f"{source_name}_to_{target_name}"] = matrix[source_label][target_label]
    result["net_win_delta"] = result["candidate_wins"] - result["reference_wins"]
    return result


def _macro_primitive(macro: Mapping[str, Any] | None) -> tuple[int, ...]:
    if macro is None:
        return ()
    targets = tuple(int(value) for value in macro["targets"])
    counters = tuple(int(value) for value in macro["counters"])
    if len(targets) != len(counters) or any(value < 0 for value in counters):
        raise ValueError("malformed macro signature")
    return tuple(
        serial for serial, count in zip(targets, counters, strict=True)
        for _ in range(count)
    )


def _slot_difference(left: Sequence[Any], right: Sequence[Any]) -> tuple[int, int]:
    missing = object()
    width = max(len(left), len(right))
    flips = sum(
        (left[index] if index < len(left) else missing)
        != (right[index] if index < len(right) else missing)
        for index in range(width)
    )
    return width, flips


def action_difference(
    *, fp32_indices: tuple[int, ...], fp32_stopped: bool,
    fp16_indices: tuple[int, ...], fp16_stopped: bool,
    fp32_macro: Mapping[str, Any] | None,
    fp16_macro: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare two greedy actions generated from the exact same legal state."""

    root_slots, root_slot_flips = _slot_difference(fp32_indices, fp16_indices)
    engine_root_flip = fp32_indices != fp16_indices
    decoder_flip = engine_root_flip or fp32_stopped != fp16_stopped
    macro_presence_flip = (fp32_macro is None) != (fp16_macro is None)
    allocation_comparable = (
        not engine_root_flip and fp32_macro is not None and fp16_macro is not None
    )
    allocation_flip = bool(
        allocation_comparable
        and (
            tuple(fp32_macro["targets"]) != tuple(fp16_macro["targets"])
            or tuple(fp32_macro["counters"]) != tuple(fp16_macro["counters"])
        )
    )
    primitive_examined = primitive_flipped = 0
    if allocation_comparable:
        primitive_examined, primitive_flipped = _slot_difference(
            _macro_primitive(fp32_macro), _macro_primitive(fp16_macro)
        )
    return {
        "engine_root_action_flip": engine_root_flip,
        "decoder_action_flip": decoder_flip,
        "effective_action_flip": engine_root_flip or macro_presence_flip or allocation_flip,
        "first_token_flip": (
            bool(fp32_indices or fp16_indices)
            and (fp32_indices[0] if fp32_indices else None)
            != (fp16_indices[0] if fp16_indices else None)
        ),
        "stop_flip": fp32_stopped != fp16_stopped,
        "root_token_slots_examined": root_slots,
        "root_token_slots_flipped": root_slot_flips,
        "macro_presence_flip": macro_presence_flip,
        "allocation_comparable": allocation_comparable,
        "allocation_flip": allocation_flip,
        "primitive_slots_examined": primitive_examined,
        "primitive_slots_flipped": primitive_flipped,
    }


def empty_action_summary() -> dict[str, Any]:
    return {
        **{name: 0 for name in NUMERIC_ACTION_FIELDS},
        "games_examined": [],
        "affected_game_ids": [],
        "decoder_only_affected_game_ids": [],
        "flip_events": [],
    }


def merge_action_summaries(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    merged = empty_action_summary()
    games: set[str] = set()
    aggregated_game_count = 0
    affected: set[str] = set()
    decoder_affected: set[str] = set()
    events: list[dict[str, Any]] = []
    for summary in summaries:
        for name in NUMERIC_ACTION_FIELDS:
            merged[name] += int(summary.get(name, 0))
        raw_games = summary.get("games_examined", [])
        if isinstance(raw_games, int):
            # Persisted chunk summaries contain only their already-aggregated
            # cardinality. Keep it numeric so synthetic IDs cannot collide
            # across chunks and under-count the final report.
            aggregated_game_count += raw_games
        else:
            games.update(str(value) for value in raw_games)
        affected.update(str(value) for value in summary.get("affected_game_ids", []))
        decoder_affected.update(
            str(value) for value in summary.get("decoder_only_affected_game_ids", [])
        )
        events.extend(dict(value) for value in summary.get("flip_events", []))
    merged["games_examined_ids"] = sorted(games)
    merged["games_examined"] = len(games) + aggregated_game_count
    merged["affected_game_ids"] = sorted(affected)
    merged["affected_games"] = len(affected)
    merged["decoder_only_affected_game_ids"] = sorted(decoder_affected)
    merged["decoder_only_affected_games"] = len(decoder_affected)
    merged["flip_events"] = events
    decisions = merged["strategic_decisions_examined"]
    merged["engine_root_action_flip_fraction"] = (
        merged["engine_root_action_flips"] / decisions if decisions else 0.0
    )
    merged["effective_action_flip_fraction"] = (
        merged["effective_action_flips"] / decisions if decisions else 0.0
    )
    allocation_count = merged["allocation_decisions_examined"]
    merged["allocation_flip_fraction"] = (
        merged["allocation_flips"] / allocation_count if allocation_count else 0.0
    )
    return merged


def _macro_signature_from_wire(wire: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if wire is None:
        return None
    return {
        "targets": [int(item["serial"]) for item in wire["targets"]],
        "counters": [int(value) for value in wire["counters"]],
    }


class QuantizationPairCollector(FullSemanticRolloutCollector):
    """FP32-primary collector with an exact-same-state FP16 shadow forward."""

    def __init__(self, *args, shadow_model, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shadow_model = shadow_model.eval().requires_grad_(False)
        self.shadow_macro_planner = MacroPlanner(self.shadow_model.allocation_head)
        self._pair_summary = empty_action_summary()
        self._pair_games: set[str] = set()
        self._pair_affected: set[str] = set()
        self._pair_decoder_affected: set[str] = set()

    def _shadow_macro(
        self, *, item, message: Mapping[str, Any], action, validated, state,
        option_embeddings: torch.Tensor, batch_index: int,
    ) -> dict[str, Any] | None:
        root_options = (message.get("select") or {}).get("option") or []
        phantom_root = next((
            selected for selected in action.indices
            if 0 <= selected < len(root_options)
            and isinstance(root_options[selected], Mapping)
            and root_options[selected].get("attackId") == PHANTOM_DIVE_ATTACK_ID
        ), None)
        raw_bench = message.get("visible_opponent_bench") or []
        if not phantom_macro_eligible(
            phantom_root=phantom_root,
            target_count=len(raw_bench),
            action_boundary_mode=item.job.action_boundary_mode,
            chance_before_allocation=bool(message.get("chance_before_phantom_allocation")),
        ):
            return None
        if len(raw_bench) > PHANTOM_DIVE_MAX_TARGETS:
            return None
        identities = sorted(
            StableTargetIdentity(
                1 - int(message["actor"]), int(raw["serial"]), int(raw["id"]), slot
            )
            for slot, raw in enumerate(raw_bench)
        )
        raw_by_serial = {
            int(raw["serial"]): {**raw, "benchSlot": slot}
            for slot, raw in enumerate(raw_bench)
        }
        visible = [
            with_public_prize(raw_by_serial[target.serial], self.prizes)
            for target in identities
        ]
        target_rows = []
        for target in identities:
            card_mask = (
                validated.card_mask[batch_index]
                & validated.card_cat[batch_index, :, 2].eq(2)
                & validated.card_cat[batch_index, :, 3].eq(6)
                & validated.card_cat[batch_index, :, 4].eq(target.initial_bench_slot + 1)
            )
            positions = card_mask.nonzero(as_tuple=False).flatten()
            if positions.numel() != 1:
                raise RuntimeError(
                    "shadow Phantom target does not map to exactly one state token"
                )
            target_rows.append(state.cards[batch_index, int(positions[0])])
        planned = self.shadow_macro_planner.plan(
            state_summary=state.summary[batch_index],
            root_option=option_embeddings[batch_index, phantom_root],
            target_embeddings=torch.stack(target_rows),
            target_identities=identities,
            visible_targets=visible,
            greedy=True,
        )
        return {
            "targets": [target.serial for target in planned.allocation.target_ids],
            "counters": list(planned.allocation.counters),
        }

    def _infer_focal(self, requests):
        cpu_features = collate_canonical_records(
            [message["record"] for _, message in requests]
        )
        batch = move_batch(cpu_features, self.device)
        (
            shadow_actions, shadow_validated, shadow_state, shadow_options,
            _shadow_values, _shadow_auxiliary, _shadow_timing,
        ) = sample_actions_with_encoding(
            self.shadow_model, batch, greedy=True, generators=None
        )
        shadow_macros = [
            self._shadow_macro(
                item=item, message=message, action=shadow_actions[index],
                validated=shadow_validated, state=shadow_state,
                option_embeddings=shadow_options, batch_index=index,
            )
            for index, (item, message) in enumerate(requests)
        ]
        primary = super()._infer_focal(requests)
        for index, ((item, message), (fp32_action, fp32_wire, _timing)) in enumerate(
            zip(requests, primary, strict=True)
        ):
            fp16_action = shadow_actions[index]
            fp32_macro = _macro_signature_from_wire(fp32_wire)
            fp16_macro = shadow_macros[index]
            difference = action_difference(
                fp32_indices=tuple(fp32_action.indices),
                fp32_stopped=bool(fp32_action.stopped),
                fp16_indices=tuple(fp16_action.indices),
                fp16_stopped=bool(fp16_action.stopped),
                fp32_macro=fp32_macro,
                fp16_macro=fp16_macro,
            )
            game_id = item.job.game_id
            self._pair_games.add(game_id)
            self._pair_summary["strategic_decisions_examined"] += 1
            self._pair_summary["engine_root_action_flips"] += int(
                difference["engine_root_action_flip"]
            )
            self._pair_summary["decoder_action_flips"] += int(
                difference["decoder_action_flip"]
            )
            self._pair_summary["effective_action_flips"] += int(
                difference["effective_action_flip"]
            )
            self._pair_summary["first_token_flips"] += int(
                difference["first_token_flip"]
            )
            self._pair_summary["stop_flips"] += int(difference["stop_flip"])
            self._pair_summary["root_token_slots_examined"] += int(
                difference["root_token_slots_examined"]
            )
            self._pair_summary["root_token_slots_flipped"] += int(
                difference["root_token_slots_flipped"]
            )
            self._pair_summary["macro_presence_flips"] += int(
                difference["macro_presence_flip"]
            )
            self._pair_summary["allocation_decisions_examined"] += int(
                difference["allocation_comparable"]
            )
            self._pair_summary["allocation_flips"] += int(
                difference["allocation_flip"]
            )
            self._pair_summary["primitive_slots_examined"] += int(
                difference["primitive_slots_examined"]
            )
            self._pair_summary["primitive_slots_flipped"] += int(
                difference["primitive_slots_flipped"]
            )
            if difference["effective_action_flip"]:
                self._pair_affected.add(game_id)
            if difference["decoder_action_flip"] and not difference["effective_action_flip"]:
                self._pair_decoder_affected.add(game_id)
            if difference["decoder_action_flip"] or difference["effective_action_flip"]:
                self._pair_summary["flip_events"].append({
                    "game_id": game_id,
                    "selection_index": int(message.get("selection_index", -1)),
                    "turn": int(message.get("turn", -1)),
                    "fp32_indices": list(fp32_action.indices),
                    "fp32_stopped": bool(fp32_action.stopped),
                    "fp16_indices": list(fp16_action.indices),
                    "fp16_stopped": bool(fp16_action.stopped),
                    "fp32_macro": fp32_macro,
                    "fp16_macro": fp16_macro,
                    **difference,
                })
        return primary

    def action_summary(self) -> dict[str, Any]:
        return merge_action_summaries([{
            **self._pair_summary,
            "games_examined": sorted(self._pair_games),
            "affected_game_ids": sorted(self._pair_affected),
            "decoder_only_affected_game_ids": sorted(self._pair_decoder_affected),
        }])


def _no_parameter_alias(primary: torch.nn.Module, shadow: torch.nn.Module) -> None:
    primary_objects = {id(value) for value in primary.parameters()}
    shadow_objects = {id(value) for value in shadow.parameters()}
    if primary_objects & shadow_objects:
        raise RuntimeError("FP32 primary and FP16 shadow share Parameter objects")
    primary_storage = {
        value.untyped_storage().data_ptr() for value in primary.parameters()
    }
    shadow_storage = {
        value.untyped_storage().data_ptr() for value in shadow.parameters()
    }
    if primary_storage & shadow_storage:
        raise RuntimeError("FP32 primary and FP16 shadow share tensor storage")


def _episode_rows(episodes: Sequence[Any], opponent_hash: str) -> list[dict[str, Any]]:
    return [{
        "game_id": episode.job.game_id,
        "replica": int(episode.job.game_id[1:3]) - 1,
        "opponent_id": episode.job.opponent_id,
        "opponent_exact_deck_sha256": exact_deck_sha256(episode.job.opponent_deck),
        "opponent_effective_policy_sha256": opponent_hash,
        "engine_seed": episode.job.seed,
        "search_seed": episode.job.search_seed,
        "policy_seed": episode.job.policy_seed,
        "focal_won_toss": episode.job.focal_won_toss,
        "focal_first": runner._episode_focal_first(episode),
        "outcome": 1 if episode.reward == 1.0 else -1 if episode.reward == -1.0 else 0,
        "turns": episode.turns,
        "valid": episode.valid,
        "error": episode.error,
        "fallback_reason": episode.diagnostics.get("macro_fallback_reason"),
    } for episode in episodes]


def _resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (root / path).resolve()


def run(
    *, checkpoint: Path, reference_report: Path, output_root: Path,
    inference_device: str = "cuda:0", torch_threads: int = 2,
    chunk_games: int = 32, total_games: int = EXPECTED_GAMES,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    checkpoint = checkpoint.resolve()
    reference_report = reference_report.resolve()
    output_root = output_root.resolve()
    device = torch.device(inference_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("paired diagnostic requested unavailable CUDA inference")
    if total_games < 1 or total_games > EXPECTED_GAMES or total_games % chunk_games:
        raise ValueError("total_games must be 1..2048 and divisible by chunk_games")
    if output_root.exists() and not resume:
        raise FileExistsError(output_root)
    reference = json.loads(reference_report.read_text(encoding="utf-8"))
    reference_rows = reference.get("entries")
    reference_audit = reference.get("candidate_deployment_identity_audit") or {}
    if (
        reference.get("status") != "PASS"
        or reference.get("completed_games") != EXPECTED_GAMES
        or not isinstance(reference_rows, list)
        or len(reference_rows) != EXPECTED_GAMES
        or reference_audit.get("status") != "PASS"
        or reference_audit.get("contract_id") != FP16_CONTRACT_ID
        or reference_audit.get("checkpoint_update") != EXPECTED_UPDATE
    ):
        raise RuntimeError("FP16 reference report is not the completed audited U40 run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("update") != EXPECTED_UPDATE:
        raise RuntimeError("paired diagnostic requires update-000040")
    if reference.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("FP16 reference and raw-FP32 run do not share source checkpoint")

    output_root.mkdir(parents=True, exist_ok=resume)
    fp32_package = output_root / "u000040_raw_fp32_storage_fp32_runtime_package"
    if resume:
        fp32_manifest = json.loads((fp32_package / "manifest.json").read_text())
    else:
        fp32_manifest = export_fp32_diagnostic(
            checkpoint=checkpoint,
            output=fp32_package,
        )
        # Record the actual execution device without expanding the shared
        # raw-FP32 exporter API used by other diagnostics.
        fp32_manifest["evaluation_inference_device"] = str(device)
        _atomic_json(fp32_package / "manifest.json", fp32_manifest)
    if (
        fp32_manifest.get("diagnostic_contract_id") != RAW_FP32_CONTRACT_ID
        or fp32_manifest.get("checkpoint_update") != EXPECTED_UPDATE
        or fp32_manifest.get("storage_dtype") != "fp32"
        or fp32_manifest.get("runtime_dtype") != "fp32"
        or fp32_manifest.get("kaggle_strength_evidence") is not False
        or fp32_manifest.get("promotion_eligible") is not False
    ):
        raise RuntimeError("raw-FP32 package did not fail closed as diagnostic-only")
    fp32_model = _load_package(fp32_package, runner.focal_deck())

    fp16_package = _resolve_path(str(reference["package"]), root)
    fp16_manifest = json.loads((fp16_package / "manifest.json").read_text())
    fp16_model = _load_package(fp16_package, runner.focal_deck())
    fp16_portable = fp16_package / "strategy/model.bin"
    fp16_payload = torch.load(fp16_portable, map_location="cpu", weights_only=True)
    fp16_audit = audit_candidate_deployment(
        manifest=fp16_manifest,
        payload=fp16_payload,
        runtime_model=fp16_model,
        source_checkpoint_sha256=_sha256(checkpoint),
        portable_checkpoint_sha256=_sha256(fp16_portable),
    )
    if (
        fp16_audit.status != "PASS"
        or fp16_audit.effective_candidate_sha256
        != reference_audit.get("effective_candidate_sha256")
    ):
        raise RuntimeError("FP16 shadow does not match the completed reference identity")
    _no_parameter_alias(fp32_model, fp16_model)

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    fp32_model = fp32_model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    fp16_model = fp16_model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if any(value.dtype != torch.float32 for value in fp32_model.parameters()):
        raise RuntimeError("raw-FP32 runtime contains non-FP32 parameters")
    if any(value.dtype != torch.float32 for value in fp16_model.parameters()):
        raise RuntimeError("FP16-round-trip runtime contains non-FP32 parameters")
    opponent = runner.load_frozen_opponent(device)
    opponent_audit = opponent._policy_identity_audit
    if (
        opponent_audit.status != "PASS"
        or opponent_audit.requested_policy_id != "Policy-0809"
    ):
        raise RuntimeError("FATAL: full Policy-0809 opponent identity audit failed")

    jobs, schedule_sha256 = build_frozen_jobs(
        focal_deck_id=runner.FOCAL_DECK_ID,
        focal_deck=runner.focal_deck(),
        runtime_root=runner.runtime_root(),
        source_policy_update=EXPECTED_UPDATE,
        # Intentional A/B pin: seed identity comes from the completed FP16 run.
        focal_deployment_identity=fp16_audit.effective_candidate_sha256,
        opponent_effective_policy_sha256=opponent_audit.effective_policy_sha256,
        evaluation_units=8,
    )
    if schedule_sha256 != reference.get("schedule_sha256"):
        raise RuntimeError("reconstructed schedule SHA-256 differs from FP16 reference")
    reference_by_id = _index_outcomes(reference_rows)
    if {job.game_id for job in jobs} != set(reference_by_id):
        raise RuntimeError("reconstructed schedule game IDs differ from FP16 reference")
    jobs = jobs[:total_games]

    rows: list[dict[str, Any]] = []
    completed_chunks = 0
    chunk_summaries: list[dict[str, Any]] = []
    if resume:
        while True:
            chunk_path = output_root / f"chunk-{completed_chunks + 1:03d}.json"
            if not chunk_path.is_file():
                break
            chunk = json.loads(chunk_path.read_text())
            entries = chunk.get("entries")
            if not isinstance(entries, list) or len(entries) != chunk_games:
                raise RuntimeError(f"malformed completed chunk: {chunk_path}")
            expected = [job.game_id for job in jobs[
                completed_chunks * chunk_games:(completed_chunks + 1) * chunk_games
            ]]
            if [row.get("game_id") for row in entries] != expected:
                raise RuntimeError("completed chunk does not match schedule prefix")
            rows.extend(entries)
            chunk_summaries.append(chunk["action_comparison"])
            completed_chunks += 1

    started = time.perf_counter()
    for chunk_number, chunk_start in enumerate(
        range(completed_chunks * chunk_games, total_games, chunk_games),
        start=completed_chunks + 1,
    ):
        chunk_jobs = jobs[chunk_start:chunk_start + chunk_games]
        collector = QuantizationPairCollector(
            fp32_model,
            opponent,
            shadow_model=fp16_model,
            device=device,
            worker_processes=min(8, len(chunk_jobs)),
            engines_per_worker=4,
            inference_channels_per_role=4,
            mode="greedy",
            coalesce_ms=5.0,
            timeout_seconds=300.0,
            record_trajectory=False,
        )
        episodes = collector.collect(chunk_jobs)
        _assert_cpu_chunk_health(episodes, expected_games=len(chunk_jobs))
        chunk_rows = _episode_rows(episodes, opponent_audit.effective_policy_sha256)
        action_summary = collector.action_summary()
        if action_summary["games_examined"] != len(chunk_jobs):
            raise RuntimeError("action shadow did not examine every game")
        rows.extend(chunk_rows)
        chunk_summaries.append(action_summary)
        paired_reference = [
            row for row in reference_rows if row["game_id"] in {x["game_id"] for x in rows}
        ]
        outcome_comparison = compare_outcomes(paired_reference, rows)
        cumulative_actions = merge_action_summaries(chunk_summaries)
        chunk_report = {
            "schema_version": SCHEMA,
            "chunk": chunk_number,
            "entries": chunk_rows,
            "action_comparison": action_summary,
            "collector_metrics": collector.metrics(),
        }
        _atomic_json(output_root / f"chunk-{chunk_number:03d}.json", chunk_report)
        progress = {
            "schema_version": SCHEMA,
            "status": "RUNNING",
            "completed_games": len(rows),
            "total_games": total_games,
            "wins": sum(row["outcome"] == 1 for row in rows),
            "losses": sum(row["outcome"] == -1 for row in rows),
            "draws": sum(row["outcome"] == 0 for row in rows),
            "win_rate": sum(row["outcome"] == 1 for row in rows) / len(rows),
            "outcome_changed_games": outcome_comparison["changed_games"],
            "fp16_win_to_fp32_loss": outcome_comparison["win_to_loss"],
            "fp16_loss_to_fp32_win": outcome_comparison["loss_to_win"],
            "strategic_decisions_examined": cumulative_actions[
                "strategic_decisions_examined"
            ],
            "effective_action_flips": cumulative_actions["effective_action_flips"],
            "effective_action_flip_fraction": cumulative_actions[
                "effective_action_flip_fraction"
            ],
            "affected_action_games": cumulative_actions["affected_games"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output_root / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        del episodes, collector
        gc.collect()

    selected_reference = [
        row for row in reference_rows if row["game_id"] in {x["game_id"] for x in rows}
    ]
    outcome_comparison = compare_outcomes(selected_reference, rows)
    action_comparison = merge_action_summaries(chunk_summaries)
    status = "PASS" if len(rows) == total_games else "FAILED"
    report = {
        "schema_version": SCHEMA,
        "status": status,
        "diagnostic_only": True,
        "kaggle_strength_evidence": False,
        "promote_evidence": False,
        "promotion_eligible": False,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_games": len(rows),
        "wins": sum(row["outcome"] == 1 for row in rows),
        "losses": sum(row["outcome"] == -1 for row in rows),
        "draws": sum(row["outcome"] == 0 for row in rows),
        "win_rate": sum(row["outcome"] == 1 for row in rows) / len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_update": EXPECTED_UPDATE,
        "fp32_package": str(fp32_package),
        "fp32_manifest": fp32_manifest,
        "fp16_reference_report": str(reference_report),
        "fp16_reference_report_sha256": _sha256(reference_report),
        "fp16_package": str(fp16_package),
        "fp16_candidate_deployment_identity_audit": fp16_audit.to_manifest(),
        "opponent_policy_identity_audit": opponent_audit.to_manifest(),
        "schedule_sha256": schedule_sha256,
        "schedule_seed_identity": "pinned_to_fp16_reference_for_paired_ablation",
        "official_engine_device": "cpu",
        "neural_inference_device": str(device),
        "outcome_comparison": outcome_comparison,
        "action_comparison": action_comparison,
        "action_comparison_scope": (
            "same-state focal strategic decisions visited by the raw-FP32 primary; "
            "forced deterministic callbacks excluded; Phantom allocation compared separately"
        ),
        "entries": rows,
    }
    _atomic_json(output_root / "action_comparison.json", action_comparison)
    _atomic_json(output_root / "report.json", report)
    _atomic_json(output_root / "progress.json", {
        key: report[key] for key in (
            "schema_version", "status", "completed_games", "wins", "losses",
            "draws", "win_rate", "elapsed_seconds",
        )
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--chunk-games", type=int, default=32)
    parser.add_argument("--total-games", type=int, default=EXPECTED_GAMES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        checkpoint=args.checkpoint,
        reference_report=args.reference_report,
        output_root=args.output_root,
        inference_device=args.inference_device,
        torch_threads=args.torch_threads,
        chunk_games=args.chunk_games,
        total_games=args.total_games,
        resume=args.resume,
    )
    print(json.dumps({
        "status": report["status"],
        "games": report["completed_games"],
        "wins": report["wins"],
        "losses": report["losses"],
        "draws": report["draws"],
        "win_rate": report["win_rate"],
        "outcome_changed_games": report["outcome_comparison"]["changed_games"],
        "effective_action_flips": report["action_comparison"]["effective_action_flips"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
