"""Raw-FP32 U-checkpoint diagnostic over eight official CPU-256 units.

This path is deliberately ineligible for Kaggle/Promote evidence.  It preserves
the source FP32 tensors instead of applying the mandatory FP16 storage round-trip.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import torch

from ..candidate_deployment import KaggleEvaluationActorCritic
from ..evaluation.frozen_jobs import build_frozen_jobs
# Importing the exporter module rather than duplicating its audited tensor inventory
# keeps this diagnostic tied to the same 0042 package composition.
from .. import export_full_semantic_candidate as exporter
from ..rollout.collector import FullSemanticRolloutCollector
from ..rollout.deck_routing import exact_deck_sha256
from ..training import run_full_semantic as runner


SCHEMA = "0042_raw_fp32_cpu2048_policy0809_diagnostic_v1"
CONTRACT_ID = "0042_raw_fp32_storage_fp32_runtime_diagnostic_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _effective_sha256(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(CONTRACT_ID.encode("ascii") + b"\0")
    for field in exporter.PORTABLE_STATE_FIELDS:
        state = payload[field]
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{field}.{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    semantic = {
        key: payload["metadata"].get(key)
        for key in (
            "actor_metadata", "opponent_meta_class_count", "own_archetype_id",
            "own_archetype_vocabulary_version", "own_archetype_taxonomy_sha256",
            "no_option_lora", "action_schema_version", "decision_gate_version",
            "canonicalizer_version", "trajectory_schema_version",
            "official_protocol_adapter_version", "feature_preprocessing_version",
            "inference_contract",
        )
    }
    digest.update(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def export_fp32_diagnostic(
    *, checkpoint: Path, output: Path,
    evaluation_inference_device: str = "cpu",
) -> dict[str, Any]:
    """Export the complete effective candidate without an FP16 quantization step."""

    if output.exists():
        raise FileExistsError(output)
    if not evaluation_inference_device.strip():
        raise ValueError("evaluation inference device must be nonempty")
    # First reuse the normal exporter for runtime/assets/schema validation.  The
    # resulting tensor payload is immediately replaced from the audited FP32 base
    # and FP32 RL checkpoint before the package can be used.
    manifest = exporter.export_candidate(
        source=runner.CANDIDATE_ROOT,
        checkpoint=checkpoint,
        output=output,
        require_frozen_selection=False,
    )
    base_payload = exporter._load_payload(exporter.BASE_CHECKPOINT)
    rl_payload = exporter._load_payload(checkpoint)
    portable_path = output / "strategy/model.bin"
    portable = torch.load(portable_path, map_location="cpu", weights_only=True)
    merged = exporter._canonical_base_state(base_payload)
    rl_state = rl_payload["state_dict"]
    for name in tuple(merged):
        if name.startswith("action_decoder."):
            merged[name] = rl_state[f"actor.{name}"].detach().cpu().float().clone()
        elif torch.is_floating_point(merged[name]):
            merged[name] = merged[name].detach().cpu().float().clone()
    portable["actor_state_dict"] = merged
    for field, prefix in (
        ("value_head_state_dict", "value_head."),
        ("allocation_head_state_dict", "allocation_head."),
        ("value_adapter_state_dict", "value_adapter."),
        ("policy_strategy_adapter_state_dict", "policy_strategy_adapter."),
    ):
        portable[field] = {
            name.removeprefix(prefix): (
                value.detach().cpu().float().clone()
                if torch.is_floating_point(value) else value.detach().cpu().clone()
            )
            for name, value in rl_state.items() if name.startswith(prefix)
        }
    floating = [
        value for field in exporter.PORTABLE_STATE_FIELDS
        for value in portable[field].values() if torch.is_floating_point(value)
    ]
    if not floating or any(value.dtype != torch.float32 for value in floating):
        raise RuntimeError("raw FP32 package contains a non-FP32 floating tensor")
    torch.save(portable, portable_path)
    manifest.update({
        "candidate": output.name,
        "portable_checkpoint_sha256": _sha256(portable_path),
        "storage_dtype": "fp32",
        "runtime_dtype": "fp32",
        "evaluation_inference_device": evaluation_inference_device,
        "diagnostic_contract_id": CONTRACT_ID,
        "diagnostic_effective_sha256": _effective_sha256(portable),
        "kaggle_strength_evidence": False,
        "promote_evidence": False,
        "promotion_eligible": False,
        "non_candidate": True,
        "selection": "RAW FP32 DIAGNOSTIC ONLY; no Kaggle/Promote strength claim",
        "frozen_evaluation": None,
    })
    manifest.pop("deployment_effective_sha256", None)
    manifest["package_file_sha256"] = exporter._package_file_hashes(output)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _load_package(package: Path, deck: tuple[int, ...]) -> KaggleEvaluationActorCritic:
    package_text = str(package)
    stale = [name for name in sys.modules if name == "strategy" or name.startswith("strategy.")]
    if stale:
        raise RuntimeError(f"stale candidate strategy modules: {stale[:3]}")
    sys.path.insert(0, package_text)
    try:
        runtime = importlib.import_module("strategy.deployment.compound_inference")
        policy = runtime.PortableCompoundSemanticPolicy.from_checkpoint(
            package / "strategy/model.bin", deck
        )
    finally:
        sys.path.remove(package_text)
        for name in tuple(sys.modules):
            if name == "strategy" or name.startswith("strategy."):
                sys.modules.pop(name, None)
    model = KaggleEvaluationActorCritic(policy).to(device="cpu", dtype=torch.float32)
    return model.eval().requires_grad_(False)


def _episode_rows(episodes: list[Any], opponent_hash: str) -> list[dict[str, Any]]:
    return [{
        "game_id": episode.job.game_id,
        "replica": int(episode.job.game_id[1:3]) - 1,
        "opponent_id": episode.job.opponent_id,
        "opponent_exact_deck_sha256": exact_deck_sha256(episode.job.opponent_deck),
        "opponent_effective_policy_sha256": opponent_hash,
        "focal_won_toss": episode.job.focal_won_toss,
        "focal_first": runner._episode_focal_first(episode),
        "outcome": 1 if episode.reward == 1.0 else -1 if episode.reward == -1.0 else 0,
        "turns": episode.turns,
        "valid": episode.valid,
        "error": episode.error,
        "fallback_reason": episode.diagnostics.get("macro_fallback_reason"),
    } for episode in episodes]


def _assert_cpu_chunk_health(episodes: list[Any], *, expected_games: int) -> None:
    """CPU equivalent of the acceptance gate, without CUDA-only telemetry."""

    if len(episodes) != expected_games:
        raise RuntimeError(
            f"CPU diagnostic returned {len(episodes)} games, expected {expected_games}"
        )
    invalid = [episode for episode in episodes if not episode.valid or episode.error]
    semantic_fallbacks = [
        episode for episode in episodes
        if episode.diagnostics.get("macro_fallback_reason")
        not in {None, runner.CHANCE_BOUNDARY_FALLBACK}
    ]
    unsupported = [
        episode for episode in episodes
        if int(episode.diagnostics.get("unsupported_effect", 0))
    ]
    pending_resets = [
        episode for episode in episodes
        if int(episode.diagnostics.get("pending_macro_reset", 0))
    ]
    if invalid or semantic_fallbacks or unsupported or pending_resets:
        raise RuntimeError(
            "CPU diagnostic health failed: "
            f"invalid={len(invalid)} semantic_fallbacks={len(semantic_fallbacks)} "
            f"unsupported={len(unsupported)} pending_resets={len(pending_resets)}"
        )
    if len({episode.job.game_id for episode in episodes}) != expected_games:
        raise RuntimeError("CPU diagnostic returned duplicate game IDs")


def run(
    *, checkpoint: Path, output_root: Path, torch_threads: int = 4,
    chunk_games: int = 32, resume: bool = False,
    elapsed_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    if output_root.exists() and not resume:
        raise FileExistsError(output_root)
    if chunk_games < 1 or 256 % chunk_games:
        raise ValueError("chunk_games must be a positive divisor of 256")
    if elapsed_offset_seconds < 0:
        raise ValueError("elapsed_offset_seconds must be nonnegative")
    output_root.mkdir(parents=True, exist_ok=resume)
    package = output_root / "u150_fp32_diagnostic_package"
    if resume:
        if not package.is_dir() or not (package / "manifest.json").is_file():
            raise RuntimeError("resume requested without an existing FP32 package")
        manifest = json.loads((package / "manifest.json").read_text())
        if (
            manifest.get("diagnostic_contract_id") != CONTRACT_ID
            or manifest.get("storage_dtype") != "fp32"
            or manifest.get("runtime_dtype") != "fp32"
            or manifest.get("rl_checkpoint_sha256") != _sha256(checkpoint)
            or manifest.get("checkpoint_update") != 150
        ):
            raise RuntimeError("resume package/checkpoint identity mismatch")
    else:
        manifest = export_fp32_diagnostic(checkpoint=checkpoint, output=package)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    model = _load_package(package, runner.focal_deck())
    opponent = runner.load_frozen_opponent(torch.device("cpu"))
    opponent_audit = opponent._policy_identity_audit
    jobs, schedule_sha256 = build_frozen_jobs(
        focal_deck_id=runner.FOCAL_DECK_ID,
        focal_deck=runner.focal_deck(),
        runtime_root=runner.runtime_root(),
        source_policy_update=int(manifest["checkpoint_update"]),
        focal_deployment_identity=manifest["diagnostic_effective_sha256"],
        opponent_effective_policy_sha256=opponent_audit.effective_policy_sha256,
        evaluation_units=8,
    )
    progress_path = output_root / "progress.json"
    rows: list[dict[str, Any]] = []
    completed_units = 0
    resumed_partial: list[dict[str, Any]] = []
    if resume:
        for unit_number in range(1, 9):
            complete = output_root / f"unit-{unit_number:02d}.json"
            partial = output_root / f"unit-{unit_number:02d}-partial.json"
            if complete.is_file():
                entries = json.loads(complete.read_text()).get("entries")
                if not isinstance(entries, list) or len(entries) != 256:
                    raise RuntimeError(f"malformed completed unit: {complete}")
                rows.extend(entries)
                completed_units += 1
                continue
            if partial.is_file():
                entries = json.loads(partial.read_text()).get("entries")
                if (
                    not isinstance(entries, list)
                    or len(entries) >= 256
                    or len(entries) % chunk_games
                ):
                    raise RuntimeError(f"malformed partial unit: {partial}")
                resumed_partial = entries
            break
    started = time.perf_counter()
    for unit in range(completed_units, 8):
        unit_started = time.perf_counter()
        unit_rows: list[dict[str, Any]] = (
            list(resumed_partial) if unit == completed_units else []
        )
        if unit_rows:
            rows.extend(unit_rows)
        unit_metrics: list[dict[str, float]] = []
        unit_jobs = jobs[unit * 256:(unit + 1) * 256]
        expected_prefix = [job.game_id for job in unit_jobs[:len(unit_rows)]]
        if [row.get("game_id") for row in unit_rows] != expected_prefix:
            raise RuntimeError("partial unit does not match the canonical schedule prefix")
        for chunk_start in range(len(unit_rows), 256, chunk_games):
            chunk = unit_jobs[chunk_start:chunk_start + chunk_games]
            collector = FullSemanticRolloutCollector(
                model, opponent, device=torch.device("cpu"),
                worker_processes=min(8, len(chunk)), engines_per_worker=4,
                inference_channels_per_role=4, mode="greedy", coalesce_ms=5.0,
                timeout_seconds=300.0, record_trajectory=False,
            )
            episodes = collector.collect(chunk)
            metrics = collector.metrics()
            _assert_cpu_chunk_health(episodes, expected_games=len(chunk))
            chunk_rows = _episode_rows(
                episodes, opponent_audit.effective_policy_sha256
            )
            unit_rows.extend(chunk_rows)
            rows.extend(chunk_rows)
            unit_metrics.append(metrics)
            progress = {
                "schema_version": SCHEMA,
                "status": "RUNNING",
                "completed_units": unit,
                "total_units": 8,
                "active_unit": unit + 1,
                "active_unit_completed_games": len(unit_rows),
                "active_unit_total_games": 256,
                "completed_games": len(rows),
                "total_games": 2048,
                "wins": sum(row["outcome"] == 1 for row in rows),
                "losses": sum(row["outcome"] == -1 for row in rows),
                "draws": sum(row["outcome"] == 0 for row in rows),
                "win_rate": sum(row["outcome"] == 1 for row in rows) / len(rows),
                "elapsed_seconds": elapsed_offset_seconds + time.perf_counter() - started,
                "last_chunk": {
                    "unit": unit + 1,
                    "chunk": chunk_start // chunk_games + 1,
                    "games": len(chunk_rows),
                    "elapsed_games_per_second": (
                        len(rows) / max(
                            elapsed_offset_seconds + time.perf_counter() - started,
                            1e-9,
                        )
                    ),
                    "collector_metrics": metrics,
                },
            }
            _atomic_json(output_root / f"unit-{unit + 1:02d}-partial.json", {
                **progress, "entries": unit_rows,
            })
            _atomic_json(progress_path, progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            del episodes, collector
            gc.collect()
        wins = sum(row["outcome"] == 1 for row in unit_rows)
        unit_report = {
            "unit": unit + 1, "games": 256, "wins": wins,
            "losses": sum(row["outcome"] == -1 for row in unit_rows),
            "draws": sum(row["outcome"] == 0 for row in unit_rows),
            "win_rate": wins / 256,
            "elapsed_seconds": time.perf_counter() - unit_started,
            "collector_metrics_by_chunk": unit_metrics,
        }
        _atomic_json(output_root / f"unit-{unit + 1:02d}.json", {
            **unit_report, "entries": unit_rows,
        })
        progress = {
            "schema_version": SCHEMA,
            "status": "RUNNING" if unit < 7 else "PASS",
            "completed_units": unit + 1,
            "total_units": 8,
            "completed_games": len(rows),
            "total_games": 2048,
            "wins": sum(row["outcome"] == 1 for row in rows),
            "losses": sum(row["outcome"] == -1 for row in rows),
            "draws": sum(row["outcome"] == 0 for row in rows),
            "win_rate": sum(row["outcome"] == 1 for row in rows) / len(rows),
            "elapsed_seconds": elapsed_offset_seconds + time.perf_counter() - started,
            "last_unit": unit_report,
        }
        _atomic_json(progress_path, progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    report = {
        **json.loads(progress_path.read_text()),
        "created_at": datetime.now(UTC).isoformat(),
        "contract_id": CONTRACT_ID,
        "diagnostic_only": True,
        "kaggle_strength_evidence": False,
        "promote_evidence": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "package": str(package),
        "package_manifest": manifest,
        "schedule_sha256": schedule_sha256,
        "opponent_policy_identity_audit": opponent_audit.to_manifest(),
        "entries": rows,
    }
    _atomic_json(output_root / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--chunk-games", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--elapsed-offset-seconds", type=float, default=0.0)
    args = parser.parse_args()
    report = run(
        checkpoint=args.checkpoint.resolve(),
        output_root=args.output_root.resolve(),
        torch_threads=args.torch_threads,
        chunk_games=args.chunk_games,
        resume=args.resume,
        elapsed_offset_seconds=args.elapsed_offset_seconds,
    )
    print(json.dumps({key: report[key] for key in (
        "status", "completed_games", "wins", "losses", "draws", "win_rate"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
