"""Audit and benchmark the CUDA-resident 0042 -> Engine 2.0 hot path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
HOT_PATH_FILES = (
    "src/official_engine_kernels.cu",
    "python/ptcg_cuda_engine/semantic0031_bridge.py",
)
AUDITED_ROUTING_FILES = {
    "python/ptcg_cuda_engine/semantic0031_resident.py": "e9bb6a274c9866e3ffff7576aea12479866d5882d9518144fef4d88a5b635da9",
    "python/ptcg_cuda_engine/semantic0031_router.py": "b627c4b01df238e08a5441deab466114d980efe30ffa0f558ba634a5fc898d2b",
}
EXPECTED_OFFICIAL_API = frozenset({
    "action_bytes", "advance_to_decision", "allocated_bytes", "apply_actions",
    "apply_packed_actions", "apply_packed_setup_actions", "batch_size", "classify",
    "decision_actors", "encode_policy_v1", "encode_semantic0031_v2_lanes",
    "game_results", "pack_actions", "prize_counts", "reset_seeded_first_min",
    "reset_seeded_first_min_masked", "reset_seeded_first_min_semantic",
    "reset_seeded_first_min_semantic_masked", "reset_seeded_interactive",
    "reset_seeded_interactive_masked", "reset_seeded_interactive_semantic",
    "reset_seeded_interactive_semantic_masked", "reset_states", "rule_pack_bytes",
    "semantic_history_raw", "state_bytes", "statuses", "turns", "upload_rule_pack",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_source_hot_path(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    rows = []
    for relative in HOT_PATH_FILES:
        old = root / "engine_cuda" / relative
        new = root / "engine_cuda_2_0" / relative
        old_hash, new_hash = _sha(old), _sha(new)
        if old_hash != new_hash:
            raise RuntimeError(f"CUDA 2.0 hot-path drift requires a performance review: {relative}")
        rows.append({"path": relative, "0042_sha256": old_hash, "0043_sha256": new_hash})
    routing_rows = []
    for relative, expected in AUDITED_ROUTING_FILES.items():
        actual = _sha(root / "engine_cuda_2_0" / relative)
        if actual != expected:
            raise RuntimeError(f"CUDA 2.0 audited multi-policy routing drift: {relative}")
        routing_rows.append({"path": relative, "0043_sha256": actual})
    integration = (PROJECT_ROOT / "cuda_engine_2/inference.py").read_text()
    resident_loader = (PROJECT_ROOT / "cuda_engine_2/resident.py").read_text()
    forbidden = (
        "semantic_runtime.features.compiler",
        "compile_canonical_row",
        "collate_canonical_records",
    )
    found = [token for token in forbidden if token in integration or token in resident_loader]
    if found:
        raise RuntimeError(f"0043 CUDA inference imports CPU feature compilation: {found}")
    return {
        "status": "PASS",
        "hot_path_files": rows,
        "audited_routing_files": routing_rows,
        "device_feature_entry": "OfficialCudaEngine.encode_semantic0031_v2_lanes",
        "resident_runner": "ptcg_cuda_engine.semantic0031_resident.run_resident_greedy_jobs",
        "resident_backend_loader": "cuda_engine_2.resident.load_resident_backend",
        "cpu_feature_compiler_imports": 0,
    }


def audit_extension_api(extension_dir: Path) -> dict[str, Any]:
    probe = """
import json, sys
sys.path.insert(0, sys.argv[1])
import _ptcg_cuda
print(json.dumps({
  'state_abi': int(_ptcg_cuda.OFFICIAL_STATE_ABI_VERSION),
  'rule_abi': int(_ptcg_cuda.OFFICIAL_RULE_ABI_VERSION),
  'semantic_schema': int(_ptcg_cuda.SEMANTIC0031_CUDA_FEATURE_SCHEMA_VERSION),
  'methods': sorted(x for x in dir(_ptcg_cuda.OfficialCudaEngine) if not x.startswith('_')),
}))
"""
    completed = subprocess.run(
        ["python3", "-c", probe, str(Path(extension_dir).resolve())],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout)
    if (
        payload["state_abi"] != 7
        or payload["rule_abi"] != 1
        or payload["semantic_schema"] != 1
        or set(payload["methods"]) != EXPECTED_OFFICIAL_API
    ):
        raise RuntimeError("CUDA Engine 2.0 extension API compatibility failed")
    return {"status": "PASS", **payload}


def _native_sample(binary: Path, *, envs: int, steps: int) -> float:
    completed = subprocess.run(
        [str(binary), "--envs", str(envs), "--steps", str(steps), "--policies", "12"],
        text=True, capture_output=True, check=True,
    )
    return float(json.loads(completed.stdout)["environment_steps_per_second"])


def compare_native_smoke(
    old_binary: Path, new_binary: Path, *, repeats: int = 5,
    envs: int = 4096, steps: int = 2000, minimum_ratio: float = 0.95,
) -> dict[str, Any]:
    if repeats < 3:
        raise ValueError("performance comparison requires at least three repeats")
    old = [_native_sample(old_binary, envs=envs, steps=steps) for _ in range(repeats)]
    new = [_native_sample(new_binary, envs=envs, steps=steps) for _ in range(repeats)]
    old_median, new_median = statistics.median(old), statistics.median(new)
    ratio = new_median / old_median
    if ratio < minimum_ratio:
        raise RuntimeError(f"CUDA Engine 2.0 native throughput regressed: ratio={ratio:.6f}")
    return {
        "status": "PASS", "envs": envs, "steps": steps, "repeats": repeats,
        "old_samples_per_second": old, "new_samples_per_second": new,
        "old_median_per_second": old_median,
        "new_median_per_second": new_median,
        "new_over_old_ratio": ratio, "minimum_ratio": minimum_ratio,
        "scope": "synthetic native engine/codec/route microbenchmark; not full-policy rollout throughput",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-binary", type=Path, required=True)
    parser.add_argument("--new-binary", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": "0043_cuda_engine_2_performance_compatibility_v1",
        "status": "PASS",
        "source_hot_path": audit_source_hot_path(),
        "extension_api": audit_extension_api(args.extension_dir),
        "native_benchmark": compare_native_smoke(
            args.old_binary.resolve(), args.new_binary.resolve(), repeats=args.repeats,
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
