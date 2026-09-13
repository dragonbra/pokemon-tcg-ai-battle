"""Export compact Public Deck Router V3 Kaggle package with a 200 MB hard gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import torch

from ..assets import AssetRegistry, sha256_file
from ..rollout.deck_routing import exact_deck_sha256
from ..semantic_runtime.deployment.public_deck_memory_v3 import (
    PUBLIC_POLICY_ID, ROUTED_UPDATES, route_manifest,
)
from .candidate import _tensor_hash
from .export_public_router_kaggle import (
    _copy_official_runtime, _deterministic_archive, _file_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "semantic_runtime"
DEFAULT_UPDATE = 170
RULES = route_manifest(DEFAULT_UPDATE)
PACKAGE_NAME = "0045-007-PINHAOLONG_V3-DRAGAPULT_EX"
OUTPUT = ROOT / "archive/submission" / PACKAGE_NAME
ARCHIVE = ROOT / f"{PACKAGE_NAME}.tar.gz"
V20 = ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/candidates"
CANDIDATE_DIRS = {
    5: "v14-u005", 25: "v14-u025", 125: "v19-u125", 140: "v19-u140",
    145: "v19-u145", 160: "v19-u160", 165: "v19-u165", 170: "v19-u170",
}
SCHEMA = "0045_public_deck_router_v3_kaggle_package_v1"
DELTA_SCHEMA = "0045_public_deck_router_v3_complete_actor_v1"
MAX_ARCHIVE_BYTES = 200_000_000


MAIN = '''"""Kaggle entrypoint for 0045 Public Deck Router V3."""
import hashlib, json, os, sys
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file(): ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401
def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()
def _manifest():
    payload=json.loads((ROOT/"manifest.json").read_text())
    if payload.get("schema_version") != "0045_public_deck_router_v3_kaggle_package_v1":
        raise RuntimeError("0045 Deck Router V3 package schema mismatch")
    expected=payload.get("package_file_sha256")
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if not isinstance(expected,dict) or actual != set(expected): raise RuntimeError("0045 Deck Router V3 package inventory mismatch")
    bad=[name for name,digest in expected.items() if _sha256(ROOT/name) != digest]
    if bad: raise RuntimeError(f"0045 Deck Router V3 package hash mismatch: {bad}")
    return payload
PACKAGE_MANIFEST=_manifest()
from strategy.deployment.public_deck_router_v3 import PublicDeckV3RoutedCompoundPolicy  # noqa: E402
DECK=[int(line) for line in (ROOT/"deck.csv").read_text().splitlines() if line.strip()]
POLICY=PublicDeckV3RoutedCompoundPolicy.from_compact_checkpoint(
    ROOT/"strategy/model.bin", ROOT/"strategy/router_deltas.bin", DECK, default_update=170
)
if POLICY.deployment_identity.get("composite_effective_sha256") != PACKAGE_MANIFEST.get("composite_effective_sha256"):
    raise RuntimeError("0045 Deck Router V3 runtime/package composite mismatch")
def read_deck_csv(): return list(DECK)
def agent(observation):
    if observation.get("select") is None:
        POLICY.reset(); return list(DECK)
    return POLICY.select(observation)
'''


def _routed_actor_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: value for name, value in payload["actor_state_dict"].items()
        if name.startswith("action_decoder.")
        or name.startswith("option_encoder.cross_attention_transformer.norm.")
        or (
            name.startswith("state_encoder.")
            and ".parametrizations." in name
            and not name.endswith(".original")
        )
    }


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def export(
    *, output: Path = OUTPUT, archive: Path = ARCHIVE,
    package_name: str = PACKAGE_NAME,
) -> dict[str, Any]:
    if not package_name.startswith("0045-007-PINHAOLONG_"):
        raise ValueError("PinHaoLong package name must declare exact deck 007")
    if output.exists() or archive.exists():
        raise FileExistsError("Deck Router V3 package output already exists")
    registry = AssetRegistry.load(PROJECT_ROOT); registry.validate_all()
    asset = next(row for row in registry.decks if row.deck_id == "007")
    deck = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    target_deck_hash = exact_deck_sha256(deck)
    if target_deck_hash != asset.content_sha256:
        raise RuntimeError("007 registry/exact-deck identity mismatch")
    candidates: dict[int, Path] = {}
    reports: dict[int, dict[str, Any]] = {}
    candidate_ids: dict[str, str] = {}
    for update, directory in CANDIDATE_DIRS.items():
        report_path = V20 / directory / "report.json"
        candidate_path = V20 / directory / "materialization/model.pt"
        report = json.loads(report_path.read_text())
        audit = report["focal_policy_identity_audit"]
        if (
            report.get("status") != "PASS" or audit.get("status") != "PASS"
            or audit.get("checkpoint_update") != update
            or sha256_file(candidate_path) != audit.get("portable_checkpoint_sha256")
        ):
            raise RuntimeError(f"U{update} qualifying candidate identity failed")
        payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
        effective = _tensor_hash(payload, audit["focal_exact_deck_sha256"])
        if effective != audit["effective_candidate_sha256"]:
            raise RuntimeError(f"U{update} effective candidate hash failed")
        candidates[update] = candidate_path; reports[update] = report
        candidate_ids[str(update)] = _tensor_hash(payload, target_deck_hash)

    identity_payload = {
        "policy_id": PUBLIC_POLICY_ID,
        "default_checkpoint_update": DEFAULT_UPDATE,
        "rules": RULES,
        "candidate_effective_sha256": candidate_ids,
    }
    composite = _json_hash(identity_payload)
    output.mkdir(parents=True)
    try:
        shutil.copytree(
            RUNTIME_ROOT, output / "strategy",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runtime_manifest.json"),
        )
        default_payload = torch.load(
            candidates[DEFAULT_UPDATE], map_location="cpu", weights_only=True
        )
        default_payload["metadata"] = dict(default_payload["metadata"])
        default_payload["metadata"].update({
            "focal_deck_id": "007",
            "focal_exact_deck_sha256": target_deck_hash,
            "own_archetype_id": 0,
        })
        torch.save(default_payload, output / "strategy/model.bin")
        routed: dict[str, Any] = {}
        for update in ROUTED_UPDATES:
            if update == DEFAULT_UPDATE: continue
            payload = torch.load(candidates[update], map_location="cpu", weights_only=True)
            routed[str(update)] = {
                "actor_routed_state_dict": _routed_actor_state(payload),
                "policy_option_lora_state_dict": payload["policy_option_lora_state_dict"],
                "allocation_head_state_dict": payload["allocation_head_state_dict"],
            }
        floating = [
            value for route in routed.values() for state in route.values()
            for value in state.values() if torch.is_floating_point(value)
        ]
        if not floating or any(value.dtype != torch.float16 for value in floating):
            raise RuntimeError("V3 compact routed deltas are not FP16 stored")
        deltas = {
            "schema_version": DELTA_SCHEMA,
            "routed_policy_deltas": routed,
            "metadata": {
                "policy_id": PUBLIC_POLICY_ID,
                "default_checkpoint_update": DEFAULT_UPDATE,
                "routed_updates": list(ROUTED_UPDATES),
                "rules": RULES,
                "storage_dtype": "fp16", "runtime_dtype": "fp32",
                "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
                "candidate_effective_sha256": candidate_ids,
                "identity_payload": identity_payload,
                "composite_effective_sha256": composite,
            },
        }
        torch.save(deltas, output / "strategy/router_deltas.bin")
        _copy_official_runtime(output)
        (output / "deck.csv").write_text(
            "".join(f"{card}\n" for card in deck), encoding="utf-8"
        )
        (output / "main.py").write_text(MAIN, encoding="utf-8")
        best = reports[DEFAULT_UPDATE]["summary"]
        manifest = {
            "schema_version": SCHEMA,
            "project_id": "0045_single_deck_expert_minimal_lora",
            "candidate": package_name,
            "policy_id": PUBLIC_POLICY_ID,
            "deck_id": "007", "deck_display_name": asset.name,
            "focal_exact_deck_sha256": asset.content_sha256,
            "default_checkpoint_update": DEFAULT_UPDATE,
            "routed_updates": list(ROUTED_UPDATES),
            "routing_rules": RULES,
            "routed_modules": [
                "state_encoder_lora", "state_encoder_ffn_lora",
                "option_encoder_lora_ffn_norm", "action_decoder", "allocation_head",
            ],
            "candidate_effective_sha256": candidate_ids,
            "composite_effective_sha256": composite,
            "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
            "storage_dtype": "fp16", "runtime_dtype": "fp32",
            "qualifying_default_benchmark": {
                "focal_deck_id": "067",
                "evidence_boundary": (
                    "checkpoint-selection evidence only; package deployment deck is exact 007"
                ),
                "games": best["games"], "wins": best["wins"],
                "losses": best["losses"], "draws": best["draws"],
                "win_rate": best["win_rate"],
            },
            "archive_max_bytes": MAX_ARCHIVE_BYTES,
            "kaggle_submitted": False,
        }
        manifest["package_file_sha256"] = _file_inventory(output)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        env = dict(os.environ); env.update({
            "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
            "OPENBLAS_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1",
        })
        subprocess.run([
            "python3", "-c",
            "import main; assert len(main.read_deck_csv())==60; "
            "assert main.POLICY.active_update==170; "
            "assert main.POLICY.policy_id=='Experimental-Public-DeckRouter-V3-Policy0814'",
        ], cwd=output, env=env, check=True)
        _deterministic_archive(output, archive)
        if archive.stat().st_size >= MAX_ARCHIVE_BYTES:
            raise RuntimeError(
                f"V3 archive exceeds 200 MB hard gate: {archive.stat().st_size}"
            )
        result = {
            **manifest, "archive": str(archive.relative_to(ROOT)),
            "archive_sha256": sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "compact_delta_bytes": (output / "strategy/router_deltas.bin").stat().st_size,
        }
        return result
    except BaseException:
        shutil.rmtree(output, ignore_errors=True); archive.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--package-name", default=PACKAGE_NAME)
    args = parser.parse_args()
    print(json.dumps(export(
        output=args.output.resolve(), archive=args.archive.resolve(),
        package_name=args.package_name,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
