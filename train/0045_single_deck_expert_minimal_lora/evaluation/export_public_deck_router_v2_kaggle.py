"""Export V15 Public Deck Router V2 as a compact Kaggle agent package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import torch

from ..assets import AssetRegistry, sha256_file
from ..semantic_runtime.deployment.public_deck_memory import (
    DEFAULT_UPDATE, PUBLIC_POLICY_ID, ROUTED_UPDATES, RULE_MANIFEST,
)
from ..policy.actor_critic import (
    DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT,
    load_actor_critic,
)
from .candidate import _expanded_actor_state_dict, _tensor_hash
from .export_public_router_kaggle import (
    _copy_official_runtime, _deterministic_archive, _file_inventory,
)
from .public_router_candidate import (
    _head_effective_hash, _head_states, _json_hash, _tensor_digest,
)
from .public_deck_router_v2 import V14_ADAPTATION, _effective_actor_state
from .run_public_deck_router_v2_policy0814_cuda512 import validate_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "semantic_runtime"
PACKAGE_NAME = "0045-007-PINHAOLONG_V2-DRAGAPULT_EX"
OUTPUT = ROOT / "archive/submission" / PACKAGE_NAME
ARCHIVE = ROOT / f"{PACKAGE_NAME}.tar.gz"
REPORT = (
    ROOT / "rl_runs/0045_single_deck_expert_minimal_lora/versions/"
    "V15_public_deck_router_v2_policy0814/artifact/"
    "public_deck_router_eval512/report.json"
)
MATERIALIZATION_ROOT = REPORT.parent / "materialization"
SCHEMA = "0045_public_deck_router_v2_kaggle_package_v1"
HEAD_SCHEMA = "0045_public_deck_router_v2_compact_heads_v1"


MAIN = '''"""Kaggle entrypoint for 0045 Public Deck Router V2."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _manifest():
    payload = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0045_public_deck_router_v2_kaggle_package_v1":
        raise RuntimeError("0045 Deck Router V2 package schema mismatch")
    expected = payload.get("package_file_sha256")
    actual = {
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not isinstance(expected, dict) or actual != set(expected):
        raise RuntimeError("0045 Deck Router V2 package inventory mismatch")
    mismatched = [name for name, digest in expected.items() if _sha256(ROOT / name) != digest]
    if mismatched:
        raise RuntimeError(f"0045 Deck Router V2 package hash mismatch: {mismatched}")
    return payload

PACKAGE_MANIFEST = _manifest()
from strategy.deployment.public_deck_router import PublicDeckRoutedCompoundPolicy  # noqa: E402
DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PublicDeckRoutedCompoundPolicy.from_compact_checkpoint(
    ROOT / "strategy/model.bin", ROOT / "strategy/router_heads.bin", DECK
)
if POLICY.deployment_identity.get("composite_effective_sha256") != PACKAGE_MANIFEST.get("composite_effective_sha256"):
    raise RuntimeError("0045 Deck Router V2 runtime/package composite mismatch")

def read_deck_csv():
    return list(DECK)

def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
'''


def _materialize_heads(
    candidates: Mapping[int, Path], report: Mapping[str, Any], output: Path,
    deck: tuple[int, ...],
) -> dict[str, Any]:
    audit = report["focal_public_router_identity_audit"]
    if (
        set(candidates) != set(ROUTED_UPDATES)
        or audit.get("policy_id") != PUBLIC_POLICY_ID
        or audit.get("rules") != RULE_MANIFEST
        or audit.get("default_checkpoint_update") != DEFAULT_UPDATE
    ):
        raise RuntimeError("Deck Router V2 qualifying identity mismatch")
    head_ids: dict[str, str] = {}
    candidate_ids: dict[str, str] = {}
    payloads: dict[int, dict[str, Any]] = {}
    shared_reference: dict[str, torch.Tensor] | None = None
    for update in ROUTED_UPDATES:
        payload = torch.load(candidates[update], map_location="cpu", weights_only=True)
        expected = audit["candidate_materializations"][str(update)]
        if sha256_file(candidates[update]) != expected["portable_checkpoint_sha256"]:
            raise RuntimeError(f"U{update} portable file identity mismatch")
        effective = _tensor_hash(payload, expected["focal_exact_deck_sha256"])
        if effective != expected["effective_candidate_sha256"]:
            raise RuntimeError(f"U{update} effective candidate identity mismatch")
        states = _head_states(payload)
        head = _head_effective_hash(states)
        if head != audit["routed_component_sha256"][str(update)]:
            raise RuntimeError(f"U{update} routed component identity mismatch")
        metadata = payload.get("metadata") or {}
        model, _ = load_actor_critic(
            checkpoint=DEFAULT_0814_ACTOR_CHECKPOINT,
            value_checkpoint=DEFAULT_0814_VALUE_CHECKPOINT,
            deck=deck, deck_id="007", device="cpu",
            own_archetype_id_override=int(metadata["own_archetype_id"]),
            adaptation=V14_ADAPTATION,
        )
        model.actor.load_state_dict({
            name: value.float() if torch.is_floating_point(value) else value
            for name, value in _expanded_actor_state_dict(
                payload["actor_state_dict"]
            ).items()
        }, strict=True)
        shared = {
            name: value for name, value in _effective_actor_state(model.actor).items()
            if not name.startswith("action_decoder.")
        }
        if shared_reference is None:
            shared_reference = shared
        elif set(shared) != set(shared_reference) or any(
            not torch.equal(shared[name], shared_reference[name]) for name in shared
        ):
            raise RuntimeError(f"U{update} shared Actor identity mismatch")
        payloads[update] = payload
        head_ids[str(update)] = head
        candidate_ids[str(update)] = effective
    assert shared_reference is not None
    shared_id = _tensor_digest(shared_reference)
    if shared_id != audit["shared_actor"]["shared_effective_sha256"]:
        raise RuntimeError("Deck Router V2 shared Actor hash mismatch")
    identity_payload = dict(audit["identity_payload"])
    if (
        identity_payload.get("shared_effective_sha256") != shared_id
        or identity_payload.get("routed_component_sha256") != head_ids
        or identity_payload.get("candidate_effective_sha256") != candidate_ids
        or _json_hash(identity_payload) != audit["composite_effective_sha256"]
    ):
        raise RuntimeError("Deck Router V2 composite identity mismatch")
    routed = {
        str(update): _head_states(payloads[update])
        for update in ROUTED_UPDATES if update != DEFAULT_UPDATE
    }
    compact = {
        "schema_version": HEAD_SCHEMA,
        "routed_head_state_dicts": routed,
        "metadata": {
            "policy_id": PUBLIC_POLICY_ID,
            "default_checkpoint_update": DEFAULT_UPDATE,
            "routed_updates": list(ROUTED_UPDATES),
            "rules": RULE_MANIFEST,
            "rules_sha256": _json_hash(RULE_MANIFEST),
            "storage_dtype": "fp16", "runtime_dtype": "fp32",
            "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
            "candidate_effective_sha256": candidate_ids,
            "head_effective_sha256": head_ids,
            "shared_effective_sha256": shared_id,
            "identity_payload": identity_payload,
            "composite_effective_sha256": audit["composite_effective_sha256"],
            "qualifying_report_schema": report.get("schema_version"),
            "qualifying_benchmark_id": report.get("benchmark_id"),
        },
    }
    floating = [
        value for state in routed.values() for field in state.values()
        for value in field.values() if torch.is_floating_point(value)
    ]
    if not floating or any(value.dtype != torch.float16 for value in floating):
        raise RuntimeError("Deck Router V2 compact heads are not FP16 stored")
    torch.save(compact, output)
    return {
        "status": "PASS", "schema_version": HEAD_SCHEMA,
        "compact_heads_sha256": sha256_file(output),
        "default_portable_checkpoint_sha256": sha256_file(candidates[DEFAULT_UPDATE]),
        "shared_effective_sha256": shared_id,
        "head_effective_sha256": head_ids,
        "candidate_effective_sha256": candidate_ids,
        "composite_effective_sha256": audit["composite_effective_sha256"],
        "storage_dtype": "fp16", "runtime_dtype": "fp32",
    }


def export(*, output: Path = OUTPUT, archive: Path = ARCHIVE) -> dict[str, Any]:
    if output.exists() or archive.exists():
        raise FileExistsError("Deck Router V2 package output already exists")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    audit = report["focal_public_router_identity_audit"]
    registry = AssetRegistry.load(PROJECT_ROOT); registry.validate_all()
    asset = next(row for row in registry.decks if row.deck_id == "007")
    deck = tuple(map(int, (PROJECT_ROOT / asset.deck_path).read_text().splitlines()))
    candidates = {
        update: MATERIALIZATION_ROOT / f"update-{update:06d}/model.pt"
        for update in ROUTED_UPDATES
    }
    if any(not path.is_file() for path in candidates.values()):
        raise FileNotFoundError("V15 frozen candidate materializations are incomplete")
    output.mkdir(parents=True)
    try:
        shutil.copytree(
            RUNTIME_ROOT, output / "strategy",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runtime_manifest.json"),
        )
        default_payload = torch.load(
            candidates[DEFAULT_UPDATE], map_location="cpu", weights_only=True
        )
        base_payload = torch.load(
            DEFAULT_0814_ACTOR_CHECKPOINT, map_location="cpu", weights_only=True
        )
        default_payload["metadata"] = dict(default_payload["metadata"])
        default_payload["metadata"]["actor_metadata"] = {
            "model_config": base_payload["metadata"]["model_config"]
        }
        torch.save(default_payload, output / "strategy/model.bin")
        compact = _materialize_heads(
            candidates, report, output / "strategy/router_heads.bin", deck
        )
        _copy_official_runtime(output)
        (output / "deck.csv").write_text(
            "".join(f"{card}\n" for card in deck), encoding="utf-8"
        )
        (output / "main.py").write_text(MAIN, encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA,
            "project_id": "0045_single_deck_expert_minimal_lora",
            "candidate": PACKAGE_NAME,
            "policy_id": PUBLIC_POLICY_ID,
            "policy_kind": "public_exact_deck_evidence_router",
            "deck_id": "007", "deck_display_name": asset.name,
            "focal_exact_deck_sha256": audit["candidate_materializations"]["70"]["focal_exact_deck_sha256"],
            "default_checkpoint_update": DEFAULT_UPDATE,
            "routed_updates": list(ROUTED_UPDATES),
            "routing_rules": RULE_MANIFEST,
            "routed_modules": audit["routed_modules"],
            "compact_deployment_audit": compact,
            "composite_effective_sha256": audit["composite_effective_sha256"],
            "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
            "storage_dtype": "fp16", "runtime_dtype": "fp32",
            "runtime_framework": "pytorch",
            "native_runtime_load_order": "torch_before_cg",
            "critic_outputs_consumed_by_actor_or_router": False,
            "qualifying_strength_evidence": {
                "status": "PASS", "report": str(REPORT.relative_to(ROOT)),
                "report_sha256": sha256_file(REPORT),
                "benchmark_id": report["benchmark_id"],
                "games": report["summary"]["games"],
                "wins": report["summary"]["wins"],
                "losses": report["summary"]["losses"],
                "draws": report["summary"]["draws"],
                "win_rate": report["summary"]["win_rate"],
                "evidence_class": audit["evidence_class"],
            },
            "historical_checkpoint_boundary": audit["historical_checkpoint_boundary"],
            "kaggle_submitted": False,
            "semantic_runtime_source": str(RUNTIME_ROOT.relative_to(ROOT)),
        }
        manifest["package_file_sha256"] = _file_inventory(output)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        environment = dict(os.environ)
        environment.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        subprocess.run([
            "python3", "-c",
            "import main; assert len(main.read_deck_csv()) == 60; "
            "assert main.POLICY.policy_id == 'Experimental-Public-DeckRouter-V2-Policy0814'; "
            "assert main.POLICY.active_update == 70; "
            "assert main.POLICY.deployment_identity['runtime_dtype'] == 'fp32'",
        ], cwd=output, env=environment, check=True)
        _deterministic_archive(output, archive)
        return {
            **manifest, "archive": str(archive.relative_to(ROOT)),
            "archive_sha256": sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
        }
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        archive.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    print(json.dumps(export(
        output=args.output.resolve(), archive=args.archive.resolve()
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
