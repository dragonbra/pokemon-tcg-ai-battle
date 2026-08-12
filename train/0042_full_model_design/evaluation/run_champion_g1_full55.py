"""Seal V22 U10 as Champion G1 and publish restart-safe full-55 CUDA-2048."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from evaluation.combat_mat_contract import validate_combat_mat_index

from ..export_full_semantic_candidate import export_candidate
from ..league.catalog import load_frozen_catalog
from ..policy.own_archetype import OwnArchetypeVocabulary
from ..training.run_remaining_archetype_curriculum import validate_terminal


ROOT = Path(__file__).resolve().parents[3]
VERSION = "V22_archaludon_ex_cinderace_048"
UPDATE = 10
SOURCE = ROOT / f"rl_runs/0042_full_model_design/versions/{VERSION}/checkpoint/update-000010.pt"
ARCHIVE = ROOT / "archive/pretrained/0042_champion_g1"
OUTPUT = ROOT / (
    "docs/evaluation/combat_mat/policy_0809/"
    "champion_g1_v22_u10_vs_frozen_0809_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1"
)
RAW = ROOT / ".tmp/evaluation/0042_champion_g1_full55"
STATE = RAW / "state.json"
BASELINE = ROOT / (
    "docs/evaluation/combat_mat/policy_0809/"
    "0809_kaggle_top100_plus_v1_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1/manifest.json"
)
REFERENCE_NUMBER = "048"
EXPECTED_OPPONENT_EFFECTIVE = (
    "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
)

base = importlib.import_module("engine_cuda.tools.evaluate_policy_0809_cuda")
renderer = importlib.import_module(
    "train.0042_full_model_design.diagnostics.render_custom_deck_cuda2048"
)


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
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state(**payload: object) -> None:
    _atomic_json(STATE, {**payload, "checked_at": time.time()})


def _number(item: Any) -> str:
    return item.root.name.split("_", 1)[0]


def evaluation_order() -> tuple[str, ...]:
    """First numbered representative for classes 0-13, then every remainder."""

    vocabulary = OwnArchetypeVocabulary.load()
    catalog = sorted(load_frozen_catalog(), key=lambda item: int(_number(item)))
    seen: set[int] = set()
    first: list[str] = []
    remainder: list[str] = []
    for item in catalog:
        number = _number(item)
        class_id = vocabulary.classify_own_deck(item.deck).value
        if class_id < 14 and class_id not in seen:
            seen.add(class_id)
            first.append(number)
        else:
            remainder.append(number)
    if seen != set(range(14)) or len(first) != 14:
        raise RuntimeError("Champion G1 phase one does not cover exact classes 0-13")
    order = tuple(first + remainder)
    if len(order) != 55 or set(order) != {f"{value:03d}" for value in range(1, 56)}:
        raise RuntimeError("Champion G1 evaluation order is not an exact 001-055 permutation")
    return order


def wait_for_v22() -> Path:
    while True:
        try:
            checkpoint = validate_terminal(VERSION, UPDATE)
            status = json.loads(
                (SOURCE.parent.parent / "artifact/status.json").read_text(encoding="utf-8")
            )
            try:
                os.kill(int(status.get("pid")), 0)
            except (OSError, TypeError, ValueError):
                return checkpoint
            _state(state="waiting_for_v22_process_exit", version=VERSION, update=UPDATE)
        except RuntimeError:
            status_path = SOURCE.parent.parent / "artifact/status.json"
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.is_file() else {}
            )
            try:
                os.kill(int(status.get("pid")), 0)
            except (OSError, TypeError, ValueError):
                raise
            _state(state="waiting_for_v22_terminal_gate", version=VERSION, update=UPDATE)
        except (FileNotFoundError, json.JSONDecodeError):
            _state(state="waiting_for_v22_u10", version=VERSION, update=UPDATE)
        time.sleep(10)


def _catalog_by_number() -> dict[str, Any]:
    return {_number(item): item for item in load_frozen_catalog()}


def seal_champion(checkpoint: Path) -> dict[str, Any]:
    digest = _sha256(checkpoint)
    archive_checkpoint = ARCHIVE / "source_update_000010.pt"
    archive_sidecar = archive_checkpoint.with_suffix(".pt.sha256")
    manifest_path = ARCHIVE / "manifest.json"
    if ARCHIVE.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("champion_id") != "Champion-G1"
            or manifest.get("source_checkpoint_sha256") != digest
            or not archive_checkpoint.is_file()
            or _sha256(archive_checkpoint) != digest
        ):
            raise RuntimeError("existing Champion G1 archive identity differs")
        return manifest

    ARCHIVE.mkdir(parents=True)
    shutil.copy2(checkpoint, archive_checkpoint)
    archive_sidecar.write_text(digest + "\n", encoding="ascii")
    if _sha256(archive_checkpoint) != digest:
        raise RuntimeError("Champion G1 archived checkpoint copy mismatch")

    item = _catalog_by_number()[REFERENCE_NUMBER]
    deck_manifest = json.loads((item.root / "manifest.json").read_text(encoding="utf-8"))
    package = ARCHIVE / "fp16_reference_048_package"
    portable = export_candidate(
        source=importlib.import_module(
            "train.0042_full_model_design.training.run_full_semantic"
        ).CANDIDATE_ROOT,
        checkpoint=checkpoint,
        output=package,
        require_frozen_selection=False,
        deployment_deck=item.deck,
        deployment_deck_id=item.deck_id,
        deployment_deck_display_name=f"048 · {item.display_name}",
        deployment_deck_source=str(item.root.relative_to(ROOT)),
    )
    deployment = portable.get("candidate_deployment_identity_audit") or portable.get(
        "deployment_identity"
    ) or {}
    manifest = {
        "schema_version": "0042_champion_generation_anchor_v1",
        "champion_id": "Champion-G1",
        "designation": "user_designated_generation_anchor",
        "promote_champion_protocol_decision": "USER_DESIGNATED_G1_NOT_AUTOMATIC_PROMOTE",
        "source_version": VERSION,
        "source_update": UPDATE,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": digest,
        "archived_checkpoint": str(archive_checkpoint.relative_to(ROOT)),
        "checkpoint_kind": "model_only",
        "reference_portable_package": str(package.relative_to(ROOT)),
        "reference_deck_number": REFERENCE_NUMBER,
        "reference_deck_exact_sha256": item.deck_sha256,
        "reference_export_manifest": portable,
        "reference_deployment_identity": deployment,
        "evaluation_order": list(evaluation_order()),
        "frozen_opponent_policy_id": "Policy-0809",
        "frozen_opponent_effective_sha256": EXPECTED_OPPONENT_EFFECTIVE,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _validate_report(report: dict[str, Any], *, number: str, checkpoint_sha: str) -> None:
    entries = report.get("entries")
    candidate = report.get("candidate_deployment_identity_audit") or {}
    opponent = report.get("opponent_policy_identity_audit") or {}
    if (
        report.get("status") != "PASS"
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("checkpoint_update") != UPDATE
        or report.get("summary", {}).get("games") != 2048
        or not isinstance(entries, list)
        or len(entries) != 2048
        or any(
            row.get("valid") is not True
            or row.get("error") is not None
            or row.get("semantic_fallback") is not False
            for row in entries
        )
        or candidate.get("status") != "PASS"
        or candidate.get("storage_dtype") != "fp16"
        or candidate.get("runtime_dtype") != "fp32"
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
        or opponent.get("effective_policy_sha256") != EXPECTED_OPPONENT_EFFECTIVE
        or report.get("focal_opponent_shared_parameter_storages") != 0
    ):
        raise RuntimeError(f"Champion G1 deck {number} CUDA-2048 evidence failed")


def _summary(number: str, item: Any, report: dict[str, Any]) -> dict[str, Any]:
    entries = report["entries"]
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    summary = report["summary"]
    manifest = json.loads((item.root / "manifest.json").read_text(encoding="utf-8"))
    return {
        "deck_number": number,
        "deck_id": item.deck_id,
        "display_name": item.display_name,
        "representative_cards": [],
        "exact_deck_sha256": item.deck_sha256,
        "schedule_games": item.games,
        "best_rank": item.best_rank,
        "observed_players": len(manifest.get("provenance", {}).get("source_ranks", [])),
        "segment": item.segment,
        "games": 2048,
        "wins": int(summary["wins"]),
        "losses": int(summary["losses"]),
        "draws": int(summary["draws"]),
        "win_rate": float(summary["win_rate"]),
        "actual_first_games": len(first),
        "actual_first_wins": sum(row["outcome"] == 1 for row in first),
        "actual_second_games": len(second),
        "actual_second_wins": sum(row["outcome"] == 1 for row in second),
        "wall_seconds": float(summary["elapsed_seconds"]),
        "schedule_sha256": report["schedule_sha256"],
        "candidate_deployment_identity_audit": report[
            "candidate_deployment_identity_audit"
        ],
        "opponent_policy_identity_audit": report["opponent_policy_identity_audit"],
        "engine": {
            "extension": report.get("collector_metrics", {}).get("extension"),
        },
        "report": f"reports/{number}.html",
        "games_file": f"games/{number}.json",
    }


def _baseline_delta(number: str, summary: dict[str, Any]) -> float | None:
    if not BASELINE.is_file():
        return None
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    rows = {str(row["deck_number"]): row for row in baseline.get("reports", [])}
    return (
        summary["win_rate"] - float(rows[number]["win_rate"])
        if number in rows else None
    )


def publish(completed_numbers: set[str], champion: dict[str, Any]) -> None:
    catalog, candidates = base._catalog()
    rows = []
    for number in sorted(completed_numbers):
        report = json.loads((RAW / number / "report.json").read_text(encoding="utf-8"))
        item = _catalog_by_number()[number]
        row = _summary(number, item, report)
        row["policy_0809_baseline_win_rate_delta"] = _baseline_delta(number, row)
        rows.append(row)
    identity = {
        "schema_version": "0042_champion_g1_identity_audit_v1",
        "status": "PASS",
        "requested_policy_id": "Champion-G1",
        "materialized_policy_id": "Champion-G1",
        "source_checkpoint_sha256": champion["source_checkpoint_sha256"],
        "frozen_opponent_policy_id": "Policy-0809",
        "frozen_opponent_effective_sha256": EXPECTED_OPPONENT_EFFECTIVE,
    }
    base._publish(
        identity,
        rows,
        catalog,
        candidates,
        output_root=OUTPUT,
        focal_policy_id="Champion-G1",
        opponent_policy_id="Frozen Policy-0809",
        focal_policy_sha256=champion["source_checkpoint_sha256"],
        requested_numbers=tuple(evaluation_order()),
        page_title="Champion G1 · Full-55 CUDA-2048",
        evidence_note=(
            "同一 V22 U10 immutable effective checkpoint 按 exact deck 绑定 001–055；"
            "首轮先覆盖 14 种正式 Meta Archetype，再补齐重复构筑。"
        ),
        comparison_label="vs 0809 EC",
    )
    validate_combat_mat_index(json.loads((OUTPUT / "manifest.json").read_text()))


def run_one(number: str, checkpoint: Path, champion: dict[str, Any]) -> None:
    item = _catalog_by_number()[number]
    root = RAW / number
    report_path = root / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_report(report, number=number, checkpoint_sha=champion["source_checkpoint_sha256"])
    else:
        if root.exists():
            attempts = RAW / "incomplete_attempts"
            attempts.mkdir(parents=True, exist_ok=True)
            destination = attempts / f"{number}-{int(time.time())}"
            root.rename(destination)
        command = [
            sys.executable, "-m",
            "train.0042_full_model_design.diagnostics.fp16_cuda2048_custom_deck_policy0809",
            "--checkpoint", str(checkpoint),
            "--deck", str(item.root / "deck.csv"),
            "--deck-id", item.deck_id,
            "--deck-display-name", f"{number} · {item.display_name}",
            "--deck-source", str(item.root.relative_to(ROOT)),
            "--output-root", str(root),
            "--evaluation-units", "8",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise RuntimeError(f"Champion G1 deck {number} evaluator exited {completed.returncode}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_report(
            report, number=number,
            checkpoint_sha=champion["source_checkpoint_sha256"],
        )

    OUTPUT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    OUTPUT.joinpath("games").mkdir(parents=True, exist_ok=True)
    html = renderer.render(report, renderer._audit_from_report(report))
    report_out = OUTPUT / f"reports/{number}.html"
    temporary = report_out.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, report_out)
    _atomic_json(OUTPUT / f"games/{number}.json", {
        "schema": "0042_champion_g1_cuda2048_games_v1",
        "summary": _summary(number, item, report),
        "games": report["entries"],
    })


def main() -> int:
    checkpoint = wait_for_v22()
    champion = seal_champion(checkpoint)
    order = evaluation_order()
    completed: set[str] = set()
    for position, number in enumerate(order, 1):
        _state(
            state="evaluating",
            champion_id="Champion-G1",
            current_deck=number,
            position=position,
            total=55,
            completed=sorted(completed),
            phase="archetype_first" if position <= 14 else "remaining_exact_decks",
        )
        run_one(number, checkpoint, champion)
        completed.add(number)
        publish(completed, champion)
    _state(
        state="complete", champion_id="Champion-G1", completed=sorted(completed),
        output=str(OUTPUT.relative_to(ROOT)), finished_at=time.time(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
