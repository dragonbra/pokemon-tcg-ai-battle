"""Run a Deck-003 Kaggle package on CPU engine workers with GPU inference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any

from evaluation.packages.loader import SubmissionPackage, load_submission_package
from evaluation.runner.batch import BatchConfig, run_batch

from .policy0814_exact_deck_cpu256_schedule import (
    CONTRACT_ID, MASTER_SEED, materialize as materialize_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
DECK_IDS = ("001", "002", "003", "007", "008", "009", "011", "071")
OPPONENT_EFFECTIVE_SHA256 = (
    "476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_card_ids() -> set[int]:
    path = ROOT / "data/official/EN_Card_Data.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            int(row[0].split(":")[-1])
            for row in csv.reader(handle)
            if row and row[0] != "Card ID"
        }


def _extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    if not (destination / "main.py").is_file():
        raise RuntimeError("archive is not a rootless Kaggle package")
    return destination


def _materialize_opponent_server(root: Path) -> Path:
    strategy = root / "strategy"
    shutil.copytree(
        PROJECT_ROOT / "semantic_runtime",
        strategy,
        ignore=shutil.ignore_patterns("runtime_manifest.json", "__pycache__", "*.pyc"),
    )
    source_model = PROJECT_ROOT / "assets/policies/definitions/policy_0814/model.pt"
    os.link(source_model, strategy / "model.bin")
    deck = PROJECT_ROOT / "assets/decks/definitions/001/deck.csv"
    shutil.copy2(deck, root / "deck.csv")
    (root / "main.py").write_text(
        """from pathlib import Path
from strategy.deployment.inference import PortableSemanticPolicy
ROOT = Path(__file__).resolve().parent
DECK = [int(x) for x in (ROOT / 'deck.csv').read_text().splitlines() if x.strip()]
POLICY = PortableSemanticPolicy.from_checkpoint(ROOT / 'strategy/model.bin', DECK)
def agent(observation):
    if observation.get('select') is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
""",
        encoding="utf-8",
    )
    return root


def _opponents(candidate: SubmissionPackage) -> tuple[SubmissionPackage, ...]:
    rows = []
    for deck_id in DECK_IDS:
        deck_path = PROJECT_ROOT / f"assets/decks/definitions/{deck_id}/deck.csv"
        deck = [int(value) for value in deck_path.read_text().splitlines() if value.strip()]
        rows.append(replace(
            candidate,
            name=f"policy_0814_deck_{deck_id}",
            deck=deck,
            deck_hash=_sha256(deck_path),
            package_hash=OPPONENT_EFFECTIVE_SHA256,
            display_name=f"Policy-0814 exact deck {deck_id}",
            package_manifest={
                "policy_id": "Policy-0814",
                "effective_policy_sha256": OPPONENT_EFFECTIVE_SHA256,
                "deck_id": deck_id,
            },
        ))
    return tuple(rows)


def _start_server(command: list[str], socket: Path, log: Path) -> tuple[subprocess.Popen, Any]:
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if socket.exists():
            return process, handle
        if process.poll() is not None:
            handle.flush()
            raise RuntimeError(f"inference server exited early; see {log}")
        time.sleep(0.1)
    process.terminate()
    raise TimeoutError(f"inference server did not create {socket}")


def _stop_server(process: subprocess.Popen, handle: Any) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    handle.close()


def _candidate_identity(candidate: SubmissionPackage, archive: Path) -> str:
    manifest = candidate.package_manifest or {}
    deployment = manifest.get("deployment_contract")
    if isinstance(deployment, dict):
        value = deployment.get("effective_candidate_sha256")
        if isinstance(value, str) and len(value) == 64:
            return value
    return _sha256(archive)


def run(archive: Path, output_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    package_root = _extract_archive(archive, output_root / "package")
    candidate = load_submission_package(package_root, _official_card_ids(), name=archive.stem)
    identity = _candidate_identity(candidate, archive)
    schedule = materialize_schedule(PROJECT_ROOT, focal_deployment_identity=identity)
    opponents = _opponents(candidate)
    counts = tuple(int(schedule["realized_deck_counts"][deck_id]) for deck_id in DECK_IDS)
    if smoke:
        opponents = opponents[:1]
        counts = (1,)

    opponent_root = _materialize_opponent_server(output_root / "opponent_server")
    candidate_socket = Path("/tmp") / f"0045-candidate-{os.getpid()}.sock"
    opponent_socket = Path("/tmp") / f"0045-opponent-{os.getpid()}.sock"
    candidate_process = candidate_handle = opponent_process = opponent_handle = None
    try:
        candidate_process, candidate_handle = _start_server([
            sys.executable, "-m",
            "pokemon_tcg_ai.evaluation.compound_package_inference_server",
            "--package-root", str(package_root), "--socket", str(candidate_socket),
        ], candidate_socket, output_root / "candidate_inference.log")
        opponent_process, opponent_handle = _start_server([
            sys.executable, "-m", "evaluation.runner.inference_server",
            "--candidate", str(opponent_root), "--socket", str(opponent_socket),
            "--device", "cuda:0", "--batch-size", "32", "--batch-wait-ms", "2",
            "--inference-dtype", "fp32", "--profile",
        ], opponent_socket, output_root / "opponent_inference.log")
        result = run_batch(BatchConfig(
            candidate=candidate,
            opponents=opponents,
            games_per_opponent=1,
            games_by_opponent=counts,
            output_root=output_root / "report",
            visualize=False,
            keep_temp=smoke,
            max_steps=4000,
            workers=8,
            worker_cpu_threads=1,
            worker_timeout_seconds=300,
            worker_crash_retries=1,
            candidate_inference_socket=candidate_socket,
            opponent_inference_socket=opponent_socket,
            candidate_inference_device="cuda:0",
            candidate_inference_dtype="fp32",
            opponent_inference_dtype="fp32",
            candidate_inference_batch_size=32,
            inference_ability_repeat_limit=20,
            engine_turn_draw_limit=100,
            opponent_pool_id="0045_policy0814_exact_decks_v18_half_distribution",
            opponent_policy_hash=OPPONENT_EFFECTIVE_SHA256,
            opponent_policy_label="Policy-0814",
            opponent_schedule_id=CONTRACT_ID + ("_smoke" if smoke else ""),
            seed=MASTER_SEED,
            seeded_engine=True,
            agent_selects_first_player=True,
            focal_seed_identity="0045_deck003_v18_cpu256_common_schedule",
        ))
    finally:
        if opponent_process is not None:
            _stop_server(opponent_process, opponent_handle)
        if candidate_process is not None:
            _stop_server(candidate_process, candidate_handle)
        candidate_socket.unlink(missing_ok=True)
        opponent_socket.unlink(missing_ok=True)

    games = list(result.game_records)
    summary = dict(result.report_data.summary)
    valid = (
        len(games) == sum(counts)
        and summary.get("errors") == 0
        and summary.get("unfinished") == 0
    )
    audit = {
        "schema_version": "0045_package_cpu256_gpu_inference_report_v1",
        "status": "PASS" if valid else "FAIL",
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "candidate_deployment_identity": identity,
        "candidate_package_hash": candidate.package_hash,
        "candidate_deployment_contract": (candidate.package_manifest or {}).get(
            "deployment_contract"
        ),
        "official_engine": True,
        "engine_device": "cpu",
        "candidate_inference": "cuda:0_fp32",
        "opponent_inference": "cuda:0_fp32",
        "opponent_policy_id": "Policy-0814",
        "opponent_effective_policy_sha256": OPPONENT_EFFECTIVE_SHA256,
        "schedule": {key: value for key, value in schedule.items() if key != "jobs"},
        "executed_counts": dict(zip(DECK_IDS[:len(counts)], counts, strict=True)),
        "batch_run_id": result.run_id,
        "report_html": str(result.report_path),
        "summary": summary,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not valid:
        raise RuntimeError(f"CPU evaluation failed integrity gate: {audit['summary']}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.archive.resolve(), args.output_root.resolve(), smoke=args.smoke), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
