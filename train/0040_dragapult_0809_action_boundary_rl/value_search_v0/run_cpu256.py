"""Run the locked U270 official-CPU baseline or Value Search V0 treatment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Iterator

from evaluation.cards import load_card_catalog
from evaluation.frozen_0806_full_evaluation import POLICY_0806_TARGET, _batch_config
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.packages.loader import load_submission_package
from evaluation.runner.batch import _policy_inference_server, run_batch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACKAGE = ROOT / ".tmp/evaluation/0040_u270_value_search_v0/package"
HOOK_MODULE = "train.0040_dragapult_0809_action_boundary_rl.value_search_v0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _candidate_server(package: Path) -> Iterator[Path]:
    socket_path = Path(tempfile.gettempdir()) / f"ptcg-u270-vs-{os.getpid()}.sock"
    command = [
        sys.executable,
        "-m",
        f"{HOOK_MODULE}.inference_server",
        "--candidate",
        str(package),
        "--socket",
        str(socket_path),
        "--device",
        "cuda:0",
        "--ability-repeat-limit",
        "20",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 90.0
    try:
        while not socket_path.exists():
            if process.poll() is not None:
                detail = process.stderr.read() if process.stderr else ""
                raise RuntimeError(f"U270 Value Search server failed: {detail}")
            if time.monotonic() >= deadline:
                raise RuntimeError("U270 Value Search server startup timed out")
            time.sleep(0.05)
        yield socket_path
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        socket_path.unlink(missing_ok=True)


@contextmanager
def _hook_environment(enabled: bool, log_root: Path) -> Iterator[None]:
    names = (
        "EVALUATION_VALUE_SEARCH_V0_MODULE",
        "EVALUATION_VALUE_SEARCH_V0_LOG_ROOT",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        if enabled:
            os.environ[names[0]] = HOOK_MODULE
            os.environ[names[1]] = str(log_root)
        else:
            for name in names:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(*, mode: str, games: int, output: Path, package_root: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    log_root = output / "search_logs"
    official_ids = set(load_card_catalog(ROOT / "data/official/EN_Card_Data.csv"))
    candidate = load_submission_package(package_root, official_ids)
    manifest = candidate.package_manifest or {}
    if manifest.get("checkpoint_update") != 270:
        raise RuntimeError("candidate package is not update 270")
    candidate = dataclasses.replace(
        candidate,
        name=str(manifest["deck_id"]),
        display_name=str(manifest["deck_display_name"]),
    )
    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    if games == 256:
        opponents = None
        counts = None
    elif games > 0:
        opponents = (catalog.opponents[0],)
        counts = (games,)
    else:
        raise ValueError("games must be positive")

    started = time.perf_counter()
    with _candidate_server(package_root) as candidate_socket:
        with _policy_inference_server(
            root=catalog.opponent_policy.root,
            device="cuda:0",
            batch_size=128,
            batch_wait_ms=1.0,
            label="opponent",
            ability_repeat_limit=20,
            inference_dtype="fp32",
        ) as opponent_socket:
            config = _batch_config(
                catalog,
                candidate,
                POLICY_0806_TARGET,
                output_root=output / "report",
                workers=min(16, games),
                batch_size=128,
                batch_wait_ms=1.0,
                inference_dtype="fp32",
                candidate_socket=candidate_socket,
                opponent_socket=opponent_socket,
                opponents=opponents,
                counts=counts,
                engine_pool_size=1,
            )
            config = dataclasses.replace(
                config,
                candidate_inference_device="cuda:0",
                opponent_inference_device="cuda:0",
                worker_timeout_seconds=900.0,
                share_policy_inference_server=False,
                keep_temp=games != 256,
                **(
                    {}
                    if games == 256
                    else {
                        "opponent_pool_id": "0040_u270_value_search_v0_diagnostic",
                        "opponent_schedule_id": f"diagnostic-first-slot-{games}",
                    }
                ),
            )
            with _hook_environment(mode == "on", log_root):
                result = run_batch(config)
    wall_seconds = time.perf_counter() - started
    rows = list(result.report_data.games)
    wins = sum(row.get("winner") == 0 for row in rows)
    losses = sum(row.get("winner") == 1 for row in rows)
    draws = len(rows) - wins - losses
    errors = sum(
        row.get("status") != "finished" or bool(row.get("error_kind")) for row in rows
    )
    summary = {
        "schema_version": "0040_u270_value_search_v0_cpu_result_v1",
        "mode": mode,
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(rows),
        "errors": errors,
        "error_details": [
            {
                "game_id": item.get("game_id"),
                "error_kind": item.get("error_kind"),
                "error": item.get("error"),
            }
            for item in result.game_records
            if item.get("error_kind") or item.get("status") != "finished"
        ],
        "wall_seconds": wall_seconds,
        "evaluation_seed": config.seed,
        "opponent_pool_id": config.opponent_pool_id,
        "opponent_schedule_id": config.opponent_schedule_id,
        "checkpoint_update": manifest["checkpoint_update"],
        "rl_checkpoint_sha256": manifest["rl_checkpoint_sha256"],
        "portable_checkpoint_sha256": manifest["portable_checkpoint_sha256"],
        "deployment_effective_sha256": manifest["deployment_effective_sha256"],
        "package_manifest_sha256": _sha256(package_root / "manifest.json"),
        "report": str(result.report_path.relative_to(ROOT)),
        "run_id": result.run_id,
    }
    (output / "games.json").write_text(
        json.dumps({"summary": summary, "entries": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    run(mode=args.mode, games=args.games, output=args.output, package_root=args.package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
