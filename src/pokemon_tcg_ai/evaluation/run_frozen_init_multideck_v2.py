"""Run and progressively publish Frozen-0045-Init on seven focal decks."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

from .benchmark_v2_schedule import CONTRACT_ID
from .render_frozen_init_multideck_v2 import DECK_ORDER, publish
from .run_benchmark_v2 import run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    status_path = output_root / "status.json"
    status: dict[str, Any] = {
        "schema_version": "0045_frozen_init_multideck_benchmark_v2_status_v1",
        "state": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "current_deck": None,
        "deck_order": list(DECK_ORDER),
        "completed_decks": [],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "contract_id": CONTRACT_ID,
    }
    _write(status_path, status); publish(output_root)
    try:
        for deck_id in DECK_ORDER:
            status["current_deck"] = deck_id
            _write(status_path, status); publish(output_root)
            report = run(
                deck_id=deck_id,
                output_root=output_root / "reports" / deck_id,
                checkpoint=checkpoint,
                checkpoint_update=0,
            )
            status["completed_decks"].append(deck_id)
            status.setdefault("summaries", {})[deck_id] = report["summary"]
            status["last_completed_at"] = datetime.now(UTC).isoformat()
            _write(status_path, status); publish(output_root)
        status["state"] = "complete"
        status["current_deck"] = None
        status["completed_at"] = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": "0045_frozen_init_multideck_benchmark_v2_manifest_v1",
            "status": "PASS",
            "checkpoint_sha256": status["checkpoint_sha256"],
            "contract_id": CONTRACT_ID,
            "deck_order": list(DECK_ORDER),
            "reports": {
                deck_id: {
                    "path": f"reports/{deck_id}/report.json",
                    "sha256": _sha256(output_root / "reports" / deck_id / "report.json"),
                }
                for deck_id in DECK_ORDER
            },
        }
        _write(output_root / "manifest.json", manifest)
        _write(status_path, status); publish(output_root)
        return 0
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["traceback"] = traceback.format_exc()
        status["failed_at"] = datetime.now(UTC).isoformat()
        _write(status_path, status); publish(output_root)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
