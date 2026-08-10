"""Record why aggregate paired games do not satisfy the lockstep Gate D."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_summary(
    *, aggregate: Mapping[str, Any], gate_c: Mapping[str, Any]
) -> dict[str, Any]:
    summary = aggregate["summary"]
    return {
        "schema_version": "0038_gate_d_end_to_end_v1",
        "gate": "D",
        "status": "NOT_REACHED",
        "reason": (
            "Gate A and Gate C already fail.  Existing 256 paired outcomes were not "
            "instrumented with a per-decision CUDA/package lockstep trace, so they cannot "
            "locate the first strategic divergence and are not promoted to Gate D evidence."
        ),
        "existing_aggregate_context_only": {
            "games": summary["games"],
            "official_package_wins": summary["wins"],
            "official_package_losses": summary["losses"],
            "official_package_win_rate": summary["win_rate"],
            "cuda_same_seed_wins": summary["rl_cuda_same_seed_wins"],
            "cuda_same_seed_losses": summary["rl_cuda_same_seed_losses"],
            "cuda_same_seed_win_rate": summary["rl_cuda_same_seed_win_rate"],
            "paired_outcome_agreement": summary["paired_outcome_agreement"],
            "paired_outcome_agreement_rate": summary[
                "paired_outcome_agreement_rate"
            ],
            "cuda_loss_to_official_win": summary["rl_loss_to_cpu_win"],
            "cuda_win_to_official_loss": summary["rl_win_to_cpu_loss"],
            "errors": summary["errors"],
            "unfinished": summary["unfinished"],
        },
        "earliest_available_fixed_snapshot_divergence": gate_c.get(
            "first_divergence"
        ),
        "earliest_available_greedy_divergence": (
            gate_c.get("policy") or {}
        ).get("first_greedy_divergence"),
        "required_before_rerun": [
            "Gate A rule continuation parity",
            "Gate C exact observation tensor parity",
            "per-decision first-divergence recorder",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--gate-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    aggregate_path = args.aggregate.resolve()
    gate_c_path = args.gate_c.resolve()
    result = build_summary(
        aggregate=json.loads(aggregate_path.read_text(encoding="utf-8")),
        gate_c=json.loads(gate_c_path.read_text(encoding="utf-8")),
    )
    result["sources"] = {
        "aggregate": str(aggregate_path),
        "aggregate_sha256": _sha256(aggregate_path),
        "gate_c": str(gate_c_path),
        "gate_c_sha256": _sha256(gate_c_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_summary"]
