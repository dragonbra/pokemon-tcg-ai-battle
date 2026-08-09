"""Condense fixed-snapshot CPU/CUDA tensor and policy parity evidence.

The full scaffold report intentionally retains the first mismatching tensors and
their causal-history context.  Release gating should consume this smaller
record so that it cannot accidentally hide an input mismatch behind aggregate
logit agreement.
"""

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


def summarize(report: Mapping[str, Any]) -> dict[str, Any]:
    details = report.get("tensor_mismatch_details") or []
    first_tensor = min(
        (row for row in details if isinstance(row, Mapping)),
        key=lambda row: int(row.get("decision", 1 << 30)),
        default=None,
    )
    model = report.get("model") or {}
    value = report.get("value_model") or {}
    mismatch_counts = dict(report.get("tensor_mismatch_counts") or {})
    inputs_equal = not mismatch_counts and int(
        report.get("fixed_action_cuda_state_errors", 0)
    ) == 0
    model_equal = bool(model.get("passed"))
    status = "PASS" if inputs_equal and model_equal else "FAIL"

    first_divergence: dict[str, Any] | None = None
    if first_tensor is not None:
        first_divergence = {
            "category": "preprocessing/state",
            "stage": "observation_tensor",
            "engine_decision": int(first_tensor["decision"]),
            "field": first_tensor.get("field"),
            "flat_index": first_tensor.get("flat_index"),
            "official": first_tensor.get("expected"),
            "cuda": first_tensor.get("actual"),
        }
    elif model.get("first_failure"):
        first_divergence = {
            "category": "numeric/backend",
            "stage": "root_logits",
            **dict(model["first_failure"]),
        }

    return {
        "schema_version": "0038_gate_c_fixed_snapshot_v1",
        "gate": "C",
        "status": status,
        "case": report.get("case"),
        "decisions_replayed": report.get("decisions_replayed"),
        "focal_decisions": report.get("model_focal_decisions"),
        "fixed_primitive_state_errors": report.get("fixed_action_cuda_state_errors"),
        "legal_option_mask_mismatches": mismatch_counts.get("option_mask", 0),
        "first_divergence": first_divergence,
        "tensor_mismatch_counts": mismatch_counts,
        "policy": {
            "root_logits_tolerance_failures": model.get("root_logits_tolerance_failures"),
            "max_abs_logit_error": model.get("max_root_logit_absolute_error"),
            "mean_abs_logit_error": model.get("mean_root_logit_absolute_error"),
            "top1_divergences": model.get("first_step_top1_divergences"),
            "top2_set_divergences": model.get("first_step_top2_set_divergences"),
            "greedy_action_divergences": model.get("greedy_action_divergences"),
            "first_greedy_divergence": model.get("first_greedy_divergence"),
            "minimum_official_top1_top2_margin": model.get(
                "minimum_cpu_top1_top2_margin"
            ),
            "minimum_cuda_top1_top2_margin": model.get(
                "minimum_cuda_top1_top2_margin"
            ),
        },
        "value": {
            "deployed_in_package": model.get("value") != "not_deployed_in_current_candidate",
            "decisions": value.get("decisions"),
            "max_abs_error": value.get("max_value_absolute_error"),
            "mean_abs_error": value.get("mean_value_absolute_error"),
            "sign_divergences": value.get("value_sign_divergences"),
            "first_divergence": value.get("first_divergence"),
        },
        "numeric_only_comparison_reached": inputs_equal,
        "reason": (
            None if status == "PASS" else
            "exact categorical/mask inputs differ; numeric-backend attribution is not valid yet"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.input.resolve()
    result = summarize(json.loads(source.read_text(encoding="utf-8")))
    result["source"] = str(source)
    result["source_sha256"] = _sha256(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["summarize"]
