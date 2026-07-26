"""Deterministic version-local and project-level checkpoint selection."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_REQUIRED_GATES=("legality","invariance","counterfactual","numerical","runtime")


def select_checkpoint(candidates: Sequence[Mapping[str,Any]], *, composite_tolerance: float=1e-9) -> dict[str,Any]:
    eligible=[];rejected=[]
    for value in candidates:
        candidate=dict(value);gates=candidate.get("gates",{})
        reasons=[gate for gate in _REQUIRED_GATES if not isinstance(gates,Mapping) or gates.get(gate) is not True]
        if reasons:rejected.append({"candidate":candidate,"reasons":reasons})
        else:eligible.append(candidate)
    eligible.sort(key=lambda x:(-round(float(x["decision_quality_composite"])/composite_tolerance) if composite_tolerance else -float(x["decision_quality_composite"]),-float(x["validation_exact"]),float(x["validation_loss"]),int(x["epoch"]),str(x["sha256"])))
    return {"schema_version":"checkpoint_selection_v1","selected":eligible[0] if eligible else None,"ranking":eligible,"rejected":rejected,"state":"selected" if eligible else "no_eligible_checkpoint"}


__all__=["select_checkpoint"]
