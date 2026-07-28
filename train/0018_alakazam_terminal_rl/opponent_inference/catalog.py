from __future__ import annotations

from enum import StrEnum

from evaluation.packages.loader import SubmissionPackage


class OpponentInferenceKind(StrEnum):
    CPU = "cpu"
    SOURCE_CAUSAL = "source_causal"
    FULL_ACTION = "full_action"
    ID_ONLY = "id_only"


def classify_opponent(package: SubmissionPackage) -> OpponentInferenceKind:
    """Classify only package layouts whose inference contract is understood."""
    root = package.root
    strategy = root / "strategy"
    if (strategy / "model.bin").is_file():
        if (strategy / "portable_inference.py").is_file() or (
            strategy / "source_inference.py"
        ).is_file():
            return OpponentInferenceKind.SOURCE_CAUSAL
        if (strategy / "full_action_inference.py").is_file():
            return OpponentInferenceKind.FULL_ACTION
    if (
        (root / "policy.pt").is_file()
        and (root / "idonly_policy.py").is_file()
        and not (root / "policy_left.pt").exists()
        and not (root / "policy_right.pt").exists()
    ):
        return OpponentInferenceKind.ID_ONLY
    return OpponentInferenceKind.CPU


__all__ = ["OpponentInferenceKind", "classify_opponent"]
