"""Conservative classification before tensor compilation or policy inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import factorial
from typing import Any


class DecisionClass(StrEnum):
    TERMINAL = "TERMINAL"
    LEGAL_EMPTY_PASS = "LEGAL_EMPTY_PASS"
    MASK_ERROR = "MASK_ERROR"
    FORCED = "FORCED"
    STRATEGIC = "STRATEGIC"
    CHANCE_BOUNDARY = "CHANCE_BOUNDARY"
    INFORMATION_BOUNDARY = "INFORMATION_BOUNDARY"
    PRIORITY_TRANSFER = "PRIORITY_TRANSFER"


@dataclass(frozen=True, slots=True)
class DecisionGateResult:
    classification: DecisionClass
    raw_option_count: int
    legal_completion_count: int
    canonical_completion_count: int
    stop_legal: bool
    cancel_legal: bool
    decline_legal: bool
    pass_legal: bool
    forced_action: tuple[int, ...] | None = None
    boundary: DecisionClass | None = None
    reason: str = ""

    @property
    def calls_policy(self) -> bool:
        return self.classification is DecisionClass.STRATEGIC


def _ordered_completion_count(n: int, minimum: int, maximum: int) -> int:
    return sum(factorial(n) // factorial(n - size) for size in range(minimum, maximum + 1))


def _token(value: Any) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "")


def _explicit_controls(options: Sequence[Any]) -> tuple[bool, bool, bool]:
    cancel = decline = passed = False
    for option in options:
        if not isinstance(option, Mapping):
            continue
        values = {_token(option.get(key)) for key in ("type", "name", "label", "context")}
        option_type = option.get("type")
        cancel |= bool(values & {"cancel", "abort"})
        decline |= option_type == 2 or bool(values & {"no", "decline", "refuse"})
        passed |= option_type == 14 or bool(values & {"pass", "end", "done", "skip"})
    return cancel, decline, passed


class DecisionGate:
    """Fail closed unless a callback is provably a single mandatory completion."""

    def classify(
        self,
        observation: Mapping[str, Any],
        *,
        canonical_completion_count: int | None = None,
        boundary: DecisionClass | None = None,
    ) -> DecisionGateResult:
        current = observation.get("current")
        select = observation.get("select")
        if isinstance(current, Mapping):
            result = current.get("result")
            if isinstance(result, int) and not isinstance(result, bool) and result >= 0:
                return DecisionGateResult(
                    DecisionClass.TERMINAL, 0, 0, 0, False, False, False, False,
                    boundary=boundary, reason="official current.result is terminal",
                )
        if not isinstance(select, Mapping):
            return DecisionGateResult(
                DecisionClass.MASK_ERROR, 0, 0, 0, False, False, False, False,
                boundary=boundary, reason="non-terminal observation has no select payload",
            )
        raw_options = select.get("option")
        if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
            return DecisionGateResult(
                DecisionClass.MASK_ERROR, 0, 0, 0, False, False, False, False,
                boundary=boundary, reason="select.option is not a sequence",
            )
        options = list(raw_options)
        minimum = select.get("minCount", 0)
        maximum = select.get("maxCount", len(options))
        if (
            isinstance(minimum, bool) or not isinstance(minimum, int)
            or isinstance(maximum, bool) or not isinstance(maximum, int)
            or minimum < 0 or maximum < minimum or maximum > len(options)
        ):
            return DecisionGateResult(
                DecisionClass.MASK_ERROR, len(options), 0, 0, False, False, False, False,
                boundary=boundary, reason="invalid minCount/maxCount",
            )
        stop = minimum < maximum
        cancel, decline, passed = _explicit_controls(options)
        legal = _ordered_completion_count(len(options), minimum, maximum)
        canonical = legal if canonical_completion_count is None else canonical_completion_count
        if canonical < 0:
            raise ValueError("canonical completion count cannot be negative")
        if not options:
            classification = (
                DecisionClass.LEGAL_EMPTY_PASS if minimum == maximum == 0
                else DecisionClass.MASK_ERROR
            )
            return DecisionGateResult(
                classification, 0, legal, canonical, stop, cancel, decline, True,
                forced_action=() if classification is DecisionClass.LEGAL_EMPTY_PASS else None,
                boundary=boundary,
                reason="empty optional selection" if classification is DecisionClass.LEGAL_EMPTY_PASS
                else "mandatory selection has an empty mask",
            )
        forced = canonical == 1 and not (stop or cancel or decline or passed)
        action = tuple(range(minimum)) if forced and legal == 1 else None
        return DecisionGateResult(
            DecisionClass.FORCED if forced else DecisionClass.STRATEGIC,
            len(options), legal, canonical, stop, cancel, decline, passed,
            forced_action=action, boundary=boundary,
            reason="one mandatory canonical completion" if forced else "policy choice remains",
        )


__all__ = ["DecisionClass", "DecisionGate", "DecisionGateResult"]
