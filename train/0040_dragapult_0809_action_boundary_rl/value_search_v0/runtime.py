"""Official-CPU worker-side Search lifecycle for the U270 V0 experiment."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .eligibility import CandidateGroup, policy_conditioned_group
from .hidden import build_nonprivileged_hypothesis


FORBIDDEN_LOG_TYPES = frozenset({0, 4, 5, 22})  # shuffle, draw, hidden draw, coin
HIDDEN_AREAS = frozenset({1, 6})  # deck, prize


def _pokemon_summary(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": card.get("id"),
        "serial": card.get("serial"),
        "hp": card.get("hp"),
        "max_hp": card.get("maxHp"),
        "energy_cards": [
            item.get("id") for item in card.get("energyCards") or ()
            if isinstance(item, Mapping)
        ],
        "energies": list(card.get("energies") or ()),
        "tools": [
            item.get("id") for item in card.get("tools") or ()
            if isinstance(item, Mapping)
        ],
    }


def _visible_board_summary(observation: Mapping[str, Any]) -> dict[str, Any]:
    current = observation.get("current") or {}
    focal = current.get("yourIndex")
    players = current.get("players") or ()
    summaries: list[dict[str, Any]] = []
    for index, player in enumerate(players):
        if not isinstance(player, Mapping):
            continue
        summary = {
            "player": index,
            "active": [
                _pokemon_summary(card) for card in player.get("active") or ()
                if isinstance(card, Mapping)
            ],
            "bench": [
                _pokemon_summary(card) for card in player.get("bench") or ()
                if isinstance(card, Mapping)
            ],
            "deck_count": player.get("deckCount"),
            "hand_count": player.get("handCount"),
            "prize_count": len(player.get("prize") or ()),
            "discard": [
                card.get("id") for card in player.get("discard") or ()
                if isinstance(card, Mapping)
            ],
        }
        if index == focal:
            summary["focal_hand"] = [
                card.get("id") for card in player.get("hand") or ()
                if isinstance(card, Mapping)
            ]
        summaries.append(summary)
    return {"players": summaries}


def _candidate_option_payloads(
    observation: Mapping[str, Any], group: CandidateGroup
) -> list[list[Any]]:
    options = (observation.get("select") or {}).get("option") or ()
    return [
        [options[index] if 0 <= index < len(options) else None for index in selection]
        for selection in group.selections
    ]


def select_value_winner(
    group: CandidateGroup, values: Sequence[float]
) -> tuple[list[int], float, float]:
    if len(values) != len(group.selections):
        raise ValueError("candidate Value count does not match official selections")
    policy_position = group.selections.index(group.policy_selection)
    best_value = max(values)
    winners = [index for index, value in enumerate(values) if value == best_value]
    winner_position = policy_position if policy_position in winners else winners[0]
    return (
        list(group.selections[winner_position]),
        float(values[policy_position]),
        float(best_value),
    )


def _active_player(current: Mapping[str, Any]) -> int | None:
    turn = current.get("turn")
    first = current.get("firstPlayer")
    if (
        isinstance(turn, bool)
        or not isinstance(turn, int)
        or turn <= 0
        or first not in (0, 1)
    ):
        return None
    return int(first) if turn % 2 else 1 - int(first)


def _branch_reject_reason(
    root: Mapping[str, Any], branch: Mapping[str, Any], focal: int
) -> str | None:
    root_current = root.get("current")
    current = branch.get("current")
    select = branch.get("select")
    if not isinstance(root_current, Mapping) or not isinstance(current, Mapping):
        return "MISSING_CURRENT_STATE"
    if current.get("result") != -1:
        return "TERMINAL_BRANCH"
    if not isinstance(select, Mapping) or current.get("yourIndex") != focal:
        return "PERSPECTIVE_FLIP"
    if (
        current.get("turn") != root_current.get("turn")
        or _active_player(current) != focal
        or _active_player(root_current) != focal
    ):
        return "TURN_CHANGE"
    for log in branch.get("logs") or ():
        if not isinstance(log, Mapping):
            return "INVALID_BRANCH_LOG"
        if log.get("type") in FORBIDDEN_LOG_TYPES:
            return "RNG_OR_DRAW_LOG"
        if log.get("type") in {6, 7} and (
            log.get("fromArea") in HIDDEN_AREAS or log.get("toArea") in HIDDEN_AREAS
        ):
            return "HIDDEN_ZONE_TRANSITION"
    return None


class WorkerValueSearchV0:
    def __init__(
        self,
        *,
        game_id: str,
        registered_deck: Sequence[int],
        candidate_agent: Any,
        log_root: Path,
    ) -> None:
        self.game_id = game_id
        self.registered_deck = tuple(int(card_id) for card_id in registered_deck)
        self.candidate_agent = candidate_agent
        self.log_path = log_root / f"{game_id}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        from cg.api import all_card_data

        self.basic_ids = frozenset(
            int(card.cardId) for card in all_card_data() if bool(card.basic)
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def _base_record(
        self,
        observation: Mapping[str, Any],
        policy_selection: Sequence[int],
        step: int,
    ) -> dict[str, Any]:
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        return {
            "schema_version": "0040_u270_value_search_v0_decision_v1",
            "game_id": self.game_id,
            "step": step,
            "turn": current.get("turn"),
            "focal_player": current.get("yourIndex"),
            "select_type": select.get("type"),
            "select_context": select.get("context"),
            "effect_id": (
                select.get("effect", {}).get("id")
                if isinstance(select.get("effect"), Mapping) else None
            ),
            "policy_selection": list(policy_selection),
        }

    def rerank(
        self,
        *,
        game_module: Any,
        observation: dict[str, Any],
        policy_selection: list[int],
        step: int,
    ) -> tuple[list[int], dict[str, Any]]:
        started = time.perf_counter()
        record = self._base_record(observation, policy_selection, step)
        record.update({"eligible": False, "override": False})
        try:
            group, reason = policy_conditioned_group(
                observation,
                policy_selection,
                is_basic_card=self.basic_ids.__contains__,
            )
            record["eligibility_reason"] = reason
            if group is None:
                return self._finish(record, policy_selection, started)
            record.update(
                {
                    "decision_family": group.family.value,
                    "candidate_selections": [list(item) for item in group.selections],
                    "candidate_option_payloads": _candidate_option_payloads(
                        observation, group
                    ),
                    "candidate_count": len(group.selections),
                    "board_summary": _visible_board_summary(observation),
                }
            )
            result = self._search_and_value(observation, group)
            record.update(result)
            if result.get("fallback_reason") is not None:
                return self._finish(record, policy_selection, started)
            values = result["candidate_values"]
            if any(not math.isfinite(value) for value in values):
                record["fallback_reason"] = "VALUE_NONFINITE"
                return self._finish(record, policy_selection, started)
            winner, policy_value, best_value = select_value_winner(group, values)
            record.update(
                {
                    "eligible": True,
                    "search_winner": winner,
                    "policy_candidate_value": policy_value,
                    "best_value": best_value,
                    "value_margin": best_value - policy_value,
                    "override": winner != list(policy_selection),
                }
            )
            return self._finish(record, winner, started)
        except BaseException as exc:
            record["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return self._finish(record, policy_selection, started)

    def _search_and_value(
        self, observation: dict[str, Any], group: CandidateGroup
    ) -> dict[str, Any]:
        from cg.api import (
            search_begin,
            search_end,
            search_release,
            search_step,
            to_observation_class,
        )

        focal = int(observation["current"]["yourIndex"])
        hypothesis = build_nonprivileged_hypothesis(observation, self.registered_deck)
        root = None
        branch_ids: list[int] = []
        branch_observations: list[dict[str, Any]] = []
        search_started = time.perf_counter()
        try:
            root = search_begin(
                to_observation_class(observation),
                list(hypothesis.your_deck),
                list(hypothesis.your_prize),
                list(hypothesis.opponent_deck),
                list(hypothesis.opponent_prize),
                list(hypothesis.opponent_hand),
                list(hypothesis.opponent_active),
                False,
            )
            for selection in group.selections:
                branch = search_step(root.searchId, list(selection))
                branch_ids.append(int(branch.searchId))
                branch_observation = dataclasses.asdict(branch.observation)
                reason = _branch_reject_reason(observation, branch_observation, focal)
                if reason is not None:
                    return {
                        "fallback_reason": reason,
                        "search_seconds": time.perf_counter() - search_started,
                    }
                branch_observations.append(branch_observation)
            search_seconds = time.perf_counter() - search_started
            value_started = time.perf_counter()
            values = self.candidate_agent.value_search_v0_values(branch_observations)
            value_seconds = time.perf_counter() - value_started
            return {
                "fallback_reason": None,
                "candidate_values": values,
                "branch_metadata": [
                    {
                        "turn": item["current"].get("turn"),
                        "select_player": item["current"].get("yourIndex"),
                        "select_type": (item.get("select") or {}).get("type"),
                        "select_context": (item.get("select") or {}).get("context"),
                        "terminal": item["current"].get("result") != -1,
                    }
                    for item in branch_observations
                ],
                "search_seconds": search_seconds,
                "value_seconds": value_seconds,
            }
        finally:
            if root is not None:
                for search_id in reversed(branch_ids):
                    search_release(search_id)
                search_release(int(root.searchId))
                search_end()

    def _finish(
        self, record: dict[str, Any], action: list[int], started: float
    ) -> tuple[list[int], dict[str, Any]]:
        record["total_seconds"] = time.perf_counter() - started
        record["executed_selection"] = list(action)
        self._write(record)
        return list(action), record


def create_worker_runtime(*, request: Any, candidate_agent: Any) -> WorkerValueSearchV0:
    log_root = Path(
        os.environ.get(
            "EVALUATION_VALUE_SEARCH_V0_LOG_ROOT",
            ".tmp/evaluation/0040_u270_value_search_v0/search_logs",
        )
    )
    return WorkerValueSearchV0(
        game_id=str(request.game_id),
        registered_deck=request.candidate.deck,
        candidate_agent=candidate_agent,
        log_root=log_root,
    )


__all__ = ["WorkerValueSearchV0", "create_worker_runtime"]
