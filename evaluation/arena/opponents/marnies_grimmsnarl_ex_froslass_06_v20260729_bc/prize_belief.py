"""Deterministic prize-card belief tracking for PTCG replay observations.

The ledger deliberately uses only information available to the acting player:

* the registered 60-card list;
* that player's visible zones and dynamic counts;
* a complete ``select.deck`` snapshot when the engine exposes one; and
* the number of face-down prize cards that remain.

Before a complete deck snapshot, per-card prize counts follow a hypergeometric
posterior.  A valid full-deck snapshot collapses that posterior to an exact
count by 60-card conservation.  If a face-down prize is subsequently taken and
its identity cannot be isolated from the next observation, the exact posterior
is thinned with the corresponding hypergeometric transition rather than being
silently treated as exact.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


PRIZE_FEATURE_NAMES = (
    "registered_count",
    "visible_nonprize_count",
    "hidden_pool_count",
    "expected_deck_count",
    "expected_prize_count",
    "prize_min_count",
    "prize_max_count",
    "probability_at_least_one_prized",
    "expected_accessible_count",
    "effective_prize_count",
    "exact_evidence",
    "exact_evidence_age",
)

SOURCE_CODES = {
    "hypergeometric": 0,
    "exact_snapshot": 1,
    "retained_exact": 2,
    "thinned_snapshot": 3,
    "empty": 4,
}


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def card_id(card: Any) -> int:
    if isinstance(card, (int, float)) and not isinstance(card, bool):
        return max(0, _as_int(card))
    return max(0, _as_int(_get(card, "id", _get(card, "cardId", 0))))


def _normalise_pmf(values: Iterable[float], size: int) -> tuple[float, ...]:
    result = [max(0.0, float(value)) for value in values]
    if len(result) < size:
        result.extend([0.0] * (size - len(result)))
    elif len(result) > size:
        result = result[:size]
    total = sum(result)
    if total <= 0.0:
        result = [1.0] + [0.0] * (size - 1)
    else:
        result = [value / total for value in result]
    return tuple(result)


def hypergeometric_pmf(population: int, successes: int, draws: int) -> tuple[float, ...]:
    """Return P(X=j) for successes drawn without replacement.

    The returned tuple has ``successes + 1`` entries.  Impossible states are
    represented by zeroes, which keeps every card ID's tensor width implicit
    in its registered multiplicity rather than a global maximum.
    """

    population = max(0, int(population))
    successes = min(max(0, int(successes)), population)
    draws = min(max(0, int(draws)), population)
    denominator = math.comb(population, draws) if population else 1
    values: list[float] = []
    for count in range(successes + 1):
        failures_drawn = draws - count
        if count > draws or failures_drawn < 0 or failures_drawn > population - successes:
            values.append(0.0)
            continue
        values.append(
            math.comb(successes, count)
            * math.comb(population - successes, failures_drawn)
            / denominator
        )
    return _normalise_pmf(values, successes + 1)


def thin_prize_pmf(
    pmf: Iterable[float],
    old_prize_count: int,
    new_prize_count: int,
) -> tuple[float, ...]:
    """Condition a prize-count PMF after unknown face-down prizes are removed."""

    source = tuple(max(0.0, float(value)) for value in pmf)
    size = max(1, len(source))
    old_prize_count = max(0, int(old_prize_count))
    new_prize_count = min(max(0, int(new_prize_count)), old_prize_count)
    if new_prize_count == old_prize_count:
        return _normalise_pmf(source, size)
    result = [0.0] * size
    for old_successes, probability in enumerate(source):
        if probability <= 0.0 or old_successes > old_prize_count:
            continue
        transition = hypergeometric_pmf(old_prize_count, old_successes, new_prize_count)
        for remaining, conditional in enumerate(transition):
            result[remaining] += probability * conditional
    return _normalise_pmf(result, size)


@dataclass(frozen=True)
class PrizeCardBelief:
    card_id: int
    registered_count: int
    visible_nonprize_count: int
    hidden_pool_count: int
    pmf: tuple[float, ...]
    source: str
    exact_age: int

    @property
    def expected_prize_count(self) -> float:
        return sum(index * probability for index, probability in enumerate(self.pmf))

    @property
    def prize_min_count(self) -> int:
        return next((index for index, probability in enumerate(self.pmf) if probability > 1e-9), 0)

    @property
    def prize_max_count(self) -> int:
        return max(
            (index for index, probability in enumerate(self.pmf) if probability > 1e-9),
            default=0,
        )

    @property
    def probability_at_least_one(self) -> float:
        return max(0.0, min(1.0, 1.0 - (self.pmf[0] if self.pmf else 1.0)))

    @property
    def exact(self) -> bool:
        return sum(probability > 1e-9 for probability in self.pmf) == 1


@dataclass(frozen=True)
class PrizeLedgerObservation:
    card_ids: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    beliefs: tuple[PrizeCardBelief, ...]
    source: str
    source_code: int
    effective_prize_count: int
    actual_prize_count: int
    deck_count: int
    visible_nonprize_count: int
    expected_visible_nonprize_count: int
    visible_count_gap: int
    snapshot_valid: bool
    exact: bool


class PrizeLedger:
    """Cross-decision posterior over the acting player's prize-card multiset."""

    def __init__(self, registered_deck: Iterable[int]):
        self.registered_deck = tuple(int(value) for value in registered_deck)
        if len(self.registered_deck) != 60:
            raise ValueError(f"registered deck must contain 60 cards, got {len(self.registered_deck)}")
        if any(value <= 0 for value in self.registered_deck):
            raise ValueError("registered deck card IDs must be positive")
        self.registered = Counter(self.registered_deck)
        self.card_ids = tuple(sorted(self.registered))
        self.reset()

    def reset(self) -> None:
        self._pmfs: dict[int, tuple[float, ...]] = {}
        self._source = "hypergeometric"
        self._effective_prize_count: int | None = None
        self._exact_age = 0
        self._observations = 0

    @staticmethod
    def _actor_and_player(obs: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]:
        current = _get(obs, "current", {}) or {}
        select = _get(obs, "select", {}) or {}
        actor = _as_int(_get(current, "yourIndex", -1), -1)
        players = _as_list(_get(current, "players", []))
        own = players[actor] if actor in (0, 1) and actor < len(players) and isinstance(players[actor], dict) else {}
        return actor, own, current, select

    def _visible_nonprize(
        self,
        actor: int,
        own: dict[str, Any],
        current: dict[str, Any],
        select: dict[str, Any],
        expected_count: int,
    ) -> Counter[int]:
        counts: Counter[int] = Counter()
        serials: set[tuple[int, int]] = set()

        def add(card: Any, *, recurse: bool = True) -> bool:
            cid = card_id(card)
            if cid not in self.registered:
                return False
            owner = _as_int(_get(card, "playerIndex", actor), actor)
            if owner not in (-1, actor):
                return False
            serial_raw = _get(card, "serial", None)
            serial = _as_int(serial_raw, -1)
            key = (owner, serial)
            if serial >= 0 and key in serials:
                return False
            if counts[cid] >= self.registered[cid]:
                return False
            if serial >= 0:
                serials.add(key)
            counts[cid] += 1
            if recurse and isinstance(card, dict):
                for child_key in ("energyCards", "tools", "preEvolution"):
                    for child in _as_list(card.get(child_key)):
                        add(child, recurse=True)
            return True

        for zone in ("active", "bench", "hand", "discard", "lostZone", "lost", "removed"):
            for card in _as_list(own.get(zone)):
                add(card)
        for card in _as_list(current.get("stadium")):
            if _as_int(_get(card, "playerIndex", -1), -1) == actor:
                add(card)
        for card in _as_list(current.get("looking")):
            add(card)

        # A Trainer or Energy resolving an effect may temporarily be in neither
        # hand nor discard.  Add transient sources only until 60-card
        # conservation is satisfied; serial de-duplication prevents counting an
        # attached Energy both as an attachment and as select.effect.
        if sum(counts.values()) < expected_count:
            transient: list[Any] = []
            for key in ("effect", "contextCard"):
                value = select.get(key)
                if value is not None:
                    transient.append(value)
            for card in transient:
                if sum(counts.values()) >= expected_count:
                    break
                add(card)
        return counts

    @staticmethod
    def _effective_prizes(current: dict[str, Any], actual: int, deck_count: int) -> int:
        turn = _as_int(current.get("turn"), -1)
        # During setup the six prizes have not yet moved out of the 53-card
        # remainder, but the future prize sample is already the relevant belief.
        if actual == 0 and turn == 0 and deck_count >= 53:
            return 6
        return min(max(actual, 0), 6)

    def _snapshot_prize_counts(
        self,
        select: dict[str, Any],
        deck_count: int,
        visible: Counter[int],
        prize_count: int,
    ) -> Counter[int] | None:
        raw_snapshot = _as_list(select.get("deck"))
        if not raw_snapshot or len(raw_snapshot) != deck_count:
            return None
        snapshot = Counter(card_id(card) for card in raw_snapshot)
        if 0 in snapshot or any(cid not in self.registered for cid in snapshot):
            return None
        result: Counter[int] = Counter()
        for cid in self.card_ids:
            count = self.registered[cid] - visible[cid] - snapshot[cid]
            if count < 0:
                return None
            result[cid] = count
        if sum(result.values()) != prize_count:
            return None
        return result

    def _hypergeometric_update(
        self,
        visible: Counter[int],
        effective_prizes: int,
        deck_count: int,
    ) -> None:
        hidden = {cid: max(0, self.registered[cid] - visible[cid]) for cid in self.card_ids}
        expected_pool = max(0, deck_count + effective_prizes)
        observed_pool = sum(hidden.values())
        population = expected_pool if observed_pool == expected_pool else observed_pool
        draws = min(effective_prizes, population)
        self._pmfs = {
            cid: hypergeometric_pmf(population, min(hidden[cid], population), draws)
            for cid in self.card_ids
        }
        self._source = "empty" if draws == 0 else "hypergeometric"
        self._exact_age = 0

    def _thin_update(self, new_prize_count: int) -> None:
        old_prize_count = int(self._effective_prize_count or 0)
        self._pmfs = {
            cid: thin_prize_pmf(pmf, old_prize_count, new_prize_count)
            for cid, pmf in self._pmfs.items()
        }
        self._source = "thinned_snapshot"
        self._exact_age += 1

    def observe(self, obs: dict[str, Any]) -> PrizeLedgerObservation:
        actor, own, current, select = self._actor_and_player(obs)
        if actor not in (0, 1) or not own:
            raise ValueError("observation does not contain a valid acting player")
        deck_count = min(max(_as_int(own.get("deckCount"), 0), 0), 60)
        actual_prizes = min(len(_as_list(own.get("prize"))), 6)
        effective_prizes = self._effective_prizes(current, actual_prizes, deck_count)
        expected_visible = max(0, 60 - deck_count - actual_prizes)
        visible = self._visible_nonprize(actor, own, current, select, expected_visible)
        visible_total = sum(visible.values())
        snapshot = self._snapshot_prize_counts(
            select,
            deck_count,
            visible,
            effective_prizes,
        )

        if snapshot is not None:
            self._pmfs = {
                cid: tuple(1.0 if index == snapshot[cid] else 0.0 for index in range(self.registered[cid] + 1))
                for cid in self.card_ids
            }
            self._source = "exact_snapshot"
            self._exact_age = 0
        elif not self._pmfs:
            self._hypergeometric_update(visible, effective_prizes, deck_count)
        elif self._effective_prize_count is not None and effective_prizes < self._effective_prize_count:
            self._thin_update(effective_prizes)
        elif self._effective_prize_count is not None and effective_prizes > self._effective_prize_count:
            self._hypergeometric_update(visible, effective_prizes, deck_count)
        elif self._source in ("exact_snapshot", "retained_exact"):
            self._source = "retained_exact"
            self._exact_age += 1
        elif self._source == "thinned_snapshot":
            self._exact_age += 1
        else:
            self._hypergeometric_update(visible, effective_prizes, deck_count)

        self._effective_prize_count = effective_prizes
        self._observations += 1
        beliefs: list[PrizeCardBelief] = []
        features: list[tuple[float, ...]] = []
        all_exact = True
        for cid in self.card_ids:
            registered = self.registered[cid]
            visible_count = min(visible[cid], registered)
            hidden_count = max(0, registered - visible_count)
            belief = PrizeCardBelief(
                card_id=cid,
                registered_count=registered,
                visible_nonprize_count=visible_count,
                hidden_pool_count=hidden_count,
                pmf=_normalise_pmf(self._pmfs.get(cid, (1.0,)), registered + 1),
                source=self._source,
                exact_age=self._exact_age,
            )
            beliefs.append(belief)
            all_exact &= belief.exact
            expected_prize = belief.expected_prize_count
            expected_deck = max(0.0, hidden_count - expected_prize)
            accessible = max(0.0, registered - expected_prize)
            features.append(
                (
                    registered / 4.0,
                    visible_count / 4.0,
                    hidden_count / 4.0,
                    expected_deck / 4.0,
                    expected_prize / 4.0,
                    belief.prize_min_count / 4.0,
                    belief.prize_max_count / 4.0,
                    belief.probability_at_least_one,
                    accessible / 4.0,
                    effective_prizes / 6.0,
                    1.0 if belief.exact else 0.0,
                    min(self._exact_age, 20) / 20.0,
                )
            )
        return PrizeLedgerObservation(
            card_ids=self.card_ids,
            features=tuple(features),
            beliefs=tuple(beliefs),
            source=self._source,
            source_code=SOURCE_CODES[self._source],
            effective_prize_count=effective_prizes,
            actual_prize_count=actual_prizes,
            deck_count=deck_count,
            visible_nonprize_count=visible_total,
            expected_visible_nonprize_count=expected_visible,
            visible_count_gap=expected_visible - visible_total,
            snapshot_valid=snapshot is not None,
            exact=all_exact,
        )


__all__ = [
    "PRIZE_FEATURE_NAMES",
    "SOURCE_CODES",
    "PrizeCardBelief",
    "PrizeLedger",
    "PrizeLedgerObservation",
    "card_id",
    "hypergeometric_pmf",
    "thin_prize_pmf",
]
