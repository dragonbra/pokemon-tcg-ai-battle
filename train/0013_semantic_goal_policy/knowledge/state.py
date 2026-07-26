"""Shared offline/online causal knowledge state."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .ledger import IdentityLedgerEntry, KnowledgeStage, KnownCount, immutable_entries
from .visibility import ViewKind, classify_deck_view, classify_log


@dataclass(frozen=True, slots=True)
class KnownCard:
    card_id: int
    serial: int
    source_event: int


@dataclass(frozen=True, slots=True)
class OpponentHand:
    known: tuple[KnownCard, ...]
    unknown_slots: int


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    decision_index: int
    self_ledger: Mapping[int, IdentityLedgerEntry]
    opponent_hand: OpponentHand
    deck_membership_known: bool
    deck_order_known: bool
    recent_events: tuple[Mapping[str, Any], ...]


class CausalKnowledgeState:
    def __init__(self, actor: int, registered_deck: Sequence[int], event_window: int = 64) -> None:
        if actor < 0 or len(registered_deck) != 60:
            raise ValueError("invalid causal state registration")
        self.actor = actor
        self.initial = Counter(registered_deck)
        self.event_window = event_window
        self.decision_index = 0
        self._current: KnowledgeSnapshot | None = None
        self._pending_view: Counter[int] | None = None
        self._known_hand: dict[int, KnownCard] = {}
        self._unknown_hand = 0
        self._events: list[Mapping[str, Any]] = []
        self._deck_known = False
        self._deck_order = False
        self._pending_action: tuple[int, ...] | None = None

    @classmethod
    def new_game(cls, actor: int, registered_deck: Sequence[int], event_window: int = 64) -> CausalKnowledgeState:
        return cls(actor, registered_deck, event_window)

    def _visible_counts(self, observation: Mapping[str, Any]) -> dict[int, Counter[str]]:
        current = observation.get("current")
        if not isinstance(current, Mapping):
            raise ValueError("current must be a mapping")
        if current.get("yourIndex") != self.actor:
            raise ValueError("causal state actor mismatch")
        players = current.get("players")
        if not isinstance(players, Sequence) or self.actor >= len(players):
            raise ValueError("causal state players malformed")
        own = players[self.actor]
        if not isinstance(own, Mapping):
            raise ValueError("actor player state malformed")
        output: dict[int, Counter[str]] = {}
        for zone in ("active", "bench", "hand", "discard", "prize"):
            values = own.get(zone, [])
            if not isinstance(values, Sequence):
                continue
            for item in values:
                if isinstance(item, Mapping):
                    card_id = item.get("id")
                    if isinstance(card_id, int) and not isinstance(card_id, bool):
                        output.setdefault(card_id, Counter())[zone] += 1
        return output

    def _consume_log(self, log: Mapping[str, Any]) -> None:
        kind = classify_log(log)
        self._events.append(MappingProxyType(dict(log)))
        player = log.get("playerIndex")
        if kind == "hand_view" and player != self.actor:
            cards = log.get("cards", [])
            if not isinstance(cards, Sequence):
                raise ValueError("hand view cards malformed")
            self._known_hand.clear()
            for item in cards:
                if not isinstance(item, Mapping):
                    raise ValueError("hand view card malformed")
                card_id, serial = item.get("id"), item.get("serial")
                if not isinstance(card_id, int) or not isinstance(serial, int):
                    raise ValueError("hand view identity malformed")
                self._known_hand[serial] = KnownCard(card_id, serial, self.decision_index)
            self._unknown_hand = 0
        elif kind in {"move", "discard"} and player != self.actor and log.get("fromArea") == "hand":
            serial = log.get("serial")
            if isinstance(serial, int):
                self._known_hand.pop(serial, None)
            elif self._unknown_hand:
                self._unknown_hand -= 1
        elif kind == "draw" and player != self.actor:
            self._unknown_hand += 1
        elif kind == "shuffle":
            if player == self.actor and log.get("area", "deck") == "deck":
                self._deck_order = False
            if player != self.actor and log.get("area") in {"hand", "hand_to_deck"}:
                self._known_hand.clear()

    def consume(self, observation: Mapping[str, Any]) -> KnowledgeSnapshot:
        if self._pending_action is not None:
            self._pending_action = None
        visible = self._visible_counts(observation)
        current = observation["current"]
        players = current["players"]
        opponent = players[1 - self.actor] if len(players) == 2 else None
        logs = observation.get("logs", [])
        if not isinstance(logs, Sequence):
            raise ValueError("logs must be a sequence")
        for log in logs:
            if not isinstance(log, Mapping):
                raise ValueError("log must be a mapping")
            self._consume_log(log)
        if isinstance(opponent, Mapping):
            hand_count = opponent.get("handCount", 0)
            if not isinstance(hand_count, int) or isinstance(hand_count, bool) or hand_count < 0:
                raise ValueError("opponent hand count malformed")
            known_count = min(len(self._known_hand), hand_count)
            if known_count < len(self._known_hand):
                self._known_hand = dict(list(sorted(self._known_hand.items()))[:known_count])
            self._unknown_hand = max(self._unknown_hand, hand_count - len(self._known_hand))
            self._unknown_hand = hand_count - len(self._known_hand)
        entries: dict[int, IdentityLedgerEntry] = {}
        for card_id, initial in sorted(self.initial.items()):
            zones = visible.get(card_id, Counter())
            visible_total = sum(zones.values())
            if self._deck_known and self._pending_view is not None:
                deck_count = self._pending_view.get(card_id, 0)
                prize = initial - visible_total - deck_count
                if prize < 0:
                    raise ValueError("resource conservation violated")
                deck = KnownCount.exact(deck_count, KnowledgeStage.INFERRED_EXACT, self.decision_index)
                prize_count = KnownCount.exact(prize, KnowledgeStage.INFERRED_EXACT, self.decision_index)
            else:
                deck = KnownCount.unknown(initial)
                prize_count = KnownCount.unknown(initial)
            entries[card_id] = IdentityLedgerEntry(card_id, initial, MappingProxyType(dict(zones)), deck, prize_count)
        snapshot = KnowledgeSnapshot(self.decision_index, immutable_entries(entries), OpponentHand(tuple(sorted(self._known_hand.values(), key=lambda item: item.serial)), self._unknown_hand), self._deck_known, self._deck_order, tuple(self._events[-self.event_window:]))
        self._current = snapshot
        select = observation.get("select", {})
        if not isinstance(select, Mapping):
            raise ValueError("select must be a mapping")
        view_kind = classify_deck_view(select)
        if view_kind in {ViewKind.FULL_MEMBERSHIP, ViewKind.ORDERED_VIEW}:
            deck_view = select.get("deck")
            if deck_view is not None:
                counts: Counter[int] = Counter()
                for item in deck_view:
                    if not isinstance(item, Mapping) or not isinstance(item.get("id"), int):
                        raise ValueError("full deck membership contains malformed card")
                    counts[item["id"]] += 1
                if any(counts[card_id] > count for card_id, count in self.initial.items()) or any(card_id not in self.initial for card_id in counts):
                    raise ValueError("full deck membership violates registered deck")
                self._pending_view = counts
                self._deck_known = True
                self._deck_order = view_kind is ViewKind.ORDERED_VIEW
        self.decision_index += 1
        return snapshot

    def record_pending(self, action: Sequence[int], decision_index: int) -> None:
        if self._pending_action is not None:
            raise ValueError("pending action already exists")
        if self._current is None or decision_index != self._current.decision_index:
            raise ValueError("pending action decision mismatch")
        self._pending_action = tuple(action)

    def reconcile(self, observation: Mapping[str, Any]) -> KnowledgeSnapshot:
        return self.consume(observation)

    def encode_current(self) -> KnowledgeSnapshot:
        if self._current is not None:
            return self._current
        entries = {card_id: IdentityLedgerEntry(card_id, count, MappingProxyType({}), KnownCount.unknown(count), KnownCount.unknown(count)) for card_id, count in self.initial.items()}
        return KnowledgeSnapshot(0, immutable_entries(entries), OpponentHand((), 0), False, False, ())


__all__ = ["CausalKnowledgeState", "KnowledgeSnapshot", "KnownCard", "KnowledgeStage", "OpponentHand"]
