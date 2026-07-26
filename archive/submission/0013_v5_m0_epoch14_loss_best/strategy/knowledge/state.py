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
        self._exact_deck: Counter[int] | None = None
        self._exact_prize: Counter[int] | None = None
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
        seen_serials: set[int] = set()

        def add_entity(item: object, zone: str) -> None:
            if not isinstance(item, Mapping):
                return
            serial = item.get("serial")
            if isinstance(serial, int) and not isinstance(serial, bool):
                if serial in seen_serials:
                    return
                seen_serials.add(serial)
            card_id = item.get("id")
            if isinstance(card_id, int) and not isinstance(card_id, bool):
                output.setdefault(card_id, Counter())[zone] += 1
            for field in ("energyCards", "preEvolution", "tools"):
                children = item.get(field, ())
                if isinstance(children, Sequence):
                    for child in children:
                        add_entity(child, zone)

        for zone in ("active", "bench", "hand", "discard", "prize"):
            values = own.get(zone, [])
            if isinstance(values, Sequence):
                for item in values:
                    add_entity(item, zone)
        stadium = current.get("stadium", ())
        if isinstance(stadium, Sequence):
            for item in stadium:
                if isinstance(item, Mapping) and item.get("playerIndex") == self.actor:
                    add_entity(item, "stadium")
        select = observation.get("select", {})
        if isinstance(select, Mapping):
            add_entity(select.get("contextCard"), "playing")
        return output

    def _remove_exact(self, zone: str, card_id: int) -> None:
        value = self._exact_deck if zone == "deck" else self._exact_prize
        if value is None:
            return
        if value[card_id] <= 0:
            if zone == "deck":
                self._exact_deck = None
                self._deck_order = False
            else:
                self._exact_prize = None
            return
        value[card_id] -= 1
        if value[card_id] == 0:
            del value[card_id]

    def _add_exact(self, zone: str, card_id: int) -> None:
        value = self._exact_deck if zone == "deck" else self._exact_prize
        if value is not None:
            value[card_id] += 1

    def _consume_self_resource_log(self, log: Mapping[str, Any]) -> None:
        log_type = log.get("type")
        if log.get("playerIndex") != self.actor or isinstance(log_type, bool):
            return
        card_id = log.get("cardId")
        card_id = card_id if isinstance(card_id, int) and not isinstance(card_id, bool) else None
        if log_type == 0:
            self._deck_order = False
            return
        if log_type == 4 and card_id is not None:
            self._remove_exact("deck", card_id)
            self._deck_order = False
            return
        if log_type not in {6, 7}:
            return
        from_area, to_area = log.get("fromArea"), log.get("toArea")
        touched = {from_area, to_area} & {1, 6}
        if not touched:
            return
        self._deck_order = False
        if log_type == 7 or card_id is None:
            if 1 in touched:
                self._exact_deck = None
            if 6 in touched:
                self._exact_prize = None
            return
        if from_area == 1:
            self._remove_exact("deck", card_id)
        elif from_area == 6:
            self._remove_exact("prize", card_id)
        if to_area == 1:
            self._add_exact("deck", card_id)
        elif to_area == 6:
            self._add_exact("prize", card_id)

    def _consume_log(self, log: Mapping[str, Any]) -> None:
        kind = classify_log(log)
        self._consume_self_resource_log(log)
        self._events.append(MappingProxyType(dict(log)))
        if len(self._events) > self.event_window:
            del self._events[:-self.event_window]
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
        own = players[self.actor] if self.actor < len(players) else None
        if not isinstance(own, Mapping):
            raise ValueError("actor player state malformed")
        deck_total = own.get("deckCount")
        prize_zone = own.get("prize", ())
        if (
            isinstance(deck_total, bool)
            or not isinstance(deck_total, int)
            or deck_total < 0
            or not isinstance(prize_zone, Sequence)
        ):
            raise ValueError("actor hidden-zone counts malformed")
        select = observation.get("select", {})
        if not isinstance(select, Mapping):
            raise ValueError("select must be a mapping")
        view_kind = classify_deck_view(select)
        deck_view = select.get("deck")
        if (
            view_kind in {ViewKind.FULL_MEMBERSHIP, ViewKind.ORDERED_VIEW}
            and isinstance(deck_view, Sequence)
            and len(deck_view) == deck_total
        ):
            exact_deck: Counter[int] = Counter()
            for item in deck_view:
                if not isinstance(item, Mapping) or not isinstance(item.get("id"), int):
                    raise ValueError("full deck membership contains malformed card")
                exact_deck[item["id"]] += 1
            if (
                any(exact_deck[card_id] > count for card_id, count in self.initial.items())
                or any(card_id not in self.initial for card_id in exact_deck)
            ):
                raise ValueError("full deck membership violates registered deck")
            self._exact_deck = exact_deck
            self._deck_order = view_kind is ViewKind.ORDERED_VIEW
            inferred_prize: Counter[int] = Counter()
            prize_valid = True
            for card_id, initial in self.initial.items():
                zones = visible.get(card_id, Counter())
                outside_hidden = sum(
                    count for zone, count in zones.items() if zone != "prize"
                )
                count = initial - outside_hidden - exact_deck.get(card_id, 0)
                if count < 0:
                    prize_valid = False
                    break
                if count:
                    inferred_prize[card_id] = count
            self._exact_prize = (
                inferred_prize
                if prize_valid and sum(inferred_prize.values()) == len(prize_zone)
                else None
            )
        if self._exact_deck is not None and sum(self._exact_deck.values()) != deck_total:
            self._exact_deck = None
            self._deck_order = False
        if self._exact_prize is not None and sum(self._exact_prize.values()) != len(prize_zone):
            self._exact_prize = None
        if self._exact_deck is not None and self._exact_prize is not None:
            conservation_valid = all(
                self._exact_deck.get(card_id, 0)
                + self._exact_prize.get(card_id, 0)
                + sum(
                    count
                    for zone, count in visible.get(card_id, Counter()).items()
                    if zone != "prize"
                )
                == initial
                for card_id, initial in self.initial.items()
            )
            if not conservation_valid:
                self._exact_deck = None
                self._exact_prize = None
                self._deck_order = False
        self._deck_known = self._exact_deck is not None
        entries: dict[int, IdentityLedgerEntry] = {}
        for card_id, initial in sorted(self.initial.items()):
            zones = visible.get(card_id, Counter())
            deck = (
                KnownCount.exact(
                    self._exact_deck.get(card_id, 0),
                    KnowledgeStage.INFERRED_EXACT,
                    self.decision_index,
                )
                if self._exact_deck is not None
                else KnownCount.unknown(initial)
            )
            prize_count = (
                KnownCount.exact(
                    self._exact_prize.get(card_id, 0),
                    KnowledgeStage.INFERRED_EXACT,
                    self.decision_index,
                )
                if self._exact_prize is not None
                else KnownCount.unknown(initial)
            )
            entries[card_id] = IdentityLedgerEntry(
                card_id, initial, MappingProxyType(dict(zones)), deck, prize_count
            )
        snapshot = KnowledgeSnapshot(self.decision_index, immutable_entries(entries), OpponentHand(tuple(sorted(self._known_hand.values(), key=lambda item: item.serial)), self._unknown_hand), self._deck_known, self._deck_order, tuple(self._events[-self.event_window:]))
        self._current = snapshot
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
