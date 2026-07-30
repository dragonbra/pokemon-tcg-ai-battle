"""Audited transition state layered over the accepted 0013 own-resource ledger."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import ledger as _ledger_types

HAND_AREA = 2
DRAW_REVERSE = 5
MOVE_CARD = 6
MOVE_CARD_REVERSE = 7
LOG_NAMES = (
    "shuffle", "has_basic", "turn_start", "turn_end", "draw", "draw_hidden",
    "move_card", "move_hidden", "switch", "change", "play", "attach", "evolve",
    "devolve", "move_attached", "attack", "hp_change", "poisoned", "burned",
    "asleep", "paralyzed", "confused", "coin", "result",
)


@dataclass(frozen=True, slots=True)
class KnownOpponentCard:
    card_id: int
    serial: int
    source_event: int


@dataclass(frozen=True, slots=True)
class TypedEvent:
    source_event: int
    decision_index: int
    local_log_ordinal: int
    log_type: int
    name: str
    actor: int | None
    card_id: int | None
    from_area: int | None
    to_area: int | None
    identity_visible: bool
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CausalSnapshot:
    decision_index: int
    perspective_actor: int
    self_ledger: Mapping[int, Any]
    known_opponent_hand: tuple[KnownOpponentCard, ...]
    unknown_opponent_hand: int
    recent_events: tuple[TypedEvent, ...]
    deck_membership_known: bool
    deck_order_known: bool


class CausalKnowledge:
    """Consume each actor-visible incremental log exactly once in chronological row order."""

    def __init__(self, actor: int, registered_deck: Sequence[int], event_window: int = 64):
        self.actor = actor
        self.initial = Counter(registered_deck)
        self.event_window = event_window
        self._decision_index = 0
        self._exact_deck: Counter[int] | None = None
        self._exact_prize: Counter[int] | None = None
        self._deck_source_event: int | None = None
        self._prize_source_event: int | None = None
        self._deck_order_known = False
        self._known_opponent_hand: dict[int, KnownOpponentCard] = {}
        self._unknown_opponent_hand = 0
        self._events: list[TypedEvent] = []
        self._event_index = 0

    def _consume_opponent_hand(self, log: Mapping[str, Any], opponent: int) -> None:
        log_type = log.get("type")
        if log.get("playerIndex") != opponent or isinstance(log_type, bool):
            return
        if log_type == DRAW_REVERSE:
            self._unknown_opponent_hand += 1
            return
        if log_type not in {MOVE_CARD, MOVE_CARD_REVERSE}:
            return
        from_area, to_area = log.get("fromArea"), log.get("toArea")
        serial = log.get("serial")
        card_id = log.get("cardId")
        if from_area == HAND_AREA:
            if log_type == MOVE_CARD and isinstance(serial, int) and not isinstance(serial, bool):
                if self._known_opponent_hand.pop(serial, None) is None and self._unknown_opponent_hand:
                    self._unknown_opponent_hand -= 1
            else:
                # A redacted departure could be any previously revealed card. With no
                # candidate-set channel, retaining an arbitrary identity would leak certainty.
                self._known_opponent_hand.clear()
                if self._unknown_opponent_hand:
                    self._unknown_opponent_hand -= 1
        if to_area == HAND_AREA:
            if (
                log_type == MOVE_CARD
                and isinstance(serial, int) and not isinstance(serial, bool)
                and isinstance(card_id, int) and not isinstance(card_id, bool)
            ):
                self._known_opponent_hand[serial] = KnownOpponentCard(
                    card_id, serial, self._event_index
                )
            else:
                self._unknown_opponent_hand += 1

    def _typed_event(self, log: Mapping[str, Any], local_log_ordinal: int) -> TypedEvent:
        raw_type = log.get("type")
        if isinstance(raw_type, bool) or not isinstance(raw_type, int):
            raise ValueError(f"unclassified engine log type: {raw_type!r}")
        if not 0 <= raw_type < len(LOG_NAMES):
            raise ValueError(f"engine log type outside audited enum: {raw_type}")
        actor = log.get("playerIndex")
        actor = actor if isinstance(actor, int) and not isinstance(actor, bool) else None
        card_id = log.get("cardId")
        card_id = card_id if isinstance(card_id, int) and not isinstance(card_id, bool) else None
        return TypedEvent(
            source_event=self._event_index,
            decision_index=self._decision_index,
            local_log_ordinal=local_log_ordinal,
            log_type=raw_type,
            name=LOG_NAMES[raw_type],
            actor=actor,
            card_id=card_id,
            from_area=log.get("fromArea") if isinstance(log.get("fromArea"), int) else None,
            to_area=log.get("toArea") if isinstance(log.get("toArea"), int) else None,
            identity_visible=card_id is not None,
            payload=MappingProxyType(dict(log)),
        )

    @staticmethod
    def _card_id(item: object) -> int | None:
        if not isinstance(item, Mapping):
            return None
        value = item.get("id")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _visible_counts(self, observation: Mapping[str, Any]) -> dict[int, Counter[str]]:
        current = observation["current"]
        own = current["players"][self.actor]
        output: dict[int, Counter[str]] = {}
        seen_serials: set[int] = set()

        def add(item: object, zone: str) -> None:
            if not isinstance(item, Mapping):
                return
            serial = item.get("serial")
            if isinstance(serial, int) and not isinstance(serial, bool):
                if serial in seen_serials:
                    return
                seen_serials.add(serial)
            card_id = self._card_id(item)
            if card_id is not None:
                output.setdefault(card_id, Counter())[zone] += 1
            for field in ("energyCards", "preEvolution", "tools"):
                children = item.get(field, ())
                if isinstance(children, Sequence):
                    for child in children:
                        add(child, zone)

        for zone in ("active", "bench", "hand", "discard"):
            values = own.get(zone, ())
            if isinstance(values, Sequence):
                for item in values:
                    add(item, zone)
        stadium = current.get("stadium", ())
        if isinstance(stadium, Sequence):
            for item in stadium:
                if isinstance(item, Mapping) and item.get("playerIndex") == self.actor:
                    add(item, "stadium")
        looking = current.get("looking", ())
        if isinstance(looking, Sequence):
            for item in looking:
                add(item, "looking")
        select = observation.get("select", {})
        if isinstance(select, Mapping):
            add(select.get("contextCard"), "playing")
            add(select.get("effect"), "playing")
        return output

    def _remove_exact(self, zone: str, card_id: int) -> None:
        value = self._exact_deck if zone == "deck" else self._exact_prize
        if value is None:
            return
        if value[card_id] <= 0:
            if zone == "deck":
                self._exact_deck = None
                self._deck_source_event = None
                self._deck_order_known = False
            else:
                self._exact_prize = None
                self._prize_source_event = None
            return
        value[card_id] -= 1
        if value[card_id] == 0:
            del value[card_id]

    def _add_exact(self, zone: str, card_id: int) -> None:
        value = self._exact_deck if zone == "deck" else self._exact_prize
        if value is not None:
            value[card_id] += 1

    def _consume_self_hidden_zones(self, log: Mapping[str, Any]) -> None:
        raw_type = log.get("type")
        if log.get("playerIndex") != self.actor or isinstance(raw_type, bool):
            return
        card_id = log.get("cardId")
        card_id = card_id if isinstance(card_id, int) and not isinstance(card_id, bool) else None
        if raw_type == 0:
            self._deck_order_known = False
        elif raw_type == 4 and card_id is not None:
            self._remove_exact("deck", card_id)
            if self._exact_deck is not None:
                self._deck_source_event = self._event_index
            self._deck_order_known = False
        elif raw_type in {MOVE_CARD, MOVE_CARD_REVERSE}:
            from_area, to_area = log.get("fromArea"), log.get("toArea")
            touched = {from_area, to_area} & {1, 6}
            if not touched:
                return
            self._deck_order_known = False
            if raw_type == MOVE_CARD_REVERSE or card_id is None:
                if 1 in touched:
                    self._exact_deck = None
                    self._deck_source_event = None
                if 6 in touched:
                    self._exact_prize = None
                    self._prize_source_event = None
                return
            if from_area == 1:
                self._remove_exact("deck", card_id)
            elif from_area == 6:
                self._remove_exact("prize", card_id)
            if to_area == 1:
                self._add_exact("deck", card_id)
            elif to_area == 6:
                self._add_exact("prize", card_id)
            if 1 in touched and self._exact_deck is not None:
                self._deck_source_event = self._event_index
            if 6 in touched and self._exact_prize is not None:
                self._prize_source_event = self._event_index

    def _reanchor_from_full_deck_view(
        self,
        observation: Mapping[str, Any],
        visible: Mapping[int, Counter[str]],
    ) -> None:
        current = observation["current"]
        own = current["players"][self.actor]
        deck_total = own.get("deckCount")
        select = observation.get("select", {})
        deck_view = select.get("deck") if isinstance(select, Mapping) else None
        if not isinstance(deck_total, int) or isinstance(deck_total, bool):
            raise ValueError("actor deckCount must be an integer")
        if not isinstance(deck_view, Sequence) or len(deck_view) != deck_total:
            return
        # Projection maps an absent view to []; an empty deck is still a safe exact
        # membership statement, but no order authority is claimed.
        exact = Counter()
        for item in deck_view:
            card_id = self._card_id(item)
            if card_id is None:
                raise ValueError("full deck view contains a card without identity")
            exact[card_id] += 1
        if any(card_id not in self.initial or count > self.initial[card_id] for card_id, count in exact.items()):
            raise ValueError("full deck membership violates registered deck")
        self._exact_deck = exact
        self._deck_source_event = self._event_index
        self._deck_order_known = False
        inferred = Counter()
        for card_id, initial in self.initial.items():
            outside_hidden = sum(visible.get(card_id, {}).values())
            count = initial - exact.get(card_id, 0) - outside_hidden
            if count < 0:
                self._exact_prize = None
                self._prize_source_event = None
                return
            if count:
                inferred[card_id] = count
        prize = own.get("prize", ())
        self._exact_prize = inferred if isinstance(prize, Sequence) and sum(inferred.values()) == len(prize) else None
        self._prize_source_event = self._event_index if self._exact_prize is not None else None

    def _ledger_snapshot(
        self,
        observation: Mapping[str, Any],
        visible: Mapping[int, Counter[str]],
    ) -> Mapping[int, Any]:
        own = observation["current"]["players"][self.actor]
        deck_total = own.get("deckCount")
        prize = own.get("prize", ())
        if self._exact_deck is not None and sum(self._exact_deck.values()) != deck_total:
            self._exact_deck = None
            self._deck_source_event = None
            self._deck_order_known = False
        if self._exact_prize is not None and (
            not isinstance(prize, Sequence) or sum(self._exact_prize.values()) != len(prize)
        ):
            self._exact_prize = None
            self._prize_source_event = None
        entries: dict[int, Any] = {}
        for card_id, initial in sorted(self.initial.items()):
            visible_total = sum(visible.get(card_id, {}).values())
            hidden_remaining = max(0, initial - visible_total)
            prize_total = len(prize) if isinstance(prize, Sequence) else 0
            deck_lower = max(0, hidden_remaining - prize_total)
            deck_upper = min(hidden_remaining, int(deck_total))
            prize_lower = max(0, hidden_remaining - int(deck_total))
            prize_upper = min(hidden_remaining, prize_total)
            deck = (
                _ledger_types.KnownCount(
                    self._exact_deck.get(card_id, 0),
                    self._exact_deck.get(card_id, 0),
                    self._exact_deck.get(card_id, 0),
                    _ledger_types.KnowledgeStage.INFERRED_EXACT,
                    self._deck_source_event,
                    self._event_index
                    - (
                        self._deck_source_event
                        if self._deck_source_event is not None
                        else self._event_index
                    ),
                )
                if self._exact_deck is not None
                else _ledger_types.KnownCount(
                    None,
                    deck_lower,
                    deck_upper,
                    _ledger_types.KnowledgeStage.BOUNDED,
                )
            )
            prize_count = (
                _ledger_types.KnownCount(
                    self._exact_prize.get(card_id, 0),
                    self._exact_prize.get(card_id, 0),
                    self._exact_prize.get(card_id, 0),
                    _ledger_types.KnowledgeStage.INFERRED_EXACT,
                    self._prize_source_event,
                    self._event_index
                    - (
                        self._prize_source_event
                        if self._prize_source_event is not None
                        else self._event_index
                    ),
                )
                if self._exact_prize is not None
                else _ledger_types.KnownCount(
                    None,
                    prize_lower,
                    prize_upper,
                    _ledger_types.KnowledgeStage.BOUNDED,
                )
            )
            entries[card_id] = _ledger_types.IdentityLedgerEntry(
                card_id,
                initial,
                MappingProxyType(dict(visible.get(card_id, {}))),
                deck,
                prize_count,
            )
        return MappingProxyType(entries)

    def consume(
        self,
        observation: Mapping[str, Any],
        event_cursor: Mapping[str, Any] | None = None,
    ) -> CausalSnapshot:
        current = observation.get("current")
        if not isinstance(current, Mapping) or current.get("yourIndex") != self.actor:
            raise ValueError("causal actor mismatch")
        players = current.get("players")
        if not isinstance(players, Sequence) or len(players) != 2:
            raise ValueError("0020 requires two-player observations")
        opponent = 1 - self.actor
        logs = observation.get("logs", ())
        if not isinstance(logs, Sequence):
            raise ValueError("logs must be a sequence")
        if event_cursor is not None:
            if event_cursor.get("actor_decision_index") != self._decision_index:
                raise ValueError("event cursor decision index is not chronological")
            if event_cursor.get("incoming_log_count") != len(logs):
                raise ValueError("event cursor log count mismatch")
        for local_log_ordinal, log in enumerate(logs):
            if not isinstance(log, Mapping):
                raise ValueError("log must be a mapping")
            event = self._typed_event(log, local_log_ordinal)
            self._consume_opponent_hand(log, opponent)
            self._consume_self_hidden_zones(log)
            self._events.append(event)
            self._event_index += 1
        if len(self._events) > self.event_window:
            del self._events[:-self.event_window]
        opponent_state = players[opponent]
        if not isinstance(opponent_state, Mapping):
            raise ValueError("opponent state must be a mapping")
        hand_count = opponent_state.get("handCount")
        if isinstance(hand_count, bool) or not isinstance(hand_count, int) or hand_count < 0:
            raise ValueError("opponent handCount must be non-negative")
        if len(self._known_opponent_hand) > hand_count:
            kept = sorted(self._known_opponent_hand.items())[:hand_count]
            self._known_opponent_hand = dict(kept)
        self._unknown_opponent_hand = hand_count - len(self._known_opponent_hand)
        visible = self._visible_counts(observation)
        self._reanchor_from_full_deck_view(observation, visible)
        ledger = self._ledger_snapshot(observation, visible)
        snapshot = CausalSnapshot(
            decision_index=self._decision_index,
            perspective_actor=self.actor,
            self_ledger=ledger,
            known_opponent_hand=tuple(
                sorted(self._known_opponent_hand.values(), key=lambda item: item.serial)
            ),
            unknown_opponent_hand=self._unknown_opponent_hand,
            recent_events=tuple(self._events),
            deck_membership_known=self._exact_deck is not None,
            deck_order_known=self._deck_order_known,
        )
        self._decision_index += 1
        return snapshot


__all__ = ["CausalKnowledge", "CausalSnapshot", "KnownOpponentCard", "TypedEvent"]
