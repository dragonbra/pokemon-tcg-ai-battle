from __future__ import annotations

from typing import Any, Iterable

from .cards import BUDEW, ITCHY_POLLEN_ATTACK, SUPPORTERS


class GameMemory:
    """保存单个 observation 无法恢复的时点事实。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.turn_key: tuple[int, int] | None = None
        self.turn_start_serials: set[int] = set()
        self.known_field_serials: set[int] = set()
        self.evolved_this_turn: set[int] = set()
        self.supporter_used = False
        self.stadium_used = False
        self.energy_used = False
        self.retreat_used = False
        self.attack_submitted = False
        self.item_lock_turn_key: tuple[int, int] | None = None
        self.post_ko_turn_key: tuple[int, int] | None = None
        self.seen_ko_events: set[tuple[int, int, int, int]] = set()
        self.seen_item_lock_events: set[tuple[int, int, int, int]] = set()
        self.effect_steps: dict[tuple[int, int], int] = {}

    @staticmethod
    def _field(player: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            card
            for card in [*(player.get("active") or []), *(player.get("bench") or [])]
            if card
        ]

    @staticmethod
    def _serial(card: dict[str, Any]) -> int | None:
        serial = card.get("serial")
        return int(serial) if serial is not None else None

    @staticmethod
    def _event_signature(log: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(log.get("type", -1)),
            int(log.get("playerIndex", -1)),
            int(log.get("serial", -1) or -1),
            int(log.get("attackId", -1) or -1),
        )

    @staticmethod
    def _last_opponent_segment(logs: list[dict[str, Any]], your_index: int) -> list[dict[str, Any]]:
        own_starts = [
            index
            for index, log in enumerate(logs)
            if int(log.get("type", -1)) == 2
            and int(log.get("playerIndex", -1)) == your_index
        ]
        if not own_starts:
            return []
        own_start = own_starts[-1]
        opponent_index = 1 - your_index
        opponent_starts = [
            index
            for index, log in enumerate(logs[:own_start])
            if int(log.get("type", -1)) == 2
            and int(log.get("playerIndex", -1)) == opponent_index
        ]
        if not opponent_starts:
            return []
        return logs[opponent_starts[-1] + 1 : own_start]

    @staticmethod
    def _is_own_ko(log: dict[str, Any], your_index: int) -> bool:
        return (
            int(log.get("type", -1)) == 6
            and int(log.get("playerIndex", -1)) == your_index
            and int(log.get("fromArea", -1)) in {4, 5}
            and int(log.get("toArea", -1)) == 3
        )

    @staticmethod
    def _is_item_lock(log: dict[str, Any], opponent_index: int) -> bool:
        return (
            int(log.get("type", -1)) == 15
            and int(log.get("playerIndex", -1)) == opponent_index
            and int(log.get("cardId", -1)) == BUDEW
            and int(log.get("attackId", -1)) == ITCHY_POLLEN_ATTACK
        )

    def sync(
        self,
        current: dict[str, Any],
        player: dict[str, Any],
        logs: Iterable[dict[str, Any]] = (),
    ) -> None:
        turn_key = (int(current.get("turn", -1)), int(current.get("yourIndex", 0)))
        log_list = list(logs)
        field = self._field(player)
        field_serials = {
            serial for serial in (self._serial(card) for card in field) if serial is not None
        }
        is_new_turn = turn_key != self.turn_key
        if is_new_turn:
            self.turn_key = turn_key
            self.turn_start_serials = set(field_serials)
            self.evolved_this_turn.clear()
            self.supporter_used = bool(current.get("supporterPlayed", False))
            self.stadium_used = bool(current.get("stadiumPlayed", False))
            self.energy_used = bool(current.get("energyAttached", False))
            self.retreat_used = bool(current.get("retreated", False))
            self.attack_submitted = False
            self.known_field_serials = set(field_serials)

            segment = self._last_opponent_segment(log_list, turn_key[1])
            if segment:
                self.post_ko_turn_key = (
                    turn_key if any(self._is_own_ko(log, turn_key[1]) for log in segment) else None
                )
                self.item_lock_turn_key = (
                    turn_key
                    if any(self._is_item_lock(log, 1 - turn_key[1]) for log in segment)
                    else None
                )
            else:
                new_ko = False
                new_lock = False
                for log in log_list:
                    signature = self._event_signature(log)
                    if self._is_own_ko(log, turn_key[1]) and signature not in self.seen_ko_events:
                        new_ko = True
                    if (
                        self._is_item_lock(log, 1 - turn_key[1])
                        and signature not in self.seen_item_lock_events
                    ):
                        new_lock = True
                self.post_ko_turn_key = turn_key if new_ko else None
                self.item_lock_turn_key = turn_key if new_lock else None
        else:
            self.supporter_used = self.supporter_used or bool(current.get("supporterPlayed", False))
            self.stadium_used = self.stadium_used or bool(current.get("stadiumPlayed", False))
            self.energy_used = self.energy_used or bool(current.get("energyAttached", False))
            self.retreat_used = self.retreat_used or bool(current.get("retreated", False))
            for card in field:
                serial = self._serial(card)
                if serial is None or serial in self.known_field_serials:
                    continue
                previous_serials = {
                    int(item["serial"])
                    for item in (card.get("preEvolution") or [])
                    if isinstance(item, dict) and item.get("serial") is not None
                }
                if previous_serials:
                    self.evolved_this_turn.add(serial)
                    self.evolved_this_turn.update(previous_serials)
            self.known_field_serials.update(field_serials)

        for log in log_list:
            signature = self._event_signature(log)
            if self._is_own_ko(log, turn_key[1]):
                self.seen_ko_events.add(signature)
            if self._is_item_lock(log, 1 - turn_key[1]):
                self.seen_item_lock_events.add(signature)

    def can_evolve(self, serial: int | None, appear_this_turn: bool, own_turn: int) -> bool:
        return bool(
            own_turn > 1
            and serial is not None
            and not appear_this_turn
            and serial in self.turn_start_serials
            and serial not in self.evolved_this_turn
        )

    def record_evolution(self, serials: Iterable[int | None]) -> None:
        self.evolved_this_turn.update(int(serial) for serial in serials if serial is not None)

    def record_main_action(self, option_type: int, card_id: int | None) -> None:
        if option_type == 7 and card_id in SUPPORTERS:
            self.supporter_used = True
        elif option_type == 8:
            self.energy_used = True
        elif option_type == 12:
            self.retreat_used = True
        elif option_type == 13:
            self.attack_submitted = True
        elif option_type == 7 and card_id is not None:
            # Stadium legality remains simulator-authoritative; the fact flag is
            # updated by the next observation's stadiumPlayed value.
            return

    def effect_step(self, serial: int | None, effect_id: int | None) -> int:
        if serial is None or effect_id is None:
            return 0
        return self.effect_steps.get((int(serial), int(effect_id)), 0)

    def advance_effect(self, serial: int | None, effect_id: int | None) -> None:
        if serial is None or effect_id is None:
            return
        key = (int(serial), int(effect_id))
        self.effect_steps[key] = self.effect_steps.get(key, 0) + 1
