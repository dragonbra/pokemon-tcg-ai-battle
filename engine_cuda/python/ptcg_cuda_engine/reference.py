from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

from .schema import OPCODES, RulePack


MASK64 = (1 << 64) - 1
MAX_COUNTERS = 32
INTERPRETER_BUDGET = 64


class Status(IntEnum):
    EMPTY = 0
    NEEDS_POLICY = 1
    ADVANCING = 2
    TERMINAL = 3
    ERROR = 4


class Error(IntEnum):
    NONE = 0
    INVALID_ACTION = 1
    RULE_PACK_BOUNDS = 2
    INTERPRETER_BUDGET = 3
    STACK_OVERFLOW = 4
    ROUTE_OVERFLOW = 5
    CODEC_ENTITY_OVERFLOW = 6
    CODEC_OPTION_OVERFLOW = 7
    UNSUPPORTED_OPCODE = 8
    INVALID_POLICY_ID = 9
    KNOWN_DIVERGENCE_660_1207 = 6601207


@dataclass(frozen=True)
class ReferenceResetSpec:
    seed: int
    active_card_ids: tuple[int, int] = (1, 2)
    active_hp: tuple[int, int] = (100, 100)
    deck_count: tuple[int, int] = (53, 53)
    hand_count: tuple[int, int] = (7, 7)
    policy_ids: tuple[int, int] = (0, 1)


@dataclass
class CardState:
    card_id: int
    max_hp: int
    damage: int
    owner: int
    zone: int = 3
    slot: int = 0
    flags: int = 0


@dataclass
class PlayerState:
    policy_id: int
    active_entity: int
    deck_count: int
    hand_count: int
    prize_count: int = 6
    discard_count: int = 0
    bench_count: int = 0
    turns_taken: int = 0
    actions_taken: int = 0


@dataclass
class BattleState:
    rng_state: int
    episode_id: int
    status: Status
    error: Error
    winner: int
    actor: int
    action_count: int
    players: list[PlayerState]
    cards: list[CardState]
    turn: int = 0
    decision_count: int = 0
    counters: list[int] = field(default_factory=lambda: [0] * MAX_COUNTERS)


def _next_random(state: BattleState) -> int:
    value = state.rng_state & MASK64
    value ^= value >> 12
    value ^= (value << 25) & MASK64
    value ^= value >> 27
    state.rng_state = value & MASK64
    return (value * 0x2545F4914F6CDD1D) & MASK64


def _resolve_target(encoded: int, actor: int) -> int:
    if encoded == 0:
        return actor
    if encoded == 1:
        return actor ^ 1
    if encoded == 2:
        return 0
    if encoded == 3:
        return 1
    return actor


def _fail(state: BattleState, error: Error) -> None:
    state.error = error
    state.status = Status.ERROR


class BatchedReferenceEngine:
    """Deterministic CPU mirror of the prototype opcode subset.

    This reference validates layout and interpreter behavior. It is not an
    alternative semantic oracle for the official game engine.
    """

    def __init__(self, rule_pack: RulePack, batch_size: int, policy_count: int):
        rule_pack.validate()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 1 <= policy_count <= 32:
            raise ValueError("policy_count must be in [1, 32]")
        self.rule_pack = rule_pack
        self.batch_size = batch_size
        self.policy_count = policy_count
        self.states: list[BattleState] = []

    def reset(self, specs: Iterable[ReferenceResetSpec]) -> list[BattleState]:
        rows = list(specs)
        if len(rows) != self.batch_size:
            raise ValueError("reset spec count does not match batch size")
        states: list[BattleState] = []
        for env, spec in enumerate(rows):
            rng_state = (((spec.seed & 0xFFFFFFFF) << 32) ^ (env + 0x9E3779B97F4A7C15)) & MASK64
            if rng_state == 0:
                rng_state = 0xD1B54A32D192ED03
            players = [
                PlayerState(
                    policy_id=spec.policy_ids[player],
                    active_entity=player,
                    deck_count=spec.deck_count[player],
                    hand_count=spec.hand_count[player],
                )
                for player in range(2)
            ]
            cards = [
                CardState(
                    card_id=spec.active_card_ids[player],
                    max_hp=spec.active_hp[player] or 100,
                    damage=0,
                    owner=player,
                )
                for player in range(2)
            ]
            state = BattleState(
                rng_state=rng_state,
                episode_id=((spec.seed & 0xFFFFFFFF) << 32) | env,
                status=Status.NEEDS_POLICY,
                error=Error.NONE,
                winner=-1,
                actor=0,
                action_count=len(self.rule_pack.actions),
                players=players,
                cards=cards,
            )
            state.actor = _next_random(state) & 1
            if any(policy < 0 or policy >= self.policy_count for policy in spec.policy_ids):
                _fail(state, Error.INVALID_POLICY_ID)
            states.append(state)
        self.states = states
        return states

    def step(self, action_indices: Iterable[int]) -> list[BattleState]:
        actions = list(action_indices)
        if len(actions) != self.batch_size:
            raise ValueError("action count does not match batch size")
        if not self.states:
            raise RuntimeError("reset must be called before step")
        for state, selected in zip(self.states, actions):
            if state.status != Status.NEEDS_POLICY:
                continue
            if selected < 0 or selected >= len(self.rule_pack.actions):
                _fail(state, Error.INVALID_ACTION)
                continue
            self._execute(state, selected)
        return self.states

    def _execute(self, state: BattleState, action_index: int) -> None:
        action = self.rule_pack.actions[action_index]
        state.status = Status.ADVANCING
        state.players[state.actor].actions_taken += 1
        state.decision_count += 1
        pc = 0
        budget = 0
        while pc < len(action.program) and budget < INTERPRETER_BUDGET:
            instruction = action.program[pc]
            pc += 1
            budget += 1
            opcode = instruction.opcode
            target = _resolve_target(instruction.target, state.actor)

            if opcode == OPCODES["NOP"]:
                pass
            elif opcode == OPCODES["DRAW"]:
                player = state.players[target]
                drawn = min(max(0, instruction.arg0), player.deck_count)
                player.deck_count -= drawn
                player.hand_count += drawn
            elif opcode == OPCODES["DAMAGE_ACTIVE"]:
                active = state.cards[state.players[target].active_entity]
                active.damage = min(65535, active.damage + max(0, instruction.arg0))
            elif opcode == OPCODES["HEAL_ACTIVE"]:
                active = state.cards[state.players[target].active_entity]
                active.damage = max(0, active.damage - max(0, instruction.arg0))
            elif opcode == OPCODES["END_TURN"]:
                state.players[state.actor].turns_taken += 1
                state.actor ^= 1
                state.turn += 1
                state.status = Status.NEEDS_POLICY
            elif opcode == OPCODES["CHECK_KNOCKOUT"]:
                active = state.cards[state.players[target].active_entity]
                if active.max_hp > 0 and active.damage >= active.max_hp:
                    state.winner = target ^ 1
                    state.status = Status.TERMINAL
                    return
            elif opcode == OPCODES["SET_WINNER"]:
                state.winner = target
                state.status = Status.TERMINAL
                return
            elif opcode == OPCODES["EMIT_DECISION"]:
                state.status = Status.NEEDS_POLICY
                return
            elif opcode == OPCODES["SET_COUNTER"]:
                if not 0 <= instruction.arg0 < MAX_COUNTERS:
                    _fail(state, Error.RULE_PACK_BOUNDS)
                    return
                state.counters[instruction.arg0] = instruction.arg1 & 0xFFFFFFFF
            elif opcode == OPCODES["ADD_COUNTER"]:
                if not 0 <= instruction.arg0 < MAX_COUNTERS:
                    _fail(state, Error.RULE_PACK_BOUNDS)
                    return
                state.counters[instruction.arg0] = (
                    state.counters[instruction.arg0] + instruction.arg1
                ) & 0xFFFFFFFF
            elif opcode == OPCODES["DELAY_EFFECT"]:
                if instruction.arg0 == 660 and instruction.arg1 == 1207:
                    _fail(state, Error.KNOWN_DIVERGENCE_660_1207)
                else:
                    _fail(state, Error.UNSUPPORTED_OPCODE)
                return
            elif opcode == OPCODES["MOVE_CARD"]:
                moved = False
                for card in state.cards:
                    if card.card_id == instruction.arg0 and card.owner == target:
                        card.zone = instruction.arg1
                        moved = True
                        break
                if not moved and instruction.flags & 1:
                    _fail(state, Error.RULE_PACK_BOUNDS)
                    return
            elif opcode == OPCODES["CONDITIONAL_COUNTER_GE"]:
                if not 0 <= instruction.arg0 < MAX_COUNTERS:
                    _fail(state, Error.RULE_PACK_BOUNDS)
                    return
                if state.counters[instruction.arg0] >= instruction.arg1:
                    destination = pc + instruction.arg2
                    if not 0 <= destination <= len(action.program):
                        _fail(state, Error.RULE_PACK_BOUNDS)
                        return
                    pc = destination
            elif opcode == OPCODES["RANDOM_BRANCH"]:
                threshold = min(10000, max(0, instruction.arg0))
                delta = instruction.arg1 if _next_random(state) % 10000 < threshold else instruction.arg2
                destination = pc + delta
                if not 0 <= destination <= len(action.program):
                    _fail(state, Error.RULE_PACK_BOUNDS)
                    return
                pc = destination
            elif opcode == OPCODES["HALT"]:
                if state.status == Status.ADVANCING:
                    state.status = Status.NEEDS_POLICY
                return
            else:
                _fail(state, Error.UNSUPPORTED_OPCODE)
                return

            if state.status in (Status.TERMINAL, Status.ERROR):
                return

        if pc < len(action.program):
            _fail(state, Error.INTERPRETER_BUDGET)
        elif state.status == Status.ADVANCING:
            state.status = Status.NEEDS_POLICY

    def route(self, capacity: int) -> tuple[list[list[int]], list[int]]:
        if capacity <= 0:
            raise ValueError("route capacity must be positive")
        routes = [[-1] * capacity for _ in range(self.policy_count)]
        counts = [0] * self.policy_count
        for env, state in enumerate(self.states):
            if state.status != Status.NEEDS_POLICY:
                continue
            policy = state.players[state.actor].policy_id
            if not 0 <= policy < self.policy_count:
                _fail(state, Error.INVALID_POLICY_ID)
                continue
            slot = counts[policy]
            counts[policy] += 1
            if slot >= capacity:
                _fail(state, Error.ROUTE_OVERFLOW)
                continue
            routes[policy][slot] = env
        return routes, counts

    def encode_policy_v1(self) -> list[dict[str, list[int] | list[float]]]:
        encoded: list[dict[str, list[int] | list[float]]] = []
        for state in self.states:
            actor = state.actor
            global_cat = [0] * 8
            global_num = [0.0] * 16
            global_cat[:8] = [
                actor + 1,
                int(state.status),
                state.turn + 1,
                state.players[actor].policy_id + 1,
                state.players[actor ^ 1].policy_id + 1,
                state.winner + 2,
                int(state.error),
                1,
            ]
            global_num[:6] = [
                state.players[actor].deck_count / 60.0,
                state.players[actor ^ 1].deck_count / 60.0,
                state.players[actor].hand_count / 60.0,
                state.players[actor ^ 1].hand_count / 60.0,
                state.turn / 100.0,
                state.decision_count / 1000.0,
            ]
            entity_cat = [0] * (128 * 6)
            entity_num = [0.0] * (128 * 10)
            entity_parent = [-1] * 128
            entity_mask = [0] * 128
            for index, card in enumerate(state.cards[:128]):
                self_owned = card.owner == actor
                zone = 0
                if card.zone == 3:
                    zone = 1 if self_owned else 6
                elif card.zone == 4:
                    zone = 2 if self_owned else 7
                elif card.zone == 2:
                    zone = 3 if self_owned else 0
                elif card.zone == 5:
                    zone = 4 if self_owned else 8
                elif card.zone == 6:
                    zone = 5 if self_owned else 9
                elif card.zone == 8:
                    zone = 10
                cat_offset = index * 6
                num_offset = index * 10
                entity_cat[cat_offset : cat_offset + 6] = [
                    card.card_id,
                    1 if self_owned else 2,
                    zone,
                    card.slot + 1,
                    2,
                    card.flags + 1,
                ]
                entity_num[num_offset : num_offset + 3] = [
                    card.damage / card.max_hp if card.max_hp else 0.0,
                    card.max_hp / 400.0,
                    card.damage / 400.0,
                ]
                entity_mask[index] = 1
            option_cat = [0] * (80 * 12)
            option_num = [0.0] * (80 * 4)
            option_equiv = [-1] * 80
            option_mask = [0] * 80
            for index, action in enumerate(self.rule_pack.actions):
                offset = index * 12
                option_cat[offset] = action.option_type
                option_cat[offset + 3] = 1
                option_cat[offset + 4] = action.card_id
                option_cat[offset + 11] = index + 1
                option_equiv[index] = action.action_id
                option_mask[index] = 1
            encoded.append(
                {
                    "global_cat": global_cat,
                    "global_num": global_num,
                    "entity_cat": entity_cat,
                    "entity_num": entity_num,
                    "entity_parent": entity_parent,
                    "entity_mask": entity_mask,
                    "option_cat": option_cat,
                    "option_num": option_num,
                    "option_equiv": option_equiv,
                    "option_mask": option_mask,
                    "min_count": [1],
                    "max_count": [1],
                }
            )
        return encoded

    def digests(self) -> list[int]:
        return [self._digest(state) for state in self.states]

    @staticmethod
    def _digest(state: BattleState) -> int:
        digest = 1469598103934665603

        def add(value: int) -> None:
            nonlocal digest
            digest ^= value & MASK64
            digest = (digest * 1099511628211) & MASK64

        add(state.rng_state)
        add(state.turn)
        add(state.decision_count)
        add(int(state.status) & 0xFFFFFFFF)
        add(int(state.error) & 0xFFFFFFFF)
        add(state.winner & 0xFFFFFFFF)
        add(state.actor)
        for player in state.players:
            add(player.policy_id)
            add(player.deck_count)
            add(player.hand_count)
        for card in state.cards:
            add(card.card_id)
            add(card.damage)
            add(card.zone)
        return digest
