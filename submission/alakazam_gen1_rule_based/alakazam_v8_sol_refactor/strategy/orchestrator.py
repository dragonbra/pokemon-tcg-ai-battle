from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .cards import NIGHTTIME_MINE, SUPPORTERS, make_deck_spec
from .effects.dispatcher import select_effect
from .facts import build_turn_facts
from .memory import GameMemory
from .model import ActionKind, Decision
from .options import decode_options, match_intent
from .planner import build_turn_plan
from .policies import (
    propose_commit,
    propose_continuity,
    propose_control,
    propose_resources,
    propose_setup,
)
from .profiles import BASELINE_PROFILE, StrategyProfile
from .routes import analyze_routes


class StrategyOrchestrator:
    """把 observation 编排为事实、计划、策略意图和一个合法提交。"""

    def __init__(self, deck: Sequence[int], profile: StrategyProfile = BASELINE_PROFILE) -> None:
        self.deck = tuple(int(card_id) for card_id in deck)
        self.deck_spec = make_deck_spec(self.deck)
        self.profile = profile
        self.memory = GameMemory()
        self.previous_plan = None

    def reset(self) -> None:
        self.memory.reset()
        self.previous_plan = None

    def choose(self, obs: dict[str, Any]) -> Decision:
        select = obs.get("select") or {}
        select_type = int(select.get("type", 0))
        context = int(select.get("context", 0))
        facts = build_turn_facts(obs, self.memory, self.deck_spec)
        if select_type != 0 or context != 0:
            options = decode_options(select, facts)
            routes = analyze_routes(facts, options)
            plan = build_turn_plan(facts, routes, self.profile, self.previous_plan)
            decision = select_effect(obs, facts, plan, self.memory, self.profile)
            self.previous_plan = plan
            return decision

        options = decode_options(select, facts)
        routes = analyze_routes(facts, options)
        plan = build_turn_plan(facts, routes, self.profile, self.previous_plan)
        self.previous_plan = plan
        intents = self._intents(facts, routes, plan, options)
        for intent in intents:
            matched = match_intent(intent, options)
            if matched is None:
                continue
            self._record_main_action(matched)
            return Decision((matched.index,), intent.rule_id, plan.kind, intent.purpose)

        end = next((option for option in options if option.action_kind == ActionKind.END), None)
        if end:
            return Decision((end.index,), "fallback.end_turn", plan.kind, "no_planned_action")
        return Decision((), "fallback.no_legal_action", plan.kind, "no_legal_option")

    def choose_main(self, obs: dict[str, Any]) -> Decision:
        return self.choose(obs)

    def _intents(self, facts, routes, plan, options):
        # 阶段顺序是冲突解决协议；每个策略模块只提出自己负责的意图。
        phase_groups = (
            propose_continuity(facts, plan, routes, options, self.profile),
            propose_control(facts, plan, routes, options, self.profile),
            propose_setup(facts, plan, routes, options, self.profile),
            propose_resources(facts, plan, routes, options, self.profile),
            propose_commit(facts, plan, routes, options, self.profile),
        )
        return tuple(intent for group in phase_groups for intent in group)

    def _record_main_action(self, option) -> None:
        self.memory.record_main_action(option.raw_type, option.card_id)
        if option.action_kind == ActionKind.EVOLVE:
            target = option.target
            if target:
                self.memory.record_evolution((target.serial,))
        if option.action_kind == ActionKind.PLAY and option.card_id in SUPPORTERS:
            self.memory.supporter_used = True
        if option.action_kind == ActionKind.PLAY and option.card_id == NIGHTTIME_MINE:
            self.memory.stadium_used = True
        if option.action_kind == ActionKind.ATTACK:
            self.memory.attack_submitted = True
