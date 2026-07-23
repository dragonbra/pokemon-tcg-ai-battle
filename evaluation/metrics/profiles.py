from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib

from .base import MetricPlugin
from .correctness import CorrectnessPlugin
from .length import LengthPlugin
from .library_pressure import LibraryPressurePlugin
from .outcome import OutcomePlugin
from .powerful_hand import PowerfulHandPlugin
from .post_ko_relay import PostKORelayPlugin
from .rare_candy import RareCandyPlugin
from .run_away_draw import RunAwayDrawPlugin


CORE_PROFILE_ID = "core"
AUTO_ITERATION_PROFILE_ID = "auto_iteration_v8_setup_relay"

CORE_METRIC_IDS = (
    "outcome",
    "length",
    "correctness",
    "powerful_hand",
    "rare_candy",
    "post_ko_relay",
    "run_away_draw",
    "library_pressure",
)
AUTO_ITERATION_METRIC_IDS = (*CORE_METRIC_IDS, "setup_relay", "attack_quality")


@dataclass(frozen=True)
class MetricPriority:
    metric_id: str
    priority: str
    direction: str


@dataclass(frozen=True)
class SemanticGroup:
    group_id: str
    title: str
    description: str

    def manifest(self) -> dict[str, str]:
        return {
            "id": self.group_id,
            "title": self.title,
            "description": self.description,
        }


@dataclass(frozen=True)
class MetricSemantic:
    semantic_id: str
    metric_id: str
    group_id: str
    title: str
    role: str
    direction: str
    value_source: str
    tracking_target: str
    display_kind: str = "aggregate_ratio"

    def manifest(self) -> dict[str, str]:
        return {
            "semantic_id": self.semantic_id,
            "metric_id": self.metric_id,
            "group_id": self.group_id,
            "title": self.title,
            "role": self.role,
            "direction": self.direction,
            "value_source": self.value_source,
            "tracking_target": self.tracking_target,
            "display_kind": self.display_kind,
        }


@dataclass(frozen=True)
class MetricProfile:
    profile_id: str
    revision: int
    description: str
    priorities: tuple[MetricPriority, ...]
    metric_ids: tuple[str, ...]
    plugin_factories: tuple[Callable[[], MetricPlugin], ...]
    semantic_groups: tuple[SemanticGroup, ...] = ()
    metric_semantics: tuple[MetricSemantic, ...] = ()

    def manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "id": self.profile_id,
            "revision": self.revision,
            "description": self.description,
            "metric_ids": list(self.metric_ids),
            "priorities": [
                {
                    "metric_id": priority.metric_id,
                    "priority": priority.priority,
                    "direction": priority.direction,
                }
                for priority in self.priorities
            ],
        }
        if self.semantic_groups:
            manifest["semantic_groups"] = [
                group.manifest() for group in self.semantic_groups
            ]
        if self.metric_semantics:
            manifest["metric_semantics"] = [
                semantic.manifest() for semantic in self.metric_semantics
            ]
        return manifest


def _lazy_factory(module_name: str, class_name: str) -> Callable[[], MetricPlugin]:
    def factory() -> MetricPlugin:
        module = importlib.import_module(module_name)
        plugin_class = getattr(module, class_name)
        return plugin_class()

    return factory


def _core_factories() -> tuple[Callable[[], MetricPlugin], ...]:
    return (
        OutcomePlugin,
        LengthPlugin,
        CorrectnessPlugin,
        PowerfulHandPlugin,
        RareCandyPlugin,
        PostKORelayPlugin,
        RunAwayDrawPlugin,
        LibraryPressurePlugin,
    )


CORE_PROFILE = MetricProfile(
    profile_id=CORE_PROFILE_ID,
    revision=1,
    description="Evaluation framework core metrics",
    priorities=(),
    metric_ids=CORE_METRIC_IDS,
    plugin_factories=_core_factories(),
)

AUTO_ITERATION_PROFILE = MetricProfile(
    profile_id=AUTO_ITERATION_PROFILE_ID,
    revision=3,
    description="V8 setup and relay semantic metric profile",
    priorities=(
        MetricPriority("outcome", "result_guardrail", "higher"),
        MetricPriority("powerful_hand", "target", "higher"),
        MetricPriority("post_ko_relay", "target", "higher"),
        MetricPriority("attack_quality", "penalty", "lower"),
    ),
    metric_ids=AUTO_ITERATION_METRIC_IDS,
    plugin_factories=(
        *_core_factories(),
        _lazy_factory("evaluation.metrics.setup_relay", "SetupRelayPlugin"),
        _lazy_factory("evaluation.metrics.attack_quality", "AttackQualityPlugin"),
    ),
    semantic_groups=(
        SemanticGroup(
            "result_correctness_guardrail",
            "结果与正确性护栏",
            "先检查 G0 correctness，再检查总体、实际先手和实际后手结果；异常不能被过程指标抵消。",
        ),
        SemanticGroup(
            "stage_1_setup",
            "阶段一：二回合基础能力",
            "观察二回合实际提交 Powerful Hand、Dunsparce 替代路线和额外过牌。",
        ),
        SemanticGroup(
            "stage_2_post_ko_relay",
            "阶段二：Post-KO 接力能力",
            "观察可攻击 Alakazam 被击倒后，下一回合另一只 Alakazam 是否立即接力攻击。",
        ),
        SemanticGroup(
            "stage_3_attack_quality",
            "阶段三：攻击质量惩罚项",
            "观察已经提交的攻击是否真正推进 Prize race；未拿奖赏率越低越好。",
        ),
        SemanticGroup(
            "auxiliary_health_audit",
            "辅助健康与审计指标",
            "提供进化链、局部错误、牌库压力和对局长度等解释性信号，不单独决定晋级。",
        ),
    ),
    metric_semantics=(
        MetricSemantic(
            "outcome",
            "outcome",
            "result_correctness_guardrail",
            "结果护栏：总体胜率",
            "guardrail",
            "higher",
            "aggregate.value",
            "总体、实际先手、实际后手胜率；同时保留 error、unfinished 和完整样本数。",
            "outcome_turn_order",
        ),
        MetricSemantic(
            "correctness",
            "correctness",
            "result_correctness_guardrail",
            "G0 正确性门槛",
            "guardrail",
            "lower",
            "aggregate.value",
            "非法动作、agent error、运行时错误和未完成对局必须先通过 correctness gate。",
        ),
        MetricSemantic(
            "powerful_hand",
            "powerful_hand",
            "stage_1_setup",
            "二回合 Alakazam 实际攻击",
            "target",
            "higher",
            "aggregate.value",
            "实际选择 attackId=1072 的对局数 / 全部评测对局数；同时展示到达二回合分母。",
            "powerful_hand_turn_order",
        ),
        MetricSemantic(
            "dunsparce_bridge",
            "setup_relay",
            "stage_1_setup",
            "无 Abra 的 Dunsparce 换位路线",
            "diagnostic",
            "higher",
            "payload.dunsparce_bridge",
            "在机会样本中完成 Dudunsparce 接力并于二回合实际提交 Powerful Hand。",
            "payload_ratio",
        ),
        MetricSemantic(
            "second_turn_draws",
            "setup_relay",
            "stage_1_setup",
            "二回合额外过牌张数",
            "diagnostic",
            "higher",
            "payload.second_turn_draws",
            "展示全部对局和实际到达二回合样本；正常回合抽牌单独作为审计字段。",
            "draw_summary",
        ),
        MetricSemantic(
            "post_ko_success",
            "post_ko_relay",
            "stage_2_post_ko_relay",
            "Post-KO 立即接力成功率",
            "target",
            "higher",
            "payload.success_rate",
            "下一回合另一只 Alakazam 合法来到 Active 并实际提交攻击 / Post-KO 机会数。",
            "payload_success_ratio",
        ),
        MetricSemantic(
            "recoverable_discard_miss",
            "post_ko_relay",
            "stage_2_post_ko_relay",
            "recoverable_discard_miss",
            "diagnostic",
            "lower",
            "payload.failure_counts.recoverable_discard_miss",
            "优先定位弃牌区仍有攻击线资源却没有形成接力的失败次数。",
            "payload_scalar",
        ),
        MetricSemantic(
            "attack_quality",
            "attack_quality",
            "stage_3_attack_quality",
            "攻击但未拿奖赏率",
            "penalty",
            "lower",
            "payload.non_prize_attacks.rate",
            "已结算且 Prize 状态明确的未拿奖赏攻击数 / 已结算且 Prize 状态明确的攻击总数。",
            "payload_ratio",
        ),
        MetricSemantic(
            "attack_quality_powerful",
            "attack_quality",
            "stage_3_attack_quality",
            "Powerful Hand 子集未拿奖赏率",
            "penalty",
            "lower",
            "payload.powerful_hand.non_prize_attacks",
            "重点降低 Powerful Hand 已结算攻击中的未拿奖赏比例。",
            "powerful_hand_attack_ratio",
        ),
        MetricSemantic(
            "rare_candy",
            "rare_candy",
            "auxiliary_health_audit",
            "Rare Candy 进化链完成率",
            "health",
            "higher",
            "aggregate.value",
            "辅助观察二回合 Rare Candy 进化链是否完成，不单独解释为策略改善。",
        ),
        MetricSemantic(
            "run_away_draw",
            "run_away_draw",
            "auxiliary_health_audit",
            "空 Bench Run Away Draw 错误次数",
            "health",
            "lower",
            "aggregate.numerator",
            "明确的局部策略错误，目标为零；每次触发都应保留 case 证据。",
            "scalar",
        ),
        MetricSemantic(
            "library_pressure",
            "library_pressure",
            "auxiliary_health_audit",
            "低牌库区间消耗量",
            "audit",
            "lower",
            "aggregate.value",
            "记录牌库低于 15/10 张后的消耗，结合终局闭环解释，不能单独排序。",
            "scalar",
        ),
        MetricSemantic(
            "length",
            "length",
            "auxiliary_health_audit",
            "平均对局长度",
            "audit",
            "diagnostic",
            "aggregate.value",
            "用于发现异常过长、过早结束或运行时行为变化。",
            "scalar",
        ),
    ),
)


_PROFILES = {
    CORE_PROFILE_ID: CORE_PROFILE,
    AUTO_ITERATION_PROFILE_ID: AUTO_ITERATION_PROFILE,
}


def available_metric_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_metric_profile(profile_id: str) -> MetricProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown metric profile: {profile_id}") from exc
