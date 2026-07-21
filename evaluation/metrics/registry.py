from __future__ import annotations

from collections.abc import Iterable, Mapping
import html
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

from .base import (
    AggregateMetric,
    GameContext,
    GameMetric,
    MetricPlugin,
    MetricPresentation,
)
from .correctness import CorrectnessPlugin
from .length import LengthPlugin
from .outcome import OutcomePlugin
from .powerful_hand import PowerfulHandPlugin
from .rare_candy import RareCandyPlugin
from .post_ko_relay import PostKORelayPlugin
from .run_away_draw import RunAwayDrawPlugin
from .library_pressure import LibraryPressurePlugin


CORE_PLUGIN_FACTORIES = {
    "outcome": OutcomePlugin,
    "length": LengthPlugin,
    "correctness": CorrectnessPlugin,
    "powerful_hand": PowerfulHandPlugin,
    "rare_candy": RareCandyPlugin,
    "post_ko_relay": PostKORelayPlugin,
    "run_away_draw": RunAwayDrawPlugin,
    "library_pressure": LibraryPressurePlugin,
}
CORE_METRIC_IDS = tuple(CORE_PLUGIN_FACTORIES)


class MetricRegistry:
    def __init__(
        self,
        plugins: Iterable[MetricPlugin] = (),
        *,
        trusted_plugins: Iterable[str] = (),
    ) -> None:
        self._plugins: dict[str, MetricPlugin] = {}
        self._trusted_plugins: dict[str, bool] = {}
        self._presentation_errors: tuple[dict[str, object], ...] = ()
        trusted_ids = set(trusted_plugins)
        for plugin in plugins:
            self.register(plugin, trusted=plugin.metric_id in trusted_ids)

    def register(self, plugin: MetricPlugin, *, trusted: bool = False) -> None:
        if plugin.metric_id in self._plugins:
            raise ValueError(f"metric already registered: {plugin.metric_id}")
        self._plugins[plugin.metric_id] = plugin
        self._trusted_plugins[plugin.metric_id] = trusted

    def get(self, metric_id: str) -> MetricPlugin:
        return self._plugins[metric_id]

    def analyze(self, trace: dict, context: GameContext) -> dict[str, GameMetric]:
        results: dict[str, GameMetric] = {}
        for metric_id, plugin in self._plugins.items():
            try:
                results[metric_id] = plugin.analyze_game(trace, context)
            except Exception as exc:
                results[metric_id] = GameMetric(
                    metric_id=metric_id,
                    status="error",
                    numerator=0,
                    denominator=1,
                    value="metric_error",
                    evidence=(),
                    diagnostics=(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "game_id": context.game_id,
                            "opponent": context.opponent_name,
                        },
                    ),
                    payload={
                        "games": 1,
                        "error_games": 1,
                        "unfinished_games": 0,
                        "lifecycle_status": "error",
                        "error_reasons": {type(exc).__name__: 1},
                        "exception_type": type(exc).__name__,
                    },
                )
        return results

    def aggregate(
        self,
        results: dict[str, list[GameMetric] | GameMetric],
    ) -> dict[str, AggregateMetric]:
        aggregates: dict[str, AggregateMetric] = {}
        for metric_id, metric_results in results.items():
            values = metric_results if isinstance(metric_results, list) else [metric_results]
            try:
                aggregates[metric_id] = self.get(metric_id).aggregate(values)
            except Exception as exc:
                attempts = len(values)
                input_unfinished_games = sum(
                    result.status == "unfinished" for result in values
                )
                aggregates[metric_id] = AggregateMetric(
                    metric_id=metric_id,
                    numerator=0,
                    denominator=attempts,
                    value="metric_error",
                    by_opponent={},
                    payload={
                        "games": attempts,
                        "error_count": 1,
                        "error_games": attempts,
                        "unfinished_games": input_unfinished_games,
                        "error_reasons": {type(exc).__name__: 1},
                        "exception_type": type(exc).__name__,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
        return aggregates

    def present(
        self,
        aggregates: Mapping[str, AggregateMetric],
        results: Mapping[str, list[GameMetric] | GameMetric],
    ) -> dict[str, MetricPresentation]:
        presentations: dict[str, MetricPresentation] = {}
        presentation_errors: list[dict[str, object]] = []
        for metric_id, aggregate in aggregates.items():
            plugin = self.get(metric_id)
            renderer = getattr(plugin, "render", None)
            metric_results = results.get(metric_id, ())
            values = metric_results if isinstance(metric_results, list) else [metric_results]
            try:
                if callable(renderer) and self._trusted_plugins.get(metric_id, False):
                    presentation = renderer(aggregate, tuple(values))
                    if not isinstance(presentation, MetricPresentation):
                        raise TypeError("metric render must return MetricPresentation")
                else:
                    presentation = generic_metric_presentation(aggregate)
            except Exception as exc:
                presentation_errors.append(
                    {
                        "metric_id": metric_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                presentations[metric_id] = generic_metric_presentation(aggregate)
            else:
                presentations[metric_id] = presentation
        self._presentation_errors = tuple(presentation_errors)
        return presentations

    @property
    def presentation_errors(self) -> tuple[dict[str, object], ...]:
        return self._presentation_errors

    @property
    def plugins(self) -> tuple[MetricPlugin, ...]:
        return tuple(self._plugins.values())


def core_metric_plugins() -> tuple[MetricPlugin, ...]:
    """通过显式 factory 创建核心插件，保持 catalog 顺序稳定。"""
    return tuple(factory() for factory in CORE_PLUGIN_FACTORIES.values())


def generic_metric_presentation(aggregate: AggregateMetric) -> MetricPresentation:
    """为旧 plugin 提供不依赖专属 renderer 的稳定展示。"""
    metric_id = aggregate.metric_id
    title = f"指标：{metric_id}"
    markdown = "\n".join(
        (
            f"### {title}",
            "",
            "| 分子 | 分母 | value |",
            "| ---: | ---: | ---: |",
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    aggregate.numerator,
                    aggregate.denominator,
                    aggregate.value,
                )
            )
            + " |",
            "",
        )
    )
    html_id = html.escape(metric_id)
    html_numerator = html.escape(str(aggregate.numerator))
    html_denominator = html.escape(str(aggregate.denominator))
    html_value = html.escape(str(aggregate.value))
    html_fragment = (
        f'<section class="metric-plugin"><h3>{html_id}</h3>'
        '<table><thead><tr><th>分子</th><th>分母</th><th>value</th></tr></thead>'
        f"<tbody><tr><td>{html_numerator}</td><td>{html_denominator}</td>"
        f"<td>{html_value}</td></tr></tbody></table></section>"
    )
    return MetricPresentation(metric_id, title, markdown, html_fragment)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _load_module(module_path: str | Path) -> tuple[ModuleType, str | None]:
    raw_path = str(module_path)
    class_name: str | None = None
    if ":" in raw_path and not Path(raw_path).is_file():
        raw_path, class_name = raw_path.rsplit(":", 1)
    path = Path(raw_path)
    if path.is_file():
        unique_name = f"evaluation_metric_plugin_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(unique_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load metric module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module, class_name
    return importlib.import_module(raw_path), class_name


def load_metric_plugin(module_path: str | Path) -> MetricPlugin:
    """加载一个额外 MetricPlugin；模块必须提供唯一的本地实现类。"""
    module, explicit_class = _load_module(module_path)
    if explicit_class:
        candidate = getattr(module, explicit_class, None)
        candidates = [candidate] if inspect.isclass(candidate) else []
    else:
        candidates = [
            value
            for value in vars(module).values()
            if inspect.isclass(value)
            and value.__module__ == module.__name__
            and isinstance(getattr(value, "metric_id", None), str)
            and callable(getattr(value, "analyze_game", None))
            and callable(getattr(value, "aggregate", None))
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"metric module must expose exactly one plugin class: {module_path}"
        )
    try:
        plugin = candidates[0]()
    except TypeError as exc:
        raise ValueError("metric plugin class must have a no-argument factory") from exc
    if not isinstance(plugin, MetricPlugin):
        raise TypeError(f"metric plugin does not implement MetricPlugin: {module_path}")
    if not isinstance(plugin.metric_id, str) or not plugin.metric_id:
        raise ValueError("metric plugin metric_id must be a non-empty string")
    return plugin


def create_metric_registry(
    extra_module_paths: Iterable[str | Path] = (),
    profile_id: str = "core",
) -> MetricRegistry:
    """按 profile 创建 registry，并只追加额外插件，不允许覆盖已有 ID。"""
    from .profiles import get_metric_profile

    profile = get_metric_profile(profile_id)
    registry = MetricRegistry()
    for factory in profile.plugin_factories:
        registry.register(factory(), trusted=True)
    for module_path in extra_module_paths:
        plugin = load_metric_plugin(module_path)
        if plugin.metric_id in profile.metric_ids:
            reason = (
                "core metric"
                if plugin.metric_id in CORE_METRIC_IDS
                else "selected metric profile"
            )
            raise ValueError(
                f"dynamic plugin cannot override {reason}: {plugin.metric_id}"
            )
        registry.register(plugin, trusted=False)
    return registry


plugin_factory = load_metric_plugin
