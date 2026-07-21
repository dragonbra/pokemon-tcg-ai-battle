from __future__ import annotations

from collections.abc import Iterable
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

from .base import AggregateMetric, GameContext, GameMetric, MetricPlugin
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
    def __init__(self, plugins: Iterable[MetricPlugin] = ()) -> None:
        self._plugins: dict[str, MetricPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: MetricPlugin) -> None:
        if plugin.metric_id in self._plugins:
            raise ValueError(f"metric already registered: {plugin.metric_id}")
        self._plugins[plugin.metric_id] = plugin

    def get(self, metric_id: str) -> MetricPlugin:
        return self._plugins[metric_id]

    def analyze(self, trace: dict, context: GameContext) -> dict[str, GameMetric]:
        return {
            metric_id: plugin.analyze_game(trace, context)
            for metric_id, plugin in self._plugins.items()
        }

    def aggregate(
        self,
        results: dict[str, list[GameMetric] | GameMetric],
    ) -> dict[str, AggregateMetric]:
        return {
            metric_id: self.get(metric_id).aggregate(
                metric_results if isinstance(metric_results, list) else [metric_results]
            )
            for metric_id, metric_results in results.items()
        }

    @property
    def plugins(self) -> tuple[MetricPlugin, ...]:
        return tuple(self._plugins.values())


def core_metric_plugins() -> tuple[MetricPlugin, ...]:
    """通过显式 factory 创建核心插件，保持 catalog 顺序稳定。"""
    return tuple(factory() for factory in CORE_PLUGIN_FACTORIES.values())


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


def create_metric_registry(extra_module_paths: Iterable[str | Path] = ()) -> MetricRegistry:
    """创建核心 registry，并只追加额外插件，不允许覆盖核心 ID。"""
    registry = MetricRegistry(core_metric_plugins())
    for module_path in extra_module_paths:
        plugin = load_metric_plugin(module_path)
        if plugin.metric_id in CORE_METRIC_IDS:
            raise ValueError(f"dynamic plugin cannot override core metric: {plugin.metric_id}")
        registry.register(plugin)
    return registry


plugin_factory = load_metric_plugin
