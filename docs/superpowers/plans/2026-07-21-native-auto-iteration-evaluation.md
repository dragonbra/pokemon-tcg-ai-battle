# Native AutoIteration Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本地 `evaluation/` 框架中原生提供 AutoIteration V8 Setup and Relay profile 的指标计算、结构化结果和指标专属展示。

**Architecture:** 扩展现有 `GameMetric/AggregateMetric`，增加 `MetricProfile` 和指标 presentation hook。批处理根据 profile 创建 registry，单局分析结果进入轻量 `games.jsonl` 引用，完整聚合 payload 进入 `metrics.json`，报告层组合通用表格和 plugin 专属 Markdown/HTML section。晋级决策仍不属于 evaluation。

**Tech Stack:** Python 3.11+、标准库 `dataclasses`/`unittest`/`json`、现有 evaluation worker/trace/reporting 实现。

## Global Constraints

- 只修改 `evaluation/`、对应 `tests/`、`evaluation/README.md` 和本设计/计划文档；不修改 `work/`、`submission/`、`scripts/` 或 `engine/`。
- 不实现 `promote`、`observe`、`reject`、`measurement_only` 或任何 candidate/control 晋级逻辑。
- 默认 `core` profile、现有核心 metric IDs、默认 CLI 和旧动态 plugin 必须保持兼容。
- AutoIteration 第二个己方回合先手使用 engine turn 3，后手使用 engine turn 4；Powerful Hand 成功必须是实际选择 `attackId=1072`。
- error、unfinished、unknown prize 和未到达目标回合不得静默从分母删除。
- 指标事件定义或分母变化必须通过 profile revision 隔离。
- 生产代码遵循 TDD：每个新增行为先写失败测试并确认失败，再写最小实现。
- 不自动执行 `git commit` 或 `git push`。每个任务结束只进行 diff/status 检查。
- 子代理约束：机械的单文件/单指标任务使用显式 `gpt-5.6-luna`；跨模块最终审查才可使用更强模型；不把紧耦合的 registry/batch 主链拆给多个并行写入代理。

---

## 文件结构

### 修改

- `evaluation/metrics/base.py`: 结果 payload、profile priority 和 presentation 数据类型。
- `evaluation/metrics/registry.py`: profile registry、presentation fallback、plugin 计算异常隔离。
- `evaluation/metrics/outcome.py`: 总体/先后手 payload。
- `evaluation/metrics/powerful_hand.py`: 先后手和 reached-turn payload。
- `evaluation/metrics/post_ko_relay.py`: 先后手和保守失败分类 payload。
- `evaluation/runner/batch.py`: profile 选择、metric refs、presentation 构建和 manifest。
- `evaluation/cli.py`: `--metric-profile` 解析和 BatchConfig 传递。
- `evaluation/reporting/models.py`: profile/presentation 的 ReportData 和安全转换。
- `evaluation/reporting/markdown.py`: profile metadata 和 plugin section。
- `evaluation/reporting/html.py`: profile metadata 和 plugin section。
- `evaluation/README.md`: profile CLI、输出和 AutoIteration 口径说明。

### 新增

- `evaluation/metrics/profiles.py`: `MetricProfile`、优先级、内置 profile catalog。
- `evaluation/metrics/setup_relay.py`: setup、Dunsparce bridge、extra/normal draw。
- `evaluation/metrics/attack_quality.py`: attack submission/resolution/Prize quality。
- `tests/test_evaluation_profiles.py`: profile catalog、registry 和 manifest metadata。
- `tests/test_evaluation_auto_iteration_metrics.py`: AutoIteration synthetic trace 指标。
- `tests/test_evaluation_metric_presentations.py`: plugin presentation 与报告渲染。

不新增依赖，不修改已有策略测试或策略代码。

## Task 1: 扩展 metric contract 与 registry presentation

**Files:**
- Modify: `evaluation/metrics/base.py`
- Modify: `evaluation/metrics/registry.py`
- Modify: `evaluation/runner/batch.py`
- Test: `tests/test_evaluation_metrics.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- `GameMetric(..., payload: Mapping[str, object] = {})` 和 `AggregateMetric(..., payload: Mapping[str, object] = {})` 保持旧位置参数兼容。
- 新增 `MetricPresentation(metric_id, title, markdown, html)`。
- `MetricRegistry.present(aggregates, results)` 返回 `dict[str, MetricPresentation]`；没有 `render` 的旧 plugin 使用通用展示。
- `MetricRegistry.analyze` 捕获单个 plugin 异常并生成该 plugin 的 error metric，不吞掉其他 plugin。

- [ ] **Step 1: Write the failing tests**

在 `tests/test_evaluation_metrics.py` 增加：

```python
def test_metric_payload_and_presentation_are_available_without_breaking_old_plugin(self):
    class PresentedPlugin:
        metric_id = "presented"

        def analyze_game(self, trace, context):
            return GameMetric(self.metric_id, "success", 1, 2, 0.5, (), (), {"seen": True})

        def aggregate(self, results):
            return AggregateMetric(self.metric_id, 1, 2, 0.5, {}, {"total": 1})

        def render(self, aggregate, results):
            return MetricPresentation(self.metric_id, "Presented", "## Presented", "<h2>Presented</h2>")

    registry = MetricRegistry((PresentedPlugin(),))
    result = registry.analyze({}, context())["presented"]
    aggregate = registry.aggregate({"presented": [result]})["presented"]
    presentation = registry.present({"presented": aggregate}, {"presented": [result]})["presented"]
    self.assertEqual(result.payload["seen"], True)
    self.assertEqual(aggregate.payload["total"], 1)
    self.assertIn("Presented", presentation.markdown)
```

同时增加没有 `render` 方法的旧 plugin 通用 fallback 和一个抛出异常的 plugin 测试，断言异常结果 `status == "error"`、其他 plugin 仍返回成功结果。

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_evaluation_metrics tests.test_evaluation_batch -v`

Expected: FAIL because `MetricPresentation`、payload fields 和 `MetricRegistry.present` 尚不存在。

- [ ] **Step 3: Write minimal implementation**

在 `base.py` 末尾增加带默认空 mapping 的 payload 和 `MetricPresentation`；在 registry 增加：

```python
def present(self, aggregates, results):
    views = {}
    for metric_id, aggregate in aggregates.items():
        plugin = self.get(metric_id)
        metric_results = results.get(metric_id, ())
        renderer = getattr(plugin, "render", None)
        if callable(renderer):
            views[metric_id] = renderer(aggregate, tuple(metric_results))
        else:
            views[metric_id] = generic_metric_presentation(aggregate)
    return views
```

旧 plugin 的构造和 `asdict()` 输出必须继续有效；`batch._metric_refs()` 增加 `numerator`、`denominator` 和 JSON-safe `payload`。

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_evaluation_metrics tests.test_evaluation_batch -v`

Expected: 新增测试和已有 evaluation metric/batch 测试全部 PASS。

- [ ] **Step 5: Refactor and inspect diff**

运行 `git diff --check` 和 `git status --short`，确认没有修改 `work/`、`submission/`、`scripts/`、`engine/`；不执行 commit。

## Task 2: Add MetricProfile catalog and CLI selection

**Files:**
- Create: `evaluation/metrics/profiles.py`
- Modify: `evaluation/metrics/registry.py`
- Modify: `evaluation/metrics/__init__.py`
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/cli.py`
- Test: `tests/test_evaluation_profiles.py`
- Test: `tests/test_evaluation_cli.py`

**Interfaces:**
- `MetricPriority(metric_id, priority, direction)`。
- `MetricProfile(profile_id, revision, description, priorities, plugin_factories)`。
- `available_metric_profiles() -> tuple[str, ...]`。
- `get_metric_profile(profile_id: str) -> MetricProfile`。
- `create_metric_registry(extra_module_paths=(), profile_id="core") -> MetricRegistry`。
- `BatchConfig.metric_profile_id: str = "core"`。
- CLI 新增 `--metric-profile`，默认 `core`。

- [ ] **Step 1: Write the failing tests**

在 `tests/test_evaluation_profiles.py` 增加：

```python
def test_auto_iteration_profile_declares_revision_priorities_and_metrics(self):
    profile = get_metric_profile("auto_iteration_v8_setup_relay")
    self.assertEqual(profile.revision, 2)
    self.assertEqual(profile.metric_ids, (...))
    self.assertEqual(profile.priorities[0].metric_id, "outcome")

def test_unknown_profile_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "unknown metric profile"):
        get_metric_profile("missing")
```

在 `tests/test_evaluation_cli.py` 的 parser fixture 中断言 `--metric-profile auto_iteration_v8_setup_relay` 被传到 BatchConfig。

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_evaluation_profiles tests.test_evaluation_cli -v`

Expected: FAIL because profile catalog and CLI argument do not exist。

- [ ] **Step 3: Write minimal implementation**

在 `profiles.py` 定义 profile dataclass、`CORE_PROFILE` 和 AutoIteration profile 的延迟导入 factory tuple；factory 只在 registry 真正创建 profile 时导入对应 plugin 类，避免 catalog 模块与后续新增 plugin 形成循环导入。先让 profile 选择和 manifest metadata 工作。`registry.create_metric_registry` 根据 profile factory 创建 plugin，再追加 dynamic modules；重复 ID 和动态覆盖都抛出 `ValueError`。

在 `BatchConfig` 保存 `metric_profile_id`，`_metric_registry` 将其传给 registry；CLI parser 增加：

```python
run.add_argument(
    "--metric-profile",
    default="core",
    choices=available_metric_profiles(),
    help="选择内置 metric profile",
)
```

choices 错误必须在 CLI 层清晰显示，直接 API 仍使用 `ValueError`。

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_evaluation_profiles tests.test_evaluation_cli -v`

Expected: profile catalog、CLI parser 和既有 CLI 测试全部 PASS。

- [ ] **Step 5: Refactor and inspect diff**

检查 `core` profile 的 plugin 顺序仍等于 `CORE_METRIC_IDS`，并运行 `python3 -m compileall -q evaluation`。

## Task 3: Implement setup/relay preparation and directional outcome payloads

**Files:**
- Create: `evaluation/metrics/setup_relay.py`
- Modify: `evaluation/metrics/profiles.py`
- Modify: `evaluation/metrics/outcome.py`
- Modify: `evaluation/metrics/powerful_hand.py`
- Test: `tests/test_evaluation_auto_iteration_metrics.py`

**Interfaces:**
- `SetupRelayPlugin.metric_id == "setup_relay"`。
- `SetupRelayPlugin.analyze_game(trace, context) -> GameMetric`。
- `SetupRelayPlugin.aggregate(results) -> AggregateMetric`。
- setup payload keys: `opening_components`、`all_four`、`bridge_opportunity`、`bridge_completed`、`draws_to_second_turn`。
- aggregate payload keys: `by_turn_order`、`opening_four_components`、`dunsparce_bridge`、`second_turn_draws`。

- [ ] **Step 1: Write failing synthetic-trace tests**

在新测试文件构造最小 step trace，覆盖：先手 turn 3 实际 attack 1072、后手 turn 4 实际 attack 1072、未到达第二回合、Active Dunsparce -> 观察 Dudunsparce/Active Alakazam 的保守 bridge、Ability draw 与 turn-start draw 分离。

核心断言形状：

```python
result = PowerfulHandPlugin().analyze_game(trace, context(candidate_first=False))
self.assertEqual(result.numerator, 1)
self.assertEqual(result.diagnostics[0]["target_turn"], 4)
self.assertEqual(result.payload["reached_second_turn"], True)
```

以及：

```python
aggregate = SetupRelayPlugin().aggregate(results)
self.assertEqual(aggregate.payload["second_turn_draws"]["all_games"]["games"], 2)
self.assertEqual(aggregate.payload["second_turn_draws"]["normal_draw_cards"], 1)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_evaluation_auto_iteration_metrics -v`

Expected: FAIL because setup plugin不存在且 existing PowerfulHand 没有 payload/target_turn 字段。

- [ ] **Step 3: Implement minimal setup plugin and directional fields**

沿用 `evaluation.metrics.trace_utils` 和旧 AutoIteration 逻辑的 observation/log 读取方式，不复制 agent 策略代码。单局 plugin 只使用 `trace_steps`、`current`、`player_at`、`logs`、`selected_options` 等事实；无法确认时返回 `unavailable`、`None` 或 `evidence_strength="proxy"`。

扩展 `OutcomePlugin.aggregate` 和 `PowerfulHandPlugin.aggregate` 的 payload，按 `candidate_first` 从 diagnostics 分组 `first`/`second`；PowerfulHand 记录 `target_turn`、`reached_second_turn` 和 `powerful_hand_available`。

在 AutoIteration profile factory 中注册 `OutcomePlugin`、现有健康指标、增强后的 PowerfulHand、`SetupRelayPlugin`、增强后的 `PostKORelayPlugin` 和 `AttackQualityPlugin`。

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_evaluation_auto_iteration_metrics tests.test_evaluation_metrics -v`

Expected: synthetic AutoIteration tests和既有 metric tests全部 PASS。

- [ ] **Step 5: Refactor and inspect diff**

确认 setup plugin 不把正常抽牌计入 extra draw，不把“合法 option 存在”计入 Powerful Hand 成功；运行 `python3 -m compileall -q evaluation`。

## Task 4: Implement post-KO classification and attack quality

**Files:**
- Create: `evaluation/metrics/attack_quality.py`
- Modify: `evaluation/metrics/post_ko_relay.py`
- Modify: `evaluation/metrics/profiles.py`
- Test: `tests/test_evaluation_auto_iteration_metrics.py`

**Interfaces:**
- `AttackQualityPlugin.metric_id == "attack_quality"`。
- attack payload keys: `attack_submissions`、`resolved_attacks`、`unresolved_attacks`、`unknown_prize_attacks`、`non_prize_attacks`、`powerful_hand`、`attacks`。
- relay payload keys: `opportunities`、`successes`、`failure_counts`、`evidence_strength`、`by_turn_order`。

- [ ] **Step 1: Write failing tests**

构造至少四个 synthetic cases：实际攻击后 type 15 结算且 Prize 减少、结算但 Prize 不变、攻击未结算、Prize 状态未知；断言只有已结算且 Prize 明确的攻击进入 non-prize 分母。

构造 post-KO cases 覆盖五个固定分类，并断言没有选项级事实时 `evidence_strength == "proxy"`，不会把 discard 中出现攻击线自动判成 recoverable action miss。

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_evaluation_auto_iteration_metrics -v`

Expected: FAIL because `AttackQualityPlugin` 和 relay classification 尚不存在。

- [ ] **Step 3: Implement minimal plugins**

使用 trace 中 agent step 的 selected attack option 与后续 logs 的 `type=15`、player index、Prize 数量快照配对；不能配对时只增加 unresolved/unknown audit count，不猜测结果。

在 `PostKORelayPlugin` 保留现有 zero-ready numerator/denominator 兼容字段，新增事件级 payload 和五类保守分类；分类只在 trace facts 足够时升级，否则使用 `nonterminal_no_field_route` 或 `evidence_strength="proxy"`。

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_evaluation_auto_iteration_metrics tests.test_evaluation_metrics -v`

Expected: attack quality、relay classification 和既有 post-KO tests全部 PASS。

- [ ] **Step 5: Refactor and inspect diff**

检查 attack quality 不读取对手构筑假设，不将 `attackId=1072` 的合法性替代为成功；运行 `git diff --check`。

## Task 5: Wire report data and plugin-specific Markdown/HTML visualization

**Files:**
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/reporting/models.py`
- Modify: `evaluation/reporting/markdown.py`
- Modify: `evaluation/reporting/html.py`
- Create: `tests/test_evaluation_metric_presentations.py`
- Modify: `tests/test_evaluation_batch.py`
- Modify: `tests/test_evaluation_reporting.py`

**Interfaces:**
- `ReportData.metric_profile: Mapping[str, object]`。
- `ReportData.presentations: Mapping[str, MetricPresentation]`。
- batch manifest `metric_profile` 包含 id、revision、metric_ids、priorities。
- `metrics.json` 每项包含 `payload`；`games.jsonl.metric_refs` 每项包含 status/numerator/denominator/value/payload。

- [ ] **Step 1: Write failing tests**

增加 batch fixture，选择 AutoIteration profile 后断言：

```python
self.assertEqual(result.manifest["metric_profile"]["id"], "auto_iteration_v8_setup_relay")
self.assertEqual(result.report_data.metric_profile["revision"], 2)
self.assertIn("setup_relay", result.report_data.presentations)
self.assertIn("payload", result.metric_results["setup_relay"])
```

报告测试断言 Markdown 出现 setup/attack quality 专属 section，HTML 包含对应 title 和 `report-data` JSON，同时不出现 `promote`、`reject`、`observe` 等晋级结果。

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_evaluation_metric_presentations tests.test_evaluation_batch tests.test_evaluation_reporting -v`

Expected: FAIL because ReportData 没有 profile/presentation 字段且 batch 没有调用 presentation。

- [ ] **Step 3: Implement report wiring**

batch 在 aggregate 后调用 `registry.present()`，构造 ReportData 并将 profile manifest 写入 `manifest.json`。markdown/html renderer 在固定总体和通用指标 section 后按 registry 顺序拼接 presentation；动态内容经 `json_ready`、`_cell` 或 `_text` 处理。

所有 presentation 失败只记录 `presentation_errors`，不能丢弃 metrics.json 或让其他 section 消失。

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_evaluation_metric_presentations tests.test_evaluation_batch tests.test_evaluation_reporting -v`

Expected: presentation、batch artifact 和旧 reporting tests全部 PASS。

- [ ] **Step 5: Refactor and inspect diff**

确认 `ReportData` 旧构造位置参数保持兼容或只增加尾部默认字段；确认 report 中没有晋级决策字段；运行 `git diff --check`。

## Task 6: Documentation, CLI smoke and full acceptance

**Files:**
- Modify: `evaluation/README.md`
- Test: `tests/test_evaluation_profiles.py`
- Test: `tests/test_evaluation_cli.py`

- [ ] **Step 1: Write failing documentation/CLI acceptance tests**

增加 CLI help 断言 `--metric-profile` 出现；增加 profile catalog 的 README contract test（读取 README 文本，断言包含 profile id、revision、`metrics.json`、`report.html` 和“不负责晋级”说明）。

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_evaluation_profiles tests.test_evaluation_cli -v`

Expected: README/CLI acceptance tests在文档或参数未完成时失败。

- [ ] **Step 3: Update evaluation README**

记录：

```bash
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --output /tmp/ptcg-evaluation
```

说明 profile revision、二回合 turn 3/4、实际 `attackId=1072`、分母、报告文件和晋级决策由调用层负责。

- [ ] **Step 4: Run focused and full verification**

依次运行：

```bash
python3 -m unittest tests.test_evaluation_profiles tests.test_evaluation_auto_iteration_metrics tests.test_evaluation_metric_presentations -v
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q evaluation
python3 scripts/check_assets.py
git diff --check
```

Expected: 所有命令 exit 0；资产检查不应因为 evaluation Python 文件变化而产生策略资产差异。

- [ ] **Step 5: Run a minimal real batch and inspect artifacts**

使用已有合法 package 和一个 catalog opponent，将输出写入 `/tmp/ptcg-auto-iteration-evaluation`，命令：

```bash
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents sue_alakazam \
  --games 1 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output /tmp/ptcg-auto-iteration-evaluation
```

检查 run 目录存在 `manifest.json`、`summary.json`、`games.jsonl`、`metrics.json`、`cases.jsonl`、`report.md`、`report.html`；检查 `manifest.metric_profile`、`metrics.*.payload` 和 `games.metric_refs.*.payload`。不把 trace 或 report 写入仓库。

- [ ] **Step 6: Final self-review**

用 `git status --short` 和 `git diff --stat` 确认变更范围；用 `rg` 检查 `evaluation/` 内没有 promotion decision 实现、绝对路径依赖或对手 ID 预设；确认未自动 commit/push。

## Plan self-review

- Spec coverage: Tasks 1-2 cover result/profile contracts; Tasks 3-4 cover all V8 metrics; Task 5 covers machine artifacts and visualization; Task 6 covers docs and acceptance.
- Placeholder scan: no `TBD`, `TODO`, “appropriate error handling” or undefined task reference is used as an implementation step.
- Type consistency: `MetricProfile` produces plugin factories; `MetricRegistry` consumes them; `BatchConfig.metric_profile_id` selects them; `ReportData` consumes registry presentations; report renderers consume `ReportData.presentations`.
- Scope: no task edits strategy, engine, external evaluator or promotion decision logic.

## Final status

本计划已完成执行。实现后的最终验收包括：

- [x] AutoIteration V8 profile、revision 2、指标 payload 和专属 Markdown/HTML presentation。
- [x] error、unfinished、unknown prize、未到达目标回合和 plugin exception 的审计字段与分母保留。
- [x] 默认 `core` 兼容、动态 plugin 追加/安全 fallback 和 profile 指标覆盖保护。
- [x] 451 个 unittest、`compileall`、`check_assets.py`、`git diff --check` 通过。
- [x] 最新真实单局 batch 产出 manifest、summary、games、metrics、cases、Markdown/HTML 和 trace。
- [x] Luna 只读 review 修复后结论为 Ready to merge。
