# Native AutoIteration Evaluation Design

## 目标

将 `work/auto-iteration/` 当前约定的评测内容原生实现到本仓库的
`evaluation/` 框架，使 AutoIteration 调用层只需要选择候选 package、对手矩阵和
metric profile，就能获得可复现的单局事实、聚合指标、关键 case 以及 Markdown/HTML
展示。

本次改造只负责测量和展示，不负责 candidate/control 晋级。`promote`、`observe`、
`reject`、`measurement_only` 以及任何 promotion threshold 仍由调用层解释。

## 范围边界

### 包含

- profile 标识、revision、指标优先级和 profile manifest；
- metric plugin 的结构化单局 payload、聚合 payload 和指标专属展示；
- 当前 V8 Setup and Relay profile 所需的 outcome、二回合 setup、Powerful Hand、
  post-KO relay、attack quality 及已有 correctness/health metrics；
- 总体、实际先手、实际后手、机会样本、事件样本、对手诊断和审计字段；
- `metrics.json`、`games.jsonl`、`cases.jsonl`、`report.md`、`report.html` 中的一致契约；
- 原生 CLI 的 `--metric-profile` 接口和动态额外 plugin 兼容性；
- evaluation 专项测试、README 和 profile 使用说明。

### 不包含

- 晋级状态、candidate/control 比较和自动接受/拒绝；
- 修改 `scripts/auto_iteration_report.py`、`scripts/alakazam_auto_iter.py` 或任何
  `work/`、`submission/` 策略代码；
- 修改官方 engine、worker trace 的运行语义或 Kaggle 上传流程；
- 将所有历史 profile 的旧数字强行转换为新口径。

## 已有系统约束

- worker 已经在独立进程中产生完整 trace，metric 层只消费 trace 和
  `GameContext`，不导入 agent 模块。
- 默认 core profile 的现有指标 ID 和默认 CLI 行为必须保持兼容。
- AutoIteration 的第二个己方回合固定为先手 engine turn 3、后手 engine turn 4；成功
  必须是实际选择 `attackId=1072`，不能用“有合法选项”替代。
- error、unfinished、unknown prize 和未到达目标回合不得静默从分母删除。
- profile revision 是指标事件定义、分母或替代卡牌语义发生变化时的比较隔离边界。
- 完整 trace 继续由 `TraceStore` 临时保存并按既有 case 规则最多保留三份。

## 架构

### Profile

新增 `MetricProfile`，由 profile id、revision、说明、优先级和 plugin factories 组成。
profile 只定义“测量什么以及怎样展示”，不含 promotion 决策。

```python
@dataclass(frozen=True)
class MetricProfile:
    profile_id: str
    revision: int
    description: str
    priorities: tuple[MetricPriority, ...]
    plugin_factories: tuple[Callable[[], MetricPlugin], ...]
```

内置 profile 至少包括：

- `core`：现有全部核心指标，作为默认兼容 profile；
- `auto_iteration_v8_setup_relay`：V8 profile revision 2 的完整指标集合。

CLI 使用 `--metric-profile` 选择内置 profile，默认值为 `core`。现有
`--metric-module` 仍用于追加外部 plugin，外部 plugin 不能覆盖 core 或当前 profile 已
注册的 metric ID。

### Metric plugin

保留现有 `analyze_game()` 和 `aggregate()` 生命周期，同时给结果增加结构化 payload。
新增字段放在 dataclass 末尾并提供默认值，以保留现有外部 plugin 的位置参数兼容性。

```python
@dataclass(frozen=True)
class GameMetric:
    metric_id: str
    status: str
    numerator: int
    denominator: int
    value: float | int | str | None
    evidence: tuple
    diagnostics: tuple
    payload: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class AggregateMetric:
    metric_id: str
    numerator: int
    denominator: int
    value: float | int | str | None
    by_opponent: dict[str, dict[str, object]]
    payload: Mapping[str, object] = field(default_factory=dict)
```

需要专属展示的 plugin 实现
`render(aggregate: AggregateMetric, results: tuple[GameMetric, ...])`，返回
`MetricPresentation`；
旧 plugin 没有该方法时使用通用的分子/分母/value 表格，不破坏动态 plugin。

```python
@dataclass(frozen=True)
class MetricPresentation:
    metric_id: str
    title: str
    markdown: str
    html: str
```

registry 负责调用 presentation hook 和统一异常处理。指标计算异常会变成该局的
`status="error"` metric，并进入现有 case/error 流程；展示异常不能修改 metrics.json，
只在 manifest/report diagnostics 中记录 `presentation_error`。

### 批处理数据流

```text
CLI --metric-profile
  -> MetricProfile
  -> MetricRegistry
  -> worker trace
  -> analyze_game(trace, context)
  -> games.jsonl.metric_refs（含 status/value/numerator/denominator/payload 摘要）
  -> aggregate(results)
  -> metrics.json
  -> plugin presentations + generic report sections
  -> report.md / report.html
```

`ReportData` 增加 profile metadata 和 presentation collection；现有报告消费者只读取
已有字段时仍可工作。`metrics.json` 保留完整聚合 payload，`games.jsonl` 的
`metric_refs` 保留单局的轻量 payload、分子、分母和状态，展示层不重新推断指标。

## AutoIteration V8 profile 指标

### Outcome

使用实际结果记录总体、先手、后手的 games、wins、losses、draws、errors、unfinished 和
win rate。先后手分组使用 `GameContext.candidate_first`，不假设交替局序等于实际结果。

### Powerful Hand

复用现有实际 option selection 识别逻辑，确认己方第二回合是否选择
`attackId=1072`。payload 同时记录：

- `all_games`：成功数 / 全部对局数；
- `reached_second_turn`：成功数 / 实际到达第二回合数；
- `by_turn_order`：先手、后手实际样本；
- `reason_counts`：selected、not selected、unavailable、second turn not reached；
- `powerful_hand_available` 与实际选择分离的审计字段。

### Setup and relay preparation

新增 setup plugin，单局识别：

- 第一回合起始 Active Abra；
- 第一回合起始手牌 Rare Candy；
- Alakazam 或 Poke Pad/Hilda/Dawn 检索路线；
- Psychic Energy 或 Hilda 路线；
- 四组件同时满足，仅作为状态观察；
- 无 Active Abra 且 Active Dunsparce 的 bridge opportunity；
- 观察到 Dudunsparce、第二回合 Active Alakazam 且实际 Powerful Hand 的保守 bridge
  代理；
- 从己方第一回合开始到第二回合结束的 Ability/card-effect extra draws，正常回合抽牌
  单独保存。

聚合结果必须同时提供全样本平均值、实际到达第二回合平均值和先后手平均值；未到达目标
回合的全样本平均按 0 纳入，并明确写入分母。

### Post-KO relay

在现有 KO event 检测基础上扩展 opportunity、success、先后手以及失败分类：

- `field_route_miss`；
- `recoverable_discard_miss`；
- `recoverable_route_incomplete`；
- `nonterminal_no_field_route`；
- `terminal_no_resource`。

当前轻量 trace 无法证明合法回收 option 时，分类必须保守并在 payload 中标记证据强度，
不能把候选资源直接写成“正确回收机会未使用”。

### Attack quality

新增 attack quality plugin，追踪实际 attack option、后续 type 15 结算、Prize 状态
变化和 Powerful Hand 子集：

- submissions；
- resolved/unresolved；
- unknown prize；
- resolved 且 Prize 未减少的 non-prize attacks；
- Powerful Hand 对应子集；
- attack id、prize delta、damage evidence。

主比例只使用“已结算且 Prize 状态明确”的攻击作为分母。未完成和未知状态保留为审计
字段，不进入比例。

### Health metrics

`correctness`、`length`、`rare_candy`、`run_away_draw`、`library_pressure` 继续由
核心 profile 提供。AutoIteration profile 可以展示它们，但不产生任何 promotion gate。

## 报告契约

`manifest.json` 增加：

```json
{
  "metric_profile": {
    "id": "auto_iteration_v8_setup_relay",
    "revision": 2,
    "metric_ids": ["outcome", "length", "correctness", "powerful_hand", "rare_candy", "post_ko_relay", "run_away_draw", "library_pressure", "setup_relay", "attack_quality"],
    "priorities": ["outcome", "powerful_hand", "post_ko_relay", "attack_quality"]
  }
}
```

`metrics.json` 的每个 metric 保留旧字段，并增加：

- `payload`：机器可读的 profile-specific 结构；
- `by_opponent`：对手诊断；
- `diagnostics`：事件和分母审计。

`games.jsonl.metric_refs.<metric_id>` 使用以下轻量字段：`status`、`numerator`、
`denominator`、`value` 和 `payload`。完整 evidence 仍只保留在临时/选中的 trace 中，避免
长期 games 文件膨胀。

`report.md` 和 `report.html` 由公共 renderer 输出总体/对手矩阵、通用指标表，再按
plugin presentation 插入专属 section。HTML presentation 只能由可信 plugin 产生，所有
动态字符串仍通过公共转义辅助函数处理。

## 文件边界

### 修改

- `evaluation/metrics/base.py`：扩展结果契约和 presentation 类型；
- `evaluation/metrics/registry.py`：profile registry、presentation fallback 和错误隔离；
- `evaluation/metrics/outcome.py`、`powerful_hand.py`、`post_ko_relay.py`：补充 profile
  所需结构化字段；
- `evaluation/runner/batch.py`、`evaluation/cli.py`：选择 profile、写入 manifest 和
  report data；
- `evaluation/reporting/models.py`、`markdown.py`、`html.py`：渲染 profile metadata 和
  plugin sections；
- `evaluation/README.md`：记录 profile CLI 和结果契约。

### 新增

- `evaluation/metrics/profiles.py`：profile 数据结构和内置 profile catalog；
- `evaluation/metrics/setup_relay.py`：二回合 setup、bridge、draw 和 relay 观测；
- `evaluation/metrics/attack_quality.py`：攻击质量指标；
- `tests/test_evaluation_profiles.py`：profile 选择和 manifest 契约；
- `tests/test_evaluation_auto_iteration_metrics.py`：V8 指标 fixture、分母和分类；
- `tests/test_evaluation_metric_presentations.py`：Markdown/HTML presentation；
- `docs/superpowers/plans/2026-07-21-native-auto-iteration-evaluation.md`：实现计划。

不修改 `work/`、`submission/`、`scripts/` 和 `engine/`。

## 测试和验收

必须先为每个新增行为写失败测试，再实现最小代码。验收至少包括：

1. 默认 core profile 的既有 metric IDs、CLI 和旧动态 plugin 测试全部通过；
2. `auto_iteration_v8_setup_relay` 能从 synthetic trace 识别实际 `attackId=1072`，
   先手 engine turn 3、后手 engine turn 4；
3. Powerful Hand 的全样本和 reached-turn 分母不同且数值正确；
4. setup 四组件、Dunsparce bridge 和 extra/normal draw 不混淆；
5. post-KO 五类分类和保守证据标记稳定；
6. attack quality 正确区分提交、结算、Prize unknown 和 non-prize；
7. profile revision、优先级和 metric IDs 写入 manifest；
8. Markdown/HTML 包含 plugin 专属 section，且不执行晋级决策；
9. `python3 -m unittest discover -s tests -p 'test_*.py'`、`python3 -m compileall -q evaluation`
   和 `python3 scripts/check_assets.py` 通过；
10. 一个最小真实 evaluation batch 能生成完整 report artifacts，且运行结束后临时 trace
    按既有策略清理。

## 风险与处理

- 旧 trace 的 action/log 字段不完整：指标返回 `unavailable` 和审计 reason，禁止猜测；
- 新旧 profile 语义不同：通过 revision 和 manifest 隔离，不在 evaluation 内比较；
- 动态 plugin 缺少 presentation：使用通用表格，不让展示缺失阻断指标计算；
- plugin 计算异常：局部标记 metric error，保留其他指标和 case；
- AutoIteration 调用层尚未切换 CLI：保留现有 `--metric-module` 和默认 profile，原有
  调用仍可运行，切换只需要选择 `--metric-profile`。
