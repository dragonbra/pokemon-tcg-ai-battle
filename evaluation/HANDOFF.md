# Evaluation Framework 交接说明

本文是给项目级 Agent 和 AutoIteration 调用层的最短使用指南。Evaluation 负责运行对局、
通过 metric plugin 聚合指标、保留证据并生成报告；它不负责 candidate/control 比较，也不
负责 promote、observe、reject 等晋级决策。

## 首选入口

要求 Python 3.11+，在仓库根目录执行：

```bash
python3 -m evaluation list-opponents
python3 -m evaluation validate work/alakazam_v8_current
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output rl/runs/evaluation/alakazam_candidate
```

`validate` 必须先通过。`run` 完成后会打印 `run_id` 和报告目录。输出根目录位于
`rl/runs/evaluation/` 时，实验目录会自动获得顺序前缀，例如：

```text
rl/runs/evaluation/0001-alakazam_candidate/run-<uuid>/
```

直接打开该目录下的 `report.html` 即可查看完整语义报告：

```bash
open rl/runs/evaluation/0001-alakazam_candidate/run-<uuid>/report.html
```

正式复盘建议把 `--output` 指向 `rl/runs/evaluation/<label>`，以获得自动编号；对应的
training 和 candidate 产物也应使用相同的 `0001-<label>` 目录名。CLI
强制 `--opponents all --games 10` 或更高，即固定 17 个 opponent、至少 170 局；小样本
探索请使用训练日志、离线数据或 simulator smoke，不通过 repo evaluation CLI。

## 报告产物

每次运行在 `<numbered-output>/<run_id>/` 写入：

- `manifest.json`：候选、对手、参数、profile id/revision、metric ids、hash 和保留 trace。
- `summary.json`：总体结果、错误、未完成对局和按 opponent 汇总。
- `games.jsonl`：逐局轻量记录和 `metric_refs`，用于审计具体局面。
- `metrics.json`：各 plugin 的聚合 numerator、denominator、value 和 payload 原始数据。
- `cases.jsonl`：被选中的失败案例与 evidence；对应 trace 位于 `traces/`。
- `report.md`：无浏览器依赖的文本报告。
- `report.html`：独立 HTML，包含语义指标、原始指标、插件审计和案例。

默认只保留被选中案例的少量完整 trace；只有调试时才加 `--keep-temp`。

## AutoIteration 指标语义

使用 `auto_iteration_v8_setup_relay`（当前 revision 2）时，HTML 会把同一个 metric plugin
展开到下面几组。每行仍保留原始 `metric_id`、角色、方向、分子/分母和追踪目标，调用层
可以据此自行做晋级解释。

- 结果与正确性护栏：G0 correctness、总体胜率、实际先手/后手结果；错误和未完成对局不能
  被过程指标抵消。
- 阶段一：二回合实际选择 `attackId=1072` 的 Powerful Hand、起手四组件、无 Abra 时的
  Dunsparce bridge、全部样本/到达二回合/先手/后手的额外过牌，以及正常回合抽牌审计。
- 阶段二：Post-KO 机会中下一回合另一只 Alakazam 是否成功接力攻击。语义展示使用
  `payload.success_rate`；`metrics.json` 中旧的顶层聚合字段继续保留，便于兼容和审计。
- 阶段三：全部已结算攻击和 Powerful Hand 子集的未拿奖赏率，作为越低越好的惩罚项。
- 辅助健康与审计：Rare Candy 进化链、空 Bench Run Away Draw、低牌库消耗和对局长度。

实际先手使用 engine turn 3，实际后手使用 engine turn 4。若事件没有有效机会，报告显示
`未定义（无有效样本）`、`无机会` 或 `无有效攻击`，不会把零机会伪装成 0%。

## 调用层边界

调用层只需要消费 `manifest.json`、`metrics.json`、`summary.json` 和 `report.html`，并根据
自己的 profile revision 和业务规则解释 promote/observe/reject。不要把这些决策写进
Evaluation，也不要修改 `engine/source/`、候选策略或 opponent package 来适配指标。

需要在 Python 中嵌入完整批处理时，使用 `evaluation.runner.batch.BatchConfig` 和
`run_batch`；它们与 CLI 使用同一个 metric profile 和报告写入路径。只做已聚合数据的报告
重绘时，可以使用 `evaluation.reporting.ReportData`、`write_report`，但这不会重新运行对局。

更多 profile、plugin 和 trace 生命周期说明见 [`evaluation/README.md`](README.md)。
