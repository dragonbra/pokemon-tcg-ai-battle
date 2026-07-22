# AutoIteration

AutoIteration 是一套长期策略研究和版本迭代约定，不是某一轮实验的指标清单，也不
是一个只按胜率排序的跑局脚本。

## 下一个 session 的阅读顺序

开始新的 iteration 前，按以下顺序读取：

1. [`GOALS.md`](GOALS.md)：确认长期目标和当前目标层级；
2. [`FRAMEWORK.md`](FRAMEWORK.md)：理解从目标、问题归因到 promotion 的通用流程；
3. [`ITERATION_RULES.md`](ITERATION_RULES.md)：检查跨轮次不变的硬约束；
4. [`metrics/README.md`](metrics/README.md)：确认指标 profile 的加载方式；
5. 当前 active profile：本轮为 [`metrics/v8-setup-relay.md`](metrics/v8-setup-relay.md)；
6. [`templates/ITERATION.md`](templates/ITERATION.md)：按模板建立本轮记录。

不要默认沿用上一轮指标。长期目标和迭代规则是稳定层；具体关注指标属于独立 profile，
每一轮必须明确引用哪个 profile。

## 目录结构

```text
work/auto-iteration/
├── README.md
├── FRAMEWORK.md
├── GOALS.md
├── ITERATION_RULES.md
├── metrics/
│   ├── README.md
│   └── v8-setup-relay.md
├── templates/
│   └── ITERATION.md
└── history_iterations/
    ├── index.html
    └── iteration-001/
        ├── index.html
        ├── iteration.md
        └── result.json
```

| 路径 | 责任 | 变化频率 |
| --- | --- | --- |
| `FRAMEWORK.md` | 通用的研究流程、证据和结论解释 | 低 |
| `GOALS.md` | 长期目标和目标层级 | 很低 |
| `ITERATION_RULES.md` | 每轮都必须遵守的约束和晋级规则 | 低 |
| `metrics/` | 某一阶段实际关注的指标 profile | 随研究问题替换 |
| `templates/` | 每轮报告的记录结构 | 低 |
| `history_iterations/` | 每轮轻量结果、HTML 页面和跨轮次汇总 | 每次评测新增 |

## 启动一轮迭代

每轮开始时先完成以下动作：

- 说明本轮属于策略、卡组、测量还是评测配置迭代；
- 固定 control、candidate、卡组状态和评测配置；
- 读取长期目标，并说明本轮缺口如何影响长期目标；
- 选择一个 active metric profile，不在结果出来后临时换指标；
- 按 active profile 写明指标优先级；低优先级惩罚项不能抵消高优先级护栏的回退；
- 提出一个主要假设和至少一个替代路径；
- 用 [`templates/ITERATION.md`](templates/ITERATION.md) 记录 focused、full 和外部验证计划。

## 结果归档

AutoIteration 文档只保存研究约定和轻量决策记录。每轮评测结果放在
`history_iterations/<iteration-id>/`，并由 `history_iterations/index.html` 汇总。
原始大批 trace 仍放在 `/tmp` 或 replay 目录，不复制进这个目录。

旧的独立报告生成脚本已经退役。当前完整语义报告由 `evaluation` 的 metric profile 直接
生成，并写入带全局编号的 RL run：

```bash
python3 -m evaluation run \
  --candidate work/<candidate-name> \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/_runs/<run-name>/evaluation
```

`history_iterations/` 保留旧规则策略时期的历史页面，不再由当前工作流重绘。新报告中的
`metrics.json` 保存可复核的分母、分类、攻击质量惩罚项和按对手胜率，`report.html` 展示
组件状态和详细分母。二回合和攻击质量主指标仍合并全部对手。当前 trace 无法证明
弃牌区资源一定存在合法回收选项，因此 `recoverable_discard_miss` 只能作为保守候选，
不能直接当作 agent 错误；同理，攻击但未拿奖赏只能作为结果惩罚项，原因需要后续
evaluator 的选项级事实。

本目录不实现 agent，也不规定 `evaluation/` 的具体代码结构；未来评测实现应逐步满足
这里定义的记录、分母、归因和证据要求。
