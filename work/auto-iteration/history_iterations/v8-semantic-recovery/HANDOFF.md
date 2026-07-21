# V8 语义恢复交接文档

## 交接范围

本阶段的目标是让 `work/alakazam_v8_current` 在新重构框架下尽量还原
`submission/alakazam_v8_luna_deck_opt` 的旧版策略语义。评测严格使用隔壁
`ptcg-agent-kaggle/eval/alakazam_replay.py`，Full 口径固定为 17 个对手 × 10 局，共
170 局；仓库内 `evaluation/` runner 没有用于这些结论。

所有历史结果位于本目录，每条记录的 Markdown、HTML 和 `result.json` 都放在对应目录内。
先看 [`index.html`](index.html) 的趋势，再看 [`iteration-023-full/iteration-023-full.md`](iteration-023-full/iteration-023-full.md)
和 [`iteration-024-full/iteration-024-full.md`](iteration-024-full/iteration-024-full.md) 的
最近两个 full 结果。

## 结论摘要

| 版本 | W/L/D | 胜率 | Meta | Powerful Hand | post-KO zero-ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target Baseline | 118/50/2 | 69.4% | 70.4% | 32/170 | 未建立同口径 |
| Start Baseline | 8/128/34 | 4.7% | 未记录 | 0/170 | 0/142 |
| 当前最佳 full：iteration-023-full | 65/105/0 | 38.2% | 37.1% | 10/170 | 125/169 |
| 最近 full：iteration-024-full | 50/117/3 | 29.4% | 30.1% | 8/170 | 157/188 |

`iteration-024-full` 的 3 个 draw 是对手侧 `effect 1197` evaluator 异常；candidate
errors 仍为 0。它没有超过 `iteration-023-full`，因此没有 promotion。当前源码保留了
Nighttime Mine 的语义修复和对应回归测试，但不应把它称为性能改善。

## 已完成的语义恢复

- 修复重构后 `KADABRA` 未导入导致的错误终局。
- 恢复 Active Alakazam 已可攻击时，先把合法 Bench Dunsparce 进化为 Dudunsparce 的
  过牌/接力语义。
- 恢复 Bench Dudunsparce 的 `Run Away Draw` 在攻击前作为普通过牌动作的语义。
- 恢复 Nighttime Mine 在合法可用时立即提交的旧版语义；对应测试为
  `test_nighttime_mine_is_played_before_a_legal_attack`。
- V8 agent、effects、facts 回归测试当前共 46 个通过。

## 仍未对齐的重点

动作分布对照显示，重构框架仍没有忠实覆盖旧版的非攻击 Active 路线：

- 旧版有明显的 Shaymin 出场、Fezandipiti ex 出场、Xerosic 和 Retreat；当前
  `iteration-023/024` trace 中这些动作几乎或完全没有出现。
- 新版 continuity 只在有限的 post-KO 条件下处理 Fezandipiti，未完整复现旧版的
  “先出场/补手牌/再把准备好的 Alakazam 接回 Active”路径。
- 新版没有统一的 Retreat intent。`ActionKind.RETREAT` 和 option decoder 已存在，
  但策略层没有覆盖旧版 `_retreat_improves_attack` 的明确交接条件。
- Xerosic 在简单 fixture 中可命中，但 full trace 没有形成旧版频率；需要审查
  `TurnFacts` 的伤害、对手手牌计数、Supporter budget、攻击路线和意图优先级是否在
  真实 observation 中同时满足。
- Nighttime Mine 的机制测试已通过，但 `iteration-024-full` 回退，说明“语义正确”
  与“整体结果改善”必须分开验证，不能直接继续放大该动作的优先级。

## 给 gpt-5.6-sol 的 review 任务

建议 review 按以下顺序进行，每次只提出一个主要假设并先写失败测试：

1. 对照 `submission/alakazam_v8_luna_deck_opt/main.py` 的旧版主行动排序，完整恢复
   Shaymin/Fezandipiti 的非攻击 Active 出场、过牌和 Retreat handoff；不要只添加一个
   card ID 分支，要确认合法选项、区域、手填 Energy 和回合预算都能穿过 orchestrator。
2. 给 Retreat 建立明确的 `ActionIntent` 及单元测试：只有 Bench 有已经具备攻击条件的
   Alakazam、当前 Active 不能合理继续攻击且本回合未 retreat 时才提交；不要把
   Dudunsparce 的 `Trading Places` 误当作普通换位。
3. 追踪 Xerosic 从事实到意图到匹配的完整数据流，使用真实 trace case 验证：Active
   Alakazam 已受伤、对手手牌至少 6、当前 Active 不能 KO 且没有更好的 Boss KO 时，
   Xerosic 应在攻击前使用。
4. 重新检查 Nighttime Mine 的设计文档边界和 full 结果，不要因为 focused 机制通过就
   把当前 `iteration-024` 晋级为 baseline。
5. 每个候选都先跑已有 V8 单测，再跑 12 局 focused，最后才跑 fresh 17×10 full；只
   使用 Auto-Iteration Sample，记录实际先后手，不要求局序交替。

## 评审基准与停止条件

- correctness gate 必须为 candidate errors `0`。
- 长期性能最低目标是达到或超过 Target 的 170 局口径，而不是只改善某个 focused
  指标。
- 任何局部语义修复都不能改变分母或把 unavailable case 归为 agent error。
- `iteration-023-full` 是当前性能参考，`iteration-024-full` 是最近代码状态的观察
  结果；两者都不是 Target。

## 常用入口

```bash
python3 -m unittest -v tests.test_alakazam_v8_agent \
  tests.test_alakazam_v8_effects tests.test_alakazam_v8_facts
python3 scripts/render_v8_history.py \
  --history-dir work/auto-iteration/history_iterations/v8-semantic-recovery
```

不要把 `/tmp` 中的完整 trace 复制进仓库。磁盘剩余空间低于 10 GiB 时，按 FIFO 删除最
早的完整临时评测目录，只保留轻量 history 和最新调试证据。
