# Alakazam V8 语义恢复设计

## 目标

在不使用当前仓库 `evaluation/` 运行框架的前提下，把
`work/alakazam_v8_current/` 的模块化策略恢复到至少不低于
`submission/alakazam_v8_luna_deck_opt/` 的可观察语义和 Sample 评测水平。

Target 使用旧版 `submission/alakazam_v8_luna_deck_opt/`；Start 使用当前
`work/alakazam_v8_current/`。隔壁仓库
`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py`
是本轮唯一对局运行入口。

## 已确认基线

本轮使用 Auto-Iteration Sample 的固定范围：17 个 opponent，每个 10 局，共 170
局。标准 runner 的 swap 配置用于形成实际先手/后手统计；交替局序不是策略约束，也
不是晋级目标。

截至 2026-07-21 的 fresh baseline：

| 版本 | W/L/D | 原始胜率 | 错误 | 二回合 Powerful Hand |
|---|---:|---:|---:|---:|
| Target `alakazam_v8_luna_deck_opt` | 118/50/2 | 69.4% | 0 | 32/170 (18.8%) |
| Start `alakazam_v8_current` | 8/128/34 | 4.7% | 34 | 0/170 |

Start 的 34 个错误均为 `NameError: name 'KADABRA' is not defined`，首次证据指向
`strategy/effects/dispatcher.py` 的 Night Stretcher 分支。未报错局还有
142/142 次击倒后无 ready attacker，说明 correctness 修复后仍需继续恢复策略语义。

## 方法

### 行为 oracle 与模块边界

- 旧版 `main.py` 只作为行为 oracle，用于逐动作比较、读取已验证的卡牌语义和寻找
  回归 case，不整体复制回新入口。
- `work/docs/` 的最终设计、规则报告和 AGENTS.md 是语义约束；当旧实现与官方规则
  冲突时，以官方规则为准并单独记录差异。
- 保留新框架的 Facts -> Routes/Plan -> Policies -> Effects 边界。
- 每轮只提出一个主要策略假设；静态契约、合法动作和跨局状态属于 correctness gate，
  不用胜率抵消。

### 语义恢复顺序

1. 修复模块导入和 effect dispatch 的 correctness 错误。
2. 恢复主要进化与攻击链：Active Kadabra/Abra 的合法进化、Rare Candy -> Alakazam、
   Psychic Energy 附加、Psychic Draw，以及攻击提交终止回合。
3. 恢复连续攻击能力：Bench Abra/Kadabra 的接力进化、必要的 Supporter/Item 搜索和
   不提前浪费攻击窗口。
4. 恢复 Dunsparce -> Dudunsparce -> Run Away Draw 的换位语义；Trading Places 永远
   不是普通换位，不能在其后继续攻击。
5. 恢复 Fezandipiti、Enhanced Hammer、Lana's Aid、Sacred Ash、Boss's Orders、
   Xerosic 及牌库保护边界。

每个阶段都从旧版对应分支和实际 trace 的 first divergence 开始，只修改达到测试所需
的最小模块。

## 验证循环

每轮遵循：

1. 在修改生产代码前先添加一个描述目标语义的最小失败测试，并确认它按预期失败。
2. 写最小实现，运行目标测试和相关策略测试。
3. 运行 `py_compile`、资产检查和入口契约检查。
4. 使用隔壁 evaluator 做 focused trace；只在 correctness gate 通过后运行完整 17x10
   Sample。
5. 用 `scripts/alakazam_auto_iter.py analyze` 生成轻量指标和 case 摘要，不把当前
   `evaluation/` 当作 runner。
6. 比较 Target、Start 和 candidate 的 W/L/D、错误、二回合 Powerful Hand、post-KO
   ready attacker 以及失败归因。

候选保留条件：错误为 0；没有新增非法动作、跨局污染或未完成对局；核心过程指标没有
不可解释回退；并且 fresh 17x10 Sample 达到或超过 Target 的结果水平。独立随机批次不
视为逐局配对，结论必须注明样本限制；若结果接近，则追加同口径 Sample 复核。

## 存储约束

- 完整对局 trace 只写入 `/tmp`，不写入仓库，不复制到 `work/auto-iteration/history_iterations/`。
- 每轮只保留轻量 `analysis.md`、`metrics.json`、`decision.md` 和必要的单局 case。
- 每轮开始和完整评测结束后运行 `df -h / /tmp`。
- 当根文件系统可用空间低于 10 GB 时，停止新增完整 trace，按目录修改时间 FIFO 删除
  最早的本轮临时评测目录；删除后再次检查空间，再继续评测。
- 不删除 Target、当前 candidate 的轻量记录和源代码；完整 trace 需要时重新生成。

## 非目标

- 不修改 `evaluation/` 以适配本轮任务。
- 不修改隔壁评测仓库或官方引擎源码。
- 不自动提交 Kaggle，不自动执行 git commit 或 git push。
- 不把随机一次 Sample 结果解释成所有 matchup 的长期因果证明。
