# Alakazam V7 实验提交

V7 基于 V6 的运行时和固定卡组，只重构策略。卡组文件必须与 V6 完全一致；策略重点
是按真实宝可梦 TCG 回合预算规划进化、附能、过牌、换位和攻击。

## 本版重点

- Active Abra 不攻击；带能量的 Active Abra 保留给完整的 Rare Candy + Alakazam 路线。
- Active 是 Shaymin 等非攻击者时，带 Psychic Energy 的 Bench Abra 保留给 Rare Candy；其它无能量 Bench Abra 先自然进化成 Kadabra。
- 第一只 Alakazam 可以攻击后，Bench Abra 默认自然进化为 Kadabra 并使用 Ability，
  不连续消耗 Rare Candy 制造没有立即收益的第二只 Alakazam。
- 只有观察到对手 `Budew (235)` 使用 `Itchy Pollen (323)` 后，才在我方下一个回合
  进入 Item Lock 路线：所有 Item 暂停，优先自然进化，并把 Enriching Energy 给
  Dunsparce/Dudunsparce 过牌。
- Dudunsparce 只有在确实存在接班宝可梦时才使用 Run Away Draw；Bench 为空时禁止把
  唯一的 Active 洗回牌库。普通 Retreat 可以在终局闭环中把带 Psychic Energy 的
  Alakazam 送到 Active；Trading Places 永不选择。
- Kadabra 非 KO 攻击仅是最后手段。牌库 10 张是保护线，只有当过牌能在本回合闭合
  最后 Prize 时才允许越过保护线。

## AutoIter 工作流

AutoIter 固定本目录的 `deck.csv`，只接受有明确 replay case 假设的最小策略改动。
每轮先分析 trace，再做 case 回归，之后用同一对手矩阵运行 control/candidate 对照；
胜率和 Meta 加权胜率是重要 guardrail，但不会替代具体 case 分析。

分析已有 trace：

```bash
python3 scripts/alakazam_auto_iter.py analyze \
  --report-dir /path/to/ptcg-agent-kaggle/reports/alakazam_v7 \
  --output-dir reports/kaggle/alakazam-v7-auto-iter/iter-00-baseline \
  --agent-label alakazam_v7
```

运行本地 evaluator（不会提交 Kaggle）：

```bash
python3 scripts/alakazam_auto_iter.py run \
  --evaluator-root /path/to/ptcg-agent-kaggle \
  --agent submission/alakazam_v7_auto_iter/main.py \
  --cg-path submission/alakazam_v7_auto_iter \
  --label alakazam_v7_auto_iter \
  --opponents romanrozen_v9,pilkwang_v2 \
  --games 10 \
  --output-dir /tmp/alakazam-v7-auto-iter-focus
```

比较两份已经生成的报告：

```bash
python3 scripts/alakazam_auto_iter.py compare \
  --control reports/kaggle/alakazam-v7-auto-iter/iter-01/control \
  --candidate reports/kaggle/alakazam-v7-auto-iter/iter-01/candidate \
  --output-dir reports/kaggle/alakazam-v7-auto-iter/iter-01/comparison
```

`metrics.json`、`cases.jsonl` 和 `analysis.md` 只保存精炼结果；完整 trace 留在隔壁
评测仓库。`empty_bench_run_away_draw_count` 是硬错误，第二回合 `Powerful Hand`、
击倒后 ready attacker 和胜率是节奏与结果 guardrail。

每轮的改动、指标和 `accept/observe/reject` 决策持续记录在 [`ITER_PROCESS.md`](ITER_PROCESS.md)。

## 验证

```bash
python3 scripts/check_assets.py
python3 -m unittest -v tests.test_alakazam_v7_strategy tests.test_alakazam_v6_strategy
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v7
```

V7 的行为验收样例和 replay case 映射见 [`REFACTOR_DESIGN.md`](REFACTOR_DESIGN.md)。
官方 Kaggle replay 才是策略表现依据，本地 battle 只用于合法性、崩溃和状态泄漏检查。
