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
  --output-dir docs/reports/kaggle/alakazam-v7-auto-iter/iter-00-baseline
```

默认从 trace 中实际出现的 agent `role` 自动识别标签。只有在确认参数与 trace 中的完整
role（包括日期后缀）完全一致时才传 `--agent-label`；不匹配的旧标签会被 analyzer 回退到
实际 role，避免把所有我方动作误判成对手动作并生成虚假的 0% 指标。

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

`run` 默认只保存 evaluator summary，不生成数百 MB 的逐局 trace。只有需要复盘的最新一轮才显式追加
`--save-traces`；旧完整 trace 在关键 case 抽取后移动到 macOS Trash。

比较两份已经生成的报告：

```bash
python3 scripts/alakazam_auto_iter.py compare \
  --control docs/reports/kaggle/alakazam-v7-auto-iter/iter-01/control \
  --candidate docs/reports/kaggle/alakazam-v7-auto-iter/iter-01/candidate \
  --output-dir docs/reports/kaggle/alakazam-v7-auto-iter/iter-01/comparison
```

`metrics.json`、`cases.jsonl` 和 `analysis.md` 只保存精炼结果；需要保存时，完整 trace 留在隔壁
评测仓库 `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/`。
本 repo 正常保留每一轮的轻量文档和指标；外部评测目录只保留最新一轮完整 trace，旧轮次
在关键 case 已抽取后清理，不把数百 MB 的逐局 JSON 提交进本 repo。`empty_bench_run_away_draw_count` 是硬错误，第二回合 `Powerful Hand`、
击倒后 ready attacker 和胜率是节奏与结果 guardrail。

每轮的改动、指标和 `accept/observe/reject` 决策持续记录在 [`ITER_PROCESS.md`](ITER_PROCESS.md)。

## Best artifact 与定时提交

只有被接受的 iteration 才能更新 `BEST_STRATEGY.json`。更新时先从当前工作区生成带
iteration label 的不可变归档：

```bash
bash scripts/promote_v7_best.sh 20 iter-20-example
```

该命令会把归档路径和 SHA-256 写入 marker，并同步一份 immutable archive 到
`~/Library/Application Support/pokemon-tcg-ai-battle/v7-best/`。每天 08:05 的
LaunchAgent 从这个不受 `~/Documents` 访问限制的目录校验 marker 并提交到 Kaggle，
不会重新打包正在继续迭代的工作区；因此工作区的后续改动不会改变定时任务实际提交的
版本。首次安装或修改 LaunchAgent 后运行：

```bash
bash scripts/install_v7_best_schedule.sh
```

当前 best 仍为 `iter-15-patch-priority`。

定时 wrapper 在 Kaggle CLI 返回非零时会把带 iteration、label、archive 和 exit code 的
失败信息写到 `~/Library/Application Support/pokemon-tcg-ai-battle/v7-best/*.failed`，并
保留完整输出在 LaunchAgent 日志中；因此次日可以确认是提交失败还是尚未触发。

## 验证

```bash
python3 scripts/check_assets.py
python3 -m unittest -v tests.test_alakazam_v7_strategy tests.test_alakazam_v6_strategy
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v7_auto_iter
bash scripts/submit_kaggle_v7_best.sh --dry-run
"$HOME/Library/Application Support/pokemon-tcg-ai-battle/v7-best/submit_v7_best_launchd.sh" --dry-run
```

V7 的行为验收样例和 replay case 映射见 [`REFACTOR_DESIGN.md`](REFACTOR_DESIGN.md)。
官方 Kaggle replay 才是策略表现依据，本地 battle 只用于合法性、崩溃和状态泄漏检查。
