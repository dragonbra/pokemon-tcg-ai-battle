# V7 AutoIter Trace 分析与轮换计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改固定 `deck.csv` 的前提下，补齐 iter-38/39 的可审计 AutoIter 结果，并让外部评测目录只保留最新一轮完整 trace。

**Architecture:** 复用 `scripts/alakazam_auto_iter.py` 的统一分析口径，从隔壁评测仓库读取完整 `game_*.json`，将轻量指标与 case 摘要写入当前 repo 的对应 iteration 目录。原始 trace 不进入当前 repo；清理时使用 macOS Trash 的可恢复移动，并在文档中记录保留边界。

**Tech Stack:** Python 3.11、现有 AutoIter analyzer、Markdown/JSONL、macOS Trash、`check_assets.py`、unittest。

## Global Constraints

- 固定 `submission/alakazam_v7_auto_iter/deck.csv`，策略迭代不得改动卡组。
- 当前 repo 保留每轮 `decision.md`、`advisor.md`、`analysis.md`、`metrics.json`、`cases.jsonl` 等轻量记录。
- 完整逐局 trace 保留在 `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/`，仅保留最新完整评测轮次。
- 旧 trace 只能在确认关键 case 已抽取后移动到 Trash，不执行不可恢复的递归删除。
- 不提交 Kaggle，不自动 commit。

---

### Task 1: 核对数据与分析口径

**Files:**
- Read: `scripts/alakazam_auto_iter.py`
- Read: `reports/kaggle/alakazam-v7-auto-iter/iter-37/analysis.md`
- Read: `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-38-second-turn-bench-insurance-20260720/`
- Read: `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-39-fresh-bench-second-turn-20260720/`

- [ ] 确认两个全量目录都包含 17 个对手、每个 10 个 `game_*.json`，并记录 summary 中的胜负、异常和 trace 覆盖。
- [ ] 确认 analyzer 的 CLI 参数、full-trace 与 focused-trace 的区别，以及 post-KO/second-turn 指标是否可以在同一批数据上计算。
- [ ] 确认当前生产代码来自 iter-39 candidate，且 `deck.csv` 的 git diff 为空。

### Task 2: 生成 iter-38/39 轻量报告

**Files:**
- Create: `reports/kaggle/alakazam-v7-auto-iter/iter-38/{metrics.json,analysis.md,cases.jsonl,advisor.md,decision.md}`
- Create: `reports/kaggle/alakazam-v7-auto-iter/iter-39/{metrics.json,analysis.md,cases.jsonl,advisor.md,decision.md}`
- Modify: `scripts/alakazam_auto_iter.py` only if an existing analyzer defect prevents the documented metrics from being generated.

- [ ] 对 iter-38 全量 trace 运行现有 analyzer，输出统一 metrics、失败 case 和分析摘要。
- [ ] 对 iter-39 全量 trace 运行现有 analyzer，输出统一 metrics、失败 case 和分析摘要。
- [ ] 在每个 `decision.md` 中简洁记录改动、control 版本、control/candidate 指标、采纳状态和证据来源。
- [ ] advisor 必须结合卡牌描述解释至少一个主要 failure case；不能把违反“Abra 不攻击”的建议写成可采纳方案。

### Task 3: 更新迭代总账并审计 deck 不变

**Files:**
- Modify: `submission/alakazam_v7_auto_iter/ITER_PROCESS.md`
- Modify: `reports/kaggle/alakazam-v7-auto-iter/README.md` if trace retention policy needs clarification.

- [ ] 追加 iter-38 和 iter-39 的最小改动、前后指标、随机样本说明、action error、采纳状态。
- [ ] 明确 iter-38 因 broad Poffin gate 退回，iter-39 因 narrow fresh-bench gate 的结果和当前处理结论。
- [ ] 用 `git diff -- submission/alakazam_v7_auto_iter/deck.csv` 验证固定卡组未被改动。

### Task 4: 可恢复地轮换外部 trace

**Files:**
- External directory: `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/`

- [ ] 在移动前确认 iter-38/39 的轻量报告和关键 case 已落盘，并确认 iter-39 是最新完整 trace。
- [ ] 将 iter-08、iter-26、iter-27、iter-32、iter-33、iter-37、iter-38 等旧完整目录逐个移动到 `/Users/hejinyu/.Trash/`，保留 iter-39 全量目录；不移动轻量 summary 目录，除非它确实包含完整逐局 JSON。
- [ ] 移动后核对外部目录大小和剩余完整 trace，记录可恢复位置与清理结果。

### Task 5: 验证

**Commands:**
- `python3 scripts/check_assets.py`
- `python3 -m py_compile scripts/alakazam_auto_iter.py submission/alakazam_v7_auto_iter/main.py`
- `python3 -m unittest tests.test_alakazam_auto_iter tests.test_alakazam_v7_auto_iter_strategy tests.test_v7_best_submission`
- `git diff --check`

- [ ] 所有命令通过；若 analyzer 或测试发现问题，先修复并重新运行对应验证。
- [ ] 最终报告当前最佳 iteration、iter-38/39 的采纳结论和仍需下一轮处理的 failure case；不宣称达到 90%，除非有同口径证据。
