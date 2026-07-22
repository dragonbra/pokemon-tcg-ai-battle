# Kaggle 公共卡组 Arena

Arena 收集 `pokemon-tcg-ai-battle` 官方 Code 页面中的来源，保留 Notebook、输出文件和 submission 压缩包，并把其中可运行的 agent 适配为本仓库 evaluation 可加载的独立 package。`pokemon-tcg-ai-battle-challenge-strategy` 只在实际发现 submission 压缩包时纳入。

## 常用命令

```bash
python3 -m arena collect --include-challenge-archives --discussion-pages 2
python3 -m arena validate
python3 -m arena run --phase smoke --games 10 --checkpoint-every 10 --seed 20260722
python3 -m arena run --phase formal --games 20 --checkpoint-every 10 --seed 20260722
python3 -m arena run --phase continuous --games 20 --limit 100000 --checkpoint-every 10 --seed 20260722
python3 -m arena run --resume <run-id> --phase continuous --limit 100000
python3 -m arena report --run-id <run-id>
python3 -m arena discussions --pages 2
```

`collect` 会保存 Code metadata、Notebook/output、leaderboard 和当前账号 submissions 快照，以及 Discussion 主题和回复树。`arena/catalog/kaggle-sources.md` 是所有来源的 Markdown 目录；没有可运行 package 的来源也会保留并标明原因。public score 属于 Kaggle submission/team，不会强行挂到无法可靠映射的 Notebook；这种情况在目录中留空，并以独立 snapshot 为准。

## Rating 和筛选

默认榜单是 `official_gaussian_approx` 语义近似：初始 `mu=600`、`sigma=200`，匹配优先选择相近分数；同一逐局事实另算初始 600、K=32 的 `elo_compat`。官方没有公开完整 sigma 更新公式，因此报告会明确标注近似，不把它描述为官方源码复刻。

低于 `mu=300` 的卡组只有在至少 20 场有效对局、连续两个 checkpoint 都低于阈值时才标记 `demoted`。来源、package、错误局和 Rating 事件永远不删除；默认 Eligible 视图排除 demoted，All 视图和低质量附录保留它们。`--include-demoted` 可让持续调度继续使用这些卡组。

## 产物位置

- `arena/db/arena.sqlite3`：索引和可查询事实。
- `arena/sources/`：原始 Kaggle 来源和 immutable metadata。
- `arena/packages/`：规范化独立 package；使用本地官方 baseline `cg/`，不使用下载包内的运行时。
- `arena/runs/<run-id>/`：逐局 JSONL、Rating 事件和 checkpoint。
- `arena/reports/<run-id>/`：不可变 All/Eligible HTML/Markdown；`latest-all.*` 和 `latest-eligible.*` 是当前快照。

Arena 的本地结果用于兼容性、对局多样性和相对强度分析，不能替代 Kaggle leaderboard 的官方 public score。Arena 不执行 Kaggle submission，也不会自动改写 `evaluation/configs/opponents.json`。
