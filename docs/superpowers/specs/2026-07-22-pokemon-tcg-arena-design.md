# Kaggle 公共卡组 Arena 设计说明

## 目标

在当前仓库建立一个可持续运行的本地 Arena，收集
`pokemon-tcg-ai-battle` 官方 Code 页面中的公开实现，并补充扫描
`pokemon-tcg-ai-battle-challenge-strategy` 方案中实际包含 submission 压缩包的内容。
Arena 使用本仓库的官方引擎和 evaluation worker，让所有可兼容卡组进行初筛、循环匹配和
长期对战，形成可审计的胜率、Rating、卡组克制关系和来源资料。

Arena 负责来源管理、适配、调度、Rating、候选池筛选和综合报告；现有 `evaluation/` 继续
负责标准 package 校验、worker 隔离、官方引擎对局和基础指标，不把 Arena 的晋级判断写回
evaluation。

## 已确认的官方行为

2026-07-22 通过 Kaggle 官方 competition pages CLI 读取
`pokemon-tcg-ai-battle` 的 Evaluation 页面，确认以下公开契约：

- Submission 首先进行与自身副本的 Validation Episode；失败的 Submission 标记为 Error。
- 正常 Submission 以 `mu0=600` 加入全部 Submission 池。
- 官方技能 Rating 是高斯分布 `N(mu, sigma)`，`mu` 是技能估计值，`sigma` 是不确定性。
- 匹配尽量选择 Rating 接近的 Submission；新 Submission 会获得更高对局频率。
- 胜负更新同时参考上一轮 `mu` 的预期结果和双方 `sigma`；平局让双方 `mu` 向均值靠拢。
- Episode 的胜利分差不会参与技能 Rating 更新。

官方页面没有公布 sigma 初值、完整更新公式或匹配权重。因此第一版默认使用
`official_gaussian_approx`，明确记录其参数和实现版本；同时提供可重放的
`elo_compat` 结果作为兼容排序。报告不得把近似算法宣称为官方源码复刻。

## 目录和持久化边界

Arena 所有新增内容位于顶层 `arena/`：

```text
arena/
├── __init__.py
├── __main__.py
├── README.md
├── collect.py
├── catalog.py
├── adapters.py
├── scheduler.py
├── rating.py
├── runner.py
├── reports.py
├── db/arena.sqlite3
├── catalog/kaggle-sources.md
├── catalog/kaggle-sources.json
├── catalog/leaderboard snapshots under sources/_meta/
├── catalog/overrides.json
├── sources/<competition>/<owner>/<slug>/
├── packages/<deck-id>/
├── runs/<run-id>/
└── reports/<run-id>/
```

原始 Notebook、submission 压缩包、解压后的兼容 package、package hash、对局结果、Rating
事件和生成报告都保留。SQLite 是可查询索引，JSON/JSONL 是可审计事实来源；报告可以从已
保存数据重绘，不要求重新运行对局。

## 来源收集和状态

`arena collect` 使用 Kaggle CLI/API 分页取得 competition Code 清单，并建立不可变的来源
快照。主来源是 `pokemon-tcg-ai-battle`；challenge-strategy 只纳入其中真实包含 submission
压缩包的方案。每个来源记录：

- URL、标题、作者、kernel slug、版本、创建/更新时间；
- Notebook votes、views、comments；
- public score、用户当前 public score、采集时间和字段来源；
- 原始下载文件 hash、是否发现压缩包；
- 适配状态、验证结果、失败原因和人工修复说明。

来源状态固定为：

- `exact_submission`：原始 submission 可直接运行；
- `notebook_reconstructed`：Notebook 含完整 agent/deck，使用本地标准 `cg/` 适配；
- `metadata_only`：只有介绍、分析或不可运行代码，不进入 Arena；
- `invalid`：存在可执行意图但无法通过 package/运行验证。

适配只允许修复路径、导入、入口函数和 package 外壳，不改变策略决策逻辑。原始来源与
适配后文件必须分开保存，并通过 hash 和 manifest 关联。

## 标准 package 和验证

每个进入 Arena 的卡组都必须形成独立 package：

```text
arena/packages/<deck-id>/
├── main.py
├── deck.csv
└── cg/
```

package 必须通过现有 validator 的 60 张牌、官方卡牌 ID、入口加载、`cg` runtime 和
物理文件 hash 检查。每个 package 先运行自身副本的 validation 对局；该对局只验证可运行性，
不进入 Rating 和公开胜率。缺失、异常或超时要记录为状态，不静默丢弃。

我们的 `work/alakazam_v8_current` 作为 `internal_reference` 加入 Arena，参与对局和
Rating，但不计入公开卡组汇总，也不自动改变现有 evaluation opponent catalog。

## 对局阶段和调度

### 初筛

- 对每个可运行卡组进行自我验证；
- 每个不同卡组配对 10 局；
- 双方先后手各 5 局；
- 保存逐局结果、状态、错误、步数、双方 package hash、engine/cg hash 和执行参数；
- 对局结果只使用 win/loss/draw，不用 Prize 分差更新 Rating。

### 正式和持续循环

正式轮次默认每配对 20 局，允许通过 CLI 增加到 50 局。初始循环完成后，调度器根据当前
Rating 选择相近对手，并优先补足尚未充分交手的配对；新卡组具有更高初始对局优先级。
每场结束后立即持久化 Rating 事件和 pair stats，checkpoint 后更新排行榜和 HTML。

调度器必须可恢复：每个 run 有固定 manifest、参数和随机种子；已完成的 game_id 不重复执行，
中断后从下一个未完成配对继续。随机选择只影响后续调度，不覆盖已写入的事实结果。

## Rating

默认引擎配置：

```text
method: official_gaussian_approx
initial_mu: 600
initial_sigma: 200
low_quality_threshold: 300
```

每个卡组保存当前 `mu`、`sigma`、排名、有效局数、胜率和状态；每场比赛写入完整的
`rating_event`，包括比赛前参数、结果、预期结果、变化量、比赛后参数和算法 revision。

`official_gaussian_approx` 采用高斯技能估计的语义：结果的意外程度和双方不确定性共同
影响 mu/sigma，平局向均值收敛。所有参数必须在 manifest 中显式保存，便于重放和调整。
`elo_compat` 从同一组逐局结果重算，默认初始分 600、K=32、平局按 0.5 处理；它只作为
兼容对照，不替代默认排行榜。

## 低质量卡组和双版本统计

低质量阈值是 `mu < 300`。为避免少量随机局误淘汰，卡组只有在至少 20 场有效 Rating 对局
且连续两个 checkpoint 都低于 300 时，才标记为 `demoted`。被标记后不删除任何来源、
package 或对局；默认不再进入 active opponent pool，但可以使用 `--include-demoted` 继续
对战。

每轮生成两个语义独立的汇总：

- `all`：所有可运行卡组，包括已 demoted 卡组；
- `eligible`：排除稳定低于 300 分的卡组。

内部参考卡组不计入公开群体平均值，但在两个报告中单独展示。

## 卡组分类和图片

卡组默认依据 Pokémon 卡牌数量和主要进化线自动推断主要宝可梦；无法可靠推断时使用
`Unknown Archetype`。`catalog/overrides.json` 可以人工覆盖 archetype、显示名、主要
宝可梦、图片 URL 和 active 状态。显示名保留主要宝可梦、作者和 Notebook 版本，例如：
`Lucario — Kiyota sample rule-based v1`。

HTML 使用官方卡牌图片 URL 懒加载，同时保存 card ID 和名称；图片无法访问时显示文字和
card ID 占位，不把大批图片二进制复制进仓库。

## HTML/Markdown 报告

每次 checkpoint 更新 `arena/reports/latest.html` 和 `latest.md`，run 完成后写入带 run_id
的不可变报告。报告包含：

- 当前 Rating 排行榜、mu/sigma、排名变化和对局数；
- All/Eligible 切换和低质量淘汰附录；
- 每个卡组总胜率、先后手拆分和错误/未完成对局；
- 卡组对卡组的 W/L/D 矩阵和胜率热力图；
- 强弱克制关系摘要，注明样本数不足；
- Kaggle 来源标题、URL、作者、votes、public score 和适配状态；
- 主要宝可梦图片和人工分类说明；
- 官方高斯近似与 Elo 兼容排行榜的并列结果；
- run manifest、engine/package hash、Rating revision 和数据更新时间。

报告是静态自包含 HTML，排行榜数据嵌入页面；持续运行时通过覆盖 latest 文件实现阶段性
更新，不启动后台服务，也不自动打开浏览器。

## CLI 边界

第一版 CLI：

```bash
python3 -m arena collect
python3 -m arena validate
python3 -m arena run --phase smoke
python3 -m arena run --phase formal
python3 -m arena run --resume <run-id>
python3 -m arena report
python3 -m arena report --include-demoted
```

后续可接 macOS `launchd` 做定时抓取和循环对战，但第一版先保证 CLI、断点恢复、报告重绘
和数据审计可靠。Arena 不执行 Kaggle submission，不把下载凭据或 `.env` 写入仓库。

## 成功标准

第一阶段完成后必须具备：

1. 官方 Code 来源快照和资料索引；
2. 每个来源明确的可运行/不可运行状态；
3. 至少一轮可恢复的初筛循环赛；
4. 默认高斯 Rating、Elo 对照和逐场 Rating 事件；
5. 300 分阈值下的 All/Eligible 双版本统计；
6. 排行榜、克制矩阵、来源元数据和卡牌图片 HTML；
7. `alakazam_v8_current` 内部参考结果；
8. 失败、超时、适配和样本不足均可审计，不通过静默删除掩盖。
