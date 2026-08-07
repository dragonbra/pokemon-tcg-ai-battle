# Evaluation Framework

`evaluation/` 用于本地候选策略的可重复结构化评测，不是 Kaggle 正式提交目录。

## CLI

```bash
python3 -m evaluation list-opponents
python3 -m evaluation --pool opponents list-opponents
python3 -m evaluation validate evaluation/arena/candidates/<candidate>
python3 -m evaluation run \
  --candidate evaluation/arena/candidates/<candidate> \
  --opponents all \
  --games 10 \
  --output .tmp/evaluation/<candidate>
```

默认 pool 是 `frozen`：固定 0019 Epoch 13 Foundation policy、`source_id=0` 与 49 个 exact-deck
identity。49 个 opponent 共用一个常驻 batched GPU policy service；被评测 candidate 使用第二个
常驻 GPU service。每个 request 都携带本方 exact 60-card deck，因此共享权重不会丢失卡组身份。
CPU worker 只执行未修改的 official engine 状态转移。Frozen 模式默认双方使用 `cuda:0`，可用
`--candidate-device` 与 `--opponent-device` 显式选择 GPU；缺少双边共享 GPU inference 时 fail
closed。

本地 evaluation 默认先从未改动的 `engine/source/` 构建并锁定
`engine/build/seeded_official/0001/libcg.so`，再把该绝对路径传给每个 worker；manifest 内的
source、adapter、compiler 和 library hash 用于审计 `0001` 的实际内容。candidate 始终占物理 player 0，opponent 始终占 player 1；harness
只在官方 `IS_FIRST` 选择上强制先后手，不交换 deck 槽位。每个 matchup 的相邻两局组成一对，
共享 engine seed 和 Search seed，仅交换先后手；policy seed 按单局独立派生。报告 manifest
记录 official source、adapter、动态库 hash、ABI、三套 seed 公式和换手合同。相同 runtime hash、
两套 exact deck、三套 seed 与确定性策略动作会产生相同 trace；随机策略还必须使用记录的
policy RNG 流。该保证是本地评测合同，不扩展到 Kaggle 托管 runtime。

标准 checkpoint 验收使用 `--opponents all --games 10`，即 49×10=490 局、每套先后手各 5 局。
48×2=96 局只用于快速 diagnostic。旧异构 Kaggle-derived pool 保留在 `arena/opponents/`，通过
全局参数 `--pool opponents` 显式选择，只作为 secondary external-generalization evidence。

`run` 也支持 `--control PACKAGE`（只在报告中展示对比）、调试用 `--visualize` / `--keep-temp`、
`--max-steps N` 和可重复的 `--metric-module MODULE[:Class]`。默认不生成可视化帧。每次 CLI
正式报告固定要求完整启用 catalog、每个至少 10 局；写入 `.tmp/evaluation/` 的 smoke 与诊断
允许显式选择子集。自 0013 起，正式报告写入
`experiments/<project_id>/evaluation/V<n>_<tag>.html`，并由同名
`rl_runs/<project_id>/versions/V<n>_<tag>/artifact/evaluation.json` 反向引用。control
不会触发 promotion、reject 或其他自动晋级决定。动态模块必须是独立的 `MetricPlugin`，
只能追加统计，不能使用任何核心 metric ID。

`--workers N` 可以覆盖并行对局数；默认按当前进程可用的 CPU affinity 自动选择，最多为
本机已校准的 `8`。无论并行度如何，每局仍在独立 worker
进程中加载双方策略和官方 engine runtime，指标分析与报告写入保持 catalog/game 的固定
顺序。默认 `--worker-cpu-threads 1`，以限制 PyTorch、OpenMP、MKL、OpenBLAS 和 NumExpr
内部线程，避免多个进程过度订阅 CPU。实际 workers、线程限制和 wall time 都内嵌在报告的
manifest 数据中；其他硬件可显式覆盖并先把小规模校准报告写到 `.tmp/evaluation/benchmarks/`。

legacy CPU `opponents` 模式仍遵守每局独立策略进程和内部线程限制。Frozen 模式则把双方策略
推理移到两个 GPU 服务，但每局 official engine 仍在独立 worker，报告仍按固定 catalog/game
顺序落盘。本机 8 个物理核心、当时固定 18×10 局官方 engine 对战的实测如下。该表是历史
legacy CPU 吞吐基准，
不代表当前 catalog 规模；数字只证明吞吐，不构成
策略强度证据；其他硬件需重新校准。

| 调度方式 | 完成情况 | 总耗时 | 吞吐 |
|---|---:|---:|---:|
| 串行 | 180/180 | 234.41 秒 | 0.77 games/s |
| 4 workers，未限制内部线程 | worker error | 488.41 秒 | 不采用 |
| 4 workers × 1 thread | 180/180 | 60.29 秒 | 2.99 games/s |
| 8 workers × 1 thread | 180/180 | 40.23 秒 | 4.47 games/s |
| 当前 report-only 默认 8 × 1 | 180/180，0 error | 38.27 秒 | 4.70 games/s |

Frozen 双边 GPU dynamic batching 已进入正式 CLI；报告 manifest 分别记录 candidate/opponent
inference mode、device、batch 参数，以及 Frozen pool/catalog/policy hash。

需要 V8 setup/relay 语义指标时使用内置的 `auto_iteration_v8_setup_relay` profile
（保留该名称作为兼容 ID），当前为 `revision 7`：

```bash
python3 -m evaluation run \
  --candidate evaluation/arena/candidates/<candidate> \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --output .tmp/evaluation/setup-relay-smoke
```

这个 profile 在核心 outcome、health 和 correctness 指标之外提供 setup/relay、
Powerful Hand、post-KO relay 和 attack quality 的结构化 payload。Powerful Hand 的
第二个己方回合按实际先后手使用 engine turn 3/4，只有实际选择 `attackId=1072`
才计为成功；未到达目标回合、未完成对局和未知 Prize 状态会保留在审计字段及相应
分母中，不会被静默删除。profile 的 id、revision、metric ids、聚合 payload、单局轻量
`metric_refs` 和专属语义展示全部内嵌在独立 `report.html` 中。

Frozen League 全量循环评测使用 `--metric-profile league_deck_quality`（revision 1）。该 profile
不复用胡地专属 Powerful Hand 作为全局指标，而是统一提取首次攻击回合、攻击连续率、可攻击却
未提交的回合、Prize/attack、多 Prize 回合、KO 后攻击间隔、Bench/进化/能量场面、牌库消耗、
Supporter/手填能量利用、对手攻击受阻、关键主攻成形、Ability 动作、伤害事件和主动离场。
`evaluation/metrics/league_profiles.py` 保留对历史 51 套 Frozen-0019 deck 的策略类别覆盖；每类只从
通用原子指标中选择重点解释字段，并记录独立的 reward warning。过程指标只用于解释 checkpoint、
定位退化和形成 reward 假设，不能覆盖 official-engine 胜负护栏。

`length` 把官方 engine 的先手玩家 phase turn 与紧随其后的后手玩家 phase turn 合成一个完整
回合，使用 `ceil(engine_turn / 2)` 作为结束回合。页面分别展示获胜和失败对局的平均结束回合与
分布图，并按 candidate 在该局是先攻还是后攻分色。终局先后手 phase 仍保留在单局 payload 中，
但不作为可见图表分组。error/unfinished 不进入胜负回合统计，
observed engine turn 仍保留在单局 payload 中；action selection 次数只用于性能与异常审计。

正式 opponents 全量循环评测长期保存在 `arena/combat_mat/`：`index.html` 是矩阵入口，
`reports/<package>/<run_id>/report.html` 保留每个 package 面对完整 catalog 的源报告（这是独立的
`arena/combat_mat` 矩阵合同，不同于 RL 正式实验报告），
`matrix.json` 保存聚合数据。正式池变化后需要按新 catalog 重新运行所有 package（包含自身，
每个有向单元格 10 局）；这些重要结果不得放在 `.tmp/`。页面展示总体、先攻和后攻胜率，
并将先攻与后攻玩家阶段按 `ceil(engine_turn / 2)` 合成完整回合。

该 profile 的 `report.html` 会按语义分组展示：结果与正确性护栏、
阶段一二回合基础能力、阶段二 Post-KO 接力能力、阶段三攻击质量惩罚项，以及辅助健康与审计
指标。每一行同时保留中文语义名、原始 `metric_id`、角色、方向、语义值和追踪目标；阶段三
攻击质量把分子/分母直接合并到语义值中，不再单设重复列。
“辅助健康与审计指标”是页面最后一个可见 section；原始指标表、failure class、case、control
差异和 plugin presentation 不再重复渲染到它后面，只保留在 HTML 内嵌 JSON 中供审计。
页面的语义值按 profile 定义选择 payload 字段。例如 `post_ko_relay` 的语义展示值
使用 `payload.success_rate`（成功数 / 机会数），而报告内嵌原始 metrics 中旧的顶层失败率
仍原样保留，便于审计。

Evaluation 只负责指标测量、证据审计和可视化，不执行自动迭代或规则策略修改，也不负责
晋级决策、淘汰决策或 candidate/control 比较。

## Package 与 catalog

被评测 candidate 与 legacy opponent 必须是标准 package：根目录包含 `main.py`、60 行
`deck.csv` 和完整的 `cg/` runtime。Frozen opponent 是特例：每个 deck identity 只保存
`deck.csv` 与 manifest，共享 `_policy/` 的入口、模型和 `cg` identity。`validate` 会打印 package
名称、deck hash、cg tree hash 与 60-card 校验结果；任何预检、卡组或 cg hash 兼容错误都会在
启动对局前退出。

默认 Frozen catalog 是 [`configs/frozen_0806.json`](configs/frozen_0806.json)，引用
`arena/frozen_pools/0806_kaggle_top100_plus_v1/` 下的 55 个轻量 exact-deck identity，并分别绑定
唯一的 Policy-0806 candidate runtime 和 Policy-0019 opponent runtime。旧 51 套 Frozen-0019
及其 Combat Mat 已迁入 `archive/evaluation/pre_0806_0019/`。legacy catalog 是
[`configs/opponents.json`](configs/opponents.json)。`all`
按所选 catalog 启用项的固定顺序选择；传入逗号分隔名称时，
只能选择 catalog 中已启用的项。新增的候选 opponent 先作为标准 package 放入
`arena/candidates/<name>/`，完成验证并经用户确认后，才按实际关键宝可梦分配
`<archetype>_<NN>` 正式名称并迁入 `arena/opponents/`。正式目录名使用 ASCII snake_case 和
两位序号，例如 `alakazam_dudunsparce_01`；catalog 同时保存易读 display name 和 1–2 个
确实存在于该 deck 的代表宝可梦 card ID。Report 的紧凑胜率图使用这些元数据展示小卡图；
图片加载失败时仍保留文字名称和胜率。候选在获准收编前保持原名。这里的 arena candidate
描述存放与准入状态，CLI `--candidate` 则描述本次被测对象。

Frozen-0806 的独立配置是 [`configs/frozen_0806.json`](configs/frozen_0806.json)，资产位于
`arena/frozen_pools/0806_kaggle_top100_plus_v1/`。它固定保存 55 个 unique exact deck 和
256 局整数频率：240 局按 2026-08-06 Top 100 的 41 个 exact deck 做最大余数分配，16 局来自
101–500 名的 14 个潜力 exact deck。该表同时作为 Arena、PPO rollout batch 和最终冻结评测的
分布合同；运行时不得重新按权重抽样。对手角色固定引用 Policy-0019，主视角固定引用
Policy-0806。默认 CLI 已使用这份固定 schedule；`--games` 只适用于 legacy opponent pool，
Frozen-0806 每次固定运行 256 局。

## 产物和 trace 生命周期

正式 RL 实验的 `run` 由 `BatchConfig/run_batch` 在
`rl_runs/evaluation/<000N-project>/V<n>_<tag>.html` 只写入一个 HTML 文件，即
`report_only` 默认产物契约。manifest、summary、逐局轻量记录、聚合 metrics、case 摘要和
presentation 数据都内嵌在该独立 HTML 中。每次正式报告完成后，同项目的 `index.html` 会根据
全部 `V*.html` 自动重建，按版本号展示 candidate/run、局数、胜负、error、unfinished、胜率和
完成率，并提供到每份完整报告的相对链接。`.tmp/evaluation/<purpose>` 等非正式输出仍在
`run-<id>/report.html` 下隔离保存。指标插件通过稳定的 metric ID 注册；
`--metric-module` 的模块路径由 metrics registry 动态加载，只能追加统计，不得覆盖核心
输出；动态模块的 HTML renderer 默认不受信任，会退回安全的通用指标展示。只有内置
profile plugin 才能提供专属 HTML section。核心指标固定为 outcome、length、correctness、powerful_hand、rare_candy、
post_ko_relay、run_away_draw 和 library_pressure。

完整逐局 trace 只在评测过程中临时保存，默认结束后全部清理，不再长期复制三场 case
trace。`--keep-temp` 仅用于调试，不能把临时文件当成长期资料；若同时需要可视化帧，显式
增加 `--visualize`。正式长期复盘仍优先使用 Kaggle 官方 episode/replay。

该框架在报告中记录 package hash、cg hash、seeded runtime hash、参数和结果。新 seeded
runtime 合同承诺在相同平台/build、相同完整输入和相同动作序列下复现本地 engine trace；它不
声称跨编译器位级一致，也不能替代 Kaggle 的正式策略评估。
