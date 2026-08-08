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

默认 pool 是 `frozen`：固定 Frozen-0806 的 55 个 exact-deck identity 和 Policy-0806 冻结 policy。
55 个 opponent 共用一个常驻 batched GPU policy service；被评测 candidate 使用第二个常驻 GPU
service。每个 request 都携带本方 exact 60-card deck，因此共享权重不会丢失卡组身份。
CPU worker 只执行未修改的 official engine 状态转移。Frozen 模式默认双方使用 `cuda:0`，可用
`--candidate-device` 与 `--opponent-device` 显式选择 GPU；缺少双边共享 GPU inference 时 fail
closed。

本地 evaluation 默认先从未改动的 `engine/source/` 构建并锁定
`engine/build/seeded_official/0002/libcg.so`，再把该绝对路径传给每个 worker；manifest 内的
source、adapter、compiler 和 library hash 用于审计 `0002` 的实际内容。`0002` 与 `0001` 使用
相同 official ABI，额外静态链接 libstdc++/libgcc，避免纯 ctypes 进程依赖 PyTorch 或 shell 的
动态库加载顺序。candidate 始终占物理 player 0，opponent 始终占 player 1；harness
只在官方 `IS_FIRST` 选择上强制先后手，不交换 deck 槽位。每个 matchup 的相邻两局组成一对，
共享 engine seed 和 Search seed，仅交换先后手；policy seed 按单局独立派生。报告 manifest
记录 official source、adapter、动态库 hash、ABI、三套 seed 公式和换手合同。相同 runtime hash、
两套 exact deck、三套 seed 与确定性策略动作会产生相同 trace；随机策略还必须使用记录的
policy RNG 流。该保证是本地评测合同，不扩展到 Kaggle 托管 runtime。

Frozen-0806 的不可变 schedule 是一个 256 局最小单位。现阶段标准 checkpoint 强度评测固定运行
八个单位，即 2,048 局；CLI 在 `--pool frozen` 时忽略 legacy `--games` 数量参数。八个单位使用独立于
RL 的固定 evaluation seed `341512806`；每个固定 slot 生成八个独立 engine/Search seed，并严格安排
四次先手、四次后手，合计 1,024 局先手、1,024 局后手。报告 manifest 必须记录 `games=2048`、
展开后的 per-opponent counts、evaluation seed、合同 schedule hash 和 seeded runtime hash；只有
2,048/2,048 finished、0 error、0 unfinished
才能作为同合同策略强度证据。旧异构 Kaggle-derived pool 保留在 `arena/opponents/`，通过全局参数
`--pool opponents` 显式选择，只作为 secondary external-generalization evidence。

2026-08-07 启动的 Policy-0806 全池评测仍是旧 256 局合同，目前在
`arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1/` 保留 25/55 份报告；对应未发布运行与
中断日志保留在 `.tmp/evaluation/frozen_0806_full/`。这些资产是历史、未完成证据，不删除、不覆盖，
也不得与历史 seeded-512 或新的 seeded-2048 报告拼接或冒充完整评测。后续用户要求新的
Frozen-0806 评测时，默认从 2,048 局合同创建独立 run；是否专门继续旧 256/512 局批次必须另行明确指定。

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

`--engine-pool-size E` 是吞吐诊断用的实验路径：保留 `N` 个 OS worker，每个 worker 同时拥有
最多 `E` 个独立 official-engine battle pointer。它不改变正式评测默认的 `E=1` 逐局进程隔离
合同。`E>1` 时，每局/每方仍有独立 causal inference session，但物理 AF_UNIX 连接按
`min(E, 8)`/role 封顶并复用，避免扩大 E 时让 resident inference server 的连接线程无界增长。
pool subprocess timeout 按该 worker 承担的总游戏数计算，不会因为 E 增大、wave 数减少而反向
缩短。报告分别记录 `engine_pool_size`、`engine_inference_channels_per_role`、OS worker 数和最大
live environment 数；这些字段只描述吞吐合同，不构成策略强度证据。

`--worker-local-compiler` 进一步把每个 causal session 的纯 Python feature compiler 绑定到
持有 official-engine observation 的 OS worker；worker 只把 canonical record 与最小
`current/select` 控制字段发给 resident GPU server。worker 不导入 PyTorch、不执行 collate 或
H2D；统一的 `collate -> H2D -> FP16 model` 仍只在 GPU server 内按跨 worker batch 执行。
Policy-0806 的 256-game 本机吞吐校准中，N16E6、batch 128、1 ms 达到 557.45 selections/s，
256/256 completed、0 error，和 2 ms 的 555.49 selections/s 属于同一噪声带；E6 具有更低
请求延迟和 RSS，因此是当前吞吐诊断推荐点。正式评测默认仍保持 E=1 隔离合同。

`--worker-compiler-backend 0035_incremental` 可在上述 worker-local 路径中实验性启用 0035
lifetime-aware 增量 canonical compiler。它仍然只在 engine worker 中输出 raw record，不把
PyTorch、tensor bank、collate 或 H2D 移入 worker。35-decision 因果轨迹已验证与 0806 stateless
record 逐字段一致；官方引擎 4-game smoke 为 4/4、0 error。N16E6/B128/1ms 的相邻 256-game
对照为 546.16 vs 541.04 selections/s，约 +0.9%，但早先 stateless 同合同曾达到 557.45/s，说明
该差异落在运行噪声内。单次 compile 均值仅从 1.280 ms 降至 1.254 ms（约 2.0%），不足以改变
整体漏斗，因此默认仍为 `policy_stateless`；该 backend 只作为后续 compiler 优化的 admission
入口，不影响训练 checkpoint 或最终策略 package。

当前正式吞吐选择是 `--worker-local-compiler --worker-compiler-backend policy_stateless`，不是
增量 backend。相邻的 Policy-0806 N16E16/B64/2ms、128-game official-engine 对照中，中央
stateless 为 342.22 selections/s、65.72 s，worker-local stateless 为 528.77 selections/s、
42.52 s，即吞吐 +54.5%、wall -35.3%；两者均为 128/128 completed、0 error。server 的
prepare-record 阶段从 65.51 降至 0.20 ms/batch，确认收益来自消除中央 compiler 漏斗。
worker-local incremental 在同合同下为 525.77 selections/s，比 stateless 低 0.57%，因此不作为
当前推荐值。所有 optional compiler/tensor cache 保持关闭时，仍走原始 stateless canonical
语义；普通 evaluation 默认也没有改变。

`inference_profile` 还会记录闭环推理的分段时间：worker 连接等待、pickle/socket send、response
wait，server ingress、queue wait、batch coalesce、CPU collate、H2D、model、decode、handler wakeup
和 response send。该观测默认关闭，时间戳只用于吞吐诊断。Policy-0806 的
N16E6/B128/1ms、256-game profile 显示：mean batch 67.93，58.1% 的 batch 至少达到 90% 的
96-live-environment 上限，只有 15.9% 不超过 8；因此主要问题不是请求供给不足。每 batch 的
CPU collate 为 24.84 ms、GPU model 为 21.06 ms、H2D GPU time 仅 1.45 ms，而 GPU model
累计只占 83.17 秒 wall 的 16.7%。worker send 平均 0.318 ms，但 worker timestamp 到 server
完成接收/反序列化平均为 21.45 ms，另有约 10.05 ms handler wakeup；这指向中央 Python
connection-thread fan-in、pickle/object materialization 和 GIL/dispatch handoff，而非 AF_UNIX
字节复制或 PCIe 带宽。下一项优化应优先验证分片 ingress/collate pipeline，不能再通过单纯增加
E 或启用 pinned H2D 推断收益。

`--async-h2d` 是保留用于诊断的 opt-in resident-server 实验路径：它使用容量为 1 的
collate/transfer/compute 队列、三槽 grow-only pinned buffer、选择性大 tensor pin、独立 CUDA
transfer stream 和 cohort 门槛。它不进入 candidate package，也不改变模型、feature 或 action。
Policy-0806 的同合同 64-game 对照中，同步 N16E4/B128/1ms 为 479.58 selections/s，async 为
432.80 selections/s（慢 9.8%）；256-game 全量 pin 初版也只有 199.04 selections/s。因此本机
0806 默认必须保持 async H2D 关闭。该结果说明当前小模型的多字段 pin/copy 与闭环碎批成本高于
可重叠的 H2D 时间；只有后续模型/record 显著增大时才应重新校准。

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

## Seeded checkpoint 比较最低合同

同一卡组的 checkpoint 筛选默认使用 2,048 局，而不是把单个 256/512 局 rollout/probe 当作提交
依据。2,048 局由冻结的 256-slot 对手 schedule 重复八次；每个 slot 使用八个独立
engine/Search seed，并在八次 replica 中强制四次先手、四次后手。所有待比
checkpoint 必须复用完全相同的 request schedule SHA-256。evaluation base seed 使用独立 namespace，
不得复用 PPO rollout 的 update seed window。

该合同改善 checkpoint 间的 common-random-number 比较，但不是“2,048 局必然足够”的统计保证；
轨迹在策略首次选择不同时就会分叉，配对 seed 不能令之后的随机调用保持动作索引级一致。提交或
策略强度结论仍必须同时报告 W-L-D、先后手、区间、逐场景翻转和 official-engine 0-error 护栏。
