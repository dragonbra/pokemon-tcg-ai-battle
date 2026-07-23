# Evaluation Framework

`evaluation/` 用于本地候选策略的可重复结构化评测，不是 Kaggle 正式提交目录。

## CLI

```bash
python3 -m evaluation list-opponents
python3 -m evaluation validate evaluation/arena/opponents/dragapult_ex_01
python3 -m evaluation run \
  --candidate evaluation/arena/candidates/<candidate> \
  --opponents all \
  --games 10 \
  --output .tmp/evaluation/<candidate>
```

`run` 也支持 `--control PACKAGE`（只在报告中展示对比）、调试用 `--visualize` / `--keep-temp`、
`--max-steps N` 和可重复的 `--metric-module MODULE[:Class]`。默认不生成可视化帧。每次 CLI
评测固定使用完整启用 catalog（当前 23 个 opponent）、每个至少 10 局；小于 10 局或指定 opponent 子集会被
拒绝。输出到 `rl_runs/<label>/evaluation` 时，实验目录会自动获得 `0001-` 形式的
顺序前缀。训练记录与 evaluation 放在 `rl_runs/0001-<label>/`，checkpoint 放在
`rl_runs/checkpoint/0001-<label>/`，TensorBoard 放在
`rl_runs/tensorboard/0001-<label>/`；三者使用同一个编号。control
不会触发 promotion、reject 或其他自动晋级决定。动态模块必须是独立的 `MetricPlugin`，
只能追加统计，不能使用任何核心 metric ID。

`--workers N` 可以覆盖并行对局数；默认按当前进程可用的 CPU affinity 自动选择，最多为
本机已校准的 `8`。无论并行度如何，每局仍在独立 worker
进程中加载双方策略和官方 engine runtime，指标分析与报告写入保持 catalog/game 的固定
顺序。默认 `--worker-cpu-threads 1`，以限制 PyTorch、OpenMP、MKL、OpenBLAS 和 NumExpr
内部线程，避免多个进程过度订阅 CPU。实际 workers、线程限制和 wall time 都内嵌在报告的
manifest 数据中；其他硬件可显式覆盖并先把小规模校准报告写到 `.tmp/evaluation/benchmarks/`。

本机 8 个物理核心、当时固定 18×10 局官方 engine 对战的实测如下。该表是历史吞吐基准，
不代表当前 catalog 规模；数字只证明吞吐，不构成
策略强度证据；其他硬件需重新校准。

| 调度方式 | 完成情况 | 总耗时 | 吞吐 |
|---|---:|---:|---:|
| 串行 | 180/180 | 234.41 秒 | 0.77 games/s |
| 4 workers，未限制内部线程 | worker error | 488.41 秒 | 不采用 |
| 4 workers × 1 thread | 180/180 | 60.29 秒 | 2.99 games/s |
| 8 workers × 1 thread | 180/180 | 40.23 秒 | 4.47 games/s |
| 当前 report-only 默认 8 × 1 | 180/180，0 error | 38.27 秒 | 4.70 games/s |

GPU 动态 batch 原型曾达到 24.01 秒，但尚未具备通用 checkpoint 合同、正式 CLI、隔离测试和
等价性验收，因此不是正式 Evaluation 路径。

需要 V8 setup/relay 语义指标时使用内置的 `auto_iteration_v8_setup_relay` profile
（保留该名称作为兼容 ID），当前为 `revision 3`：

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

被评测卡组和 opponent 都必须是标准 package：根目录包含 `main.py`、60 行 `deck.csv` 和
完整的 `cg/` runtime。`validate` 会打印 package 名称、deck hash、cg tree hash 与
60-card 校验结果；任何预检、卡组或 cg hash 兼容错误都会在启动对局前退出。

唯一 opponent catalog 是 [`configs/opponents.json`](configs/opponents.json)。`all`
按其中启用项的固定顺序选择，且 catalog 只能引用 `arena/opponents/`；传入逗号分隔名称时，
只能选择 catalog 中已启用的项。新增的候选 opponent 先作为标准 package 放入
`arena/candidates/<name>/`，完成验证并经用户确认后，才按实际关键宝可梦分配
`<archetype>_<NN>` 正式名称并迁入 `arena/opponents/`。正式目录名使用 ASCII snake_case 和
两位序号，例如 `alakazam_dudunsparce_01`；catalog 同时保存易读 display name 和 1–2 个
确实存在于该 deck 的代表宝可梦 card ID。Report 的紧凑胜率图使用这些元数据展示小卡图；
图片加载失败时仍保留文字名称和胜率。候选在获准收编前保持原名。这里的 arena candidate
描述存放与准入状态，CLI `--candidate` 则描述本次被测对象。

## 产物和 trace 生命周期

每次 `run` 由 `BatchConfig/run_batch` 在 `--output/run_id/` 只写入 `report.html`，即
`report_only` 默认产物契约。manifest、summary、逐局轻量记录、聚合 metrics、case 摘要和
presentation 数据都内嵌在该独立 HTML 中。指标插件通过稳定的 metric ID 注册；
`--metric-module` 的模块路径由 metrics registry 动态加载，只能追加统计，不得覆盖核心
输出；动态模块的 HTML renderer 默认不受信任，会退回安全的通用指标展示。只有内置
profile plugin 才能提供专属 HTML section。核心指标固定为 outcome、length、correctness、powerful_hand、rare_candy、
post_ko_relay、run_away_draw 和 library_pressure。

完整逐局 trace 只在评测过程中临时保存，默认结束后全部清理，不再长期复制三场 case
trace。`--keep-temp` 仅用于调试，不能把临时文件当成长期资料；若同时需要可视化帧，显式
增加 `--visualize`。正式长期复盘仍优先使用 Kaggle 官方 episode/replay。

该框架在报告中记录 package hash、cg hash、参数和结果，但不承诺位级或完全
随机过程复现；本地运行也不能替代 Kaggle 的正式策略评估。
