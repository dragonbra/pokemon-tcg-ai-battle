# Evaluation Framework

`evaluation/` 用于本地候选策略的可重复结构化评测，不是 Kaggle 正式提交目录。

## CLI

```bash
python3 -m evaluation list-opponents
python3 -m evaluation validate evaluation/opponents/kiyotah_dragapult
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 30 \
  --output reports/evaluation
```

`run` 也支持 `--control PACKAGE`（只在报告中展示对比）、`--no-visualize`、
`--keep-temp`、`--max-steps N` 和可重复的 `--metric-module MODULE[:Class]`。每次 CLI
评测固定使用全部 18 个 opponent、每个至少 10 局；小于 10 局或指定 opponent 子集会被
拒绝。输出到 `rl/_runs/<label>/evaluation` 时，实验目录会自动获得 `0001-` 形式的
顺序前缀。训练记录与 evaluation 放在 `rl/_runs/0001-<label>/`，checkpoint 放在
`rl/artifact/checkpoint/0001-<label>/`，TensorBoard 放在
`rl/_runs/tensorboard/0001-<label>/`；三者使用同一个编号。control
不会触发 promotion、reject 或其他自动晋级决定。动态模块必须是独立的 `MetricPlugin`，
只能追加统计，不能使用任何核心 metric ID。

AutoIteration V8 使用内置的 `auto_iteration_v8_setup_relay` profile，当前为
`revision 2`：

```bash
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output /tmp/ptcg-auto-iteration-evaluation
```

这个 profile 在核心 outcome、health 和 correctness 指标之外提供 setup/relay、
Powerful Hand、post-KO relay 和 attack quality 的结构化 payload。Powerful Hand 的
第二个己方回合按实际先后手使用 engine turn 3/4，只有实际选择 `attackId=1072`
才计为成功；未到达目标回合、未完成对局和未知 Prize 状态会保留在审计字段及相应
分母中，不会被静默删除。profile 的 id、revision、metric ids 和优先级写入
`manifest.json`，聚合 payload 写入 `metrics.json`，单局轻量 payload 写入
`games.jsonl` 的 `metric_refs`，专属 Markdown/HTML 展示写入 `report.md` 和
`report.html`。

AutoIteration 的 `report.html` 会在原始 metric 汇总之前按语义分组展示：结果与正确性护栏、
阶段一二回合基础能力、阶段二 Post-KO 接力能力、阶段三攻击质量惩罚项，以及辅助健康与审计
指标。每一行同时保留中文语义名、原始 `metric_id`、角色、方向、分子/分母和追踪目标；复合
指标可以从同一个 plugin 展开多个观察项，例如四组件、Dunsparce bridge 和额外过牌。
页面的语义值按 AutoIteration 定义选择 payload 字段。例如 `post_ko_relay` 的语义展示值
使用 `payload.success_rate`（成功数 / 机会数），而 `metrics.json` 中旧的顶层失败率仍原样
保留，便于审计和兼容既有调用层。

Evaluation 只负责指标测量、证据审计和可视化，不负责晋级决策，也不负责
candidate/control 比较；调用层根据 profile revision 和报告数据自行解释 promote、observe、reject 等
业务状态。

## Package 与 catalog

候选和 opponent 都必须是标准 package：根目录包含 `main.py`、60 行 `deck.csv` 和
完整的 `cg/` runtime。`validate` 会打印 package 名称、deck hash、cg tree hash 与
60-card 校验结果；任何预检、卡组或 cg hash 兼容错误都会在启动对局前退出。

唯一 opponent catalog 是 [`configs/opponents.json`](configs/opponents.json)。`all`
按其中启用项的固定顺序选择；传入逗号分隔名称时，只能选择 catalog 中已启用的项。

## 产物和 trace 生命周期

每次 `run` 由 `BatchConfig/run_batch` 在 `--output/run_id/` 写入同一批 run 数据：
`manifest.json`、`summary.json`、`games.jsonl`、`metrics.json`、`cases.jsonl`、
`report.md`、`report.html` 和 `traces/`。指标插件通过稳定的 metric ID 注册；
`--metric-module` 的模块路径由 metrics registry 动态加载，只能追加统计，不得覆盖核心
输出；动态模块的 HTML renderer 默认不受信任，会退回安全的通用指标展示。只有内置
profile plugin 才能提供专属 HTML section。核心指标固定为 outcome、length、correctness、powerful_hand、rare_candy、
post_ko_relay、run_away_draw 和 library_pressure。

完整逐局 trace 只在评测过程中临时保存。默认结束后会清理临时目录，长期报告按 case
优先级最多保留三场到 `traces/`；`--keep-temp` 仅用于调试，不能把临时文件当成长期
资料。正式长期复盘仍优先使用 Kaggle 官方 episode/replay。

该框架记录 package hash、cg hash、参数、结果和已保留 trace，但不承诺位级或完全
随机过程复现；本地运行也不能替代 Kaggle 的正式策略评估。
