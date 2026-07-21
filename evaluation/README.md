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
`--keep-temp`、`--max-steps N` 和可重复的 `--metric-module MODULE[:Class]`。control
不会触发 promotion、reject 或其他自动晋级决定。动态模块必须是独立的 `MetricPlugin`，
只能追加统计，不能使用任何核心 metric ID。

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
输出。核心指标固定为 outcome、length、correctness、powerful_hand、rare_candy、
post_ko_relay、run_away_draw 和 library_pressure。

完整逐局 trace 只在评测过程中临时保存。默认结束后会清理临时目录，长期报告按 case
优先级最多保留三场到 `traces/`；`--keep-temp` 仅用于调试，不能把临时文件当成长期
资料。正式长期复盘仍优先使用 Kaggle 官方 episode/replay。

该框架记录 package hash、cg hash、参数、结果和已保留 trace，但不承诺位级或完全
随机过程复现；本地运行也不能替代 Kaggle 的正式策略评估。
