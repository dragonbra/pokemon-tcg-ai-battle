# Arena 结果归档

这里仅保存可提交卡组的轻量结果，不保存逐局 trace、worker 临时文件、完整
SQLite 数据库或 Kaggle Notebook 原始下载。

- `arena-latest-summary.*`：Arena continuous 最新 checkpoint 的排行榜和汇总。
- `alakazam_v9_top20_10games/`：当前 V9 对 Arena Top 20 公开卡组各 10 局的评测结果。
- `arena/reports/`：All/Eligible HTML 报告快照。
- `arena/db/arena.sqlite3`：只含来源和 package 索引，不含任何逐局运行数据。

原始来源、逐局状态和 checkpoint 数据默认写入被 Git 忽略的 `arena/sources/` 和
`arena/runs/`；重新收集或运行 Arena 后，可以按需重新生成它们。
