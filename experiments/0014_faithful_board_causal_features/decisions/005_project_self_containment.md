# 005 — 0014 hermetic implementation boundary

0014 的 V1 数据缓存是冻结、审计过的输入资产；其 `0013` 路径只保留为内容
hash 的历史 provenance，不是运行时依赖，也不再支持从已删除的 raw shard
重建。训练和导出只读取 0014 的 materialized cache。

0014 已将 0010 基线模型及所需的卡牌本体、资源 ledger、checkpoint 实现固化
到 `train/0014_faithful_board_causal_features/`。项目内部不得 import 0010 或
0013 的可执行模块；共享的 `rl_environment`、官方数据和官方 engine 保持为
仓库级基础设施。这样历史目录的清理不会改变 0014 的训练、冒烟、导出和评测。
