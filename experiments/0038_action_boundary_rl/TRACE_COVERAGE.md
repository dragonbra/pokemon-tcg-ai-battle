# Phantom Dive trace 覆盖审计

审计日期：2026-08-08。

正式 0037 V5 `EngineEventTrace` 持久化样本为 **0**，因此旧 rollout 不能直接重放为 0038 PPO 数据。只读扫描现存官方 Kaggle Episode replay 后找到 29 个含 Phantom Dive 分配链的唯一 replay；其中 7 个位于 tracked replay 数据，22 个位于 `.tmp/environment_daily` 快照。

- 可机械重建 root state → 六次 primitive target → canonical allocation：78 / 78 条链（100%）。
- 合法目标数分布：`n=1: 5`、`n=2: 3`、`n=3: 14`、`n=4: 18`、`n=5: 38`。
- allocation 形状：`(6): 42`、`(5,1): 15`、`(4,2): 10`、`(4,1,1): 7`、`(3,3): 1`、`(3,2,1): 1`、`(2,2,1,1): 1`、`(2,1,1,1,1): 1`。
- 当前可直接用于训练的样本：**0**。原因不是标签缺失，而是这些 Episode 的 teacher/source provenance 尚未被批准为 0038 BC 来源。

结论：trace 字段足以构建短暂 allocation BC/distillation 数据，但必须先生成 source manifest、按 Episode 去重/切分，并取得来源批准。禁止把旧 action logprob 用于新 PPO ratio，也禁止在随机 allocation head 上启动正式 PPO。
