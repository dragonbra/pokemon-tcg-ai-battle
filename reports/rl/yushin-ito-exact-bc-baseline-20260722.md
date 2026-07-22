# Yushin Ito Exact BC Baseline

实验目录：`rl/runs/0001-yushin_ito_exact_bc_baseline/`

## 数据与模型

- 专家：Yushin Ito，最新 exact submission `54773249`，卡组与 `work/alakazam_v9` 完全一致。
- Kaggle public replay：1,000 局（API 对单 submission 的上限），按整局切分为 800 train / 100 validation / 100 test。
- 最终数据：86,875 条决策记录；包含 `DONE` 行携带的有效动作，只有下一行 `action: null` 的无标签终局被跳过。
- 模型：`ptcg_features_v6 + full_action_set_v1`，d_model 256、hidden 512、2 层 Transformer、4 heads、20 epochs、AdamW 3e-4。
- 纯模型合同：single categorical CE、multi-label candidate BCE、selection-count CE；只有 simulator 合法候选 mask 和 deck/reset 协议，没有规则 teacher、search、confidence gate 或 fallback。

离线结果：best validation exact-action `78.88%`；test exact-action `77.02%`，multi-action exact `75.35%`，selection-count accuracy `99.83%`，legal-action rate `100%`。`context=8` 长组合选择的 test exact rate 只有 `1.56%`，是后续奖励/架构实验的首要改进点。

## 固定评测

运行：`run-873882eef9f04d74a8c76df4f680d1b0`，`auto_iteration_v8_setup_relay`，17 opponents × 10 games。

- 122 胜 / 42 负 / 0 平；完成 164/170，口径胜率 `71.76%`。
- 6 局 `game_error` 全部集中在 `yakitori_raging_bolt`；保留 trace 显示对手提交的多选数量违反 `minCount/maxCount`，官方引擎抛出 `IndexError`。没有 `candidate_error`。
- Powerful Hand `62/170`；Post-KO relay `210/391`；攻击未拿奖赏率 `161/641`；Run Away Draw `2/170`。

## 归档

- 工作包：`work/alakazam_bc_v1/`
- 历史 submission 源：`submission/alakazam_bc_v1/`
- Kaggle archive：`submission/dist/alakazam_bc_v1.tar.gz`
- archive SHA-256：`5b832bdc103ac347596ed7588b23b07065b6672760678cf2442acc361aeb084e`
- self-contained `strategy/model.bin` SHA-256：`7ec19da29ce1b1ba294132146bf409ae2ef739417309540a11ed9e212f49d3c8`

本报告记录的是第一次纯神经 BC baseline；后续 reward ablation 应从该 checkpoint 和同一 exact 数据快照开始，单独改变 reward 或模型结构。
