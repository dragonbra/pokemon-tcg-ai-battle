# 0038 Action Boundary 当前状态

日期：2026-08-09

0038 第一版与 CUDA 接入已经完成，当前等待 GPU 空闲后启动 canonical Frozen 合同的 V5。

- 未修改官方 `engine/source/`、ABI、observation schema 或 `select` 返回协议。
- 使用 Zero-Shot Policy-0806 与 pre-RL Value 初始化；未加载 V5/RL-updated 权重。
- DecisionGate、forced shortcut、observe-only、macro trajectory 和 Phantom Dive 层级 allocation 已落地。
- Phantom Dive 对 `n=1..5` 的 canonical allocation 数为 `1/7/28/84/210`；330 个 exhaustive official parity case 全部通过。
- allocation head 使用 1,024 场 official game、3,134 条 Zero-Shot teacher 标签完成 BC warm start。
- 公共 update-0 为 `V3_update0_chance_boundary_fallback`，PPO updates = 0。
- 历史 V3 均匀面板结果 `1308-739-1` 仅作旧证据，不再作为全局 Frozen 可比基线。
- 124/124 unit/property tests 通过；短 INTEGRATED PPO smoke 通过。
- rollout 数量、backend、并行度、inference batch 和 PPO optimizer budget 已配置化；默认扩容模式为 `fixed_optimizer_budget`。
- V4 在首个 PPO update 前 fail closed，没有产生 update-1 model checkpoint；失败原因是使用了均匀 opponent 面板且 sparse-diagnostic predicate 未导入。
- `V6_canonical_frozen_cuda_fresh_rl` 已应用户要求在首个 rollout 期间人工中止，等待 50 回合终止合同完成。它只保留 canonical update-0 baseline（1162/2048，56.74%），`checkpoint_update=0`，没有任何 PPO update。下一次启动必须使用新的严格递增版本，不得续写 V6。rollout 每 256 局精确复现环境 opponent 频率；update-0 与每 5 updates 严格复用全局 `frozen_0806_seeded_2048_v2`（seed `341512806`、007 schedule SHA `98b58b...ce9`）。

详细设计见 [`experiments/0038_action_boundary_rl/DESIGN.md`](experiments/0038_action_boundary_rl/DESIGN.md)，update-0 评估见 [`V3 report`](experiments/0038_action_boundary_rl/evaluation/V3_update0_chance_boundary_fallback.html)。
