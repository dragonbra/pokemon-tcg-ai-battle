# 0038 Action Boundary 当前状态

日期：2026-08-09

0038 第一版已经完成，当前停在正式 PPO 启动门之前。

- 未修改官方 `engine/source/`、ABI、observation schema 或 `select` 返回协议。
- 使用 Zero-Shot Policy-0806 与 pre-RL Value 初始化；未加载 V5/RL-updated 权重。
- DecisionGate、forced shortcut、observe-only、macro trajectory 和 Phantom Dive 层级 allocation 已落地。
- Phantom Dive 对 `n=1..5` 的 canonical allocation 数为 `1/7/28/84/210`；330 个 exhaustive official parity case 全部通过。
- allocation head 使用 1,024 场 official game、3,134 条 Zero-Shot teacher 标签完成 BC warm start。
- 公共 update-0 为 `V3_update0_chance_boundary_fallback`，PPO updates = 0。
- 固定评估面板为 2,048 个唯一 seed（8×256，先后手各半）。结果 `1308-739-1`，胜率 `63.87%`，Wilson 95% CI `61.76%–65.92%`，0 error、14 次安全 fallback。
- 124/124 unit/property tests 通过；短 INTEGRATED PPO smoke 通过。
- rollout 数量、backend、并行度、inference batch 和 PPO optimizer budget 已配置化；默认扩容模式为 `fixed_optimizer_budget`。
- 正式入口仍 fail closed。人工批准后预留 `V4_integrated_fresh_rl`，从 V3 common update-0 重新采集完整 on-policy trajectory。

详细设计见 [`experiments/0038_action_boundary_rl/DESIGN.md`](experiments/0038_action_boundary_rl/DESIGN.md)，update-0 评估见 [`V3 report`](experiments/0038_action_boundary_rl/evaluation/V3_update0_chance_boundary_fallback.html)。
