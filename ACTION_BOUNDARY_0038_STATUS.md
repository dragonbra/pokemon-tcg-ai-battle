# 0038 Action Boundary 当前状态

日期：2026-08-09

0038 第一版、CUDA 接入与 50 完整回合截断已适配。V8 暴露两项启动阻塞：repeat guard 跨整局累计正常 Ability，破坏 Zero-Shot parity；完整 2,048 局 feature trajectory 占用 8.94 GB、进程 RSS 达 16.24 GB。V8 已停止并判为无效实验。

- 未修改官方 `engine/source/`、ABI、observation schema 或 `select` 返回协议。
- 使用 Zero-Shot Policy-0806 与 pre-RL Value 初始化；未加载 V5/RL-updated 权重。
- DecisionGate、forced shortcut、observe-only、macro trajectory 和 Phantom Dive 层级 allocation 已落地。
- Phantom Dive 对 `n=1..5` 的 canonical allocation 数为 `1/7/28/84/210`；330 个 exhaustive official parity case 全部通过。
- allocation head 使用 1,024 场 official game、3,134 条 Zero-Shot teacher 标签完成 BC warm start。
- 公共 update-0 为 `V3_update0_chance_boundary_fallback`，PPO updates = 0。
- 历史 V3 均匀面板结果 `1308-739-1` 仅作旧证据，不再作为全局 Frozen 可比基线。
- Value Network 确认加载 0036 V2 epoch-5 pre-RL 权重（SHA-256 `e88b2f...e36360`）；V8 的异常低 Value loss 来自 compound GAE 错误按 action 而非 turn 应用 lambda，现已修正。
- repeat guard 现在只累计同一 actor、同一 official turn 内的 stable Ability identity；跨回合自动重置。
- canonical 2,048 A/B：legacy sequential `1178-870`（57.52%）；forced-only 完全一致；完整 macro `1162-886`（56.74%），旧 007 为 57.32%。
- 2,048-game stochastic + GAE + 32-step PPO smoke 通过：512 trajectory games、44,052 valid transitions、Value loss 0.520、return std 0.896、EV 0.300、behavior logprob MAE `6.2e-6`。
- 内存 smoke：staged trajectory 2.286 GB，peak RSS 5.81 GB；旧 V8 分别为 8.94 GB/16.24 GB。
- V9 完成 update 2 后确认 RSS 峰值在约 9.48 GB 形成平台、没有按 update 继续增长；attempted update 3 在 PPO 前人工停止。
- V10 每 update 只运行两个完整 256-slot unit，共 512 局，并保留全部 512 条完整 EpisodeTrajectory；不再额外模拟 1,536 局无梯度贡献的对局。每个 update 结束后显式释放上一批高维 tensor 引用。
- rollout 数量、backend、并行度、inference batch 和 PPO optimizer budget 已配置化；默认扩容模式为 `fixed_optimizer_budget`。
- V4 在首个 PPO update 前 fail closed，没有产生 update-1 model checkpoint；失败原因是使用了均匀 opponent 面板且 sparse-diagnostic predicate 未导入。
- `V6_canonical_frozen_cuda_fresh_rl` 已应用户要求在首个 rollout 期间人工中止，等待 50 回合终止合同完成。它只保留 canonical update-0 baseline（1162/2048，56.74%），`checkpoint_update=0`，没有任何 PPO update。下一次启动必须使用新的严格递增版本，不得续写 V6。rollout 每 256 局精确复现环境 opponent 频率；update-0 与每 5 updates 严格复用全局 `frozen_0806_seeded_2048_v2`（seed `341512806`、007 schedule SHA `98b58b...ce9`）。
- V7 只保留 update-0 baseline `1094-954-0`（53.42%），没有 PPO update。V8 update-0 为错误 guard 下的 `1005-1043-0`（49.07%），虽产生 update-1 checkpoint，但因 parity/内存失败不得续写或比较。
- 新正式版本为 `V10_complete_512_rollout_fresh_rl`；仍从公共 V3 PPO-update-0 起步，重新初始化 optimizer/RNG/buffer，不加载 V9 RL 权重。

详细设计见 [`experiments/0038_action_boundary_rl/DESIGN.md`](experiments/0038_action_boundary_rl/DESIGN.md)，update-0 评估见 [`V3 report`](experiments/0038_action_boundary_rl/evaluation/V3_update0_chance_boundary_fallback.html)。
