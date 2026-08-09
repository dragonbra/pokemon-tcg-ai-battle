# 0038 Accelerated RL Transfer Validation

## 当前状态

V11 在写出 U0 前被旧 checkpoint action-contract validator fail closed，没有 Frozen/rollout/PPO。V12 完成 U0 CUDA Frozen-2048（1147-901，55.996%，先手 603/1024、后手 544/1024，0 fallback）后，被旧 package trace 路径拦截。V13 使用正确的 283 行 trace，但门禁误读 `training_to_package.root.greedy_action_divergences` 的层级；实际 tensor/logits/greedy/Value 全部 PASS，修复提交为 `8c8be47`。V14 完成同一 U0（1147-901）和 package parity 后，在首批 rollout 精确发现 1 个合法 Confusion chance boundary，按旧合同在 PPO 前 fail closed，权重仍为 U0。以上失败资产均保留。

V15 完成 U1–U5 PPO 并写出 U5 model-only checkpoint；固定 Frozen-2048 在唯一一局 `seed=2057788334` 遇到同一合法 chance boundary。对局本身正常完成、无 engine error，但旧 Frozen health gate 将 collector 的诊断 invalid 误当 semantic fallback，故在持久化 U5 Frozen 结果前 fail closed。独立使用同一 U5 checkpoint 和完整固定面板复现为恰好 1 个 `chance_boundary_before_allocation`，无其他 fallback。修复后的 Frozen 保留原 seed/outcome 并单列 chance telemetry；stochastic rollout 的补采合同不变。下一正式版本重新从 common U0 开始，不续写 V15。

长程版本继续绑定不可变 283 行 fixture SHA-256 `18c1684a…56ca`。Confused Phantom Dive 的投币发生在 root 与 allocation 之间，不能把后续六次 legacy callback 伪装成 compound PPO：该次完整 trace 保留为诊断，同 opponent/seat 槽位使用新的确定性派生 seed 补采，直到每 update 恰有 512 条有效 on-policy Episode；只有 `chance_boundary_before_allocation` 在 allowlist，其他 drift/fallback 继续 fail closed。补采 provenance 按 update 独立落盘。CPU 对局不会自动调度，需用户后续指定 checkpoint。

## U0 初始化合同

| 项目 | 来源 |
|---|---|
| Zero-Shot Policy | `rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt` |
| Zero-Shot Value | 0037 使用的 0036 V2 epoch-5 pre-RL latent-query Value |
| common update-0 | `V3_update0_chance_boundary_fallback/update-000000.pt`，SHA-256 `d572c867…bb22820`，`ppo_updates=0` |
| Encoder | Policy-0806 权重，冻结 |
| Action Decoder | Policy-0806 权重，全量训练 |
| Allocation Head | common U0 的 Zero-Shot legacy BC warm start |
| Last Option LoRA | Q/V r4/alpha8，fresh zero-delta，不加载旧 RL LoRA |
| Auxiliary | 既有 directional Prize + 只读 Tempo metrics；Opponent Meta、Tempo curriculum/loss、Seat/Plan/Search 关闭 |
| Optimizer/RNG/trajectory | 全新；不读取旧 optimizer、scheduler、rollout、old logprob、GAE |

正式 checkpoint metadata 会记录完整 policy/value/common-U0、CUDA rules/extension、git、feature、Action Boundary、DecisionGate、canonicalizer、trajectory 和 official adapter hash/version。

## 优化器与加速计划

| parameter group | 原 LR | U1–2 | U3–5 | U6+ 健康上限 | trainable params（V10 实测） | gradient source |
|---|---:|---:|---:|---:|---:|---|
| Action Decoder | 1e-5 | 3e-5 | 5e-5 | 1e-4 | 1,027,202 | PPO Actor / entropy / ref-KL / Prize actor |
| Allocation Head | 1e-5 | 3e-5 | 5e-5 | 1e-4 | 621,761 | allocation policy / entropy / Prize actor |
| Last Option Q/V LoRA | 3e-5 | 9e-5 | 1.5e-4 | 3e-4 | 10,240 | shared Actor path |
| V_win | 1e-4 | 2e-4 | 2e-4 | 2e-4 | 3,397,121 | terminal win Value only |
| V_prize | 1e-4 | 2e-4 | 2e-4 | 2e-4 | 103,681 | directional Prize Value only |
| Opponent Meta Head（V10） | 1e-4 | disabled | disabled | disabled | 110,743 | V10 classification；本轮禁止 |
| Opponent conditioner（V10） | 1e-5 | disabled | disabled | disabled | 110,080 | V10 Actor conditioning；本轮禁止 |

V10 实际共 7 个 optimizer groups；上表已全部列出。Local output LayerNorm 在 V10/V13 都关闭，不构成 optimizer group。正式 `trainable_parameters.json` 会从实际构造模型重新统计，不能以本表替代 runtime inventory。

## 运行合同

- CUDA resident，每 update 512 个环境频率槽位与恰好 512 条有效 trajectory；真实 pre-allocation chance boundary 诊断局以同 opponent/seat 新 seed 补采。
- `fixed_optimizer_budget`，32 optimizer steps，physical minibatch 1024，accumulation 1。
- U0 与每 5 updates 使用相同标准 CUDA Frozen-2048。
- 不设 update 上限；前 50 updates 是观察窗口，直到用户要求停止。
- W&B online：`dragon_bra/pokemon-tcg-policy-learning`。
- 所有 update 保存 model-only checkpoint；champion 只由 CUDA Frozen-2048 选择。
- 除显式 `chance_boundary_before_allocation` 诊断排除外，fallback、invalid macro、unsupported、pending reset、NaN/Inf、replay parity failure 均 fail closed。

## 安全控制

- Behavior KL `>0.005` warning，`>0.01` 降 LR cap，`≥0.02` rollback/retry。
- Clip fraction `>10%` warning，`>30%` rollback/retry。
- rollback 恢复 update 前所有 trainable tensors、清空 optimizer state，并按 10×→5×→3× 降级。
- `STOP_REQUESTED` 只在完整 update 边界生效。

## CPU 边界

U0 不运行 CPU-2048，训练期间也不自动运行任何 CPU engine evaluation。U0 的 package gate 只复用 immutable official snapshot 来比较 repaired CUDA feature、strict package intent 和 Value，不生成新 CPU 对局。

用户查看 CUDA 曲线并指定 update 后，再单独设计 CPU/GPU 对照。没有 CPU U0 同合同结果时，不会把后续小样本 CPU smoke 表述为严格 paired `ΔCPU(U0)`。
