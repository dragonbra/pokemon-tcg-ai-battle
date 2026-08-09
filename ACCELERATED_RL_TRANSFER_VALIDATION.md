# 0038 Accelerated RL Transfer Validation

## 当前状态

V11 实现/验收准备中。本文件是长期运行的简明事实页；CUDA U0、每 5 updates Frozen 曲线和 champion 会在正式 run 产物中持续记录。CPU 对局不会自动调度，需用户后续指定 checkpoint。

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

V10 实际共 7 个 optimizer groups；上表已全部列出。Local output LayerNorm 在 V10/V11 都关闭，不构成 optimizer group。正式 `trainable_parameters.json` 会从实际构造模型重新统计，不能以本表替代 runtime inventory。

## 运行合同

- CUDA resident，512 rollout games/update，512 trajectory games/update。
- `fixed_optimizer_budget`，32 optimizer steps，physical minibatch 1024，accumulation 1。
- U0 与每 5 updates 使用相同标准 CUDA Frozen-2048。
- 不设 update 上限；前 50 updates 是观察窗口，直到用户要求停止。
- W&B online：`dragon_bra/pokemon-tcg-policy-learning`。
- 所有 update 保存 model-only checkpoint；champion 只由 CUDA Frozen-2048 选择。
- fallback、invalid macro、unsupported、pending reset、NaN/Inf、replay parity failure 均 fail closed。

## 安全控制

- Behavior KL `>0.005` warning，`>0.01` 降 LR cap，`≥0.02` rollback/retry。
- Clip fraction `>10%` warning，`>30%` rollback/retry。
- rollback 恢复 update 前所有 trainable tensors、清空 optimizer state，并按 10×→5×→3× 降级。
- `STOP_REQUESTED` 只在完整 update 边界生效。

## CPU 边界

U0 不运行 CPU-2048，训练期间也不自动运行任何 CPU engine evaluation。U0 的 package gate 只复用 immutable official snapshot 来比较 repaired CUDA feature、strict package intent 和 Value，不生成新 CPU 对局。

用户查看 CUDA 曲线并指定 update 后，再单独设计 CPU/GPU 对照。没有 CPU U0 同合同结果时，不会把后续小样本 CPU smoke 表述为严格 paired `ΔCPU(U0)`。
