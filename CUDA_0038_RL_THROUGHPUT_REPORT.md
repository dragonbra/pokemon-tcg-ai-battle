# 0038 CUDA RL 接入报告

## 结论

0038 的 Action Boundary、Phantom Dive macro 与压缩 PPO trajectory 已接入 CUDA resident engine，官方 primitive `select` 协议和模型语义不变。256 局实测无 engine error；真实 stochastic rollout 到 PPO replay/update 的短 smoke 通过。

| 路径 | 局数 | 吞吐 | 相对旧 0038 CPU |
|---|---:|---:|---:|
| CUDA stochastic rollout（含 trajectory） | 256 | 8.84 games/s | 4.96× |
| CUDA greedy Frozen evaluation | 256 | 12.98 games/s | 7.29× |
| 旧 0038 CPU greedy | 2,048 | 1.78 games/s | 1.00× |

stochastic benchmark 记录了 21,819 条真实 PolicyTransition、46,607 次官方 primitive select、2,066 次 forced shortcut 和 770 个 Phantom macro。一次已知的 `n=6` 扩展 Bench 情形按 v1 合同安全 fallback，不进入正常 macro PPO；无对局失败。

## 训练效率预估

正式设置每 update 采集 2,048 局，CUDA rollout 约需 3.9 分钟；32-step fixed optimizer PPO 预计约 0.5 分钟。每 5 updates 的固定 2,048 局 greedy evaluation 约需 2.6 分钟。折算长期平均约 4.8–5.2 分钟/update、约 12 updates/hour，实际以首个正式 update 的 W&B 分阶段计时为准。

CUDA benchmark 峰值为 2.20 GiB allocated、4.11 GiB reserved；正式 collector 采用 4×512 分块，避免一次保留全部 2,048 局 trajectory。吞吐提升只代表执行效率，不作为策略强度证据。

## 正式合同

- 初始化：`V3_update0_chance_boundary_fallback` 的 pre-PPO/update-0 model-only 权重；不加载 V5 RL 权重。
- 训练：`V4_full_stack_cuda_fresh_rl`，每 update 2,048 局，`fixed_optimizer_budget`。
- 评估：只使用 `0038_frozen_2048_v1`；update 0 与之后每 5 updates 同合同配对评估。
- 时长：无固定 update 上限，在完整 update 边界响应人工 `STOP_REQUESTED`。
- 记录：本地 JSONL 为事实源，TensorBoard 与 private W&B 只镜像聚合标量。

原始 benchmark 位于 `.tmp/evaluation/0038_cuda_integration/`，CUDA extension SHA-256 为 `8b41f2230b2e622be0278bdace8ddba7761c7900fc3474a2c3851be19f226fe3`。
