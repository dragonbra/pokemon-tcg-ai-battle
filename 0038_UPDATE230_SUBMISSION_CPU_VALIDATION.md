# 0038 U230 Submission × Official CPU 验证

## 结论

当前没有发现 U230 提交包在 Action Boundary、LoRA 合并、Meta conditioning 或 Phantom Dive macro 上改变实际提交动作的证据。

- 解包 U230 在 256 局 official seeded CPU engine 诊断中为 **137-119，53.52%**，0 error、0 unfinished。
- U230 的既有 RL-CUDA 结果在完全相同的 256 个 seed 上为 **147-109，57.42%**；差值为 **-3.91pp**，paired flips 为 44 对 54，exact McNemar `p≈0.36`。
- 原始 Policy-0806/007 control 在相同 256 seed 上，official CPU 为 **139 胜**、既有 CUDA 为 **138 胜**，但逐局 outcome 也只有 **153/256（59.77%）** 一致。由此可见，当前 CPU/CUDA backend 并不具备“同 seed 必然同 outcome”的 lockstep 合同。
- 在不经过 CUDA engine 的 trace replay 中，解包 U230 与 update-230 训练态 FP32/unmerged-LoRA reference 对 **3,649 个候选 callback** 给出完全一致的 primitive `select`，其中包括 **2,849 个多选 callback** 和 **600 个 Phantom Dive 内部 callback**；action diff 为 **0**。

因此，这次 256 局没有复现“打包后策略语义明显损坏”。CPU 与 CUDA 的净 10 胜差异仍需视为 256 局诊断下的 backend/轨迹分叉与采样波动，不能据此声称 U230 在 official CPU 上等于完整 CUDA Frozen 的 61.04%，也不能据此认定打包语义有误。

## 验证合同

- 输入产物：`archive/submission/dist/0038_dragapult_ex_rl_update230.tar.gz`
- tar SHA-256：`a40fe101023f6e936f69587cb96c5c1cc7455b5e3ad075dfe3683eb98fdd7535`
- 解包后模型 SHA-256：`5bc1de70c39e9094089054543e87a321097a89f14e60bcea62aa74d61a5073a3`
- 候选入口：只从解包根目录的 `main.py`、`deck.csv`、`cg/`、`strategy/` 启动。
- 候选推理：包内 CPU 路径；没有候选 resident service，也没有引用 `train/0038_action_boundary_rl`。
- 游戏引擎：official seeded CPU runtime，raw engine turn 100 截断。
- 对手：Frozen Policy-0806；只有对手模型使用 resident CUDA FP32 加速，不参与候选动作生成。
- 诊断面板：保留 Frozen 原始 256-slot opponent 频率，每个 slot 从正式 2,048 面板选一个既有 seed；全局交替 replica 0/1，得到 128 先手、128 后手和 256 个唯一 seed。

该 256 局是按用户要求执行的快速诊断，不替代项目规定的八个 256 单元、共 2,048 局正式 Frozen 合同。

## 结果

| 路径 | 胜-负-平 | 胜率 | Wilson 95% CI |
|---|---:|---:|---:|
| U230 解包包体 + official CPU | 137-119-0 | 53.52% | 47.40%–59.53% |
| U230 RL-CUDA，同 256 seed | 147-109-0 | 57.42% | 51.30%–63.33% |
| Policy-0806 control + official CPU | 139 胜 | 54.30% | 48.18%–60.29% |
| Policy-0806 control + CUDA，同 256 seed | 138 胜 | 53.91% | 47.79%–59.91% |
| U230 完整 CUDA Frozen-2048 | 1250-798-0 | 61.04% | 58.90%–63.12% |

U230 official CPU 的先后手结果：

- 先手：76-52，59.38%；
- 后手：61-67，47.66%；
- official game：256/256 finished；
- error：0；unfinished：0。

原始 256 任务结束时，对手 GPU 服务退出超时打断了第 254、256 局的结果文件 flush；两局均使用已保存的原 request 和完全相同的 engine/search/policy seed 单独重跑恢复。其余 254 局未重跑。恢复后的总胜数与任务结束前内存汇总的 137 胜一致。

## 语义等价证据

从 256 局中每隔 8 局选择一局，共 32 局、覆盖 19 个 opponent identity。对每局从 reset 开始按原 official observation/history 顺序同时驱动：

1. tar 解包后的实际 U230 Agent；
2. update-230 训练态 FP32 reference，保留动态 Q/V LoRA、原 allocation/meta heads；
3. 原 official CPU trace 中实际提交的 primitive action。

结果：

| 检查 | 数量 | Diff |
|---|---:|---:|
| 候选 callback | 3,649 | 0 |
| 多选 callback | 2,849 | 0 |
| Phantom 内部 callback | 600 | 0 |
| 包体自身 trace replay | 3,649 | 0 |
| 包体 vs 训练态 reference | 3,649 | 0 |

这直接覆盖了导出时最值得担心的部分：LoRA merge、FP16 storage→FP32 runtime、root decoder、Meta conditioning、DecisionGate observe-only、stable target relocation 和六次 Phantom primitive 展开。

## 证据边界与下一步

- 32 局 trace parity 是较强的动作语义证据，但不是所有可能 observation 的穷举证明。
- official CPU 与 CUDA engine 当前没有逐随机事件 lockstep parity；同 seed 只能用于 paired 诊断，不能期待逐局 outcome 100% 相同。
- 256 局对约 4pp 的差异仍然不稳定。只有重新运行完整 official CPU Frozen-2048，才能更精确估计 U230 在提交环境合同下的绝对强度；本次按用户要求没有继续长跑。
- 若 Kaggle 实际成绩仍显著低于本地 53% 左右，下一轮应优先收集 Kaggle Episode/replay，定位首个 observation/action 分叉或 runtime error，而不是仅用 aggregate score 反推 Action Boundary 错误。

## 产物

- 完整 HTML：`.tmp/evaluation/0038_u230_submission_cpu_frozen256/run-c1ae70befbc048e3a56b313783c0a83d/report.html`
- 逐局结果：`.tmp/evaluation/0038_u230_submission_cpu_frozen256/run-c1ae70befbc048e3a56b313783c0a83d/results.json`
- 包体/训练态 trace parity：`.tmp/evaluation/0038_u230_submission_cpu_frozen256/run-c1ae70befbc048e3a56b313783c0a83d/package_training_reference_parity.json`
- Policy-0806 backend control：`.tmp/evaluation/0038_u230_submission_cpu_frozen256/policy0806_control/results.json`
