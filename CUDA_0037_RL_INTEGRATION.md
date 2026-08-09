# 0037 RL 的 CUDA 加速接入与 0038 复用指南

本文说明 0037 如何把 CUDA 引擎接入 PPO rollout 与 Frozen-0806 评测，以及 0038 应如何复用共享组件。本文是工程入口说明；0037 的模型、loss 与训练合同仍以 `experiments/0037_dragapult_value_initialized_rl/DESIGN.md` 和 `DESIGN.html` 为准。

## 1. 0037 当前的数据流

```text
CUDA official-rules runtime（常驻 lanes + masked refill）
  -> GPU Semantic0031 feature compiler
  -> active-prefix compaction
  -> 共享 State Encoder + 第一个 Option block
  -> 按阵营分流：
       focal：0037 adapted 最后 Option block -> stochastic Action Decoder
                                           -> Value Head
       opponent：Frozen-0806 最后 Option block -> greedy Action Decoder
  -> focal decision sink
  -> EpisodeTrajectory
  -> 既有 prepare_episodes / GAE / PPO update
```

主要入口：

- 共享常驻调度器：`engine_cuda/python/ptcg_cuda_engine/semantic0031_resident.py`
- 共享 focal/frozen 路由：`engine_cuda/python/ptcg_cuda_engine/semantic0031_router.py`
- GPU feature/decode bridge：`engine_cuda/python/ptcg_cuda_engine/semantic0031_bridge.py`
- 0037 项目适配层：`train/0037_dragapult_value_initialized_rl/rollout/cuda_collector.py`
- 0037 训练装配入口：`train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py` 中的 `build_collector()`
- Seeded-2048 公共评测合同：`evaluation/frozen_0806_contract.py`
- 独立吞吐 benchmark：`engine_cuda/tools/benchmark_0037_cuda_resident_refill.py`

0037 默认使用 `collector_backend="cuda_resident"`、`cuda_lane_count=256`、`cuda_check_interval=8`。保留 `cpu_pool` 只作为诊断 fallback。训练 rollout 每个 update 为 512 局；Frozen greedy 评测每 5 个 update 运行一次，每个 checkpoint 使用固定的 2048 局合同。

## 2. 为什么这套接入能加速 RL

旧路径让 CPU worker 分别维护引擎、编译 feature，并频繁把小 batch 送入 GPU。新路径把多局游戏保留在 CUDA lanes 中：结束的 lane 立即 masked refill，仍在运行的 lane 不重建；Semantic0031 feature 在设备侧编译并按有效前缀压紧，actor/value 直接消费 GPU tensor。

主策略与 Frozen 对手共享相同的 State Encoder 和第一个 Option block，只在最后一个 Option block 处分流。这样既支持 0037 的 Last Option LoRA，又避免为双方重复计算大段完全相同的 frozen trunk。对手 greedy 推理不计算训练不需要的 log-prob/entropy；主视角则通过 `focal_value_fn` 和 `focal_decision_sink` 保留 PPO 所需的 action、behavior log-prob、entropy、value、turn 与 feature。

本机 RTX 5080 的已验证数据：

- 严格 FP32、Seeded-2048 Frozen 评测：约 21.89 局/秒。
- 0037 stochastic 512 局 rollout：hot loop 约 45.7 秒，包含约 2.4 GB trajectory 落地后约 53.3 秒，即端到端约 9.60 局/秒。
- 相比此前约 279 秒的 CPU rollout，模拟与采集阶段约快 5.2 倍。

端到端 PPO update 仍包含 trajectory materialization、GAE 和多轮 optimizer minibatch，因此不能用纯评测的 21.89 局/秒直接推算整个 update 时间。

## 3. 正确性与复现护栏

下列条件不能为了速度关闭：

1. **真实 action relation**：skill/effect pooling 必须读取 observation 中实际的 option mask、parent、role；不能从 card prototype 猜测当前 action 是否存在。该修复使 CUDA behavior log-prob 回放 MAE 降到约 `3e-6`，低于 PPO 护栏 `1e-4`。
2. **逐局无状态采样**：stochastic action 由 policy seed、focal decision counter 和 action step 决定，不依赖 lane 顺序或 refill 时机。
3. **严格 FP32**：设置 `torch.set_float32_matmul_precision("highest")`，关闭 TF32。当前“相同 seed 两轮逐局完全一致”的证据建立在此配置上。
4. **循环判负**：同一 physical actor 的同一完整选择在一个官方回合中第 20 次出现时记录 repeat-forfeit，并判重复方负；双方计数独立，交替行动不会互相清空；报告必须保留该元数据。
5. **50 回合终止**：评测直接读取 raw `OfficialStatePod.turn`，engine turn 100（50 个完整回合）记平局；0037 RL 为对齐已有 CPU rollout 合同使用 engine turn 99。不要从归一化 Semantic0031 feature 反推时钟。
6. **Frozen-0806 合同**：固定 evaluation seed `341512806`；256 个基础槽位 × 8 个独立 seed，共 2048 局，其中先后手各 1024。只有 2048/2048 terminal、0 error、0 unfinished，且 turn-limit draw / repeat-forfeit 元数据完整，才能作为同合同强度证据。
7. **官方边界**：没有修改 `engine/source/`。CUDA runtime 实现官方规则语义，但 CUDA 与 CPU official-engine 的逐局 parity 仍是独立证据边界，不能仅凭吞吐 benchmark 宣称二者完全等价。

## 4. 0038 如何复用

0038 **不要 import `train/0037_dragapult_value_initialized_rl/...`**。编号项目必须自包含。正确做法是：

1. 直接复用共享包 `engine_cuda` 中的 resident scheduler、Semantic0031 bridge 和 router。
2. 在 `train/0038_<name>/rollout/cuda_collector.py` 建立 0038 自己的薄适配层；可参考 0037 collector 的结构，但在 0038 内固化 trajectory/config 接口。
3. 如果 0038 仍使用相同 39-key Semantic0031 schema，且适配边界仍是最后一个 Option block，可以直接装配 `Semantic0031FocalFrozenRouter`。
4. 如果 0038 改变 action boundary、feature schema 或模型分叉点，只复用 resident lane/refill 调度器；在 0038 内实现新的 feature adapter/router，并补对应 parity 与 log-prob replay 测试。
5. 通过 `focal_value_fn` 返回 focal state value，通过 `focal_decision_sink` 收集 PPO transition，最终转换成 0038 自己的 trajectory 类型，再进入其 GAE/PPO 管线。
6. Frozen 评测统一 import `evaluation.frozen_0806_contract`，不要在 0038 复制另一套 seed/schedule 常量。

最小装配示意（接口名以实际 0038 模型为准）：

```python
from ptcg_cuda_engine.semantic0031_bridge import Semantic0031DeviceAdapter
from ptcg_cuda_engine.semantic0031_resident import run_resident_greedy_jobs
from ptcg_cuda_engine.semantic0031_router import Semantic0031ResidentRouter
from evaluation.frozen_0806_contract import FROZEN_0806_CONTRACT_ID

adapter = Semantic0031DeviceAdapter(policy.actor, focal_deck, max_select=64)
opponent_option = frozen_policy.option_encoder.cross_attention_transformer
router = Semantic0031ResidentRouter(
    focal_adapter=adapter,
    opponent_last_option_layer=opponent_option.layers[1],
    opponent_option_norm=opponent_option.norm,
    opponent_decoder=frozen_policy.action_decoder,
    same_policy=False,
)

result = run_resident_greedy_jobs(
    jobs=jobs,
    rules=rules_path.read_bytes(),
    router=router,
    lane_count=256,
    device="cuda:0",
    check_interval=8,
    focal_value_fn=value_head,
    focal_decision_sink=trajectory_sink,
)
```

上面仅展示组合关系。实际接入前，请先以 0037 的 `cuda_collector.py` 和 `run_full_semantic.py::build_collector()` 为可运行参考，按共享函数的真实签名传参。

## 5. 给 0038 Codex 的验收清单

- 不修改 `engine/source/`，不 import 0037 可执行代码。
- 为 0038 新建独立 collector、config、tests，并同步 0038 的 `DESIGN.md` 与 `DESIGN.html`。
- 跑一个固定小 batch，验证 CPU canonical actor 与 CUDA actor 的 greedy action 一致。
- stochastic rollout 回放 behavior log-prob，要求 MAE `<= 1e-4`，并执行至少一个真实 PPO minibatch。
- 同一组 evaluation seed 连跑两次，逐局 outcome、repeat-forfeit 与 terminal-state hash 完全一致。
- 用小规模 lane-count benchmark 选择本机设置，再启动正式训练；不要仅按显存容量盲目放大 lanes。
- 正式 Frozen greedy 评测使用 `frozen_0806_seeded_2048_v2`，并记录 checkpoint update、0 error、0 unfinished。

当前可参考的正式单 checkpoint 报告是 `experiments/0037_dragapult_value_initialized_rl/evaluation/V5_last_option_qv_lora_r4_eval5_50u_update_000032_cuda_seeded2048.html`；Policy-0806 全池 CUDA 评测当前为可续跑的 40/55 部分结果，位于 `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_resident_v2/`，不得当作 55/55 完整结论。
