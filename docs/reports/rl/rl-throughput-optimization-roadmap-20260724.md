# RL 训练吞吐优化路线图

- 日期：2026-07-24
- 状态：未来优化参照，不阻塞今晚在线 PPO
- 启动条件：真实在线训练闭环正确后，且 profiling 证明吞吐已成为主要瓶颈
- 硬约束：不修改 `engine/source/`，正式能力结论仍来自官方 runtime 固定 Evaluation

配套文档：今晚在线 PPO 合同见
[`online-ppo-training-design-20260724.md`](online-ppo-training-design-20260724.md)，固定评测改造见
[`../evaluation/evaluation-framework-improvements-20260724.md`](../evaluation/evaluation-framework-improvements-20260724.md)。

## 1. 决策原则

今晚先实现正确的 on-policy rollout、联合 action log-prob、reward/GAE 和 PPO update。
如果正常训练速度可以接受，本路线图暂不实施。不能为了追求一个预估 decisions/s 数字，
同时引入 engine 状态复用、actor 状态串局、GPU queue、native codec 和异步 policy version
等多个难以定位的变量。

只有下面任一条件持续出现时，才启动性能工作：

- rollout wall time 占单次 iteration 的 60% 以上；
- learner GPU utilization 长时间低于 40%，同时 CPU rollout 队列供给不足；
- learner inference 的实际 batch 长期接近 1；
- opponent 计算或 Python/JSON codec 占 collect 时间的显著部分；
- 预期训练预算因吞吐不足无法在可接受时间内获得目标样本量。

这些阈值是启动 profiling 的工程触发器，不是模型优劣指标。

## 2. 当前证据与边界

### 2.1 已验证事实

- 本机 8 物理核，BC V4 的 18×10 Evaluation：串行 234.41 秒；
- `8 workers × 1 internal thread`：40.23 秒；
- 候选常驻 GPU + 动态 batch 研究原型：24.01 秒；
- 原型 180/180 completed、0 error，约 7.50 games/s；
- 原型约 934.7 total engine actions/s，粗略折算约 467 learner decisions/s；
- GPU batch=1 约 63.56 秒，比 8-worker CPU Evaluation 更慢；GPU 的价值来自 batching，
  不是简单把单条推理移动到 CUDA。

### 2.2 尚未证明

- 没有证明本机可达到其他机器 2048 环境、1664 decisions/s 的数据；
- 没有证明 GPU Evaluation 原型符合通用 checkpoint/package 合同；
- 没有证明同进程多 battle handle 长期状态隔离、内存稳定或等价；
- 没有证明 native codec 是当前首要瓶颈；
- 没有证明 opponent actor pool 对所有规则/搜索策略都能正确 reset。

因此路线图中的吞吐目标是 milestone，不是承诺。

## 3. 目标架构

长期训练系统可以演进为：

```text
N 个官方 engine battle slots
        |
        +---- learner observations ----> trainer-owned GPU batch inference
        |
        +---- opponent requests -------> persistent CPU actor pool
                                              |
trainer 同时执行 codec / GPU work <-----------+
        |
        +---- actions -> engine slots -> transitions -> rollout buffer
        |
        +---- frozen policy_version batch -> PPO update
```

核心是把 batch 边界放在 trainer 内部：trainer 从多个可行动环境一次拿到大量 learner
observation，直接做 GPU batch inference。外部 GPU server 适合 Evaluation 或过渡实现，
但不是长期在线 PPO 的首选架构。

## 4. 优化阶段

### Phase 0：建立可重复 profiling 基线

在任何重构前记录：

- episodes/s、learner decisions/s、total selections/s；
- collect/encode/learner forward/opponent/wait/update wall time；
- CPU 每核利用率、RSS、GPU utilization、显存；
- learner inference batch size p50/p90/p95；
- episode 长度和 opponent 分解；
- error、unfinished、illegal action、reset failure；
- 固定 seeds、opponent distribution、policy checkpoint hash。

先用 4/8/16 个普通独立环境 benchmark，确认瓶颈再选择下一阶段。

### Phase 1：Trainer 内 GPU batch inference

目标：模型只加载到 GPU 一次，多个环境的 learner observation 在 trainer 内合并成 batch。

需要实现：

- environment 到 trainer 的 observation/action queue；
- batch collator 和 legal mask；
- 无等待窗口的“当前 ready states”批处理；
- 每条请求携带 episode、decision、policy version；
- session history 在环境侧或 trainer session table 中按 episode 隔离；
- GPU forward 返回 action、old log-prob 和 old value；
- batch size、queue time、encode time 和 forward time 指标。

先比较 8/16/32 环境。若实际 batch 仍很小，继续增加环境或优化调度；不先增加人为
1–3ms batching window。

### Phase 2：常驻 CPU opponent actor pool

目标：opponent 策略只加载一次，反复接收决策请求，减少进程启动和重复模型/规则加载。

必须满足：

- actor 数先测 8/12/16，不按逻辑 CPU 数盲目放大；
- 每个 actor 固定 `OMP/MKL/OpenBLAS/NUMEXPR=1`；
- `(opponent_name, episode_id)` 固定路由到同一 actor；
- terminal 后显式清空 history、search state、effect serial 和 RNG/session；
- actor 崩溃只使关联 episode 失效，并能安全重建；
- 搜索型 opponent 的跨回合状态不串局；
- 记录 submit time、compute time、实际 wait time 和被 GPU 工作覆盖的时间。

CPU/GPU 重叠的正确顺序是：先提交 opponent 请求，同时执行 learner GPU batch 和 codec，
最后才等待尚未完成的 opponent 结果。

### Phase 3：官方 `.so` 的训练专用 vector wrapper

目标：不修改官方源码，在一个 native 进程中 `GameInitialize()` 一次，并持有多个独立
battle handles/slots。

官方 ABI 暴露了独立 battle data handle 的可能性，但 `GameInitialize()` 不是可重复调用
接口；同进程反复初始化曾出现全局表容量错误。因此第一步只能是可行性实验：

1. 单进程初始化一次；
2. 创建 2 个 battle handles，交错 observe/select/finish；
3. 扩到 8/16/32 handles；
4. 与独立进程基线比较合法 option、terminal result 和 trace schema；
5. 运行长时间 soak，监控 RSS、handle 释放和全局状态污染；
6. 任何串局、泄漏或非确定性异常都停止该方向。

wrapper 只能调用官方公开 runtime ABI，构建产物写到 `engine/build/` 或临时目录，不编辑、
格式化或 patch `engine/source/`。

### Phase 4：Native PolicyCodec

目标：减少 JSON parsing、Python object construction、重复 tensor allocation 和 collate。

优化顺序：

1. profiling 区分 engine 序列化、`json.loads`、feature encode、tensor allocation 和 H2D；
2. 先做 Python 侧预分配、pinned memory 和 batch tensor reuse；
3. 再考虑 C++/PyBind codec，把 observation 直接写进固定 PolicyCodec tensor；
4. codec 输出必须逐字段和 Python reference 对比；
5. feature schema 或 shape 改动同步更新 checkpoint metadata 和 `rl/model/DESIGN.html`。

不修改官方 engine 意味着无法假设能消除 engine 内部的 state copy、hidden-info removal、
Base64 或 JSON 生成。Native codec 主要优化 engine 输出之后的训练数据路径。

### Phase 5：异步流水线与内存优化

前四阶段稳定后再考虑：

- double-buffered rollout batches；
- pinned memory 与 non-blocking H2D；
- trainer update 和下一批环境推进的受控重叠；
- mixed precision inference/training；
- `torch.compile` 或 CUDA Graphs（只有 shape 稳定且 profile 有收益时）；
- compact rollout storage，避免长期保存完整 JSON observation；
- checkpoint、evaluation 和 collector 的 I/O 异步化。

PPO 初版仍保持同步 policy version。不能让环境在不知情时混用更新前后权重；若未来做
异步 actor-learner，必须记录 behavior policy version，并使用匹配算法处理 policy lag。

## 5. 建议规模和目标梯度

本机第一轮不要照搬 2048 环境或 32 actor：

| 阶段 | 环境 slots | opponent actors | 目标 |
|---|---:|---:|---|
| 正确性基线 | 4–8 | 每局独立 | 跑通在线 PPO |
| GPU batch 原型 | 16–32 | 每局独立 | 实际 batch 明显大于 1 |
| actor pool | 32–64 | 8–12 | 降低 opponent/启动等待 |
| vector wrapper | 32–64 起 | 8–16 | 降低进程与 Python 调度成本 |
| 扩展验证 | 64–128 | 由 profiling 决定 | CPU/GPU 平衡、0 串局 |

以当前约 467 learner decisions/s 的研究原型为参考，第一个正式训练优化目标可以设为
600–900 decisions/s；完成 actor pool 和 codec 后再观察是否具备 700–1100 decisions/s
的条件。目标只有在相同 policy、opponent 分布、episode 协议和正确性护栏下才可比较。

## 6. 每阶段验收护栏

所有性能实验必须同时报告：

- 与官方 runtime 独立进程基线相同的合法 action contract；
- completed/error/unfinished；
- episode reset 和 policy version 审计；
- decisions/s、episodes/s、wall time；
- CPU/GPU utilization 和内存；
- 实际 inference batch size 分布；
- 固定 Evaluation 结果只作为能力证据，不拿随机胜负验证性能语义。

性能分支的晋级顺序：

```text
unit/contract tests
  -> 2/8/16 episode interleaving test
  -> 1,000+ episode soak
  -> throughput benchmark
  -> 固定 180 局 correctness/evaluation gate
  -> 才能替代上一条训练路径
```

吞吐提高但 engine error、illegal action、状态串局、policy version 混用或内存持续增长，
一律视为失败。

## 7. 与 Evaluation 的边界

RL training 可以使用 vector slots、actor pool 和 trainer-owned GPU inference，因为训练目标
是高吞吐采样，并且有独立正确性验证。

正式 Evaluation 保持：

- 每局独立 engine 和策略进程；
- 固定 catalog/game 顺序；
- 官方 runtime；
- 稳定 CPU 并行作为默认可靠路径；
- GPU inference server 只有完成 session 隔离和等价性验收后才作为可选路径；
- 不因训练路径更快而改变正式评测协议。

这条边界保证训练系统可以积极优化，同时不损失版本间结果的真实性和可比性。

## 8. WSL 与硬件扩展建议

增加 WSL 可见 CPU/RAM 只有在 profiling 对应瓶颈时才有帮助：

- CPU 长期满载、rollout 队列供给不足：增加物理核心或 WSL CPU 配额有帮助；
- RSS 接近内存上限、出现 swap：增加 RAM，优先消除 swap；
- GPU 等待 CPU：优先提高环境/actor 吞吐；
- CPU 等待 GPU：增加环境只会加重排队，应优化 batch/模型或降低并发；
- 多进程各自开多线程：先限制内部线程，不先增加资源。

每次调整 WSL CPU/RAM 后重新跑固定 benchmark；不同资源配置的数字不能直接混在同一条
吞吐曲线中。

## 9. 暂缓条件

满足以下条件时不继续做性能工程：

- 正常 PPO iteration 已能在训练预算内完成；
- rollout 不再是主要 wall-time；
- GPU utilization 与 batch size 已合理；
- 下一阶段收益预估小于引入的正确性和维护风险；
- 当前更大的瓶颈是 reward、value、policy collapse 或 opponent distribution，而不是吞吐。

此时优先改善 RL 学习问题，并保留本文件作为未来扩展参照。
