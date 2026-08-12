# CUDA 官方引擎迁移验收快照（2026-07-30）

## 当前结论

当前版本已经从旧的 `BattleState` smoke prototype 推进到独立的
`OfficialStatePod ABI v6`、私有 typed rule pack、CPU POD/CUDA 共用解释器，以及可批量
持有官方状态和 device action 的 `OfficialDeviceArena`。本机 Docker CUDA 可正常编译、
运行和做逐字节 paired test。

这仍不是可启动正式 league RL 的完整 CUDA 引擎。MainSelect 已有覆盖 Play、Attach、
Evolve、Ability、Discard、Retreat、Attack、End 的 35 个垂直切片，但还没有完成全卡牌、
全 skill 和全条件分支覆盖；全动作 100,000 decision 官方全局 replay、六牌组完整 paired battle、
device codec 和 full-GPU PPO 也尚未通过。

## 已通过证据

| Gate | 本次结果 | 边界 |
|---|---:|---|
| Python 合同测试 | 44/44 passed | schema、RNG、pool、memory、prototype codec |
| 官方 State bridge | 10,000 个真实状态，零差异 | setup 后 Main decision、128 cards、zones、options、RNG、continuation |
| 官方 setup CPU/POD | 100,000 seeds / 428,336 decisions，零差异 | setup-only；每个状态经通用 bridge，不等于完整战斗 replay |
| setup POD CPU/CUDA | 10,000 seeds，零差异 | setup 专用 POD kernel |
| effect interpreter POD/CUDA | full-state digest 一致 | fixture 覆盖，不等于 245 类全分支 official parity |
| trigger resolver POD/CUDA | 8 states，逐字节一致 | 含顺序、可选发动、checkup RNG |
| KO/Prize POD/CUDA | 5 states，逐字节一致 | 含多 KO、Lucky Bonus、换位、终局 |
| turn-flow official/POD | 6 scenarios passed | bench/tool LIFO、turn transition 等 |
| turn-flow POD/CUDA | 3 states，逐字节一致 | 含 overflow、清理、deck-out |
| Attack official/POD | 14 scenarios / 120 checks | 含 `noTargetEffect` RNG 消耗语义 |
| Attack POD/CUDA | 70 scenarios，逐字节一致 | ABI 6，digest `5138934968570255249`，errors=0 |
| MainSelect official/POD | 35 scenarios / 35 checks | 基础 POD 由官方 State bridge 生成 |
| MainSelect POD/CUDA | 35 states，逐字节一致 | digest `14876688999377837438`，含 decision 计数 |
| continuation dispatcher | opcode 47/79 CPU/CUDA 逐字节一致 | 47 MainSelect -> 79 SelectedMain -> action -> 新 79；action 前后均 `kNeedsAction` |
| end-turn battle official/POD | 1,100 seeds / 103,400 decisions，canonical 零差异 | 真实 setup-to-terminal，但 Main 只选 End；不覆盖 Play/Attack/effect |
| end-turn battle POD/CUDA | 1,100 seeds / 103,400 decisions，raw state/status 零差异 | 每步完整 state D2H 只用于 correctness，不代表生产热路径 |
| Basic/Evolve/Attach battle official/POD | 1,000 seeds / 137,271 decisions，canonical 零差异 | 单一 Marnie/Alakazam matchup；含 Basic Play、直接进化、基础能量、optional decline、Prize、换前台、SkillOrder |
| Basic/Evolve/Attach battle POD/CUDA | 1,000 seeds / 137,271 decisions，raw state/status 零差异 | 9,353 次 SkillOrder；每步 D2H 仅用于 correctness；不覆盖 Attack 和一般 Trainer/effect 选择 |
| production arena | 4,096 environments passed | state/rule/action/status 常驻和 fail-closed |
| official fixture GPU resume | 1 个挂起复制攻击状态逐字节一致 | 统一 device action dispatcher 正向路径 |

Attack 扩展到 70 场景时曾发现 benchmark 固定只启动 64 个线程；这是测试 launch grid
错误，不是 CPU/CUDA 引擎语义差异。改为按场景数计算 blocks 后，当前 70 个状态的聚合
digest 均为 `5138934968570255249`。

MainSelect oracle 发现并修复了两个会污染后续 replay 的合同问题：Main context 曾错误写成
`0(None)` 而不是 `1(Main)`；`turnActionCount` 曾在消费 action 时递增，而官方是在产生新
decision 时递增。新 dispatcher 统一在 decision-yield 边界计数。另一个 runtime smoke
参数解析错误曾使带 fixture 的 `--batch 1` 实际运行 4,096；修复后真实 batch 1 和 4,096
均通过。原非法显存访问的根因最终定位为 CUDA device stack overflow：1/2/4 KiB 失败，
8 KiB 通过，正式 runtime 取 32 KiB 下限并报告实际生效值。

扩展 Main/Attack fixture 还发现并修复了两处规则错误：continual `CannotRetreat` 曾误映射到
攻击禁止位，导致不能撤退时同时不能攻击；有色能量支付曾把错误属性能量当成无色支付。
Rainbow DNA 进化现按一般规则匹配，Ability once-turn 记录也修正为 card ID 而不是 skill ID。
官方常量 `ANGE_FLOETTE=1429` 超出冻结快照实际 card ID `1..1267`，因此该 Stadium 分支在
`3aaeaa92` 中以 `unreachable-with-proof` 记录，不伪造 card 1429 fixture。

ABI v6 新增独立 `flow_flags`，把 CUDA 内部 Main/Refresh/Trigger 返回路径与官方
`changed/isBreak/effectLoopStop/failRetreat/stateChanged/updateOrder` 六个控制位分开。
ABI 大小仍为 119,936 B，现有显存估算不变。通用 `State -> OfficialStatePod` bridge
对容量溢出、无 RNG draw count 和非法 continuation 显式报错；Main 与 Attack oracle 的
基础 POD 已改由同一官方状态生成，不再各自手工搭建两份基础状态。

`MainSelect(47)` 与 action continuation `SelectedMain(79)` 现已形成闭环。79 是 one-shot
continuation，消费 action 前弹出；每次生成新的 Main decision 时重新压入新的 79。非预期
continuation 继续 fail-closed。真实 end-turn replay 还修正了 decision/terminal boundary 上的
内部 scratch、turn-scoped `Card.turnState` 和官方 `clearSelect()` 精确语义。official/POD
比较只在 host 副本上清零固定容量 list 的 inactive 尾槽；CPU POD/CUDA 从同一起点推进仍要求
raw 全字节一致，不在生产 GPU decision boundary 增加无语义的尾槽清零。

2026-07-31 的 full-battle 扩展把 deterministic policy 从 End-only 推进到
Basic Play -> direct Evolve -> Basic Energy Attach -> End，并补齐了 optional ability、Prize
多选、Active replacement 和 SkillOrder 的 continuation/action 合同。1,000 seeds 共比较
137,271 decisions，其中 Basic Play 9,311、Evolve 13,319、Energy Attach 9,936、optional decline
7,431、Prize selection 5,439、Active replacement 2,382、SkillOrder 9,353；official/POD canonical
差异为 0，CPU POD/CUDA raw state/status 差异为 0。该证据只覆盖当前两副牌和固定策略，
不能替代 100,000 全动作 decision、全卡池或六个晋级牌组 gate。

所有本次 machine-readable 输出位于 `engine_cuda_2_0/artifacts/`，该目录被 Git ignore。
官方生成 rule pack 和正向恢复 fixture 位于 `engine_cuda_2_0/generated/private/`，同样禁止进入
Git、package 或 W&B。

## 本机显存实测

设备为 RTX 3060 Laptop 6 GiB，Windows WDDM，CUDA 在
`nvidia/cuda:13.0.0-devel-ubuntu24.04` 容器中运行。

`OfficialStatePod` 的 ABI v6 大小为 `119,936 B`，不是早期计划中的 32 KiB。4,096 个环境
加每环境 272 B device action、1 B status 和一份 rule pack 的 arena 账面分配为：

| 指标 | 结果 |
|---|---:|
| arena API allocated | 493,250,336 B / 470.4 MiB |
| effective device stack | 32,768 B / thread |
| WDDM stack reservation（本机近似） | 约 1.47 GB / 1.37 GiB |
| `cudaMemGetInfo` free-memory delta | 1,958,739,968 B / 1,868.0 MiB |
| classification mismatches | 0 |
| fail-closed mismatches | 0 |
| positive resume state mismatches | 0 |

free-memory delta 包含 device stack、CUDA context、module 和 allocator 粒度，因此大于
API 账面数组。它是目前
最可信的本机 engine-state arena 测量，但还不包含完整 device codec、模型 activation、
rollout retention 和 learner optimizer。

按当前保守 estimator（含 codec、外部 scratch/control、96 MiB 固定 reserve 和 1.2x
headroom），10 frozen BC + 1 learner、rollout retention 为 0 时：

| environments | engine operational estimate | engine + model estimate |
|---:|---:|---:|
| 4,096 | 约 2.1792 GiB | 约 2.5611 GiB |
| 16,384 | 约 4.2721 GiB | 约 4.9949 GiB |

这些规划值把本机约 1.37 GiB stack reservation 加入旧 estimator；它在不同 SM 数量和
residency 的 GPU 上并非严格常数，服务器必须重新测量。既有 4080 policy-only profile 的
峰值为 2.546 GiB allocated / 3.982 GiB
reserved；既有 CPU-engine/GPU-policy PPO 峰值为 17.425 GiB allocated / 23.402 GiB
reserved。32 GiB 足够继续开发和 PPO smoke，但正式 batch 仍必须按
512 -> 1,024 -> 2,048 -> 4,096 逐级实测。

## 尚未通过

- 官方 CPU vs CPU POD 至少 100,000 个覆盖 Play/Attach/Evolve/Ability/Retreat/Attack/effect
  的完整战斗 decision 零差异；当前已有 103,400-decision End-only gate，以及单 matchup
  137,271-decision Basic/Evolve/Attach gate，但仍不能替代全动作/全卡牌 corpus。
- 剩余 114 个 continuation 的 dispatcher 与单元 fixture；当前 opcode 47 和 79 已接通。
- MainSelect 所有动作族的全卡牌、全 skill 和全条件分支覆盖；当前仅 35 个代表垂直切片通过。
- 1,267 cards / 1,556 attacks 的逐 card/skill/branch 完整覆盖矩阵。
- 六个晋级 BC 牌组的完整 seeded CPU/CUDA paired battle。
- PolicyCodecV1 127,205 rows 的 device tensor element parity。
- 每个 legacy codec 的独立冻结 corpus 和 device parity。
- 完整 engine -> codec -> 10+ frozen BC + learner -> decode -> apply 的无 host 热路径。
- full-CUDA PPO fresh/resume smoke。
- 原生 Linux/TCC Compute Sanitizer、5090/32 GiB Nsight 和同硬件 >=3x 15-core CPU 基线。

用户已明确取消 24 小时 soak；因此只保留有界 stress，不把缺少 24 小时 soak 作为当前 RL
smoke 的阻塞项。其余 correctness、hot-path 和性能 gate 不因此放宽。

## 已知保留问题

Ninetales `660` 借用 Amarys `1207` 延迟效果的旧官方引擎 crash 仍保持精确 quarantine：
`KNOWN_DIVERGENCE_660_1207`。没有修改冻结 official oracle，也没有为 CUDA 猜测新语义。

历史 127,205 decision policy replay 的 12 次 CPU/GPU argmax flip 仍归类为 BC 神经网络
数值边界差异，不属于 engine/state/RNG parity 失败。引擎 gate 必须继续使用相同 raw action
做 paired replay；policy 风险继续采用冻结 GPU semantics 和 battle A/B。

## Go / No-Go

当前状态是 **CUDA engine development build，NO-GO for formal full-CUDA RL**。只有完成主阶段
状态机、100,000 decision、晋级牌组 paired replay 和 device codec 后，才进入 full-GPU PPO
smoke；Linux/TCC sanitizer、Nsight 和 >=3x 性能留到服务器恢复后做最终晋级。
