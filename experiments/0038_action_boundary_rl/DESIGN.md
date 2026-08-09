# 0038 Action Boundary RL 设计

## 目标和证据边界

0038 的目标是让强化学习时间只推进于真实游戏决策，而不是官方 UI callback。通用官方规则仍以攻击为回合提交；Phantom Dive 的 200 主伤害和后场六个伤害指示物共同属于同一次攻击。当前 engine runtime 事实是六个指示物通过 `context=14`、`remainDamageCounter=6..1` 的六次 primitive `select` 完成。项目策略假设是：此链没有随机、隐藏信息揭示、对手响应或优先权转移时，可作为一个 compound action。

官方 `engine/source/`、ABI、observation、option index 和返回 `list[int]` 均不改。0038 只改 Agent、rollout wrapper、trajectory、附加 head 和训练适配层。

## 零样本基线

- State Encoder、Option Encoder、Root Action Decoder：加载 Policy-0806 epoch 11。
- Value：加载 0036 V2 epoch 5 latent-query value head。
- Conditional allocation head：0038 新增；用同一 Zero-Shot policy 在 legacy sequential contract 下采集的 official-engine 标签做短 BC warm start，之后才允许 PPO。
- optimizer、scheduler、on-policy buffer：不继承。
- GRU 保持 request-local，每个真正的 root request 重新初始化；forced/macro callback 不维护跨请求 hidden。其状态变化通过完整消费 observation/history 进入下一个真实 request。

初始化的权威来源为 `friend_0806_epoch11/model.pt`（Policy-0806）和 0036 V2 epoch-5 pre-RL latent-query Value。0037 V5 或任何 `checkpoint/update-*` 都被拒绝。Last Option self/cross attention merged Q/V LoRA 固定 r/alpha=4/8，A 为 Kaiming、B 为零，因而 update-0 是精确 zero-delta；Action Decoder 全量可训练，原始 Encoder 冻结。Prize/Meta 等辅助模块使用隔离 RNG，关闭模块或 zero conditioner 时不会改变 core/LoRA 初值、root logits、legal mask 或 `V_win`。

## DecisionGate

Gate 位于官方 observation 读取后、tensor compile 前。主分类为 `TERMINAL`、`LEGAL_EMPTY_PASS`、`MASK_ERROR`、`FORCED`、`STRATEGIC`，并可附带 `CHANCE_BOUNDARY`、`INFORMATION_BOUNDARY`、`PRIORITY_TRANSFER` 边界标记。

- 0 options：terminal、合法 `min=max=0` empty pass 或 fail-closed mask error。
- 仅当 canonical completion 为 1 且没有 STOP/cancel/decline/pass 时才 forced。
- forced 路径执行 `observe_only`，更新 knowledge/history/event；不 compile tensor、不调用 Transformer/GRU/Value/服务、不写 policy transition。
- 所有其他情况保持 strategic。未知语义不做 shortcut。

## Phantom Dive compound action

Canonical action：

```text
DragapultDamageAllocation {
  target_ids: stable (player_index, serial, card_id, initial_bench_slot),
  counters: nonnegative integer vector,
  total: 6
}
```

`n` 个目标的分配数是 `C(n+5,6)`，`n=1..8` 分别为 `1,7,28,84,210,462,924,1716`。目标以 stable identity canonicalize；allocation 不包含选择顺序，因此没有排列 alias。

当前目标合同覆盖官方 callback 中 `1≤n≤8`，包括零之大空洞的扩展 Bench。候选数超过 210 时仍由同一个 conditional allocation head 在一次完整 State Encoder 结果上批量评分；不得退回六次完整 Encoder/Value/PPO transition。超出已声明支持域或 transaction 漂移必须 fail closed，不能在内部 callback 清空事务后重新调用 Policy。

若攻击者处于混乱状态，官方在 Phantom Dive root commit 后、allocation callback 前揭示硬币结果。v2 protocol adapter 将其视为真实 `CHANCE_BOUNDARY`；此路径仍需在 release parity 修复后重新定义并证明，不能用 legacy 多次 Policy/Value callback 冒充 compound transition。任何未预见的 pending-transaction 漂移必须保留 trace、使样本无效并 fail closed。

概率严格分层：

```text
log P(complete) = log P(root) + log P(allocation | state, Phantom Dive)
```

Root decoder 的 logits、mask 和归一化集合不变。只有 root 已选中 Phantom Dive 后，allocation head 才在最多 210 个 canonical allocations 内归一化。`n=1` 参数 forced，parameter logprob/entropy 均为 0。

allocation head 复用同一次 `EncodedState.summary/cards` 与 root option embedding。每个目标的可见特征宽度为 12：分配数、放置后剩余 HP、立即 KO、可拿 Prize、距 KO 损伤、Bench 位置、当前伤害、附着能量、状态、可见进化风险、HP headroom、分配比例。目标行经共享 MLP 后 sum pooling，并加入集中/分散统计，所以同时置换目标及其分配不会改变 logit。

## 官方协议适配

`MacroPlanner` 在第一次 root 决策生成完整 allocation；`PendingMacroTransaction` 以 battle ID 隔离；`OfficialProtocolExecutor` 在之后六个 callback 中：

1. 正常消费 observation/history；
2. 校验 battle、actor、context、remain count、effect 和 legal mask；
3. 按 serial/card identity 在当前 Bench/legal option 中重新定位；
4. 返回原格式单元素 `list[int]`；
5. 第六次后清理 transaction。

漂移、目标消失、空 mask、超时、terminal 或优先权异常均 fail closed 并使 rollout macro 无效。官方 primitive 调用次数不减少；focal 模型 forward 从 root+六 callback 的 7 次降为 1 次。

allocation 的 Prize 可见特征只从版本化 public card prototypes 推导：普通 Pokémon 为 1、Pokémon ex 为 2、名称以 `Mega ` 开头的 Mega Evolution Pokémon ex 为 3；不能用字符串任意包含 `mega` 的规则误判 Yanmega ex。CPU package、official wrapper 与 CUDA adapter 共用同一映射。

## PolicyTrajectory

`EngineEventTrace` 保存所有 observations、primitive selects、forced events、reward/terminal/winner 与边界。`PolicyTransition` 每个真实决策仅一条：pre-action features、canonical macro、root/parameter/joint old logprob、pre-action value、累计 reward/discount、post-commit observation、next value、done、event span 和有效性。

Forced/macro 内部 callback 不产生 Value 样本，不进入 policy loss、entropy、KL、clip fraction 或 PPO ratio，也不额外应用 gamma/lambda。Semi-MDP GAE 使用：

```text
delta_t = accumulated_reward_t + accumulated_discount_t * next_value_t - value_t
lambda_t = 1                         if next strategic action is in the same turn
           configured GAE lambda     if the transition crosses a real turn boundary
A_t = delta_t + accumulated_discount_t * lambda_t * A_(t+1)
```

`credit_clock=turn` 与 0037 合同一致。V8 曾错误地在每个 strategic action 之间应用 `lambda=0.95`，使 return standard deviation 从约 0.90 人为压到 0.65，并制造异常偏低的 Value loss；该路径已由固定 same-turn trace 回归测试封死。

terminal 时 `next_value=0`。entropy/KL/clip 的分母仅为 valid strategic transition；因此不可与 V5 历史曲线直接横向比较。

## 模块化集成目标

- `V_win`：Value Query 0 和终局 ±1 target 完全不变。
- `V_prize`：使用 latent Query 3 的小 sidecar；`r_prize=(己方拿奖-对方拿奖)/24`，以 own-turn clock 独立计算 `A_prize`，默认 `gamma_prize=lambda_prize=0.97`。`A_win` 与 `A_prize` 分别标准化，再用固定 `alpha_prize` 组合。支持 off/directional/terminal_neutral。
- `OpponentMetaHead`：只读 deployable public state/history，Frozen deck ID 只作 label。taxonomy v1 含 unknown/ambiguous 与 Frozen catalog archetype；预测分布经 detach 和 zero-initialized conditioner 进入 Action summary，PPO 不反向到 Meta Head。
- Tempo：第一版只有公开 legal Attack opportunity、第二个己方回合攻击、连续攻击 streak 和中断原因等指标；不启用 tempo reward/loss，curriculum 只允许增加自然 opportunity state 的采样。
- Loss Registry：独立登记 policy/value-win、value-prize、opponent-meta、root entropy、allocation entropy 与预留 tempo loss；每项有 flag、weight、optimizer group、日志和 checkpoint metadata。

预设为 BASE、PRIZE、META、PRIZE_META、INTEGRATED。共同 update-0 是 Zero-Shot core + fresh zero-delta LoRA + BC allocation head；optimizer、scheduler、RNG、rollout buffer 和 old-policy snapshot 每个新实验重新建立。

## Rollout 扩容与统计频率

rollout 配置不再假设固定 512 局或固定 decision 数。`engine_backend`、games/workers/envs/batch/inflight/queue/seed shards、PPO physical minibatch、gradient accumulation、epochs 和 optimizer steps 均显式配置。accelerated backend 必须先通过 observation/legal/reward-terminal-winner/RNG/full-game/macro parity，并与 official backend 使用同一 Agent action contract。

V4–V10 使用 `accelerated:cuda_resident`：规则 blob 与 CUDA state machine 驻留 GPU，设计上 Agent 应执行同一 0038 DecisionGate、hierarchical Phantom allocation 和 primitive-select 合同。CUDA collector 只应把真实 decision boundary 写入 `PolicyTrajectory`；公开奖赏数和状态位等 allocation 条件特征随 macro action 保存，用于精确重放 joint old-logprob，不引入隐藏字段。任何 actor、目标、计数、chance 或 transaction 漂移都必须 fail closed 并排除 PPO。

扩容有两种显式模式：`fixed_epochs` 会随样本增加 optimizer steps；默认优先 `fixed_optimizer_budget`，固定每 update optimizer steps，通过 effective minibatch、累积或抽样吸收更多 rollout。切换模式必须由用户确认，不能连带静默改变 LR、epochs、effective batch 或 advantage normalization 范围。

V10 的 `fixed_optimizer_budget` 每 update 执行两个完整 256-slot frequency unit，共 512 场 stochastic game，并为全部 512 场保留完整 EpisodeTrajectory；不再额外模拟不会进入 PPO 的 1,536 场。约 44k valid strategic transitions 足够固定的 `32 × 1024` optimizer budget。rollout 数量仍配置化，未来若扩大必须让新增 trajectory 真正进入样本池。update 结束后显式释放 batch、Episode 和 collector 引用，避免上一批高维 feature 与下一批 rollout 重叠；V9 update 2 观察到的 9.48 GB RSS 平台没有继续增长，但不作为新版本的常驻内存合同。

指标分三档：每 update 只聚合已有 forward 的低成本 scalar 与七阶段 wall time；Frozen point 从已有 2,048 局逐局结果离线聚合；梯度范数/cosine、allocation margin 和详细 calibration 只在固定小 minibatch 的 update 0/5/10/之后每 10 次运行。exhaustive parity、hidden leakage、RNG/state A/B 和 fault injection 仅在测试/CI。

## Frozen Evaluation

正式评估唯一合同为全局 `frozen_0806_seeded_2048_v2`：evaluation seed `341512806`，8 replicas × 原始 256-slot 环境频率分布，共 2,048 个唯一 engine seed，先后手各 1,024。007 标准 schedule SHA-256 为 `98b58bced460c1a2e622ae4b39bf506294fcb0230bb42e4117aaa6efc73c9ce9`。update 0 与之后每 5 updates 必须复用相同 seed、slot、replica、opponent 权重、seat、greedy、CUDA backend 和 termination contract；相对 update-0 报告 paired flips、McNemar、Wilson CI、seat、matchup 和 replica 方差。历史 `0038_frozen_2048_v1` 是均匀 exact-deck 面板，仅保留 V2/V3 证据，不再用于正式比较。

训练 rollout 不复用 evaluation seed，但以 256 games 为不可拆分频率单位。每个连续 256-game unit 精确包含 Frozen catalog 的 opponent slot counts，unit 内按 `(training seed, source policy update)` 随机排列并平衡 seat；V10 每 update 使用两个完整单位，共 512 个独立 engine seed。

2,048 paired games 适合判断约 2pp 的方向与稳定性，但很小差异不能自动宣称显著。按 discordant rate 0.10/0.20/0.30 的近似规划，2pp 约需 1,962/3,923/5,884 局，1pp 约需其四倍；最终检验使用实际 paired flips。

## Checkpoint 与阶段

checkpoint schema 为 `0038_model_only_checkpoint_v1`，强制记录 action、gate、canonicalizer、trajectory 与 official adapter 五个版本。model-only payload 禁止 optimizer/scheduler/RNG/rollout/replay。

公共基线为 `V3_update0_chance_boundary_fallback`，PPO updates=0，official adapter 为 `0038_official_primitive_adapter_v2_chance_fallback`。Frozen 结果为 1308-739-1、0 error、14 safe fallback；Meta 全面板聚合因 logits 容器类型错误未发布，代码已修正，遵循“不得为统计额外 forward”合同没有单独重跑。

CUDA resident Action Boundary、吞吐 benchmark 与真实 CUDA rollout→PPO replay smoke 已完成。V4 在错误的均匀 exact-deck 面板完成 update-0 后，于首个 PPO update 前因 sparse-diagnostic predicate 未导入而 fail closed；没有产生 update-1 model checkpoint，其 `63.77%` 不得与 canonical 007 `57.32%` 比较。V4 保留为失败审计记录。

`V6_canonical_frozen_cuda_fresh_rl` 已应用户要求在首个 rollout 期间人工中止，等待 50 回合终止合同完成。V6 只保留 canonical update-0 baseline（1162/2048，56.74%）和 model-only update-0 checkpoint，没有执行 PPO update，不得续写。后续正式训练仍从 V3 common update-0 加载 BC-warm-started allocation head，启用 `INTEGRATED` preset，并重新初始化 optimizer/scheduler/RNG/rollout/old-policy。V10 每 update 采集 512 局完整 trajectory、使用固定 optimizer budget；update 0 和之后每 5 updates 运行 canonical Frozen-0806 2,048，不启用 8,192 分支。run 不设 update 上限，在完整 update 边界响应人工 stop sentinel。

防挂死截断合同从 `V8_repeat_guard_turn_limit_cuda_fresh_rl` 开始作为新正式语义。`RolloutJob.full_round_draw_limit=50` 是 CPU/official wrapper 与 CUDA resident scheduler 的单一来源；CUDA 在 official turn index 99 进入 scheduler-owned terminal draw。这不是官方胜利条件，而是项目截断语义。截断局保留为有效 Episode，无 winner、reward 0；PolicyTrajectory 将累计 reward 0 归入最后真实决策，设 `done=true` 与 `next_value=0`，并在 Episode diagnostics/W&B update scalar 中分别保留 `turn_limit_draw` 和 `rollout/cuda_turn_limit_draws`。resident batch 混用不同 limit 或 limit 为负数时 fail closed。

V7 在首批 stochastic rollout 暴露了 Mega Venusaur/Meganium 的重复 Ability 链：repeat guard 虽然按 actor 累计，但 key 包含会随 legal-option 排列变化的 option index，导致同一 Ability 被拆成多个计数，最终进入 mandatory-empty mask error。V7 在 PPO update 前 fail closed，只保留 update-0 baseline `1094-954-0`（53.42%）。

V8 将 key 改为 stable semantic identity，却错误地跨整局累计相同 Ability。它在 canonical 2,048 update-0 触发 242 次判负，只得到 `1005-1043-0`（49.07%）；随后完成一次 PPO update，但该版本因 Zero-Shot parity 失败及 8.94 GB trajectory/16.24 GB RSS 内存失控而判为无效，不得续训或作为收益证据。

V9 的 repeat guard 仍按 actor 和 stable identity 识别同一 Ability、忽略 option index/目标排列，但在 official turn 变化时清空计数：只惩罚同一回合内 20 次无进展循环，正常跨回合 Ability 使用不会累计。canonical 2,048 A/B 中，legacy sequential 从错误 guard 的 `1003-1045` 恢复为 `1178-870`（57.52%，13 forfeits）；forced-only wrapper 与 legacy 完全同记录，完整 Phantom macro 为 `1162-886`（56.74%，12 forfeits），与旧 007 `1174-874`（57.32%）差 0.58pp。新正式版本为 `V9_turn_scoped_guard_bounded_trajectory_fresh_rl`，必须重新从公共 PPO-update-0 初始化。

V9 完成 update 2 后因 rollout 效率合同变更停止；其 2,048→512 trajectory 抽样不再续用。新正式版本为 `V10_complete_512_rollout_fresh_rl`，从公共 PPO-update-0 重新初始化，不加载 V9 PPO 权重。

## 2026-08-09 release-blocking semantic parity audit

本轮先固化 U230 的 pre-fix failures，再按 Gate A→B→C→D 完成 runtime 修复。修复后的证据为：

- Gate A：最新源码重编后的 Dragapult/Dusknoir、Area Zero、Zoroark/Munkidori 共 697 个 fixed primitive decisions，CPU/reference/CUDA state、status、outcome mismatch 均为 0。原 decision 129 continuation failure 来自没有源码 provenance 的 stale cached binary；缓存现在必须校验 official/CUDA source、rule blob、compile contract 和 executable hash。
- Gate B：真实 official Area Zero fixtures 对 `n=6..8` 的 3,102 个 allocation 全部通过 canonical macro、legacy sequence 与 reversed-order alias parity；加上既有 `n=1..5` 的 330 个 allocation，最终权威 public state failure 为 0。CUDA/package callback 按 `option_source → stable serial` 重定位，任何 drift 都 fail closed，不能在内部 callback 重新调用 Policy。
- Gate C：immutable 283-decision trace 的所有模型输入 tensor、legal/mask 精确一致；143 个 focal decisions 的 root top-1/top-2/greedy divergence 为 0。CUDA/package root-logit 最大误差 `6.44e-6`，训练/CUDA Value 最大误差 `1.37e-6`、无符号翻转。修复涵盖 option skill/effect relations、event history、public resource/deck/Prize knowledge 和 feature-schema attestation。
- Gate D：8 局 CPU package/CUDA lockstep、1,335 callbacks、21 个 Phantom macros/126 个内部 callbacks，first divergence、fallback、timeout 均为 0。过程中额外修复 full-deck Prize 泄漏、stale exact-deck 重建和 Zoroark copied-attack primitive alias 被 CUDA 去重的问题。
- forced `observe_only`、macro one-forward/one-Value/one-transition、joint logprob 和 boundary GAE 合同继续通过；GRU 保持 request-local。

这次没有修改 `engine/source/`、official ABI/observation schema、checkpoint tensors、reward、学习率或模型结构。不过 U230 权重是在上述 feature 修复前的 CUDA 输入分布上通过 PPO 得到的，运行时 Gate PASS 不能恢复它的训练分布。因此 U230 继续是不可提交、不可续训的诊断 checkpoint。release manifest 会在 checkpoint metadata 缺少当前 `0038_cpu_authoritative_cuda_feature_parity_v1` attestation 时 fail closed。

下一阶段只能从原始 Zero-Shot/Pretrain 权重建立修复后 U0；不得从 U215/U230 继续。完整证据、修改文件、命令与风险见仓库根 `SEMANTIC_PARITY_AUDIT.md`。

## V13 Accelerated RL Transfer Acceptance

V11 在保存 U0 前因旧 checkpoint validator 不接受 repaired action-contract metadata 而 fail closed；V12 完成 U0 CUDA Frozen 后又因 package gate 指向 220 行中间 trace 而未进入 PPO。两个目录都保留为失败审计。V13 绑定 SHA-256 固定的 283 行 fixture，并从 common update-0 新建 on-policy 轨迹：Zero-Shot core、pre-RL `V_win`、BC allocation head 与 fresh zero-delta Q/V LoRA；optimizer、RNG、rollout、old logprob 和 GAE 全部重新生成。它不加载任何 U215/U225/U230/U255 或其他 PPO 权重。

本轮显式使用 `PRIZE` preset：保留 0038 已有 directional Prize auxiliary 和只读 Tempo metrics，但关闭 Opponent Meta、Meta conditioning、Tempo curriculum/loss 以及所有后续 Seat/Plan/Search 结构。训练合同固定为 CUDA resident、每 update 两个 256-slot unit 共 512 个 opponent/seat 槽位、恰好 512 条有效 trajectory 入池、`fixed_optimizer_budget=32`、physical minibatch 1024、accumulation 1。Confused 会在 Phantom Dive root 与 allocation 之间揭示真实随机结果；该局的 legacy primitive fallback trace 只作诊断且不能进入 compound PPO。运行器只允许这个明确的 chance-boundary reason，以相同 opponent/seat 和新的可审计 engine/policy/search seed 补齐原槽位；其他 macro drift、fallback、unsupported 或 pending reset 仍立即 fail closed。每次排除和补采均写入 `artifact/schedules/chance_boundary_replacements/`，不得静默丢弃或改变 256-slot 环境频率。

实际 V10 base LR 与 V13 加速计划如下：

| optimizer group | base LR | U1–2 | U3–5 | U6+ 健康上限 | gradient source |
|---|---:|---:|---:|---:|---|
| Action Decoder | 1e-5 | 3e-5 | 5e-5 | 1e-4 | policy/entropy/reference-KL/Prize actor |
| Allocation Head | 1e-5 | 3e-5 | 5e-5 | 1e-4 | allocation policy/entropy/Prize actor |
| Last Option Q/V LoRA | 3e-5 | 9e-5 | 1.5e-4 | 3e-4 | shared Actor path |
| V_win | 1e-4 | 2e-4 | 2e-4 | 2e-4 | terminal win Value only |
| V_prize | 1e-4 | 2e-4 | 2e-4 | 2e-4 | directional Prize Value only |

Behavior KL `>0.005` warning，`>0.01` 将后续 Actor cap 从 10→5→3；`≥0.02` 或 KL early-stop 会撤销本次所有 trainable tensor、清空 optimizer state 并用较低 cap 重试同一 on-policy batch。clip fraction `>10%` warning、`>30%` 同样回滚；NaN/Inf、old-logprob replay mismatch、非 chance-boundary 的 invalid/fallback、unsupported 或 pending reset 直接 fail closed。

U0 与之后每 5 updates 只运行 CUDA Frozen-2048。U0 package parity 使用已经固化的 283-decision official snapshot，不启动新的 CPU 对局：比较 repaired CUDA feature、training checkpoint、strict portable package 的 tensor、mask、root/allocation intent 与 Value。正式 run 不设 update 上限，在完整 update 边界响应 `STOP_REQUESTED`；前 50 updates 只是观察窗口。CPU engine 评测不自动调度，必须等用户查看曲线并指定 checkpoint 后另行执行。

Frozen 面板与 stochastic rollout 对真实 `chance_boundary_before_allocation` 采用不同的数据合同。Frozen 必须保留原始固定 seed、官方逐 callback 执行和最终 outcome，不允许补采；该事件单列为 `eval/core/chance_boundaries`，不计作 semantic fallback。逐局 v2 结果同时保存原始 fallback reason、chance-boundary 与 semantic-fallback 位。训练 rollout 则保留完整诊断 trace 后，在同 opponent/seat 槽位补采，因为该 legacy sequential episode 不能作为 compound PPO 样本。两条路径都继续对未知原因、stable identity drift、unsupported 和 pending reset fail closed；Frozen runtime telemetry 使用 `eval/runtime/*`，不得覆盖同一 update 的 `rollout/*`。
