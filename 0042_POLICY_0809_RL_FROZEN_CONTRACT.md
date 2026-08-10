# 0042 Policy-0809 RL 与 Frozen Evaluation 强制合同

**状态：MANDATORY / NON-NEGOTIABLE**

**适用项目：`0042_full_model_design`**

**版本：1.0**

**日期：2026-08-11**

本合同约束 0042 的 RL rollout、PPO、CUDA routing、opponent pool、periodic
evaluation、正式 Frozen CPU/CUDA evaluation、candidate materialization、报告与测试。
实现、配置、运行产物或报告只要违反本合同，就不得启动正式训练，不得作为 Frozen
强度证据，也不得进入 Promote Champion 判断。

本合同从属于仓库唯一 canonical 协议：
[`docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`](docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md)。
两者发生歧义时，以 canonical 协议中更严格的要求为准。本文件只把 canonical
policy identity 规则具体化为 0042 的 Policy-0809-only 实现合同。

---

## 1. 最终决策

0042 只允许以下两类有效 policy role：

```text
focal policy
  = 0042 当前待训练或待评估的完整 effective candidate

opponent policy
  = immutable full Policy-0809
```

强制要求：

1. RL training opponent 全部使用完整、独立、immutable `Policy-0809`。
2. periodic Frozen evaluation、正式 Frozen CPU evaluation 和正式 Frozen CUDA
   evaluation 全部使用完整、独立、immutable `Policy-0809`。
3. 0042 不再存在 Policy-0806 training arm、Frozen-0806 benchmark、0806 opponent
   resolver、0806 report label 或 0806 runtime fallback。
4. focal 与 opponent 即使在 update 0 内容相同，也必须先作为两个独立 policy role
   materialize；不能因为同源、同 shape、同 checkpoint 或性能需求而共享 mutable state。
5. 所有性能优化都必须在 policy identity 和逐 lane exact-deck 语义正确之后进行。

本合同不作任何自动 `PROMOTE`、`HOLD` 或 `REJECT` 决定。

## 2. Policy 身份

### 2.1 Opponent 身份

0042 的唯一 opponent policy ID 是：

```text
Policy-0809
```

它必须解析为 registry 中完整的 Policy-0809 effective weights，包括：

```text
prototype encoder
state encoder
option input encoder
all option transformer layers
final norm
LoRA/delta（如 registry 声明）
action decoder
reconstruction schema and metadata
```

每次 RL rollout 和 Frozen evaluation 开始前，必须使用 0042 project-local resolver
完成 materialization，并持久化：

- requested policy ID；
- base checkpoint path 与 SHA-256；
- 每个 effective component 的 source policy ID 与 content hash；
- complete effective-policy hash；
- model、observation、action schema versions；
- 显式 `status = PASS`。

缺失字段、hash 不匹配、requested/materialized identity 不一致，必须在创建 CUDA
engine 或开始 official game 前 `FATAL`。

### 2.2 Focal 身份

focal policy 是 0042 当前 update 的完整 effective candidate。它包含所有会改变部署
推理函数的模块，例如 pretrained base、decoder、Value latent blocks、Value Adapter、
Policy Strategy Adapter、allocation head，以及未来显式加入的 inference delta。

focal candidate 不能被称为 `Policy-0809`。它必须使用独立 ID，例如：

```text
Candidate-0042-update000000
Candidate-0042-update000005
```

### 2.3 Focal/Opponent 独立性

以下对象必须独立：

- Python module instance；
- parameter object；
- tensor storage；
- state dict materialization；
- train/eval mode；
- mutable prototype/feature cache；
- CUDA graph 或 resident cache 中会随 focal update 改变的内容；
- policy identity audit 和 manifest。

PPO optimizer 只能持有 focal trainable parameters。opponent 的所有 parameters 必须
`requires_grad=False`，不得出现在 optimizer、GradScaler、checkpoint delta 或 focal
rollback state 中。

即使 focal update 0 与 Policy-0809 的部分 content hash 相同，0042 也禁止直接把同一
module instance 同时绑定为 focal 和 opponent。未来若做纯 immutable compute
deduplication，必须另行给出逐 tensor content identity、无 alias mutation 证明和
lockstep parity；在取得人工批准前不得启用。

## 3. Exact-Deck Lane 语义

### 3.1 两种不同身份

每条 lane 同时拥有两个不可混淆的身份：

```text
opponent_policy_id = Policy-0809
opponent_exact_deck_id / exact_deck_sha256 = 该 lane 实际使用的 60 张卡组
```

Policy ID 决定 weights。exact-deck identity 决定该 lane 的合法、公开、actor-visible
registered-deck static fields。相同 weights 不代表相同 exact-deck conditioning。

### 3.2 禁止的首 lane 复用

mixed-deck batch 中严禁：

```python
Semantic0031DeviceAdapter(opponent, jobs[0].opponent_deck)
```

然后把该 adapter 的 static fields 用于全部 lanes。

只要 batch 中存在两个不同的 `opponent_deck`，上述行为就是语义错配。即使 opponent
weights 确实都是 Policy-0809、运行稳定、速度更快或胜率看似正常，结果仍然无效。

### 3.3 合法实现

实现必须选择以下一种语义等价路径：

1. **推荐：per-lane exact-deck static routing**
   - 单份 immutable Policy-0809 weights resident；
   - 为 catalog 中每套 exact deck 预计算并缓存 static fields；
   - 每条 lane 根据自己的 exact-deck hash/index gather 对应 static fields；
   - 将 heterogeneous lanes 合并为一次或少量大型 Policy-0809 batched forward；
   - 输出 scatter 回原 lane 顺序。
2. **正确性 fallback：按 exact deck 分组**
   - 先按 exact-deck identity 分组；
   - 每组只允许一个 exact-deck hash；
   - 每组用同一份 Policy-0809 weights 和该组自己的 static fields 推理；
   - 结果按原 schedule 顺序还原。

分组 fallback 可用于 smoke、parity 和紧急正确性验证，但不是长期性能终点。

### 3.4 Cache key

任何 deck-dependent static cache 至少必须由以下 tuple 唯一键控：

```text
(
  requested_policy_id,
  effective_policy_sha256,
  exact_deck_sha256,
  observation_schema_version,
  feature_preprocessing_version,
  device,
  runtime_dtype
)
```

不得只用 deck number、array position、display name、首 lane、architecture shape 或
focal deck 作为 cache key。cache miss、重复 key 内容冲突或 hash 不匹配必须 hard fail。

## 4. 允许的 Opponent 共享

所有 opponent lanes 的 effective weights 都是相同的完整 Policy-0809，因此允许并鼓励：

- 一份 immutable opponent model resident；
- 一份经 hash 验证的 prototype weight memory；
- 一次 heterogeneous-lane batched inference；
- 相同 exact-deck static fields 的缓存复用；
- 按 Policy-0809 分组的 routing queue；
- 不改变 lane identity 的 gather/scatter、CUDA graph 和 kernel fusion。

共享的前提是每条 lane 的公开 state、legal options、exact-deck static fields、seat、
seed 和 output 都保持自己的身份。共享 weights 不得退化为共享首 lane 的 deck features。

## 5. 禁止的 Focal/Opponent 共享

0042 focal 与 Policy-0809 opponent 禁止：

- 用 focal encoder/trunk 代替 opponent encoder/trunk；
- 用 opponent decoder 连接 focal trunk，或反向组合；
- 将 focal update 后的 tensor 写入 opponent resident model；
- 用同一 mutable module instance 同时服务两个 role；
- 让 focal `train()`、optimizer step、rollback 或 checkpoint load 影响 opponent；
- 使用 `same_policy=True`、shared-policy fast path 或等价 shortcut；
- 根据 architecture/shape 相同推断 content 相同；
- 在 identity resolution 前决定共享、routing、batch 或 cache。

router 初始化必须显式满足：

```text
same_policy = false
focal_effective_identity != opponent role identity
requested_opponent_policy_id = Policy-0809
opponent_identity_audit.status = PASS
```

## 6. RL Rollout 合同

### 6.1 Policy 与 dtype

- focal PPO behavior policy、optimizer/master weights、rollout log-prob 和 model-only
  checkpoint 保持 FP32；
- opponent 使用 registry materialized 的完整 Policy-0809 FP32 runtime；
- opponent 绝不参与 optimizer；
- sampled rollout 的每个 Episode 在开始时绑定 `Policy-0809` 和 exact-deck hash，直至
  terminal 不得改变。

### 6.2 Schedule

每个训练 frequency unit 必须保存：

- neutral opponent pool ID 与 pool snapshot hash；
- 55 套 exact deck 的 ID、SHA-256 和频次；
- 每条 lane 的 opponent policy ID；
- 每条 lane 的 exact-deck ID/hash；
- engine、Search、policy seed；
- seat/toss contract；
- source focal policy update；
- lane routing audit hash。

`opponent_policy_id` 全部必须为 `Policy-0809`。pool 构成、采样权重或 curriculum 改变
必须分配新的 version/config identity。

### 6.3 PPO 数据

trajectory 必须保留产生它的真实：

- focal source policy update；
- opponent effective-policy hash；
- opponent exact-deck hash；
- lane routing audit status。

任一 Episode 缺少这些身份、发生 semantic fallback、policy mismatch 或 deck-routing
mismatch，不得进入 PPO batch。禁止 warning 后继续或只在 metrics 中记一次 error。

## 7. Frozen-0809 Evaluation 合同

### 7.1 新 benchmark 身份

0042 的唯一 Frozen benchmark 是：

```text
Frozen Policy-0809
```

实现时必须创建新的 0042 contract ID，例如：

```text
0042_frozen_0809_seeded_agent_first_player_v1
```

不得继续使用、别名复用或在报告中展示 Frozen-0806 contract ID。旧 benchmark 的
schedule hash 和结果不能冒充新的 Frozen-0809 证据。

### 7.2 固定评测语义

新合同保留以下已确认评测原则：

- master evaluation seed `341512806`；
- 一个 256-slot exact-deck frequency unit；
- CPU-256 使用 replica `0`；
- CUDA-2048 使用 replicas `0..7`；
- 每个 replica 内 exact opponent slot frequency 不变；
- seeded toss 只固定 toss winner；
- toss winner Agent 必须真实处理 official context 41 并选择先后手；
- harness 禁止指定、交替或平衡最终 seat；
- 每局最多 50 个完整回合；
- 同一 physical actor 在一个 official turn 中第 20 次提交同一完整选择时判负；
- 失败或 unfinished game 不得换 seed 补跑；
- 只有目标局数全部 terminal、0 error、0 unfinished、0 semantic fallback 才通过。

seed derivation 必须显式包含 focal deployment identity、`Policy-0809` opponent effective
identity、exact-deck slot、replica 和 namespace。实现生成的新 base schedule hash 与完整
evaluation schedule hash必须冻结进 0042 contract/test；不得沿用旧 policy identity 下的
hash 常量。

### 7.3 Candidate deployment

所有 periodic Frozen evaluation、CPU-256、CUDA-2048 和任何 Kaggle-strength 结果都必须：

```text
source FP32 checkpoint
  -> materialize complete effective candidate
  -> merge all deployed heads/adapters/deltas
  -> FP16 storage artifact
  -> strict-load exact artifact into FP32 runtime
  -> greedy official-engine evaluation
```

报告必须保存 source checkpoint SHA-256、portable artifact SHA-256、deployment-effective
hash、storage/runtime dtype、conversion order 和 candidate deployment audit `PASS`。
raw FP32 candidate 只能用于诊断。

### 7.4 Opponent deployment

candidate conversion 不能影响 opponent。每次 Frozen evaluation 必须独立 materialize
完整 Policy-0809 opponent，并记录完整 component audit。candidate 与 opponent 都源自
0809 不构成共享两者 module/tensor 的理由。

## 8. 性能优化合同

性能优化的固定优先级：

```text
policy identity correctness
  -> per-lane exact-deck correctness
  -> CPU/CUDA lockstep parity
  -> reproducibility
  -> batching/cache/routing optimization
  -> throughput
```

### 8.1 目标架构

0042 的长期 CUDA 路径应实现：

```text
256 heterogeneous lanes
  -> all opponent lanes bind Policy-0809
  -> gather per-lane exact-deck static fields
  -> one shared immutable Policy-0809 batched forward
  -> scatter actions to original lanes

focal lanes
  -> independent 0042 candidate materialization and forward
```

### 8.2 性能证据

优化前后必须在相同 hardware、same seeds、same schedule 下报告：

- games/s；
- strategic decisions/s；
- mean/max inference batch size；
- GPU utilization；
- peak allocated/reserved bytes；
- deck-static cache hit/miss；
- unique exact decks per batch；
- lane routing audit failures；
- first-divergence parity result。

不允许用错误的首-lane static-field 路径作为性能 baseline。正确性 fallback 分组路径可
作为语义 reference，优化目标必须与该 reference 对固定 state/legal options 产生完全
一致的 greedy action，并对 stochastic 模式保持相同分布合同。

## 9. 必须修改的 0042 边界

正式启动前至少完成以下迁移：

1. `training/run_full_semantic.py`
   - 只保留一个 opponent policy constant：`Policy-0809`；
   - training、periodic eval、formal Frozen 均使用该 ID；
   - config、snapshot、W&B tags 和 report manifest 不再声明 Frozen-0806；
   - 使用新的 Frozen-0809 contract ID 与 schedule hash。
2. `rollout/cuda_collector.py`
   - 删除 mixed batch 读取 `jobs[0].opponent_deck` 的语义；
   - 实现 per-lane static routing，或在优化完成前 fail-closed/group-by-deck；
   - 输出 lane-level policy/deck routing audit。
3. `evaluation/frozen_jobs.py`
   - policy ID 改为 Policy-0809；
   - seed derivation 包含完整 focal/opponent identity；
   - 生成并 pin 新 schedule hash。
4. 所有 Frozen evaluator
   - 统一调用同一个 0042 candidate materializer、opponent resolver 和 hard gate；
   - 删除 raw FP32、旧 512、fixed/balanced-seat 或无 deployment audit 的旁路；
   - 旧入口若保留，只能显式标记 diagnostic/legacy 并禁止生成正式报告。
5. policy/deck catalog
   - 使用 0042-neutral pool ID；
   - runtime/config/report 不得借历史编号暗示 opponent weights identity；
   - 每套 deck 继续保留 exact 60-card hash 和 immutable provenance。
6. DESIGN
   - 同步更新 `experiments/0042_full_model_design/DESIGN.md` 与 `DESIGN.html`；
   - 明确 Policy-0809-only opponent、per-lane exact-deck routing、独立 focal/opponent、
     Frozen-0809 contract、FP16/FP32 candidate gate 和当前阶段；
   - 实现未通过前，状态不得写“implementation complete”。

## 10. Mandatory Regression Tests

正式训练前必须至少有以下自动测试：

1. mixed batch 包含至少两套不同 opponent decks；每条 lane 注入自己的 static fields。
2. 故意把 lane B 的 static fields 换成 lane A，必须 hard fail。
3. 55-deck/256-lane schedule 的 per-lane deck hash 与 engine deck 完全一致。
4. grouped-by-deck reference 与 heterogeneous per-lane batched path 在固定输入上 logits、
   masks、greedy action 完全一致。
5. batch permutation 后再 inverse-scatter，输出与原顺序完全一致。
6. 所有 opponent lanes 的 requested/materialized policy ID 都是 Policy-0809。
7. 任一 opponent component 替换为 focal component，必须 hard fail。
8. focal optimizer step 前后，opponent effective-policy hash 与参数 storage 均不改变。
9. focal/opponent 不共享 Parameter object 或 tensor storage。
10. training rollout、periodic eval、CPU Frozen 和 CUDA Frozen 对 Policy-0809 使用同一
    resolver，并得到同一 effective-policy hash。
11. periodic/CPU/CUDA candidate 均要求 FP16 storage、FP32 runtime 和 deployment audit
    `PASS`；raw FP32 必须被正式 gate 拒绝。
12. context 41 的 seeded toss winner、Agent choice 和 actual seat 完整记录。
13. CPU single-lane 与 CUDA single-lane 对相同 policy/deck/seed 的 first divergence
    lockstep parity 通过。
14. 256-lane mixed-deck CUDA 与正确性 reference 的终局结果和逐决策动作一致。
15. repository scan 确认 0042 active config/runtime/report 中没有 Policy-0806 或
    Frozen-0806 依赖；仅允许迁移记录和明确标记的历史说明出现该字符串。

## 11. 正式启动 Hard Gate

只有以下条件全部满足，0042 才可启动新的正式 PPO version：

```text
[ ] full Policy-0809 opponent identity audit PASS
[ ] focal/opponent independent materialization audit PASS
[ ] 55-deck per-lane exact-deck routing audit PASS
[ ] mixed-deck CUDA vs grouped reference parity PASS
[ ] CPU/CUDA single-lane lockstep parity PASS
[ ] PPO rollout FP32 contract PASS
[ ] periodic Frozen candidate FP16-storage/FP32-runtime gate PASS
[ ] new Frozen-0809 schedule/contract hashes pinned
[ ] CPU-256 and CUDA-2048 report schemas contain both identity audits
[ ] 0 error / 0 unfinished / 0 semantic fallback acceptance gate implemented
[ ] stale evaluator bypasses removed or fail-closed
[ ] DESIGN.md and DESIGN.html synchronized
[ ] focused and full 0042 tests pass
[ ] smoke PPO uses at least two opponent exact decks and proves lane routing
```

任一项未完成，0042 状态必须为 `BLOCKED_PRETRAINING_CONTRACT` 或等价 fail-closed
状态，不得通过 warning、临时 flag 或性能理由绕过。

## 12. 报告要求

每个正式 run 和 Frozen report 必须明确区分：

- focal candidate policy ID、source update 与 deployment-effective hash；
- opponent requested policy ID `Policy-0809` 与 effective-policy hash；
- exact-deck pool snapshot/hash；
- per-lane routing audit；
- sampled rollout 指标与 frozen greedy strength；
- CPU-256 与 CUDA-2048；
- candidate FP16-storage/FP32-runtime identity；
- human promotion decision（若无则明确 `NONE`）。

不得把 sampled rollout 胜率称为 checkpoint greedy 强度，不得把 CPU/CUDA 或不同
schedule 的结果合并，不得因为 candidate 和 opponent 都源自 0809 而将两者写成
`same_policy`。

---

**一句话验收标准：**0042 必须用独立 focal candidate 对战一份完整、immutable、共享于
所有 opponent lanes 的 Policy-0809；共享的是经审计的 opponent weights 与 batched
compute，每条 lane 的 exact-deck static semantics 仍然独立且正确，任何 Frozen 强度
结果都必须来自 FP16-storage/FP32-runtime candidate 和新的 Frozen-0809 合同。
