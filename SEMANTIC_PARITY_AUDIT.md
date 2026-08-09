# 0038 Release-Blocking Semantic Parity Audit

日期：2026-08-09

诊断候选：0038 V10 U230

checkpoint：`update-000230.pt`
checkpoint SHA-256：`a70b41d8b979a72ed3d79d8e739def40345efaf99dbf0ac49c9a08b81f9f8ab8`

## Release 结论（按 2026-08-09 条件转移 / 分布等价合同）

```text
deterministic runtime parity:          PASS in audited fixtures
explicit-random conditional parity:    PASS in audited representative rules
independent RNG statistics:            PASS in primitive/audited-effect scope
end-to-end distribution equivalence:   INCOMPLETE
U230 submission-ready:                 NO
U230 continuation training allowed:    NO
new RL started:                        NO
checkpoint weights changed:            NO
official engine source changed:         NO
```

本报告原先使用 A–D 表示规则、macro、snapshot 和 lockstep；后续验收合同已扩展为
canonical state、固定 snapshot、确定性转移、显式随机结果、RNG 分布、Action Boundary 和
端到端分布七道 Gate。旧 A–D 证据仍有效，但不能自动令新的 A–G 全部通过。

修复后的 CUDA、0038 Action Boundary 和 Kaggle-compatible package runtime 已在确定性 fixture
范围内实现一致的 observation、legal option、mask、greedy 策略意图和 primitive execution。
显式 random outcome 已在 shuffle/Prize、Numbing Water、Double Hit 与 Astonish 的真实规则路径中
闭合；CPU official、host POD 与 CUDA kernel 的完整 post-state、continuation、legal options、history
和 RNG 均一致。该结论是代表性 effect coverage，不是全卡池逐张穷举。端到端独立 seed 数据仍缺
CPU Prize 与双方 action-boundary/action-type/fallback 逐局字段，且胜率差区间尚未完全落入预注册
`±3pp` 等价带。因此当前必须保持 `submission-ready=NO`，不能由接近的胜率覆盖这些缺口。

但 U230 的权重是在修复前、缺失 option relation/history/resource knowledge 的 CUDA feature 分布上通过 PPO 得到的。运行时修好不能倒推这些权重的训练数据分布有效，因此 U230 只保留为诊断 checkpoint，不能提交或继续训练。下一步必须先从原始 Zero-Shot/Pretrain 权重建立修复后 U0 parity，再决定是否运行 5～10 updates canary。

机器可读 release manifest：

- `.tmp/evaluation/0038_semantic_parity_audit/final/u230_submission_manifest.json`
- 该旧 manifest 中的 A–D 均为 `PASS`；在新的 A–G 合同下它不是完整 release attestation
- blockers：旧训练 feature 分布不兼容/未 attested。runtime 修复已固化为 commit `bf37545`；后续 evaluation harness 修正见 `e65204a`

## 新 Gate 总表

| Gate | 验收对象 | 当前证据 | 状态 |
|---|---|---|---|
| A | backend-independent canonical authority state | versioned fail-closed schema；Gate A POD exact state comparison；排除字段有穷列举 | **PASS（已覆盖字段/fixtures）** |
| B | fixed snapshot feature/model parity | 283 snapshots；离散 tensor 0 mismatch；143 focal greedy 0 divergence | **PASS** |
| C | deterministic primitive transition | 3 case / 697 decisions 0 mismatch；Zoroark decision 129 当前 213 steps PASS | **PASS** |
| D | explicit random outcome conditional transition | 6 cases：shuffle/Prize、Numbing Water head/tail、Double Hit、Astonish；完整 continuation/state exact | **PASS（代表性规则范围）** |
| E | independent RNG distribution | batch 1/8/256/512，各 backend 每档 1,048,576 样本；基础 coin/position/opening/Prize/target 达到 margin；Gate D 代表 effect composition 通过 | **PASS（已审计 primitive/effect 范围）** |
| F | Action Boundary regression | forced/observe-only、Phantom `n=1..8`、stable serial、one Policy/Value/PPO contract | **PASS** |
| G | independent-seed end-to-end distribution | official CPU-2048 vs disjoint-seed CUDA-2048；分层/seed/error/局长通过；win CI 未落入 ±3pp 且缺部分过程字段 | **INCOMPLETE** |
| H | suite sensitivity | old runtime artifact decision 129 FAIL、current PASS；option_effect/continuation perturb FAIL→restore PASS | **PASS，带 provenance 限制** |

## 1. 运行路径

### CUDA rollout / Frozen

```text
training/run_full_semantic.py
  → rollout/cuda_collector.py
  → semantic0031_resident.py
      → CUDA official-compatible primitive engine
      → encode_semantic0031_v2_lanes
      → CudaActionBoundaryAdapter.pre_route
          forced observe-only / pending macro callback
      → shared actor + Value
      → CudaActionBoundaryAdapter.post_route
          one canonical allocation + pending transaction
      → one official primitive select per callback
```

### Kaggle-compatible package

```text
export_full_semantic_candidate.py
  → strict reconstruct actor / merged Q,V LoRA
  → strict copy Action Decoder / Allocation / Value / existing auxiliary heads
  → self-contained compiler, prototype cache and compound runtime

package/main.py
  → one module-global POLICY per game worker
      select=null: reset session and pending transaction
      pending callback: observe_only + stable-serial relocation
      strategic callback: DecisionGate → one encoder → root/allocation intent
```

CUDA 不执行一条改变规则 ABI 的 fused Phantom state mutation。两边都仍向各自 official-compatible engine 逐次提交 primitive `list[int]`；宏动作只合并 Agent 的策略推理和 PPO 时间边界。

### 关键实现位置

| 组件 | 路径 |
|---|---|
| DecisionGate | `train/0038_action_boundary_rl/action_boundary/decision_gate.py` |
| canonical allocation | `action_boundary/dragapult.py` |
| allocation head | `policy/allocation_head.py` |
| pending transaction | `action_boundary/macro_protocol.py` |
| CUDA adapter | `rollout/cuda_action_boundary.py` |
| official observe-only | `rollout/worker_compiler.py` |
| package compound runtime | `kaggle_runtime/compound_inference.py` |
| CUDA feature codec | `engine_cuda/src/official_engine_kernels.cu` |
| CUDA tensor binding | `engine_cuda/src/torch_binding.cpp` |
| package exporter | `export_full_semantic_candidate.py` |

Package 的 pending transaction 依赖“每局一个 worker、局内 module 保持加载”的正式 evaluator 生命周期。若宿主每个 callback 重建 Python 进程，宏事务无法保留；当前本地 official evaluator 与 Kaggle package 合同均为局内复用同一个 Agent module。

## 2. 固化的 pre-fix regression fixtures

以下失败在修复前已固化，测试会校验其 hash，不会被 post-fix 报告覆盖：

`train/0038_action_boundary_rl/tests/fixtures/semantic_parity_v1/`

- Zoroark/Munkidori seed 1 decision 129 前的 canonical state；
- 283 条固定 primitive trace；
- decision 0 CPU/CUDA field diff；
- decision 60 首个 greedy divergence；
- Value 最大差异和符号翻转样本；
- Phantom Dive `n=6..8` 协议样本；
- U230 package manifest 与 state-dict load report。

解压后的 283 trace SHA-256 为 `18c1684a3dc8158494fd9820278a85e60e351b2b274b520bb9fb413b1fa056ca`。Gate C 新增 `--reuse-trace`，只读消费该 fixture，不再重新生成或覆盖它。

## 3. Gate A：规则执行 parity

Gate A 完全绕过模型、DecisionGate 和宏动作，向 official CPU/reference/CUDA 提交相同 primitive script。

### 根因与修复

decision 129 的原始失败来自未带源码 provenance 的旧 CUDA paired executable，被 `--skip-build` 静默复用；它不是当前源码可复现的 official-vs-CUDA rule divergence。现在缓存复用必须同时校验：

- official source tree；
- CUDA include/extractor/kernel source；
- paired replay source；
- rule blob；
- container/compile contract；
- executable hash。

任一变化或 manifest 缺失都会 fail closed，要求重建。

最新源码在本机直接编译后结果：

| case | seeds | decisions | state mismatch | status mismatch | outcome mismatch |
|---|---:|---:|---:|---:|---:|
| Dragapult vs Dragapult/Dusknoir | 1 | 283 | 0 | 0 | 0 |
| Dragapult vs Area Zero/Kangaskhan | 1 | 201 | 0 | 0 | 0 |
| Dragapult vs Zoroark/Munkidori | 1 | 213 | 0 | 0 | 0 |
| **合计** | **3** | **697** | **0** | **0** | **0** |

更早的同轮 post-fix coverage 还包含 Dusknoir 4 seeds/1,033 decisions、Area Zero 20 seeds/4,199 decisions、Zoroark 1 seed/213 decisions，均为 0 mismatch；最后一次 697-decision 结果是在 Zoroark duplicate-option 修复后用最新源码重编得到的 release evidence。

证据：`.tmp/evaluation/0038_semantic_parity_audit/gate_a_final/release_summary.json`。

## 4. Gate B：Phantom Dive macro parity

Canonical allocation 使用 stable `(player, serial, card_id, initial_bench_slot)` target identity；count vector 总和为 6，不编码点击顺序。

| `n` | allocation 数 |
|---:|---:|
| 1 | 1 |
| 2 | 7 |
| 3 | 28 |
| 4 | 84 |
| 5 | 210 |
| 6 | 462 |
| 7 | 924 |
| 8 | 1,716 |

修复内容：

- macro domain 从 `n≤5` 扩展至 `n≤8`，候选再多也复用一次 State Encoder；
- official type-3 target callback 按 `option_source → stable serial` 重定位，不再误读恒为零的 `option_target`；
- target universe、actor、context、remaining、身份和 mask 漂移一律 `MacroProtocolError`；
- package/official/CUDA 漂移均禁止在内部 callback 重新调用 Policy；
- shared public Prize mapping 用于 allocation feature，不读取隐藏 Prize；
- `n=1` 仍是 forced allocation，不增加 parameter logprob/entropy；
- allocation 顺序 alias canonicalize 为同一宏动作。

真实 official Area Zero fixtures 对 `n=6,7,8` 的 3,102 个 allocation 全部比较 macro sequence、canonical legacy sequence 与 reversed-order alias，权威 public state、KO/Prize/terminal/legal successor parity failure 为 0。既有 `n=1..5` 的 330 个 allocation 也保持 0 failure。

Gate D 另验证 21 个实际 Phantom macro 均只创建一次 intent，并展开为 126 次 primitive callback；无 transaction reset 或 fallback。

证据：`.tmp/evaluation/0038_semantic_parity_audit/gate_b_final/area_zero_n1_n8.json`。

## 5. Gate C：observation → legal/mask → logits → action → Value

### 修复前

固定 trace 上 CUDA 将 `option_skill_*`/`option_effect_*` 置零，并在 event history、resource ledger、`deck_membership_known` 上继续漂移。143 个 focal snapshots 出现：

- root-logit tolerance failure 137；
- greedy action divergence 5；
- Value 最大绝对差 0.403402；
- Value 符号翻转 2。

### 根因与最小修复

1. CUDA codec 没有编码 option skill/effect prototype relations；现在由 rule tables 精确生成并导出 mask/parent/role。
2. MoveCard 的 temporary/original area 与 DeckBottom event 语义不等价；现在匹配 official public log。
3. CUDA 曾从 resident private state 重建 exact deck/Prize knowledge，造成隐藏信息泄漏；现在只维护公开可知 serial ledger，计数或公开事件不再支持 exact knowledge 时立即失效。
4. full-deck view 曾无条件推断 exact Prize；现在复现 CPU `CausalKnowledge` 的公开、非负且计数闭合推断。
5. 每行新增 versioned `feature_valid` attestation；schema/关键字段缺失不能回退为零张量。
6. official CPU 为多个 N Pokémon 产生相同 copied-attack primitive aliases，CUDA 曾去重；现在保留相同数量和顺序，等价 alias 只允许在 Agent canonical layer 合并。

### 修复后 immutable 283-snapshot

| 指标 | 结果 |
|---|---:|
| decisions replayed | 283 / 283 |
| exact tensor mismatch fields | 0 |
| CUDA fixed-action state errors | 0 |
| focal model decisions | 143 |
| CUDA vs package root top-1 divergence | 0 |
| CUDA vs package top-2-set divergence | 0 |
| CUDA vs package greedy divergence | 0 |
| CUDA/package root-logit max abs error | `6.44e-6` |
| training/CUDA Value max abs error | `1.37e-6` |
| Value sign divergence | 0 |
| training/package root greedy divergence | 0 |
| training/package root-logit max abs error | `0.003183` |
| training/package Value max abs error | `0.001199` |

FP16 storage → FP32 package runtime 的小误差没有改变 top-1/top-2 或 Value 符号。Area Zero `n=6..8` 三个真实 snapshots 也保持 root/allocation top-1/top-2 和 Value parity。

证据：

- `.tmp/evaluation/0038_semantic_parity_audit/gate_c_final/u230_fixed_283.json`
- `.tmp/evaluation/0038_semantic_parity_audit/gate_c_final/u230_area_zero_n6_n8.json`
- `.tmp/evaluation/0038_semantic_parity_audit/gate_c_final/release_summary.json`

## 6. Gate D：端到端 paired lockstep

同一个 U230、deterministic greedy、相同 seat/opponent/seed 下，official package CPU 与 CUDA runtime 在每个 callback 比较完整 feature/legal/mask、DecisionGate、chosen action、macro、primitive execution、terminal/winner；每局只用第一处分歧定位根因。

Gate D 在修复过程中依次发现并保留了以下 first-divergence：

1. CUDA adapter 未导入共享 `MacroProtocolError`；
2. Phantom callback 将 entity relation 误读为 `option_target`；
3. full-deck view 泄漏 exact Prize；
4. direct attach 后从 private state 错误重建 stale exact deck；
5. Zoroark copied attacks 被 CUDA rule layer 去重。

全部修复后的汇总：

| 指标 | 结果 |
|---|---:|
| paired games | 8 / 8 |
| callback decisions | 1,335 |
| strategic forwards | 1,153 |
| focal Value forwards | 580 |
| forced shortcuts | 56 |
| Phantom macros | 21 |
| Phantom primitive callbacks | 126 |
| first divergence | 0 |
| fallback | 0 |
| timeout | 0 |

覆盖 Dragapult/Dusknoir seeds 1–3 双座位，以及 Zoroark/Munkidori seed 1 双座位。证据：`.tmp/evaluation/0038_semantic_parity_audit/gate_d_postfix/release_summary.json`。

## 7. 模型加载与 submission manifest

隔离诊断 package：`.tmp/evaluation/0038_semantic_parity_audit/fixed_u230_package_v5/`。

- checkpoint critical missing/unexpected keys：`0 / 0`；
- actor、merged Last Option Q/V LoRA、Action Decoder、Allocation Head、Value、既有 auxiliary heads 全部有独立 component hash；
- portable state top-level inventory 精确匹配；
- 所有核心模块 strict load；
- actor/Value 均 `eval()`，dropout 关闭；
- package storage FP16、runtime FP32；
- Value 严格部署用于诊断，但不参与 `select()`；
- package action path 不存在 `strict=False` 或 broad exception → legacy action fallback；
- startup 校验 manifest/file hashes，缺失关键权重或 schema mismatch 时 fail closed。

U230 最终仍为 `release_ready=false`，因为 checkpoint metadata 没有当前
`0038_cpu_authoritative_cuda_feature_parity_v1` 的训练分布 attestation，而且已知其
rollout 来自 pre-fix CUDA features。runtime 修复是否已提交不改变这条 checkpoint
provenance 判定。

这条 provenance guard 是永久合同：以后即使 A–D 都通过，feature schema 不匹配的旧 RL checkpoint 也不能被误标为可发布。

## 8. Forced shortcut 与 PPO 时间语义

回归确认：

- 唯一 mandatory completion 才可 forced；STOP/cancel/decline/pass 存在时仍是 strategic；
- forced callback 仍消费 observation、event、公开 knowledge/history；
- forced/macro callback 不运行完整 Policy/Value，不形成 PPO row；
- Phantom 完整动作只有一个 joint old-logprob 和一条 PolicyTransition；
- forced callback 数量不改变 gamma/lambda/GAE；
- terminal/reward 在 chain 中累计到上一个真实 decision；
- GRU 仍为 request-local；shortcut 后下一 strategic observation/history/features 与逐 callback 路径一致；
- old logprob 在 PPO epochs 内冻结，forced rows 不进入 policy/entropy/KL/clip。

## 9. 修改范围

没有修改 `engine/source/`、ABI、official observation schema、checkpoint tensors、reward、学习率或模型结构。

主要修改文件：

- CUDA feature/rule adapter：`official_runtime.h`、`official_semantic_history.cuh`、`official_state_pod.cuh`、`official_core_pod.cuh`、`official_main_pod.cuh`、`official_engine_kernels.cu`、`torch_binding.cpp`；
- CUDA bridge/tools：`semantic0031_bridge.py`、paired differential runner、fixed-snapshot scaffold；
- 0038 macro/runtime：`macro_protocol.py`、`cuda_action_boundary.py`、`compound_inference.py`、package `main.py`、exporter；
- release diagnostics：`semantic_parity/inventory.py`、`gate_c_snapshot.py`、`submission_manifest.py`、Gate B/C/D runners；
- immutable fixtures 和相关 unit/property tests；
- 本报告与 `experiments/0038_action_boundary_rl/DESIGN.md/.html`。

## 10. 执行的关键命令与结果

```bash
# 最新源码 Gate A（本机 nvcc 重编，不复用旧 binary）
.tmp/evaluation/0038_semantic_parity_audit/gate_a_final/\
official_battle_ordered_matrix_cuda_paired \
  .tmp/cuda_0032_rules/official_rules.bin \
  .tmp/evaluation/0038_semantic_parity_audit/gate_a_final/cases_native.tsv \
  1 1 512 coverage-first-legal

# official Area Zero exhaustive macro parity
python3 -m train.0038_action_boundary_rl.semantic_parity.run_gate_b_area_zero \
  --output .tmp/evaluation/0038_semantic_parity_audit/gate_b_final/area_zero_n1_n8.json

# immutable 283-decision Gate C
python3 engine_cuda/tools/run_official_semantic0031_v2_parity_scaffold.py \
  --manifest .tmp/evaluation/0038_semantic_parity_audit/gate_c/semantic_trace_manifest.json \
  --package .tmp/evaluation/0038_semantic_parity_audit/fixed_u230_package_v5 \
  --training-checkpoint rl_runs/0038_action_boundary_rl/versions/\
V10_complete_512_rollout_fresh_rl/checkpoint/update-000230.pt \
  --extension-dir .tmp/engine_cuda_benchmark/build_sm120_staged \
  --trace .tmp/evaluation/0038_semantic_parity_audit/gate_c_final/\
fixed_283_primitive_trace.jsonl --reuse-trace --compare-decisions 283 \
  --require-history-wrap --strict
```

Release-focused unit/property suite：**72 tests passed**。Gate A/B/C/D 的真实执行证据不由 mock/unit test 替代。

## 11. 尚未覆盖的风险

- Gate D 的显式结果覆盖 6 条代表性 conditional rule path，不是全卡池随机效果逐张穷举。
- 混乱等 root commit 后的 chance boundary 已按 invalid/fail-closed 合同处理，但缺少真实 Phantom-confusion 长链 lockstep fixture。
- Area Zero `n=6..8` 已有真实 official exhaustive fixture和模型 snapshot，但 Gate D 的自然对局样本没有恰好命中全部 6/7/8 三种规模。
- Kaggle 宿主级 OOM/进程重启不在 package 控制内；正式 canary 仍必须计数 timeout、process restart、pending reset。
- 当前 U230 权重的训练分布污染不可修复，只能从正确 U0 重新开始。

## 12. 下一步启动门

当前 runtime parity 修复可以进入下一阶段，但不是直接发布 U230：

1. 从原始 Zero-Shot/Pretrain checkpoint 构造修复后 update-0；
2. 用同一 U0 在 CUDA 与 packaged official CPU runtime 做固定 snapshot/paired parity；
3. 生成带当前 feature/action schema attestation 的新 model-only U0 checkpoint；
4. 人工确认后最多运行 5～10 updates canary；
5. canary 重新采集完整 on-policy trajectory，不读取 U215/U230 rollout/old-logprob/optimizer；
6. canary 与 parity 通过后，才讨论正式 RL 或 Kaggle package。

因此本轮最终判定为：**semantic runtime repair PASS；U230 submission-ready NO；新 RL 尚未启动。**

## 13. Post-fix 007 CPU/CUDA 分布验证

最终完成了完整 official CPU `8×256=2,048` 面板。两边使用 exact deck 007、双方
Policy-0806、deterministic greedy、strict FP32、evaluation seed `341512806`、循环上限 20 和
engine turn 100 draw；CPU 使用 official seeded runtime，模型 forward 仍在 GPU。

| runtime | 局数 | W-L-D | 胜率 | Wilson 95% CI | first / second | lifecycle error |
|---|---:|---:|---:|---:|---:|---:|
| official CPU | 2,048 | 1149-899-0 | 56.10% | 53.94%–58.24% | 58.89% / 53.32% | 0 |
| repaired CUDA | 2,048 | 1146-902-0 | 55.96% | 53.80%–58.09% | 59.08% / 52.83% | 0 |

CPU−CUDA 仅 `+0.1465pp`。2048 个 opponent/slot/replica/seat mapping 全部通过；逐局 outcome
一致率 76.42%，CPU loss→CUDA win 为 240，CPU win→CUDA loss 为 243，配对差 95% CI
约 `[-1.96pp,+2.25pp]`。两边各有 10 次 repeat/progress-guard 裁决、0 turn-limit draw；
guard schedule 不要求相同，因为两个 backend 的随机轨迹允许分叉。

这份 shared standard panel 是强参考，但明确标记
`shared_standard_panel_reference_not_independent_gate_g`，不能冒充独立 RNG 的 Gate G。
machine-readable evidence：
`.tmp/evaluation/0038_semantic_parity_audit/standard_panel_reference/cpu2048_vs_cuda2048.json`。

CPU wall time 为 1,592.06 秒（1.286 games/s）；同标准面板 CUDA 为 105.25 秒
（19.459 games/s），端到端约 `15.13×`。新增终局诊断字段后的独立 CUDA run 为
129.71 秒（15.789 games/s），相对 CPU 约 `12.27×`。

昨天的 pre-fix CUDA-2048 为 1174-874-0（57.32%）。相同 schedule 下，修复后为
1146-902-0（55.96%）：净变化 -28 胜（-1.37pp），逐局 outcome agreement 69.14%，
且 330 个 win→loss 与 302 个 loss→win 大体对称。由于 pre-fix 输入确实缺失 option/history/
resource 语义，昨天的 CUDA **策略强度、RL 收益和 checkpoint 选择结论不能继续作为 official
部署证据**；但固定 schedule、吞吐数据、规则层 fixture 和问题定位证据仍然有用，不应笼统删除。

完整统计与路径见根目录 `0038_007_CPU_CUDA_POSTFIX_VALIDATION.md`。U230 的训练 provenance
结论不变：它仍不可提交、不可续训；本轮未启动 RL 或 Kaggle submission。

---

## 14. 条件转移与随机分布验收补充（本轮权威结论）

本节覆盖 2026-08-09 新增的 A–I 验收合同。判断目标不再是“同 seed 必须得到同一局”，而是：

```text
same canonical state + same primitive action + same explicit random outcome
→ same canonical next state

P_CPU(next_state | state, action)
≈ P_CUDA(next_state | state, action)
```

同 seed 锁步只保留为定位工具；胜率只属于 Gate G，不能掩盖 A–F 的失败。

### 14.1 Canonical authority state

新增 schema `0038_authority_state_v1`，实现位于
`train/0038_action_boundary_rl/semantic_parity/canonical_state.py`。它与 Actor 可见的
`canonical_public_observation` 分离：前者用于规则 differential，允许保存双方完整权威区域顺序；
后者继续执行隐藏信息隔离，不能把完整 deck/Prize 注入模型。

Canonical authority state 要求 game/turn/phase/seat/counters；双方 Active、Bench、hand、deck、
discard、Prize、energy、tools、pre-evolution、temporary；卡牌 owner、stable serial、zone/index、
HP、damage、status、attachments 与回合/Ability flags；Supporter/retreat/manual Energy/
once-per-turn 状态；effect/continuation stack；pending choice；ordered legal options；event history；
terminal/winner/error；reward、Prize delta 和 terminal reward。

任一 required section/field 缺失即 fail closed；未知字段不会被静默丢弃。只排除下列非规则字段：

| 排除字段 | 原因 |
|---|---|
| `backend` | provenance，单独写入 manifest |
| `capture_timestamp_ns` | wall-clock capture metadata |
| `host_pointer` / `device_pointer` | 进程/设备局部地址 |
| `struct_padding` | 无规则语义的 ABI padding |
| `debug_render_cache` | 规则不读取的派生 UI/debug cache |
| `kernel_lane` | CUDA 执行布局；游戏身份由 episode ID 表示 |

禁止把 damage、zone、serial、continuation、legal option 或 reward 字段加入 exclusion。当前真实
Gate A 的 official state→`OfficialStatePod` exact comparison 覆盖 697 个 primitive decision，
state/status/outcome mismatch 为 0；Python schema 的 fail-closed、hash、deck/Prize order 与
continuation sensitivity 测试通过。Gate A 结论限定于已导出 ABI 字段与 fixtures。

### 14.2 Fixed snapshot feature/model parity

比较顺序固定为 observation → option skill/effect relations → event/resource/deck-membership →
option order/mask → DecisionGate → representation → logits → macro → Value。离散字段 exact；
一般浮点 tensor 使用 `atol=rtol=1e-6`；FP32 logits/Value 使用 `atol=rtol=1e-5`。

283 个 immutable snapshots 的离散 tensor mismatch 为 0；143 个 focal 决策的 CUDA/package
top-1、top-2 set 和 greedy intent divergence 均为 0。CUDA 与 training FP32 root-logit max error
`6.44e-6`、Value max error `1.37e-6`。Package 因 FP16 storage→FP32 runtime root max error
`0.003183`、Value max error `0.001199`，但未改变 top-1/top-2 或 Value sign。

### 14.3 Deterministic primitive transition

固定 primitive trace 比较 pre-state hash、action、continuation、ordered legal set、event semantics、
reward delta 和 post-state。当前源码的 Dragapult/Dusknoir、Area Zero、Zoroark/Munkidori 共
697 decisions 全部 0 mismatch。

Gate H 对 decision 129 的执行身份得到一个重要 provenance 结论：

- 历史无 provenance binary `84e338…`：decision 129 精确 FAIL，`error=20/detail=403`；
- 当前 binary `5147f9…`：同 case 213 decisions PASS；
- tracked pre-fix commit `c29dd6` 干净重编 binary `6a809d…`：同样 213 decisions PASS。

因此不能写成“`c29dd6` 源码必现 Gate-A bug”。可证明的是历史部署 binary 陈旧且缺少 build
provenance；这也是后来加入 source/rules/binary manifest 与 fail-closed cache 校验的原因。
Gate-C 的 pre-fix feature failures 由 `semantic_parity_v1` 的 decision 0/60/Value fixtures 独立固化。

### 14.4 Test-only explicit random outcome injection

新增 versioned fixture：shuffle permutation、Prize indices、coin results、random target indices、
named effect results；消费 cursor 对 underflow、越界和未消费结果 fail closed。fixture header 仅被
standalone benchmark 引用，production `engine_cuda/src` 和普通 include 路径没有引用。

| explicit outcome | 真实规则路径 | 结果 |
|---|---|---|
| reverse shuffle permutation | shuffle 后继续 `Draw` | complete state exact PASS |
| Prize indices/count | `DeckToPrize` | complete state exact PASS |
| coin=head / tail | `Numbing Water` 两条 continuation | complete state exact PASS |
| coin sequence `[head,tail]` | `Double Hit`，最终 90 damage | complete state exact PASS |
| target index `2` | `Astonish` 随机弃牌目标 | complete state exact PASS |

fixture 使用隔离的 `test_only_rng_preimage` adapter：显式结果先映射为能产生该结果的初始 RNG
state，随后 CPU official、host POD 和真实 CUDA kernel 都执行未修改的原始 attack/effect/
continuation。比较包含 RNG、ordered legal options、continuation 和 history。测试 header 未被
production translation unit include，生产热路径没有新增 provider 或分支。

artifact：`.tmp/evaluation/0038_semantic_parity_audit/gate_d_random/explicit_outcomes_v2.json`。
6 个 conditional cases 全部通过，Gate D 在上述代表性规则范围内为 PASS；这不是所有随机卡的
逐张 exhaustive，也不是 production runtime 的通用 mid-chain 注入接口。

### 14.5 Statistical RNG equivalence

预注册 margin：binary/small-cardinality `±0.01`；60-card position TV upper-95 `≤0.02`；
lag-1 correlation 使用 99% envelope。CPU/CUDA 使用互不重叠 seed range；逻辑 batch
`1/8/256/512` 每个 backend、每档各 `1,048,576` 样本。

| batch | coin diff 95% CI | card-position TV / upper95 | target TV / upper95 | 99% lag envelope | 结果 |
|---:|---:|---:|---:|---:|---|
| 1 | `[-0.001873, 0.000834]` | `0.003832 / 0.018360` | `0.001798 / 0.007103` | `±0.002515` | PASS |
| 8 | `[-0.002155, 0.000551]` | `0.003871 / 0.018400` | `0.001275 / 0.006580` | `±0.002515` | PASS |
| 256 | `[-0.002359, 0.000348]` | `0.004734 / 0.019263` | `0.002144 / 0.007449` | `±0.002515` | PASS |
| 512 | `[-0.000689, 0.002018]` | `0.003930 / 0.018459` | `0.001544 / 0.006849` | `±0.002515` | PASS |

opening-hand inclusion、Prize inclusion、相邻 fingerprint duplication 和 batch-1 对
8/256/512 的分布稳定性使用相同 margin，六个 within-backend batch-stability comparisons 全部
PASS。结合 Gate D 的条件规则一致性，这证明基础 RNG primitive 与已审计 effect composition 的
分布合同成立；Gate E 在当前 primitive/代表 effect 范围内 PASS。未覆盖卡牌仍列为 coverage risk，
而不是声称全卡池数学穷举。

### 14.6 Action Boundary regression

- forced mandatory choice：0 Policy、0 Value、0 PPO row；observation/history/event 正常消费；
- `n=1..8` Phantom allocation 数量为 `1/7/28/84/210/462/924/1716`，始终一个 macro intent；
- stable serial 每 callback 重定位；target universe/actor/context/remaining 漂移即 fail closed；
- macro drift 不触发第二次 Policy；
- root + allocation joint old-logprob 与 PPO replay 定义一致；内部 callbacks 不额外折扣；
- official protocol 仍逐次返回原 ABI `list[int]`。

CUDA 当前没有绕过 official primitive semantics 的 fused Phantom state mutation；两边都逐 primitive
执行。当前 fused-vs-unfolding 因而退化为同一 macro 的两套 primitive executor parity，已有
`n=1..8` exhaustive/official fixtures 通过。

### 14.7 End-to-end distribution gate

正式独立样本比较使用 official CPU-2048（seed `341512806`）与 CUDA-2048（seed
`934151280`）。两边 2,048 个 engine seed 各自唯一、集合交集为 0；opponent×seat strata exact，
先后手各 1,024，runtime error 均为 0。

| metric | official CPU | independent CUDA | comparison |
|---|---:|---:|---:|
| win rate | 56.10% | 57.57% | CPU−CUDA `-1.4648pp`；95% CI `[-4.498,+1.569]pp` |
| first | 58.89% | 59.47% | `-0.586pp` |
| second | 53.32% | 55.66% | `-2.344pp` |
| mean full rounds | 6.861 | 6.950 | `-0.089`；95% CI `[-0.217,+0.039]` |
| runtime errors | 0 | 0 | PASS |

局长区间完全落入预注册 `±0.5` round，PASS。胜率方向没有显示十个百分点级系统偏移，但其
95% CI 未完全落入预注册 `±3pp` 等价带，因此必须记为 INCOMPLETE，不是 FAIL，也不能在看到
结果后放宽 margin。55 个 exact matchup 多数样本远低于每 backend/stratum 200，同样只作描述。

CUDA 现在零额外 forward 地序列化 resident scheduler 已经收集的 `terminal_turns`、
`engine_selections`、`terminal_prize_counts`；旧 cache 缺 schema、diagnostic arrays、schedule seed
或 model hash 会 fail closed 重跑。但 official CPU compact report 没有 final Prize differential，
Policy-0806 两侧也没有 strategic/macro/forced/action-type/fallback per-game rows。因此 Gate G
仍为 INCOMPLETE。

artifacts：

- `.tmp/evaluation/0038_semantic_parity_audit/gate_g/distribution_cpu2048_cuda2048_independent_v2.json`
- `.tmp/evaluation/0038_semantic_parity_audit/gate_g/cpu_2048_official_normalized_v2.json`
- `.tmp/evaluation/0038_semantic_parity_audit/gate_g/cuda_2048_independent_934151280_normalized_v2.json`

### 14.8 Test sensitivity

- 历史 runtime artifact 在 decision 129 FAIL；当前 runtime 同 fixture PASS；
- in-memory 修改一个 `option_effect_relations`，首个差异定位到该 stage；
- in-memory 修改 continuation opcode，首个差异定位到 continuation；
- 撤销两项扰动后恢复 PASS。

执行身份 fixture：
`train/0038_action_boundary_rl/tests/fixtures/semantic_parity_v2/gate_h_execution_identity.json`。

## 15. 本轮修改范围与命令

新增 canonical/transition/model comparators、random fixture/statistics、Gate F/G/H analyzer、
standalone CUDA RNG/outcome benchmarks、immutable fixtures 和 focused tests。未修改 official
`engine/source/`、checkpoint、reward、模型结构或正式 rollout hot path；未启动 RL，未提交 Kaggle。

关键命令：

```bash
# historical artifact: expected decision-129 FAIL；current binary: 213-step PASS
engine_cuda/build/cuda_support_audit/official_battle_ordered_matrix_cuda_paired \
  .tmp/cuda_0032_rules/official_rules.bin \
  .tmp/evaluation/0038_semantic_parity_audit/sensitivity/zoroark_seed1.tsv \
  1 1 512 coverage-first-legal

.tmp/evaluation/0038_semantic_parity_audit/gate_a_final/\
official_battle_ordered_matrix_cuda_paired \
  .tmp/cuda_0032_rules/official_rules.bin \
  .tmp/evaluation/0038_semantic_parity_audit/sensitivity/zoroark_seed1.tsv \
  1 1 512 coverage-first-legal

python3 engine_cuda/tools/run_official_rng_distribution.py \
  --samples 1048576 \
  --output .tmp/evaluation/0038_semantic_parity_audit/gate_e/rng_distribution_1048576.json

python3 engine_cuda/tools/run_official_random_outcome_injection_paired.py \
  --rules .tmp/cuda_support_audit/official_rules.bin \
  --output .tmp/evaluation/0038_semantic_parity_audit/gate_d_random/explicit_outcomes_v2.json

python3 engine_cuda/tools/evaluate_policy_0806_cuda.py \
  --deck-number 007 --evaluation-seed 934151280 \
  --output-root evaluation/arena/combat_mat/policy_0806/0806_gate_g_independent_seed_934151280 \
  --temp-root .tmp/evaluation/0038_semantic_parity_audit/gate_g/cuda_independent_934151280

python3 -m train.0038_action_boundary_rl.semantic_parity.standard_panel_reference \
  --cpu-report .tmp/evaluation/0038_007_cpu_seeded2048_conditional_parity/\
published/reports/007_dragapult_ex.html \
  --cuda-games evaluation/arena/combat_mat/policy_0806/\
0806_kaggle_top100_plus_v1_cuda_seeded_2048_postfix_bf37545/games/007.json \
  --output .tmp/evaluation/0038_semantic_parity_audit/standard_panel_reference/\
cpu2048_vs_cuda2048.json
```

Focused audit suite：72/72 PASS；`compileall` 与 `git diff --check` PASS。新增/修改集中在：

- `semantic_parity/`：canonical state、fixed snapshot、deterministic transition、Gate F/G/H、
  RNG statistics、standard-panel reference、archived inventory loader；
- `engine_cuda/benchmarks|tools|tests`：test-only random-outcome benchmark、RNG benchmark、
  terminal diagnostics serialization 和 independent evaluation seed；
- `tests/fixtures/semantic_parity_v1|v2`：hash-validated pre-fix/current regression evidence；
- `SEMANTIC_PARITY_AUDIT.md`：pre/post-fix 证据、执行命令、coverage boundary 与最终 Gate 状态。

完整 tracked/new source 清单：

| 用途 | 文件 |
|---|---|
| 报告/计划 | `SEMANTIC_PARITY_AUDIT.md`、`0038_007_CPU_CUDA_POSTFIX_VALIDATION.md`、`docs/superpowers/plans/2026-08-09-0038-conditional-transition-rng-parity.md` |
| CUDA evaluation diagnostics | `engine_cuda/tools/benchmark_0037_cuda_resident_refill.py`、`engine_cuda/tools/evaluate_policy_0806_cuda.py`、`engine_cuda/tests/test_policy_0806_cuda_evaluation.py` |
| test-only random/RNG harness | `engine_cuda/benchmarks/official_random_outcome_injection_paired.cu`、`engine_cuda/benchmarks/official_rng_distribution.cu`、`engine_cuda/include/ptcg_cuda/testing/random_outcome_fixture.cuh`、`engine_cuda/tools/run_official_random_outcome_injection_paired.py`、`engine_cuda/tools/run_official_rng_distribution.py`、`engine_cuda/tests/test_random_outcome_injection.py`、`engine_cuda/tests/test_official_rng_distribution.py` |
| parity analyzers | `semantic_parity/canonical_state.py`、`deterministic_transition.py`、`fixed_snapshot_parity.py`、`gate_f_action_boundary.py`、`gate_g_distribution.py`、`random_outcomes.py`、`sensitivity.py`、`standard_panel_reference.py`、`statistical_equivalence.py`、`inventory.py`、`gate_b_macro.py`（均位于 `train/0038_action_boundary_rl/`） |
| regression tests | `tests/test_deterministic_transition_parity.py`、`test_fixed_snapshot_feature_parity.py`、`test_gate_f_action_boundary.py`、`test_gate_g_distribution.py`、`test_random_outcome_schema.py`、`test_semantic_parity_canonical_state.py`、`test_semantic_parity_sensitivity.py`、`test_standard_panel_reference.py`、`test_statistical_equivalence.py`、`test_semantic_parity_gate_b.py`、`test_semantic_parity_inventory.py`（均位于 `train/0038_action_boundary_rl/`） |
| immutable evidence | `train/0038_action_boundary_rl/tests/fixtures/semantic_parity_v2/manifest.json`、`gate_h_execution_identity.json`、`u230_archived_macro_contract.json` |

没有修改 `engine/source/`、模型权重、checkpoint、reward、PPO/action contract 或 observation schema。

## 16. 未覆盖风险与最终判定

1. Gate D 覆盖代表性的 coin/random-target/effect 路径，不是全卡池随机效果逐张穷举；
2. Gate G 缺 CPU final Prize，以及 strategic/macro/forced/action-type/fallback per-game fields；
3. independent CPU/CUDA 2,048×2,048 的 win-rate CI 未完全落入预注册 `±3pp`，不能 post-hoc 放宽；
4. 55 个 matchup 在当前样本下无法逐项达到 `±3pp` equivalence 的统计功效；
5. 规则 fixtures 重点覆盖当前主牌表/常见对手，不是全卡池 effect 穷举；
6. U215/U230 权重来自 pre-fix CUDA feature distribution，运行时 parity 不会修复其训练 provenance。

```text
submission-ready = NO
new RL allowed    = NO
```

只有 Gate A–G 全部 PASS，且从正确 Zero-Shot/Pretrain U0 重新采集 on-policy 数据后，才可解除。
