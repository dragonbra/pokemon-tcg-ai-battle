# 0038 Release-Blocking Semantic Parity Audit

日期：2026-08-09

诊断候选：0038 V10 U230

checkpoint：`update-000230.pt`
checkpoint SHA-256：`a70b41d8b979a72ed3d79d8e739def40345efaf99dbf0ac49c9a08b81f9f8ab8`

## 结论

```text
Gate A/B/C/D runtime semantic parity: PASS
U230 submission-ready:                NO
U230 continuation training allowed:   NO
new RL started:                       NO
checkpoint weights changed:           NO
official engine source changed:        NO
```

修复后的 CUDA、0038 Action Boundary 和 Kaggle-compatible package runtime 已在本轮 Gate A→B→C→D 证据范围内实现一致的公开 observation、legal option、mask、greedy 策略意图和 primitive execution。

但 U230 的权重是在修复前、缺失 option relation/history/resource knowledge 的 CUDA feature 分布上通过 PPO 得到的。运行时修好不能倒推这些权重的训练数据分布有效，因此 U230 只保留为诊断 checkpoint，不能提交或继续训练。下一步必须先从原始 Zero-Shot/Pretrain 权重建立修复后 U0 parity，再决定是否运行 5～10 updates canary。

机器可读 release manifest：

- `.tmp/evaluation/0038_semantic_parity_audit/final/u230_submission_manifest.json`
- Gates 均为 `PASS`；`release_ready=false`
- blockers：旧训练 feature 分布不兼容/未 attested。runtime 修复已固化为 commit `bf37545`；后续 evaluation harness 修正见 `e65204a`

## Gate 总表

| Gate | 修复前 | 修复后证据 | 状态 |
|---|---|---|---|
| A：纯规则 differential | Zoroark/Munkidori seed 1 decision 129 的旧缓存 binary 返回 `error=20/detail=403` | 最新源码重新编译，3 case、697 primitive decisions，state/status/outcome mismatch 均为 0 | **PASS** |
| B：Phantom macro | 旧归档仅支持 `n≤5`，错误时可能 fresh Policy fallback | official Area Zero `n=6..8` 共 3,102 allocations；既有 `n=1..5` 共 330；canonical/alias/final public state failure 为 0 | **PASS** |
| C：固定 snapshot 模型推理 | option relation 被置零，history/resource/deck knowledge 漂移；5 次 greedy divergence | immutable 283 decisions 全量 tensor parity；143 focal decisions top-1/top-2/greedy 全一致 | **PASS** |
| D：端到端 lockstep | 首轮先后暴露 macro exception、target relation、Prize/deck knowledge、Zoroark duplicate option 问题 | 8 paired games、1,335 decisions、21 macros/126 callbacks，0 divergence/fallback/timeout | **PASS** |

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

Release-focused unit/property suite：**66 tests passed**。Gate A/B/C/D 的真实执行证据不由 mock/unit test 替代。

## 11. 尚未覆盖的风险

- Gate D 是 8 局锁步诊断，不是 2,048 局强度评估；它足以定位当前已知 deterministic semantic divergence，但不能穷尽全部卡牌效果。
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

根据人工调整后的快速验收预算，正式 CPU-2048 改为一个完整 CPU-256 频率单元；CUDA
仍运行八个固定 256 shard，共 2,048 局。两边都使用 exact deck 007、双方
Policy-0806、deterministic greedy、FP32、evaluation seed `341512806`、循环上限 20 和
engine turn 100 draw。CPU 使用 official seeded runtime；模型 forward 仍在 GPU。

| runtime | 局数 | W-L-D | 胜率 | Wilson 95% CI | error / unfinished | wall time |
|---|---:|---:|---:|---:|---:|---:|
| official CPU | 256 | 152-104-0 | 59.38% | 53.26%–65.21% | 0 / 0 | 210.01s |
| repaired CUDA | 2,048 | 1146-902-0 | 55.96% | 53.80%–58.09% | 0 / 0 | 105.25s |

CPU 与 CUDA 点估计相差 3.42pp，但 CPU-256 的区间很宽；两样本 pooled z=`1.04`、
双侧 `p=0.299`。CPU 结果也落在 CUDA 八个 shard 的 51.17%–62.11% 范围内。按 CUDA
逐 matchup 胜率计算，CPU 预期胜场为 143.25，实际 152，标准化差 `z=1.17`、
`p=0.241`。没有观察到此前担心的十个百分点级系统性偏移。

额外把 CPU-256 的相同 engine/search seed、opponent 和 seat 原样交给 CUDA 重放：

- 两边总体均为 `152-104-0`；
- 256 个 pairing key 全匹配；
- 逐局 outcome 一致 196/256（76.56%）；
- CPU win→CUDA loss 30 局，CPU loss→CUDA win 30 局，净偏差为 0。

这证明相同 seed 在两个独立 engine backend 上不要求逐局轨迹完全相同，但本次样本没有
方向性 outcome 偏差。结合 Gate A–D 的逐状态、逐字段和逐决策证据，当前在已覆盖合同内
可判定 semantic runtime parity 通过；仍不能把有限 fixture 表述成对所有卡牌/状态的数学穷尽证明。

昨天的 pre-fix CUDA-2048 为 1174-874-0（57.32%）。相同 schedule 下，修复后为
1146-902-0（55.96%）：净变化 -28 胜（-1.37pp），逐局 outcome agreement 69.14%，
且 330 个 win→loss 与 302 个 loss→win 大体对称。由于 pre-fix 输入确实缺失 option/history/
resource 语义，昨天的 CUDA **策略强度、RL 收益和 checkpoint 选择结论不能继续作为 official
部署证据**；但固定 schedule、吞吐数据、规则层 fixture 和问题定位证据仍然有用，不应笼统删除。

完整统计与路径见根目录 `0038_007_CPU_CUDA_POSTFIX_VALIDATION.md`。U230 的训练 provenance
结论不变：它仍不可提交、不可续训；本轮未启动 RL 或 Kaggle submission。
