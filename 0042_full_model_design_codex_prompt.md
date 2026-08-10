# 0042 Full Model Design — Strategy-Conditioned RL Architecture
# Implementation Task + Architecture Documentation

我们现在正式开始新的 numbered RL project：

> **0042 — Full Model Design**

本轮不是调研任务，而是正式实现任务。

请基于现有 repo 的真实代码完成 0042 模型结构落地，并为后续其他 Agent / engineer 编写一份完整的模型架构说明文档。

---

# 0. 总体目标

0042 是我们当前计划中的“完整 RL 模型结构”。

它的核心目标不是继续扩大底层 representation 的可训练范围，而是：

> **尽量保留预训练模型已经学到的 state / card / option semantics，
> 把 RL 的主要自由度放在 Value-side strategic representation 与 Policy-side strategic readout 上。**

我们希望形成一个非常明确的职责划分：

```text
Pretraining:
    learn what the state/cards/options mean

RL:
    learn how to strategically value the state
    and how to strategically choose among already-understood actions
```

因此 0042 的核心原则是：

1. Prototype / State / Option representation 保持冻结；
2. **0040 使用过的 Option Q/V LoRA 在 0042 中彻底移除；**
3. Value Network 继续训练；
4. Action Decoder 继续训练；
5. 新增 Value Adapter；
6. 新增 Policy Strategy Adapter；
7. Policy 可以读取 Value/Meta strategic information；
8. Value → Policy 通过严格 stop-gradient；
9. Own-deck strategy 使用独立的 own-archetype embedding；
10. 新模块采用 zero-gated residual，使初始行为退化为明确的 0042 Base；
11. 不改变 PPO、GAE、reward 等既有 RL 训练语义，除本设计明确要求的 Meta-retention auxiliary 外。

---

# 1. Project isolation

请遵守 repo 现有 numbered-project isolation 规则。

创建独立的 0042 project，不要原地修改：

- 0031
- 0040
- 其他已有正式 experiment

0040 可以作为当前 RL pipeline 的实现参考，但 0042 必须拥有自己的正式代码路径 / preset / tests / checkpoint schema。

请先检查 repo 中实际命名规范。

如果尚不存在既定 0042 目录，使用与现有 numbered project 一致的命名方式创建，例如：

```text
train/0042_.../
experiments/0042_.../
```

不要为了复用方便而让 0042 的正式训练依赖一个会继续变化的旧 experiment implementation。

---

# 2. 必须参考的现有结构事实

此前已经完成 Strategy Adapter architecture audit。

真实模型不是简单的：

```text
encoder
  ├─ Value(h_V)
  └─ Policy(h_pi)
```

当前 Value Network 使用 8 个 latent queries：

```text
queries [B, 8, 320]
```

其中：

```text
query 0 -> Value
query 1 -> pretrained 15-class opponent Meta Archetype
query 2 -> final Prize difference
...
```

因此在 0042 中正式定义：

```text
z_V = queries[:, 0]   # Value latent
z_M = queries[:, 1]   # Meta / opponent strategic latent
```

不要再把二者笼统混称为一个 `h_V`。

---

# 3. 0042 的正式模型结构

目标总体结构：

```text
public observation
+ registered own deck
        |
        v
PrototypeEncoder
        |
        | FROZEN
        v
StateEncoder --------------------------+
        |                               |
        | FROZEN                        |
        v                               |
OptionEncoder                           |
        |                               |
        | FROZEN                        |
        +-----------------------+       |
                                |       |
                                v       v
                           Value Network
                           latent queries
                                |
                       +--------+--------+
                       |                 |
                       v                 v
                    q0 = z_V         q1 = z_M
                       |                 |
                       |                 +----> frozen pretrained
                       |                       15-class MetaHead
                       |                            |
                       |                            v
                       |                       meta_logits
                       |                       meta_probs
                       |
                Value Adapter
                       |
                       v
                 z_V_adapted
                       |
                       v
                   ValueHead
                       |
                       v
                       V
```

Policy side：

```text
state.summary
      |
      v
ActionDecoder
recurrent hidden h_t
      |
      | + detached strategic context
      v
Policy Strategy Adapter
      |
      v
readout hidden h_readout
      |
      +----> Query projection ----+
      |                           |
      +----> STOP projection      |
                                  v
                         frozen option keys/bias
                                  |
                                  v
                              logits
```

非常重要：

> `h_readout` 只用于当前 action scoring。

GRU recurrent state 继续使用原来的 `h_t`。

---

# 4. 彻底移除 Option LoRA

这是 0042 的硬性架构决策。

0040 中存在：

```text
OptionEncoder
+ last-option self/cross attention Q/V LoRA
```

并且 LoRA 同时收到：

```text
policy loss
value loss
meta loss
```

这使 Value 与 Policy 在底层 option representation 上存在隐式共享更新。

0042 不再采用这一结构。

## 0042 必须满足

```text
PrototypeEncoder     frozen
StateEncoder         frozen
OptionEncoder        frozen
Option LoRA          NOT INSTALLED
```

正式 0042 runtime 中，最好根本不要安装 Q/V LoRA wrapper。

至少必须满足：

```python
trainable_parameter_count(OptionEncoder) == 0
```

并且 optimizer 中：

```text
NO last_option_qv_lora group
NO option_lora group
NO hidden LoRA parameters
```

请增加 hard regression assertion：

> 0042 formal preset 中 OptionEncoder 不允许存在任何 trainable parameter。

如果未来重新研究 Option adaptation，应当作为独立 ablation / 后续 experiment，而不是 0042 canonical model 的隐藏开关。

不要让旧的 Option LoRA checkpoint 被静默加载进 0042。

---

# 5. Value-side canonical design

## 5.1 Value latent

从 Value Network final latent queries 得到：

```python
z_v = queries[:, 0]    # [B, 320]
z_m = queries[:, 1]    # [B, 320]
```

两者语义必须明确区分。

## 5.2 Meta Head

继续使用预训练 Value Network 已有的：

```text
15-class pretrained Meta Archetype Head
```

输入：

```python
meta_logits = pretrained_meta_head(z_m)
```

shape：

```text
[B, 15]
```

并：

```python
meta_probs = softmax(meta_logits, dim=-1)
```

不要使用此前 dormant 的 23-class `OpponentMetaHead(state.summary)`。

不要新建一套随机初始化的 opponent classifier 来替代 pretrained 15-class head。

0042 的 Meta signal 必须来自：

```text
query 1
    ->
pretrained 15-class MetaHead
```

---

# 6. Own Archetype conditioning

0042 新增一个显式的 own-deck strategic identity。

这里不是 exact-deck embedding。

模型本身已经通过 resource/state features 知道自己的精确 deck composition。

Strategy Adapter 需要的是更高层的：

> “我正在操作哪一种 strategy/archetype？”

因此定义：

```text
OwnArchetypeId
```

初始 taxonomy 使用：

```text
14 个主流 archetype
+ Others
= 15 类
```

它可以与当前 opponent Meta Archetype 使用相同的 archetype 名称集合，但：

> **OwnArchetypeId 与 OpponentArchetypeId 必须是两个不同的 semantic type。**

不要让代码认为：

```text
OwnArchetypeId == OpponentMetaLabel
```

只是目前 vocabulary 名称恰好可以相同。

需要两个独立 embedding：

```python
E_V_own
E_pi_own
```

推荐：

```text
num_embeddings = 15
embedding_dim = 16
```

因此：

```text
E_V_own   : [B,16]
E_pi_own  : [B,16]
```

这两个 embedding：

- 参数不能共享；
- Parameter object ID 必须不同；
- optimizer group 可以不同；
- checkpoint 中分别保存。

增加 regression test：

```python
assert parameter_ids(E_V_own).isdisjoint(parameter_ids(E_pi_own))
```

Unknown / unsupported own deck 应映射到 `Others`。

Own-archetype 的解析可以依赖注册的 own exact deck，因为这是 acting player 合法已知信息。

但不要把 opponent hidden deck 信息引入 actor tensors。

---

# 7. Value Residual Adapter

Value Adapter 只作用于：

```text
z_V = query 0
```

不要作用于 q1。

Canonical definition：

```python
z_v_norm = value_adapter_norm(z_v)

deck_v = E_V_own(own_archetype_id)

adapter_input_v = concat(
    z_v_norm,        # 320
    deck_v           # 16
)

delta_v = value_adapter_mlp(adapter_input_v)

z_v_adapted = (
    z_v
    + tanh(g_v) * delta_v
)
```

推荐 shape：

```text
input = 336

MLP_V:
    Linear(336, 320)
    GELU
    Linear(320, 320)
```

输出：

```text
delta_v [B,320]
```

然后：

```python
value_logit = existing_value_head(z_v_adapted)
V = 2 * sigmoid(value_logit) - 1
```

保留现有 Value output semantics。

## 7.1 LayerNorm ownership

`value_adapter_norm` 必须属于 Value Adapter 自己。

不要复用 ValueTrunk 中已有的 trainable affine LayerNorm。

推荐：

```python
nn.LayerNorm(320, elementwise_affine=True)
```

由 Value side 自己训练。

---

# 8. Zero-gated residual initialization

Value Adapter 与 Policy Adapter 都采用：

```python
h_out = h + tanh(g) * MLP(...)
```

其中：

```python
g = nn.Parameter(torch.zeros(()))
```

也就是 scalar gate。

硬性要求：

```text
gate = 0
residual MLP = normal initialization
```

禁止：

```text
gate = 0
AND
MLP output layer = zero initialized
```

因为会导致 dead adapter：

```text
grad(g) = 0
grad(MLP) = 0
```

请保留并扩展之前 audit 中已经验证过的 two-step startup regression。

First backward：
- MLP grad = 0 可以接受；
- gate grad 必须 != 0。

Gate 离开 0 后的下一次 backward：
- MLP grad 必须 != 0。

Value Adapter 和 Policy Adapter 都必须拥有这一 regression。

---

# 9. Meta latent retention

0042 中 Policy 将正式依赖：

```text
z_M
meta_probs
```

因此不能允许 RL 过程中 q1 representation 无约束漂移。

当前 pretrained MetaHead 本身保持 frozen：

```python
requires_grad(meta_head.parameters()) == False
```

但是训练时继续计算：

```python
meta_logits = meta_head(z_m)
```

注意：

> 不要因为 MetaHead frozen 就把整个 forward 放进 `torch.no_grad()`。

我们需要 Meta CE 的梯度穿过 frozen classifier 回到：

```text
z_M
Value latent trunk
```

因此：

```text
MetaHead parameters: frozen
MetaHead forward graph: retained
```

加入：

```text
L_meta_anchor = CE(meta_logits, opponent_meta_target)
```

用于保持 q1 的 pretrained Meta semantics。

总体 RL 训练语义仍保持原先 PPO / Value update 方式。

只新增这一项：

```text
Value-side auxiliary Meta retention loss
```

不要顺便修改：

- PPO
- GAE
- gamma
- reward
- clipping
- opponent sampling
- episode semantics

## 9.1 Meta target 的安全边界

Opponent Meta target 可以来自 simulator / registered opponent deck，作为：

```text
training-only label
```

但必须满足：

```text
target NEVER enters actor input
target NEVER enters strategy context
target NEVER enters exported inference package
```

它只允许用于 loss target / metrics。

保留现有 hidden-information counterfactual tests。

## 9.2 Meta coefficient

不要在底层 module 中 hardcode Meta loss coefficient。

增加显式配置项，例如：

```text
meta_anchor_coef
```

0042 preset 必须显式声明，不要使用隐藏 default。

同时增加日志：

```text
meta loss
meta accuracy
meta entropy
ValueTrunk grad norm from meta loss
ValueTrunk grad norm from critic loss
```

不要在这次 architecture implementation 中擅自设计复杂的 adaptive loss weighting。

---

# 10. Policy Strategy Context

Strategy Adapter 不读取 q0 latent。

Policy context 使用：

```text
z_M       = query 1 hidden
meta_prob = explicit 15-class opponent belief
V         = scalar strategic advantage/risk axis
side      = first / second
own deck  = own archetype embedding
```

Canonical context：

```python
context = concat(
    side_onehot,                     # 2
    meta_context_norm(z_m.detach()), # 320
    meta_probs.detach(),             # 15
    V.detach().unsqueeze(-1),        # 1
    E_pi_own(own_archetype_id)       # 16
)
```

因此在当前 320-wide model 中：

```text
context_dim = 2 + 320 + 15 + 1 + 16 = 354
```

`meta_context_norm` 必须属于 Policy Strategy Adapter 自己。

不要复用 Value-side LayerNorm。

---

# 11. first / second input

使用明确的：

```text
first_or_second one-hot
```

shape：

```text
[B,2]
```

必须从现有合法 public / match configuration 中取得。

不要通过局面 feature 猜测先后手。

不要产生新的 hidden-information dependency。

---

# 12. Policy Strategy Adapter

当前 ActionDecoder 是 autoregressive pointer-style decoder。

不存在简单的：

```text
h_pi -> Linear -> logits
```

真正的 logits 类似：

```text
Query(h_t) dot Key(option_i)
+ option_bias(option_i)

以及

Stop(h_t)
```

因此 0042 的 Strategy Adapter 必须作用于：

> **每一次 logits evaluation 时的 decoder readout hidden**

而不是 StateEncoder summary，也不是 option token。

## 12.1 Canonical Policy Adapter

当前 decoder hidden：

```python
h_t       # [B,320]
```

构造：

```python
h_norm = policy_hidden_norm(h_t)

adapter_input_pi = concat(
    h_norm,       # 320
    context       # 354
)
```

总输入：

```text
674
```

推荐：

```text
MLP_pi:

Linear(674, 320)
GELU
Linear(320, 320)
```

然后：

```python
delta_pi = policy_adapter_mlp(adapter_input_pi)

h_readout = (
    h_t
    + tanh(g_pi) * delta_pi
)
```

---

# 13. 非常重要：不要污染 recurrent state

0042 的 canonical contract 是：

```text
Strategy Adapter modifies READOUT ONLY.
```

用于 logits：

```python
query = Query(h_readout)

option_logits =
    query @ Key(option_tokens)
    + OptionBias(option_tokens)

stop_logit =
    Stop(h_readout)
```

但是 recurrent transition 仍然必须：

```python
h_next = GRU(selected_option, h_t)
```

禁止：

```python
h_next = GRU(selected_option, h_readout)
```

第一版 Strategy Adapter 不允许改变 ActionDecoder 保存 action history 的 internal recurrent dynamics。

设计职责：

```text
GRU:
    preserve pretrained / existing autoregressive action-composition behavior

Strategy Adapter:
    modify strategic preference at the current decision readout
```

请增加 regression test 明确验证这一点。

---

# 14. Strategy Context 在 compound action 内保持固定

对同一个 root decision：

```text
z_M
meta_probs
V
side
own archetype
```

应当作为固定 Strategy Context。

计算一次，然后在该 root decision 的所有 autoregressive scoring steps 中重复使用。

```text
root decision
    |
    +-> build strategy context once
    |
    +-> score step 0 using h_readout_0
    |
    +-> GRU legacy update
    |
    +-> score step 1 using h_readout_1
    |
    ...
```

不要因为 decoder 已经选择了一个 option 就重新跑一遍 Value Network 来改变 Strategy Context。

---

# 15. Option-side representation 完全不做 Strategy Adapter

0042 第一版禁止新增：

```text
Option Adapter
Option FiLM
Option LoRA
Option-side strategic gating
```

Option token 保持 frozen pretrained semantic representation。

Strategy learning 发生在：

```text
decoder query / readout side
```

所以 `Key(option_i)` 和 `OptionBias(option_i)` 继续使用 frozen option representation。

设计语义：

```text
Pretraining defines:
    where candidate actions live in semantic space.

RL learns:
    which direction the policy query should point under the current strategy.
```

---

# 16. Value → Policy gradient boundary

Policy 可以读取：

```python
z_m.detach()
meta_probs.detach()
V.detach()
```

但是 policy loss 不允许直接改变：

```text
Value latent trunk
Value Head
Meta Head
Value Adapter
E_V_own
```

Own deck embedding 必须分离：

```text
Policy:
    E_pi_own

Value:
    E_V_own
```

---

# 17. 0042 预期 Gradient Flow Matrix

由于 Option LoRA 被完全移除，0042 应能形成干净的 gradient boundary。

请写真实 backward regression，目标至少满足：

| Loss | PrototypeEncoder | StateEncoder | OptionEncoder | ValueTrunk | ValueHead | ValueAdapter / E_V | MetaHead | ActionDecoder | StrategyAdapter / E_pi |
|---|---|---|---|---|---|---|---|---|---|
| Policy | 0 | 0 | **0** | **0** | **0** | **0** | **0** | YES | YES |
| Value | 0 | 0 | **0** | YES | YES | YES | 0 | **0** | **0** |
| Meta | 0 | 0 | **0** | YES | 0 unless naturally shared path requires otherwise | 0 unless intentionally shared | frozen params=0 | **0** | **0** |

请根据真实实现报告 tensor-level nonzero gradients。

尤其要求：

```text
Policy loss -> Value side = impossible
Value loss -> Policy side = impossible
Meta loss -> Policy side = impossible
```

---

# 18. Trainable / Frozen contract

Canonical 0042：

## Frozen

```text
PrototypeEncoder
StateEncoder
OptionEncoder
pretrained 15-class MetaHead parameters
```

## Trainable Value side

```text
Value latent queries
Value latent blocks
Value final norm
Value scalar head
Value Residual Adapter
Value Adapter LayerNorm
E_V_own
g_V
existing enabled Value auxiliaries that 0042 inherits
```

## Trainable Policy side

```text
ActionDecoder
Policy Strategy Adapter
Policy Adapter LayerNorm(s)
Meta-context LayerNorm
E_pi_own
g_pi
```

不要让 OptionEncoder 进入任何 optimizer group。

---

# 19. Optimizer construction

保留现有 RL optimizer philosophy。

不要因为 0042 顺便重构整体 PPO optimizer。

必须删除：

```text
last_option_qv_lora optimizer group
```

新增显式参数组，至少能够区分：

```text
value_adapter
policy_strategy_adapter
```

如果 config framework 支持，可暴露：

```text
value_adapter_lr
policy_adapter_lr
adapter_gate_lr
own_embedding_lr
```

如果第一版不希望增加过多超参数，可让 adapter 初始 LR 继承 parent branch：

```text
Value Adapter -> Value-side LR
Policy Adapter -> ActionDecoder-side LR
```

所有 LR 必须通过 0042 preset / optimizer config 明确可审计。

---

# 20. Zero-gate Base equivalence

由于 0042 正式移除了 0040 Option LoRA：

> 不要再声称 `0042(g=0)` 与一个已经训练过 Option LoRA 的 0040 checkpoint 等价。

0042 定义自己的：

```text
0042 Base
```

其含义：

```text
pretrained actor
+ paired pretrained Value Network
+ frozen Prototype/State/Option encoders
+ no Option LoRA
+ existing ActionDecoder
+ zero Value Adapter gate
+ zero Policy Adapter gate
```

当：

```text
g_V = 0
g_pi = 0
```

要求：

```text
policy behavior == corresponding pre-adapter 0042 Base
Value behavior  == corresponding pre-adapter Value Base
Meta logits     == pretrained Meta output
```

---

# 21. Legacy-equivalence regression

使用真实 checkpoint 和固定 fixture。

CPU FP32 canonical test：

```python
torch.testing.assert_close(
    new,
    base,
    rtol=0,
    atol=0,
)
```

至少比较：

```text
root logits
bounded V
15-class Meta logits
legal masks
greedy action
```

必要条件：

```text
eval mode
fixed input
finite adapter outputs
gate exactly zero
```

CUDA 如果无法保证 bitwise equality，可在先测 repeatability floor 后使用文档化的小 tolerance。

---

# 22. Checkpoint schema

0042 是新模型 schema。

必须正式保存：

```text
Value Adapter MLP
Value Adapter LN
g_V
E_V_own

Policy Strategy Adapter MLP
Policy hidden LN
Meta-context LN
g_pi
E_pi_own

own-archetype vocabulary/version/hash
adapter schema version
```

并继续保存所有现有 0042 需要的 trainable policy/value parameters。

明确记录：

```text
NO OPTION LORA
```

推荐 checkpoint metadata 包含：

```text
model_schema = 0042
base actor checkpoint identity/hash
paired Value checkpoint identity/hash
own archetype taxonomy version/hash
adapter dimensions
no_option_lora = true
```

## 22.1 Loading

允许一个明确的 legacy/base initialization path：

```text
old pretrained actor
+
pretrained Value
->
initialize new 0042 modules
->
assert g_V == 0
->
assert g_pi == 0
```

禁止 broad `strict=False`。

只允许明确列出的 0042 新参数缺失。

正式 0042 checkpoint：

```text
missing 0042 adapter key -> fatal
unexpected adapter key -> fatal
taxonomy mismatch -> fatal
```

不要静默丢弃 Strategy Adapter，也不要静默加载旧 Option LoRA。

---

# 23. All policy execution paths must share semantics

请排查并统一：

```text
training rollout sampling
PPO replay / action evaluation
greedy evaluation
compound action evaluation
candidate export / inference
```

它们必须全部通过同一个：

```text
StrategyContext
+
Strategy readout helper
```

不能出现 rollout 使用 Adapter、PPO replay 不使用，或训练使用 adapted hidden、export 使用 raw hidden 的 mismatch。

---

# 24. Required diagnostics

0042 从第一天开始记录：

## Gates

```text
tanh(g_V)
tanh(g_pi)
```

## Effective residual ratio

Value：

```text
|| tanh(g_V) * delta_V ||
--------------------------------
|| z_V || + eps
```

Policy：

```text
|| tanh(g_pi) * delta_pi ||
--------------------------------
|| h_t || + eps
```

不要只看 gate 大小。

## Meta

记录：

```text
Meta CE
overall accuracy
macro accuracy
mean entropy
raw-turn-0 accuracy
early/mid/late accuracy
```

raw-turn-0 必须单独保留。

## Value

继续记录已有 Value metrics，并建议保留：

```text
explained variance
correlation
calibration
early / mid / late slices
```

## Gradient diagnostics

至少提供：

```text
critic -> ValueTrunk grad norm
meta anchor -> ValueTrunk grad norm
policy adapter grad norm
Value adapter grad norm
gate grad
```

---

# 25. Counterfactual Strategy diagnostics

请增加一个 diagnostic script，不改变训练语义。

## 25.1 V sensitivity

同一个真实 state，保持其余输入不变。

人工把 Strategy Adapter 看到的 detached V 替换为：

```text
-1.0
-0.5
 0.0
+0.5
+1.0
```

比较：

```text
policy KL
entropy
top-1 action
top-k action ranking
```

用于回答 Policy 是否真的学会使用 Value 作为风险/局势调度信号。

## 25.2 Meta sensitivity

同一个 state 测试：

```text
real meta probabilities
uniform meta probabilities
alternative archetype one-hot
```

以及可选：

```text
zero / mask z_M context
```

观察 Policy KL。

用于理解高维 `z_M` 与显式 15-class Meta probs 中，Policy 实际依赖哪一类信息。

---

# 26. 必须增加的 tests

### Architecture

- OptionEncoder trainable params == 0
- no Option LoRA installed
- no Option LoRA optimizer group
- E_V and E_pi Parameter identities disjoint
- Value/Policy adapter norms disjoint

### Zero gate

- g_V = 0
- g_pi = 0
- base-equivalence
- first backward gate gradient nonzero
- second backward MLP gradient nonzero

### Gradient boundaries

- policy loss cannot update Value side
- Value loss cannot update Policy side
- Meta loss cannot update Policy side
- frozen encoders receive no gradients

### Policy semantics

- Strategy Adapter affects logits
- Strategy Adapter does NOT alter GRU recurrent transition
- compound action evaluation uses same fixed Strategy Context
- rollout / PPO replay / greedy use identical Strategy Adapter logic

### Information safety

- opponent Meta training target never appears in actor input
- hidden-zone counterfactual tests remain bit-identical where expected
- raw-turn-0 Meta regression remains available

### Checkpoint

- legacy/base initialization initializes only known new 0042 modules
- 0042 checkpoint strict round-trip
- missing Strategy Adapter key is fatal
- wrong own-archetype vocabulary/hash is fatal
- Option LoRA keys are not silently accepted as 0042 state

---

# 27. Non-goals

本任务不要修改：

- reward
- PPO algorithm
- PPO clip semantics
- GAE semantics
- gamma / lambda
- opponent sampling
- Champion protocol
- search
- 1-ply search
- CUDA search
- engine rules
- observation semantics
- exact own-deck resource ledger
- pretrained card/state/option encoding
- deployment precision protocol
- unrelated export behavior

不要顺便做大规模 architecture cleanup。

0042 的目的必须保持单一：

> **建立一个干净的 strategy-conditioned RL model。**

---

# 28. Documentation task

实现完成后，在：

```text
docs/rl/
```

创建：

```text
docs/rl/0042_full_model_design.md
```

这是正式交付物，不是开发日志。

目标读者：

> 一个第一次接触 0042、但需要继续训练 / debug / 修改模型的其他 Agent 或 engineer。

文档应该让读者不需要阅读本次聊天，也能理解：

1. 0031 pretrained model 是什么；
2. 0042 相比 0031 增加了哪些结构；
3. 哪些 pretrained representation 被刻意冻结；
4. 为什么 RL 不再微调 OptionEncoder；
5. Value Network 如何组织 q0/q1；
6. Strategy Adapter 如何工作；
7. Value 如何影响 Policy；
8. 为什么使用 detach；
9. Own archetype 的作用；
10. zero-gated residual 为什么存在；
11. 哪些模块 trainable；
12. checkpoint / inference 应如何理解。

---

# 29. 文档必须基于真实 0031 代码

在写文档之前，请真实检查：

```text
0031 pretrained project
0031 model implementation
0031 checkpoint/model contract
```

不要根据本 prompt 猜测 0031。

如果 0031 的真实结构与这里的简称有差异，请以 repo 为准。

文档中明确区分：

```text
0031 pretrained architecture
0042 RL additions
```

---

# 30. docs/rl/0042_full_model_design.md 推荐结构

至少包含以下章节：

1. **Overview** — 一句话解释 0042，并说明为什么。
2. **Baseline: 0031** — 基于真实代码介绍 0031 预训练结构。
3. **What 0042 changes** — 用表格比较 0031 与 0042。
4. **0042 Architecture** — 加入清晰 ASCII diagram。
5. **Value representation** — 解释 `q0 = Value latent`、`q1 = Meta latent`。
6. **Strategy Context** — 写清楚：
   ```text
   context = [
       first/second,
       LN(q1.detach()),
       MetaProb.detach(),
       V.detach(),
       E_pi(own archetype)
   ]
   ```
   并解释每一项语义。
7. **Why Value → Policy is detached** — Policy 可以读取 Value knowledge，但 policy loss 不通过这条路径重写 Value knowledge。
8. **Why Option LoRA was removed** — 说明 0040 的 LoRA 以及 0042 为什么明确取消。
9. **Readout-only Strategy Adapter** — 解释 `h_t -> h_readout -> Query/STOP`，但 GRU 仍消费 `h_t`。
10. **Zero-gated residual design** — 解释 `h' = h + tanh(g) * MLP(...)`，`g=0 initially`。
11. **Own archetype vs opponent archetype** — 明确两个 semantic type 与两套独立 embedding。
12. **Frozen / trainable map** — 一眼可见哪些参数训练、哪些冻结。
13. **Gradient flow** — 画出 Policy / Value / Meta Anchor 三条更新路径。
14. **Initialization and checkpoint contract** — 解释 0042 Base、zero-gate equivalence、strict loading。
15. **Diagnostics** — gate、effective residual ratio、Meta/Value diagnostics、counterfactual tests。
16. **Design philosophy** — 总结：
   ```text
   0031 learns semantic competence.
   0042 preserves that competence and gives RL explicit strategic control.
   ```

文档中特别说明：

> V is not treated as a hard-coded rule such as “negative V means gamble”.
> It is an input that allows RL to learn whether and how behavior should depend on the estimated game value.

以及：

> The frozen OptionEncoder defines the semantic geometry of candidate actions.
> RL changes the policy query/readout inside that fixed geometry.

---

# 31. Documentation quality requirements

`docs/rl/0042_full_model_design.md`：

- 不要写成聊天记录；
- 不要写成 TODO dump；
- 不要假定读者知道 0040 audit；
- 不要只复制代码；
- 不要堆实现细节而缺少设计原理；
- 所有结构名称与真实代码一致；
- 所有 tensor shape 与真实实现一致；
- 给出关键代码路径；
- 明确写出 architecture invariant；
- 使用 ASCII graph / tables 帮助理解；
- 适合作为后续 Agent 的 onboarding document。

---

# 32. Acceptance criteria

完成后，0042 必须满足以下硬条件。

## Architecture

```text
[ ] PrototypeEncoder frozen
[ ] StateEncoder frozen
[ ] OptionEncoder frozen
[ ] Option LoRA absent
[ ] OptionEncoder trainable params = 0

[ ] q0 explicitly represented as Value latent
[ ] q1 explicitly represented as Meta latent

[ ] Value Adapter exists on q0
[ ] Value gate starts at zero
[ ] E_V_own exists

[ ] pretrained 15-class MetaHead uses q1
[ ] MetaHead params remain frozen
[ ] Meta auxiliary can backprop into q1 / Value trunk

[ ] Policy Strategy Adapter exists
[ ] Policy uses q1.detach()
[ ] Policy uses Meta probabilities.detach()
[ ] Policy uses adapted V.detach()
[ ] Policy uses first/second
[ ] Policy uses E_pi_own

[ ] E_V_own and E_pi_own are independent

[ ] Strategy Adapter changes readout hidden
[ ] Strategy Adapter does NOT replace GRU recurrent hidden
```

## Optimizer

```text
[ ] no Option LoRA optimizer group
[ ] no frozen OptionEncoder parameter in optimizer
[ ] adapter parameters are explicitly registered
```

## Gradient

```text
[ ] Policy loss cannot update Value side
[ ] Value loss cannot update Policy side
[ ] Meta loss cannot update Policy side
[ ] frozen representation stays frozen
```

## Equivalence

```text
[ ] zero-gate 0042 Base equivalence passes
[ ] gate startup regression passes
```

## Policy execution

```text
[ ] rollout uses Strategy Adapter
[ ] PPO replay uses Strategy Adapter
[ ] greedy eval uses Strategy Adapter
[ ] export/inference uses Strategy Adapter
[ ] compound action path uses fixed root Strategy Context
```

## Checkpoint

```text
[ ] strict 0042 save/load
[ ] adapter params cannot silently disappear
[ ] taxonomy mismatch fails
[ ] legacy Option LoRA is not silently accepted
```

## Documentation

```text
[ ] docs/rl/0042_full_model_design.md exists
[ ] accurately compares 0031 vs 0042
[ ] explains design rationale
[ ] explains frozen/trainable boundary
[ ] explains q0/q1
[ ] explains Value→Policy detach
[ ] explains removal of Option LoRA
[ ] explains readout-only strategy adapter
[ ] explains zero-gate
[ ] usable as onboarding material for another Agent
```

---

# 33. Final report

实现结束后，不要只回复“done”。

给出结构化报告：

## A. Files changed
逐文件说明用途。

## B. Final architecture
给最终真实 ASCII graph。

## C. Parameter inventory
列：
```text
module
parameter count
trainable count
optimizer group
```
特别证明：
```text
OptionEncoder trainable = 0
```

## D. Gradient-flow regression
给真实 backward matrix。

## E. Zero-gate regression
给实际测试结果。

## F. Policy-path parity
说明 rollout / PPO replay / greedy / export 是否统一。

## G. Checkpoint schema
说明新增 key / metadata。

## H. Documentation
确认：
```text
docs/rl/0042_full_model_design.md
```
并简述其章节。

## I. Remaining issues
如果存在任何 BLOCKER、semantic ambiguity、checkpoint incompatibility、missing own-archetype mapping、Meta label plumbing problem 或 inference mismatch，必须明确报告。

不要静默选择一个“差不多能跑”的实现。

---

# Final architecture principle

请始终用下面这句话判断实现是否偏离设计：

> **0042 should freeze the semantic space learned by pretraining, train Value to understand the strategic situation, and let Policy read that strategic understanding through an explicit zero-gated, stop-gradient Strategy Adapter.**

以及：

> **The OptionEncoder defines what candidate actions mean; the Strategy Adapter learns which of those actions should be preferred under the current strategic context.**

这是 0042 模型结构的核心。
