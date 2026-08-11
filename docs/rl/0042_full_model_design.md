# 0042 Full Model Design

Status: V1 sealed at U250; unbounded V5 U250 fresh-optimizer branch is active.

从新 clone、环境与 CUDA Engine 准备、preflight、正式 PPO、Frozen-0809 评测到安全停止的
完整操作流程见 [`0042_full_model_design_operations_manual.md`](0042_full_model_design_operations_manual.md)。

## 1. Overview

0042 preserves the semantic action space learned by the 0031 pretrained model, lets the paired
Value network continue learning the strategic situation, and gives the policy an explicit,
zero-gated, stop-gradient path for reading that strategic information.

The central invariant is:

> The frozen OptionEncoder defines what candidate actions mean. The Strategy Adapter learns which
> of those actions should be preferred under the current strategic context.

## 2. Baseline: the real 0031 model

The source actor is `SemanticPolicy` in
`train/0031_rule_faithful_semantic_foundation_pretraining/model/policy.py`. It is loaded from
`archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt` with SHA-256
`926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f`.

0031 is not a single pooled state vector. Its actual path is:

```text
public causal DecisionBatch
  -> OfficialPrototypeEncoder
  -> StateEncoder
       state.tokens  [B, S, 320]
       state.summary [B, 320]
  -> OptionEncoder (TransformerDecoder over state.tokens)
       options       [B, O, 320]
  -> ActionDecoder
       h_t            [B, 320]
       Query/STOP     [B, O + 1]
       GRU consumes selected option and raw h_t
```

`StateEncoder` retains global, public card-instance, exact own-deck resource, and causal event
tokens. `OptionEncoder` binds every legal option to public state instances and official card,
attack, skill, and effect prototypes, then retrieves from the full state memory. `ActionDecoder`
is an autoregressive pointer with an explicit STOP action and official `minCount`/`maxCount`
constraints.

The paired Value artifact is
`archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt`, SHA-256
`f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486`. It contains eight learned
queries, two latent decoder blocks, width 320, eight attention heads, and dropout 0. The relevant
pretrained meanings are:

| Query | Shape | Existing consumer |
|---|---:|---|
| `q0 = z_V` | `[B, 320]` | scalar Value head |
| `q1 = z_M` | `[B, 320]` | frozen 15-class archetype head |
| `q2` | `[B, 320]` | frozen 13-class final-difference head |

Value is `2 * sigmoid(value_logit) - 1`, so its range is `[-1, 1]`.

## 3. What 0042 changes

| Area | 0031 / paired Value | 0042 |
|---|---|---|
| semantic encoders | pretrained | frozen exactly |
| Option LoRA | not part of 0031 | absent and fail-closed |
| Value q0 | direct Value head | own-archetype Value residual before the same head |
| Meta q1 | 15-class head | same frozen head; CE anchor updates q1 trunk |
| decoder | raw recurrent hidden scores actions | readout-only residual scores Query/STOP |
| recurrent state | raw GRU state | unchanged raw GRU state |
| policy context | none | side, detached q1/Meta/V, own archetype |
| own identity | exact resource ledger in state | plus a strategic 15-class identity |

0042 does not change reward, PPO clipping, GAE, opponent sampling, engine rules, observation
semantics, search, or the Champion protocol.

## 4. 0042 architecture

```text
                         frozen 0031 semantic space
public state -> PrototypeEncoder -> StateEncoder -> state.tokens/state.summary
                                      |
                                      +-> frozen OptionEncoder -> options [B,O,320]
                                      |
                                      +-> trainable Value latent decoder -> queries [B,8,320]
                                             | q0                         | q1
                                             v                            v
                                ValueAdapter(LN, E_V_own)       frozen MetaHead
                                             |                            |
                                      z_V_adapted                  meta_logits [B,15]
                                             |                            |
                                      ValueHead -> V [B]                  |
                                             |                            |
                                             +-------------+--------------+
                                                           | detach
state.summary -> ActionDecoder.initialize -> raw h_t [B,320]|
        selected option -> GRU(raw h_t) <-------------------|-- raw transition only
                                                           v
                         StrategyContext(side, z_M, MetaProb, V, E_pi_own)
                                                           |
                           PolicyStrategyAdapter(raw h_t) -> h_readout [B,320]
                                                           |
                                      Query/STOP logits [B,O+1]
```

## 5. Value Adapter

The Value Adapter owns all of its normalization and own-archetype parameters:

```text
deck_v  = E_V_own(own_archetype_id)             # [B,16]
delta_v = MLP_V([LN_V(q0), deck_v])              # 336 -> 320 -> 320
q0'     = q0 + tanh(g_V) * delta_v
V       = 2 * sigmoid(ValueHead(q0')) - 1
```

`g_V` is a scalar initialized to exactly zero. The MLP uses normal PyTorch initialization; its
last layer is not zero-initialized. The LayerNorm is affine and belongs only to the Value Adapter.

## 6. Strategy Context and Policy Adapter

One fixed context is built once for each root decision:

```text
context = [
  one_hot(public relative first/second),          # 2
  LN_pi_meta(q1.detach()),                        # 320
  softmax(meta_logits.detach()),                  # 15
  V.detach(),                                     # 1
  E_pi_own(own_archetype_id),                     # 16
]                                                  # total 354

delta_pi = MLP_pi([LN_pi_hidden(h_t), context])   # 674 -> 320 -> 320
h_readout = h_t + tanh(g_pi) * delta_pi
```

`h_readout` is used only by the decoder Query projection and STOP head. `consume()` still applies
the legacy GRU to the selected option and raw `h_t`. Therefore strategic conditioning changes
which action is scored, but cannot silently redefine the decoder's recurrent state machine.

V is not treated as a hard-coded rule such as "negative V means gamble". It is an input that
allows RL to learn whether and how behavior should depend on the estimated game value.

## 7. Stop-gradient boundary

`StrategyContext.build()` detaches q1, Meta logits (before softmax), and adapted V. Policy loss can
train `ActionDecoder`, `PolicyStrategyAdapter`, its LayerNorms, gate, and `E_pi_own`; it cannot
reach the Value latent decoder, Value head, Meta head, Value Adapter, or `E_V_own` through context.

Using adapter-owned LayerNorms is intentional. Reusing a trainable Value-side LayerNorm after
`q1.detach()` would still allow policy gradient into that LayerNorm's affine parameters.

## 8. Own archetype and opponent archetype

Both currently use a versioned 15-class vocabulary (14 named archetypes plus `other`), but they are
different semantic types:

- `OwnArchetypeId` is legal actor-known context resolved from the registered exact own deck.
- opponent archetype is a training/evaluation target derived from the registered opponent deck.
- `E_V_own` and `E_pi_own` are different `nn.Embedding(15, 16)` objects with disjoint Parameters.
- no opponent target, opponent deck ID, or hidden deck list is accepted by actor forward.

The taxonomy is local to 0042 at `policy/assets/own_archetypes_v1.json`; its required SHA-256 is
`44865cc88612d91154adffe13f523a63103ab0c0d1a3d981af671b96ce4a8c36`. Unsupported own decks map
to class 14 (`other`). The code does not alias the own-ID Python type to a Meta target type.

## 9. Frozen and trainable map

Counts below come from the real paired checkpoint. State/Option totals include shared prototype
Parameters, so they must not be summed. The complete actor-critic has 61,016,835 unique Parameters,
of which 5,681,447 are trainable.

| Module | Parameters | Trainable | Optimizer group | LR | WD |
|---|---:|---:|---|---:|---:|
| PrototypeEncoder | 28,418,560 | 0 | none | - | - |
| StateEncoder (includes shared prototypes) | 47,115,840 | 0 | none | - | - |
| OptionEncoder (includes shared prototypes) | 36,627,840 | 0 | none | - | - |
| ActionDecoder | 1,027,202 | 1,027,202 | `action_decoder` | explicit per run | 0 |
| Value latent decoder + heads | 3,407,389 | 3,397,121 | `value_win` | `1e-4` | 0 |
| frozen 15-class MetaHead subset | 5,455 | 0 | none | - | - |
| Value Adapter / `E_V_own` | 211,441 | 211,441 | `value_adapter` | `1e-4` | 0 |
| Policy Adapter / `E_pi_own` | 320,241 | 320,241 | `policy_strategy_adapter` | explicit per run | 0 |
| allocation head | 621,761 | 621,761 | `allocation_head` | explicit per run | 0 |
| Prize Value auxiliary | 103,681 | 103,681 | `value_prize` | `1e-4` | 0 |

No frozen Parameter is held by the optimizer, no Parameter appears in two optimizer groups, and no
OptionEncoder Parameter or parametrization appears in any group.

## 10. Gradient flow

Independent backward probes on a real public replay produced:

| Loss | Prototype | State | Option | Value trunk/q | ValueHead | ValueAdapter | MetaHead params | Decoder | PolicyAdapter |
|---|---|---|---|---|---|---|---|---|---|
| policy | no | no | no | no | no | no | no | yes | yes |
| Value | no | no | no | yes | yes | yes | no | no | no |
| Meta anchor | no | no | no | yes | no | no | no | no | no |

The Meta head parameters are frozen, but its forward is not wrapped in `no_grad`; cross-entropy can
therefore update q1, the learned queries, and shared Value latent blocks.

## 11. Zero-gate behavior

At initialization, both residuals are exactly zero and 0042 equals its pre-adapter base in CPU
FP32 eval mode. The real-checkpoint regression uses exact equality (`rtol=0`, `atol=0`) for Value,
Meta logits, root logits, masks, and greedy action.

The two-step startup regression measured:

```text
first g gradient                -0.19204865396022797 (nonzero)
first residual MLP gradient L1   0.0
g after one SGD step             0.019204866141080856
second residual MLP gradient L1  0.2096054796129465 (nonzero)
```

This guards against the dead-adapter combination of a zero gate and zero residual output layer.

## 12. Loss and execution contracts

The existing PPO/GAE/reward semantics are retained. 0042 adds a 15-class q1 cross-entropy anchor
with `meta_anchor_coef=0.10`. Rollout stores the produced Meta logits for accuracy/entropy
diagnostics and constructs the target only in batch preparation from the registered opponent deck.

Sampling, greedy decoding, PPO replay, CUDA rollout, compound-action evaluation, candidate export,
and package inference all use the same fixed-root Strategy Context and readout-only scoring rule.
The portable candidate includes actor, Value head, allocation head, both adapters, taxonomy
metadata, and own archetype ID. FP16 storage and FP32 runtime materialization remain unchanged.

## 13. Checkpoint and initialization

Schema: `0042_strategy_conditioned_model_only_v1`.

Saved state contains the full trainable decoder, Value network, allocation/Prize heads, Value
Adapter, and Policy Strategy Adapter. Metadata includes source actor/value hashes, model and adapter
schema versions, width 320, embedding width 16, taxonomy version/hash, action contracts, and
`no_option_lora=true`.

Loading fails on a missing or unexpected adapter key, wrong taxonomy hash, wrong schema, or any
state key containing `lora` or `.parametrizations.`. Optimizer, scheduler, scaler, RNG, rollout, and
replay state are forbidden. Update 0 starts from the paired pretrained actor/Value and a hash-pinned
pre-PPO allocation-head sidecar; optimizer and on-policy data start fresh.

## 14. Information safety and baseline evidence

A hidden-zone counterfactual changes opponent private hand/deck/Prize identities while keeping the
public observation constant. Compiled features, q1 Meta logits, Value, and policy logits remain
bit-identical.

The local post-hoc validation baseline has 106,307 decisions (episode weight 1,013):

| Metric | Result |
|---|---:|
| Value AUROC | 0.840918 |
| Value explained variance vs signed outcome | 0.343675 |
| Value Pearson correlation | 0.588943 |
| Value Brier | 0.164131 |
| Value ECE-10 | 0.030905 |
| Meta weighted accuracy | 0.981400 |
| Meta macro class accuracy | 0.965780 |

Meta accuracy/entropy by raw turn is 0.9502/0.1575 for turns 0-5, 0.9981/0.00636 for turns 6-11,
and 0.9977/0.00962 for turn 12+. The more important leakage probe uses only the first actor
decision: accuracy is 0.3534 with entropy 2.0658 nats across 1,013 games; when no archetype trigger
card is publicly visible, accuracy is 0.3477 across 1,001 games. This does not suggest full-deck
leakage.

Evidence limitation: this is a post-hoc local split whose source catalog differs from the missing
original catalog used for the paired V9 checkpoint. It is calibration evidence, not a formal model
selection or strength result.

## 15. PPO Protocol V2 and Policy-0809 Frozen contract

The current V6 formal focal policy uses exact deck `dragapult_ex_0042_v6` (exact-deck SHA256
`7bdb3bb183008d9204efc68ad77b6df1039ad760c98476aa82d5809f5c446ca3`) and runs without a configured
update limit. Each update collects one complete 256-game frequency unit, retains every valid decision,
and performs up to three complete data epochs. Each epoch uses a fresh permutation, sampling
without replacement, keeps the final short minibatch, and gives every valid decision exactly one
optimizer opportunity. Epoch 2 and 3 are admitted only after a deterministic rollout-wide
behavior-policy KL guard containing 4,096 shuffled decisions plus every compound/macro decision;
target/hard guards are `0.015/0.025`. Update 1 and every tenth update perform an additional
all-decision pre-update old-logprob audit, while intervening updates audit the same guard set.
Probe rows never count as optimizer samples.

Training is FP32 with AdamW, independently configured decoder/Policy-Adapter/allocation LRs,
Value-only LR `1e-4`, no shared trainable
parameters, minibatch size 2,048 decisions, clip `0.10`, entropy coefficient `0.003`, Value
coefficient `0.5`, max grad norm `0.5`, no scheduler, and zero weight decay. Frozen old logprob,
old Value, normalized advantage, and return targets are computed once per rollout and never
refreshed across epochs.

The 2,048 decisions are one logical optimizer minibatch. On the current 16 GB WSL host it is
executed as at most two 1,024-row physical forward/backward graphs using the same logical weight
denominator, followed by exactly one gradient clip and one AdamW step. Thus optimizer-step count,
coverage, PPO ratio semantics, and the 723-row tail remain unchanged. Behavior probes use 512-row
inference chunks. Formal launch hard-requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

All training opponents resolve independently as full immutable `Policy-0809`, effective identity
`0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96`. Focal and opponent
materializations share no Parameter or tensor storage. Resident CUDA receives no batch-global
deck adapter: every job/role resource ledger is checked on GPU against its own exact 60-card deck,
and a mismatch is fatal before PPO. Before neural forward, heterogeneous ready lanes are compacted
by current actor role. The focal model processes only focal rows; the independent immutable
Full0809 opponent processes only opponent rows; outputs are scattered back to original lane order.
This is request batching, not cross-policy weight sharing. The contiguous rollout feature store
remains on CUDA through minibatch gathering, and Phantom allocation-head evaluation is grouped by
shape instead of launching one GPU operation per macro.

At update 0 and every 10 updates, the candidate is merged and exported with FP16 floating storage,
strictly rematerialized as FP32 runtime, assigned an effective deployment hash, and evaluated by
the independent Frozen-0809 CUDA-2048 contract. The identity-bound schedule includes candidate
effective identity, opponent effective identity, exact opponent deck, slot, replica, and seed
namespace. Smoke artifacts are never candidates and promotion remains manual.

The real 256-game preflight completed 21,203 valid decisions (82.824/game), observed all 55 exact
opponent decks, audited all 512 focal/opponent job roles, and reported zero bulk feature D2H. Each
of three instrumented epochs used all 21,203 decisions exactly once in 11 minibatches with a final
723-decision tail. The role-compacted production preflight reached 7.77 games/s with mean ready
batch 131.85. A 256-game, 46,061-decision exact parity benchmark against the previous
double-full-batch routing passed with no divergence and improved 6.04 to 6.68 games/s (+10.6%).
The preflight kept 1,119,219,772 feature bytes on CUDA and transferred 2,839,059 bytes of compact
control/trajectory scalars to the host. Thus there is no CPU semantic feature compiler followed by
feature re-upload; the remaining scalar transfer is required for engine control and Episode/GAE
materialization and is not a feature round trip.

The post-reset one-update validation completed all 33 logical optimizer steps at 0.94 iter/s,
100% coverage and 3x reuse. PPO peak allocated/reserved memory was 6,783,030,272 / 8,516,534,272
bytes, compared with about 15.8 GB and `dxgkio_make_resident -12` for the rejected physical
2,048-row path. Frozen Prototype/State/Option encoders and MetaHead remained bit-identical.

V1 is sealed at U250 and V5 remains the prior 007-deck branch. V6 strict-loads V5 U130 focal
model-only weights, binds the new focal exact deck, and creates a fresh optimizer and fresh
on-policy rollout. The selected protocol uses decoder LR `2e-5`, Policy Strategy Adapter LR
`4e-5`, and allocation LR `2e-5`, while Value groups stay at `1e-4`. This choice came from one
identical 256-game, 22,719-decision U250 rollout: the selected arm completed all three epochs with
behavior KL `4.30e-4`, root KL `5.62e-5`, macro KL `2.36e-3`, clip fraction `0.253%`, and no
meaningful global clipping. U250 model weights are strict-loaded only after the trainer snapshots
Policy-0809/U0 as the reference-KL policy; optimizer state, RNG state, and rollout data start fresh.
The independent V4 one-update smoke then completed 22,182 decisions, 33 optimizer steps, 100%
coverage and 3x reuse with behavior KL `3.82e-4`, clip fraction `0.344%`, and unchanged frozen
representation. V6 has no configured update limit and retains the same 256-game / three-epoch
Protocol V2 settings.

## 16. Deck pool and diagnostics

`league/decks/` uses the exact 60-card contents and numbering from
`evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks`: exactly 55 directories numbered
`001` through `055`. Its local manifests deliberately identify the policy-neutral
`0042_policy_0809_neutral_55_v1` environment pool; this deck distribution does not define or
weaken the full Policy-0809 opponent weights. V6 focal is a separate explicit training identity,
`dragapult_ex_0042_v6`, stored under `focal_decks/`; it is not substituted into the opponent
catalog and does not change any full Policy-0809 opponent weights.

Operational diagnostics include gate values, effective residual ratios, gate gradients, policy and
Value adapter gradient norms, Meta accuracy/entropy by turn and class, and Value calibration. Run
`python3 -m train.0042_full_model_design.diagnostics.strategy_sensitivity --help` for controlled V,
Meta-probability, and q1 sensitivity probes.

## 17. Code map

| Contract | Path |
|---|---|
| actor-critic assembly | `train/0042_full_model_design/policy/actor_critic.py` |
| adapters/context | `train/0042_full_model_design/policy/strategy_adapters.py` |
| readout-only decoder | `train/0042_full_model_design/semantic_policy/model/action_decoder.py` |
| rollout/replay distribution | `train/0042_full_model_design/policy/action_distribution.py` |
| PPO and optimizer | `train/0042_full_model_design/training/ppo_full_semantic.py` |
| full Policy-0809 resolver | `train/0042_full_model_design/policy_identity.py` |
| identity-bound Frozen schedule | `train/0042_full_model_design/evaluation/frozen_jobs.py` |
| per-lane exact-deck audit | `train/0042_full_model_design/rollout/deck_routing.py` |
| checkpoint contract | `train/0042_full_model_design/checkpoint.py` |
| strict storage | `train/0042_full_model_design/training/storage_full_semantic.py` |
| portable runtime | `train/0042_full_model_design/kaggle_runtime/compound_inference.py` |
| architecture tests | `train/0042_full_model_design/tests/test_strategy_architecture.py` |

0031 learns semantic competence. 0042 preserves that competence and gives RL explicit strategic
control.
