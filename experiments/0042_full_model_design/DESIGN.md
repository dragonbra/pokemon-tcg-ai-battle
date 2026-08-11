# 0042 Strategy-Conditioned Full Model

Status: **PPO Protocol V2 preflight passed; non-candidate 16-update smoke in progress**

Date: 2026-08-11

## 1. Decision summary

0042 freezes the pretrained 0031 semantic encoders, trains the paired latent-query Value network,
and lets the decoder read Value-side strategic information through a detached, zero-gated,
readout-only Policy Strategy Adapter. The OptionEncoder is frozen and has no LoRA wrapper.

This document is the authoritative project contract. The onboarding explanation is
[`docs/rl/0042_full_model_design.md`](../../docs/rl/0042_full_model_design.md).

## 2. Evidence boundary

- Official rules and the unmodified engine define legal observations, options, and terminal result.
- The 0031 source code and paired checkpoint define the pretrained actor and Value structure.
- The 0042 code and real-checkpoint backward tests define the implemented gradient boundary.
- The 106,307-decision local Value/Meta baseline is post-hoc evidence from a non-original catalog;
  it is not formal strength or model-selection evidence.
- No PPO update, W&B run, candidate promotion, or formal Frozen evaluation was performed here.

## 3. Immutable sources

| Source | Path | SHA-256 |
|---|---|---|
| 0031 Policy-0809 actor | `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt` | `926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f` |
| paired V9 Value | `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt` | `f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486` |
| own taxonomy | `train/0042_full_model_design/policy/assets/own_archetypes_v1.json` | `44865cc88612d91154adffe13f523a63103ab0c0d1a3d981af671b96ce4a8c36` |
| neutral 55-deck schedule | `train/0042_full_model_design/league/frozen_catalog.json` | `b1147f570c1df9f9b3261f3ed89d83c5a96cc5f51931a30ce9ae57c59bbc3d2c` |

The 55 local deck directories are byte-identical to
`evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks` and numbered `001` to `055`.

## 4. Model and tensor contract

```text
public causal state
  -> frozen PrototypeEncoder / StateEncoder
       state.tokens  [B,S,320]
       state.summary [B,320]
  -> frozen OptionEncoder
       options       [B,O,320]
  -> trainable Value latent decoder
       queries       [B,8,320]
       q0 z_V        [B,320] -> ValueAdapter -> ValueHead -> V [B]
       q1 z_M        [B,320] -> frozen MetaHead -> logits [B,15]

state.summary -> ActionDecoder -> raw h_t [B,320]
detached(q1, MetaProb, V) + side + E_pi_own
  -> PolicyStrategyAdapter(raw h_t) -> h_readout [B,320]
  -> Query/STOP logits [B,O+1]

selected option + raw h_t -> unchanged GRU transition
```

Value Adapter:

```text
delta_V = MLP_V([LN_V(q0), E_V_own(id)])       # 336 -> 320 -> 320
q0' = q0 + tanh(g_V) * delta_V
```

Policy Strategy Adapter:

```text
c = [one_hot(first/second), LN_M(q1.detach()),
     softmax(meta_logits.detach()), V.detach(), E_pi_own(id)]  # 354
delta_pi = MLP_pi([LN_h(h_t), c])                            # 674 -> 320 -> 320
h_readout = h_t + tanh(g_pi) * delta_pi
```

Both gates are scalar and initialized to zero. Residual MLPs use normal initialization. Each
adapter owns its affine LayerNorms. `E_V_own` and `E_pi_own` are independent `Embedding(15,16)`
tables.

## 5. Actor-visible schema and information safety

0042 does not add an opponent deck ID, full opponent deck, hidden cards, or Meta training target to
the actor input. The first/second signal is `global_cat[:,2]` from the public relative seat field.
The registered exact own deck remains legal input through the existing exact resource ledger;
`OwnArchetypeId` adds only a coarse strategic identity resolved from that known deck.

Own and opponent labels initially share taxonomy content but not semantic types or Parameters.
Unknown own decks map to class 14 (`other`). Opponent target resolution occurs only in batch
preparation. A hidden hand/deck/Prize counterfactual is bit-identical through compiled tensors,
Meta logits, Value, and policy logits.

## 6. Frozen and trainable boundary

| Module | Parameters | Trainable | Optimizer / LR |
|---|---:|---:|---|
| PrototypeEncoder | 28,418,560 | 0 | none |
| StateEncoder, including shared prototypes | 47,115,840 | 0 | none |
| OptionEncoder, including shared prototypes | 36,627,840 | **0** | none |
| ActionDecoder | 1,027,202 | 1,027,202 | `action_decoder`, `5e-6` |
| Value latent decoder and heads | 3,407,389 | 3,397,121 | `value_win`, `1e-4` |
| MetaHead subset | 5,455 | 0 | none |
| Value Adapter | 211,441 | 211,441 | `value_adapter`, `1e-4` |
| Policy Strategy Adapter | 320,241 | 320,241 | `policy_strategy_adapter`, `5e-6` |
| Allocation head | 621,761 | 621,761 | `allocation_head`, `5e-6` |
| Prize Value auxiliary | 103,681 | 103,681 | `value_prize`, `1e-4` |

The complete model has 61,016,835 unique Parameters and 5,681,447 trainable Parameters. Weight
decay is zero. No frozen or duplicate Parameter is present in the optimizer.

The pretrained MetaHead parameters stay frozen. Its forward graph stays live, so the weighted
15-class CE anchor (`meta_anchor_coef=0.10`) updates q1 queries and Value latent blocks.

## 7. Measured gradient contract

| Loss | Frozen encoders | Value trunk | ValueHead | ValueAdapter | MetaHead params | Decoder | PolicyAdapter |
|---|---|---|---|---|---|---|---|
| policy | no grad | no grad | no grad | no grad | no grad | grad | grad |
| Value | no grad | grad | grad | grad | no grad | no grad | no grad |
| Meta anchor | no grad | grad | no grad | no grad | no grad | no grad | no grad |

## 8. Preserved training semantics

PPO clipping, reference KL, entropy, GAE (`gamma=1`, `lambda=0.95`, turn clock), reward, Prize
auxiliary, action boundary, compound allocation, opponent schedule, and official engine behavior are
unchanged. Formal config is `FULL_MODEL`; no old `PRIZE`, `META`, `INTEGRATED`, Option-LoRA, or
accelerated-transfer arm exists in the canonical 0042 path.

Rollout, greedy, PPO replay, CUDA boundary, compound evaluation, export, and package inference all
build one fixed Strategy Context per root decision and use the same readout-only rule.

## 9. Initialization and checkpoint

Formal version identity is `V1_ppo_protocol_v2_baseline`. Update 0 strict-loads the paired
pretrained actor and Value plus the hash-pinned pre-PPO allocation sidecar, initializes both
adapters under a private RNG stream, and asserts `g_V=g_pi=0`. Optimizer and on-policy data start
fresh.

The model-only schema is `0042_strategy_conditioned_model_only_v1`. It saves the complete decoder,
Value network, allocation/Prize heads, and both adapters. Metadata pins base hashes, adapter schema
and dimensions, taxonomy version/hash, `no_option_lora=true`, and action contracts. Missing or
unexpected adapter tensors, taxonomy mismatch, and any LoRA/parametrization key are fatal.
Optimizer, scheduler, scaler, RNG, rollout, and replay state are forbidden.

FP16 candidate storage and FP32 runtime are preserved. The Value network and both adapters are part
of effective policy identity because q0/q1/Meta/V condition every strategic decision.

## 10. Regressions and diagnostics

- 231 project tests pass; one environment-dependent test is skipped.
- Real-checkpoint zero-gate Value, Meta, root logits, masks, and greedy action pass exact equality.
- First gate gradient is nonzero; residual MLP gradient is zero before the first gate step and
  nonzero after it.
- Real portable checkpoint export and strict FP32 materialization pass.
- Local baseline: Value AUROC 0.840918, explained variance 0.343675, Pearson 0.588943,
  Brier 0.164131, ECE-10 0.030905; Meta weighted accuracy 0.981400.
- First-decision Meta probe: 0.3534 accuracy, 2.0658 nats entropy; no trigger visible subset:
  0.3477 accuracy over 1,001 games.

Temporary machine-local evidence:

- `.tmp/strategy_adapter_v2_audit/0042_runtime_audit.json`
- `.tmp/strategy_adapter_v2_audit/0042_value_meta_baseline.json`
- `.tmp/evaluation/0042_export_smoke/`

## 11. PPO Protocol V2 and Frozen-0809

The formal focal policy is exact deck `007_dragapult_ex`. Training has no update cap. Every update
collects one complete 256-game frequency unit and retains every valid decision. PPO uses FP32
AdamW, 2,048-decision minibatches, actor LR `5e-6`, Value-only LR `1e-4`, and up to three complete
shuffle-without-replacement data epochs. Epoch 2/3 are admitted by a fixed rollout-wide behavior
KL guard set containing 4,096 shuffled rows plus every compound/macro row, with target/hard guards
`0.015/0.025`; rollout old logprob, old Value, normalized advantage, and returns stay frozen.
Update 1 and every tenth update additionally audit pre-update old-logprob parity over all decisions;
intervening updates audit the guard set. Probe rows never count as optimized samples.

All opponents are independent materializations of full immutable `Policy-0809`, effective hash
`0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96`. Resident CUDA audits
every focal/opponent job role against its own exact 60-card resource ledger and rejects any
batch-global deck substitution. Bulk rollout features remain GPU resident. PPO gathers from that
contiguous device store, and allocation-head evaluation batches Phantom macros by tensor shape.

Update 0 and every 10 updates use the same formal path: merge/export FP16 candidate storage,
strict-load FP32 runtime, verify deployment identity, then execute an identity-bound Frozen-0809
CUDA-2048 schedule. Results cannot promote a policy automatically.

The real preflight completed 256 games and 21,203 valid decisions, audited all 55 exact opponent
decks and 512 job-role ledgers, recorded zero feature D2H, and proved exact cumulative decision
usage 1/2/3 across three 11-minibatch epochs.

## 12. Current and next stage

Current stage is the final non-candidate smoke. On a passing smoke, the next stage is the fresh,
unbounded V1 Protocol V2 W&B run with model-only checkpoint retention and Frozen-0809 CUDA-2048
at every 10-update milestone. Promotion remains a separate human decision.
