# 0042 Strategy-Conditioned Full Model

Status: **V22 U10 finishing; automatic Champion G1 sealing and full-55 Frozen-0809 CUDA-2048 handoff armed**

Date: 2026-08-12

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
- The non-candidate 16-update smoke passed. Formal V1 completed its U0 Frozen-0809 CUDA-2048
  baseline and PPO update 1; no candidate has been promoted and promotion remains human-only.

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
| ActionDecoder | 1,027,202 | 1,027,202 | `action_decoder`, explicit per-run LR |
| Value latent decoder and heads | 3,407,389 | 3,397,121 | `value_win`, `1e-4` |
| MetaHead subset | 5,455 | 0 | none |
| Value Adapter | 211,441 | 211,441 | `value_adapter`, `1e-4` |
| Policy Strategy Adapter | 320,241 | 320,241 | `policy_strategy_adapter`, explicit per-run LR |
| Allocation head | 621,761 | 621,761 | `allocation_head`, explicit per-run LR |
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

The current V11 formal focal policy is catalog deck 001,
`marnie_s_grimmsnarl_ex_froslass_c20a8a46f5c6`, with exact-deck SHA256
`c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af`. The immutable
own-archetype taxonomy resolves this exact deck to class 2 (`marnies_grimmsnarl_ex`), changing the
actor-visible own-archetype input from V9 class 3 without changing tensor shapes, the action
contract, or any opponent input. The exact 60-card identity is independently bound in rollout and candidate
deployment metadata. Training has no update cap. Every update
collects one complete 256-game frequency unit and retains every valid decision. PPO uses FP32
AdamW, logical 2,048-decision minibatches, independently explicit decoder/Policy-Adapter/allocation
LRs, Value-only LR `1e-4`, and up to three complete
shuffle-without-replacement data epochs. Epoch 2/3 are admitted by a fixed rollout-wide behavior
KL guard set containing 4,096 shuffled rows plus every compound/macro row, with target/hard guards
`0.015/0.025`; rollout old logprob, old Value, normalized advantage, and returns stay frozen.
Update 1 and every tenth update additionally audit pre-update old-logprob parity over all decisions;
intervening updates audit the guard set. Probe rows never count as optimized samples.

On the 16 GB WSL CUDA host, each logical 2,048-decision optimizer step is evaluated as at most two
1,024-row physical graphs with one shared logical weight denominator, one accumulated gradient,
one clip, and one AdamW step. The 723-row tail remains one graph. This preserves the 11 logical
optimizer steps and exact decision coverage while avoiding DXG residency failure. Behavior probes
use 512-row inference chunks. Formal launch requires expandable CUDA allocator segments.

All opponents are independent materializations of full immutable `Policy-0809`, effective hash
`0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96`. Resident CUDA audits
every focal/opponent job role against its own exact 60-card resource ledger and rejects any
batch-global deck substitution. Ready lanes are compacted by actor role before forward: the focal
model evaluates only focal rows and the independent immutable opponent evaluates only opponent
rows, after which outputs are scattered back to lane order. This removes the prior double
full-batch forward without merging policy identities or changing per-lane deck fields. Bulk
rollout features remain GPU resident. PPO gathers from that contiguous device store, and
allocation-head evaluation batches Phantom macros by tensor shape.

Update 0 and every 10 updates use the same formal path: merge/export FP16 candidate storage,
strict-load FP32 runtime, verify deployment identity, then execute an identity-bound Frozen-0809
CUDA-2048 schedule. Results cannot promote a policy automatically.

Deployment-effective identity hashes the deployed FP16 tensors and runtime-semantic schema only.
Source checkpoint/file hashes, update numbers, version names, and training provenance remain
separate mandatory audit fields. U0 pins the corrected content identity and its derived schedule;
each later checkpoint derives a distinct schedule from its own effective content identity.

The real preflight completed 256 games and 21,203 valid decisions, audited all 55 exact opponent
decks and 512 job-role ledgers, recorded zero feature D2H, and proved exact cumulative decision
usage 1/2/3 across three 11-minibatch epochs. The production role-compacted preflight reached
7.77 games/s with mean ready batch 131.85. A same-schedule, per-decision parity benchmark against
the former double-full-batch route passed over 46,061 decisions and improved throughput from
6.04 to 6.68 games/s (+10.6%). The only host transfer is compact control/trajectory data
(2,839,059 bytes in the preflight); 1,119,219,772 bytes of semantic feature tensors remained on
CUDA with zero bulk feature D2H and no CPU feature recompilation/re-upload path.

A post-reset one-update PPO validation completed all 3 epochs and 33 logical optimizer steps at
0.94 iter/s, with 100% coverage, 3x reuse, peak allocated/reserved 6.78/8.52 GB, no NaN/Inf, and
bit-identical frozen encoders/MetaHead. The previous 2,048-row physical graph reached about 15.8 GB
and failed `dxgkio_make_resident`; it is not the production execution path.

## 12. Current and next stage

V1 is sealed at model-only U250 (formal Frozen-0809 CUDA-2048: 1,251-797, 61.083984%). A diagnostic
256-game rollout with 22,719 decisions compared three fresh-optimizer arms on identical data.
The selected branch uses decoder `2e-5`, Policy Strategy Adapter `4e-5`, allocation `2e-5`, and
unchanged Value groups `1e-4`; its three-epoch behavior KL was `4.30e-4`, clip fraction `0.253%`,
and all decisions received exactly 3 optimizer opportunities. The diagnostic artifact is
`.tmp/evaluation/0042_u250_lr_probe/report.json` and is not promotion evidence.

The V4 one-update smoke completed 256 games, 22,182 decisions, 33 optimizer steps, 100% coverage,
3x reuse, behavior KL `3.82e-4`, and clip fraction `0.344%`, with unchanged frozen representation.
V7's immutable U270 model-only checkpoint has SHA-256
`0ce3b0920d4fd09674d06270c100c500e7bea627ee58933f0ecb02d69a5a5198`. V8 strict-loads that
checkpoint only after `PPOTrainer` snapshots immutable Policy-0809/U0 as the unchanged reference-KL
policy. The complete learned decoder, Value network, adapters, allocation head, and Prize head
continue; optimizer, RNG, rollout, and replay state do not. Only the exact focal deck and the
runtime-derived own-archetype input change, from V7 class 6 to V9 class 3. V9 uses the same decoder
`2e-5`, Policy Adapter `4e-5`, allocation `2e-5`, and Value-group `1e-4` learning rates, runs
without an update cap, retains every model-only checkpoint, and performs identity-bound
Frozen-0809 CUDA-2048 evaluation every 10 updates. V8 completed its U0 Frozen run but stopped
before PPO because one training-FP32 versus FP16-storage Value sample differed by `0.0033997893`,
above the `0.003` gate; inputs, CPU/CUDA runtime parity, greedy actions, and Value signs agreed.
V8 remains an immutable failed version. By explicit user authorization, V9 records that deployment
numeric comparison as diagnostic and does not use it to block PPO; deployment-effective identity,
FP16-storage/FP32-runtime Frozen evaluation, full Policy-0809 opponent identity, exact-deck routing,
official-engine health, and CPU/CUDA semantic parity remain hard gates. Promotion remains a separate
human decision.

V9 was stopped by request after completing U92; V11 intentionally branches from its immutable U90
model-only checkpoint, SHA-256
`b9d754214deb679c9ab0fb45c35ee39dfee39692b8aef73f88b06c4c1f894776`, rather than from the later
stop-boundary checkpoint. V11 strict-loads only those model weights after snapshotting the same
original Policy-0809/U0 reference-KL policy, initializes fresh optimizer/RNG/on-policy state, and
switches only the focal exact deck and own-archetype class from Alakazam class 3 to Grimmsnarl class
2. It preserves V9's topology, learning rates, every-10-update Frozen CUDA-2048 cadence, explicit
FP16 numeric-drift diagnostic waiver, unbounded update duration, and retain-all model-only checkpoint
policy. V9 U30/U90/U92 results remain V9 provenance and are not V11 strength evidence. V10 is
retained as a failed formal version: it wrote its model-only U0 and synced W&B, then failed closed
before Frozen games because its transient systemd service PATH could not resolve `g++`. V11 is a
fresh version, not a V10 resume, and adds only the corrected service execution environment.

V11 terminated after its U50 Frozen evaluation. The serialized controller completed V12 catalog
003 (`Mega Lopunny ex / Mega Froslass ex`, own class 1), V13 catalog 006 (`Teal Mask Ogerpon ex /
Hero’s Cape`, class 10), and V14 catalog 009 (`Mega Lucario ex / Solrock`, class 4). V15 catalog
010 (`Festival Lead / Dipplin`, class 6) failed before update 1 because its U0 CPU/CUDA Value
diagnostic exceeded the then-active `5e-6` tolerance by `6.79e-9`; no PPO update was applied.
Per explicit user decision, numeric-closeness gates no longer exist: CPU/CUDA Value and FP32/FP16
comparisons are recorded only as diagnostics and cannot block training. Exact tensor/mask
semantics, greedy-action identity, checkpoint/deployment identity, exact-deck routing, and
independent Policy-0809 identity remain hard gates. V15 remains an immutable failed record. The
recovery controller starts V16 catalog 010 from V14 U50, then V17 catalog 011 (`Mega Kangaskhan ex
/ Crustle`, class 5). Every successful segment strict-loads
only the preceding segment's immutable `update-000050.pt`, starts fresh optimizer/RNG/on-policy
state, runs exactly 50 updates, completes the U50 FP16-storage/FP32-runtime Frozen-0809 CUDA-2048
evaluation, and then exits. The controller refuses the next transition unless the checkpoint
sidecar matches, all 2,048 Frozen games are unique and valid, there are zero errors/unfinished/
semantic fallbacks, and both candidate and full Policy-0809 identity audits pass. Focal exact-deck
hash and own-archetype class are re-derived from the exact 60 cards at every launch; numbering is
never used as the actor input. All other topology, optimizer rates, opponent pool, reference-KL
anchor, checkpoint retention, and deployment waiver settings remain those of V11.

After V17 passes its U50 terminal gate, a second controller extends the same immutable
checkpoint chain through the five remaining focal archetypes, each for exactly 10 updates: V18
catalog 019 (`Barbaracle / Cornerstone Mask Ogerpon ex`, own class 7), V19 catalog 013
(`Cynthia’s Garchomp ex / Roserade`, class 11), V20 catalog 021 (`Team Rocket’s Mewtwo ex /
Spidops`, class 9), V21 catalog 044 (`Mega Starmie ex / Mega Froslass ex`, class 12), and V22
catalog 048 (`Archaludon ex / Cinderace`, class 13). V18 strict-loads only V17
`update-000050.pt`; every later version strict-loads only the preceding version's immutable
`update-000010.pt`. Each version initializes fresh optimizer/RNG/on-policy state and completes its
U10 FP16-storage/FP32-runtime Frozen-0809 CUDA-2048 evaluation before handoff. The U10 gate applies
the same checkpoint SHA, 2,048 unique valid games, zero error/unfinished/semantic fallback,
candidate deployment identity, and independent full Policy-0809 opponent identity requirements as
the U50 controller. This continuation brings actor-visible focal training coverage to all 14
formal own-archetype classes; `Other` remains the taxonomy fallback rather than a scheduled focal
class.

After V22 completes U10 and its terminal CUDA-2048 identity gate, the user explicitly designates
that immutable model-only checkpoint as `Champion-G1`, the first generational anchor for the
continued research program. This designation is not an automatic `Promote Champion V1` score
decision and does not relabel any Frozen opponent: its archive records the exact V22/U10 source
hash, a reference FP16-storage/FP32-runtime portable package, and the deployment-effective
identity. Every evaluation view rematerializes the same Champion-G1 effective weights and binds a
different exact 60-card deck plus its derived own-archetype class; it never trains or mutates the
sealed checkpoint.

The formal Champion-G1 assessment runs 55 independent CUDA-2048 focal views against a separately
materialized complete immutable Frozen Policy-0809. Phase one scans deck numbers in ascending
order and selects the first exact deck for each formal class 0–13:
`001, 002, 003, 006, 007, 009, 010, 011, 013, 019, 020, 021, 044, 048`. Phase two then evaluates
all 41 skipped/repeated decks, including the explicit `Other` fallback, in ascending deck-number
order. Each result requires 2,048 terminal official CUDA-engine games, zero error/unsupported/
semantic fallback, candidate FP16-storage/FP32-runtime identity `PASS`, independent Policy-0809
identity `PASS`, and zero shared parameter storage. Durable output is refreshed after every deck
under `docs/evaluation/combat_mat/policy_0809/`; detail pages use the canonical Combat Mat renderer
and contain exact-60 card art, all 55 opponent matchup rows, and the ordered 14-class plus Other
aggregation. The index displays per-deck W-L-D/win rate and, where the existing same-contract 0809
EC report exists, the direct win-rate delta; missing 0809 baselines remain explicitly missing.
