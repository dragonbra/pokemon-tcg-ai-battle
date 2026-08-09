# 0040 · Dragapult 0809 Action-Boundary RL

## Objective

0040 measures the long-run PPO learning curve of exact deck 007 from the stronger paired 0809 pretraining release. It preserves the repaired 0038 Action Boundary, reward, rollout and PPO semantics. It does not add Seat residual, Opponent Meta, Plan Core, Value search, new rewards, or a new opponent pool.

The authoritative engine source and ABI remain untouched. Official callbacks still receive one primitive `list[int]` select. Agent-side DecisionGate and the pending Phantom Dive transaction only change which callbacks are Policy/Value time boundaries.

## Attested initialization

| component | immutable source | update-0 behavior |
|---|---|---|
| State/Option Encoder and Root Decoder | `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt`, SHA-256 `926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f` | encoder frozen; decoder fully trainable |
| `V_win` latent-query network | paired `value_head.pt`, SHA-256 `f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486` | 0036 V9 epoch 9; Query 0 retains terminal win/loss semantics |
| Last Option Q/V LoRA | fresh r/alpha 4/8 | A Kaiming, B zero; exact zero-delta |
| Phantom allocation head | allocation tensors only from 0038 pre-PPO U0 | BC warm start; no 0038 actor, Value, LoRA, optimizer or PPO state |
| optimizer/RNG/rollout/old policy | none | freshly initialized |

The loader is strict for every existing 0809 tensor and verifies that the Value checkpoint declares the same policy source hash. Critical missing or unexpected keys are forbidden. The only cross-project runtime input is the immutable allocation-head weight artifact; executable 0040 code is self-contained.

## Model and action contract

The input schema, tensor shapes and model flow are unchanged from repaired 0038: public canonical observation and legal-option relations feed a 320-wide semantic policy; the frozen encoders produce state/option embeddings; trainable Root Decoder and last-option Q/V LoRA produce root logits; Query 0 produces `V_win`; the conditional allocation head reuses the same encoded state for Phantom Dive.

Phantom Dive is one hierarchical policy decision:

```text
log P(action) = log P(root) + log P(allocation | state, Phantom Dive)
```

For `n=1..8`, the canonical allocation is chosen once, cached by stable Pokémon identity, and unfolded into the official primitive callbacks. It produces one encoder forward, one Value, one joint old-logprob and one PPO transition. Forced callbacks use `observe_only`; they consume public history and state changes but produce no Policy/Value/PPO sample. Chance, information, priority or transaction drift fails closed and cannot silently become a new strategic decision.

## Training objectives

Preset `PRIZE` preserves the 0038 objective registry:

- PPO clipped `L_policy_win` and terminal ±1 `L_value_win`;
- directional `V_prize`, with prize reward scaled by `1/24`, independently normalized `A_prize`, and the existing fixed actor weight;
- root and allocation entropy terms;
- read-only tempo metrics.

Opponent Meta, Meta conditioning, tempo curriculum/loss and all proposed post-0038 heads remain disabled.

## V1 long-run configuration

`V1_0809_normal_lr_long_run` uses CUDA resident rollout with the repaired CPU-authoritative feature schema and the exact Frozen Policy-0806 opponent catalog. Each update collects 512 complete games and retains all 512 trajectories. Optimization uses `fixed_optimizer_budget`, 32 optimizer steps, physical minibatch 1024, accumulation 1 and 4 PPO epochs.

| optimizer group | LR | gradient source |
|---|---:|---|
| Action Decoder | `1e-5` | policy, entropy, reference KL, Prize actor |
| Allocation Head | `1e-5` | allocation policy/entropy and Prize actor |
| Last Option Q/V LoRA | `3e-5` | shared actor path |
| `V_win` | `1e-4` | terminal win Value |
| `V_prize` | `1e-4` | directional Prize Value |

These are the normal 0038 base rates: no accelerated-transfer warmup or multiplier controller is enabled. The run has no update limit and stops only at a complete update boundary after `STOP_REQUESTED` or an error.

## Frozen evaluation and evidence boundary

U0 and every fifth update run the versioned Frozen-0806 agent-choice v3 panel: evaluation seed `341512806`, eight non-overlapping 256-game replicas, 2,048 games total, exact opponent-slot frequency, and no externally assigned seat. The actual toss winner receives context 41 and the deployed policy chooses first/second. The same fixed seeds, opponent schedule, greedy policy and engine/runtime hashes are reused at every checkpoint.

Frozen metrics are logged under `eval/*`; stochastic rollout diagnostics remain under `rollout/*`. W&B is online in private project `dragon_bra/pokemon-tcg-policy-learning`, while local `training_metrics.jsonl` is canonical. Every update saves an atomic model-only checkpoint; no optimizer, scheduler, RNG, replay or rollout buffer is serialized.

Project stage: implementation and U0/U5 guarded launch. A checkpoint is not submission-ready merely because CUDA Frozen improves; official CPU/package transfer remains a separately authorized evaluation.
