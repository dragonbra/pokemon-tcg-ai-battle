# 0045 final public model

This document is the authoritative description of the model and training code on the public
branch. The numbered experiment ledger is preserved only at tag
`archive/final-competition-repo-2026-09-13`.

## Scope

The release trains one exact-deck specialist: deck `007` (Dragapult ex), content SHA-256
`07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`. Its starting point is
the committed complete Policy-0814 Actor and Value pair. Training and evaluation opponents are
independently materialized complete Policy-0814 policies; focal and opponent tensors may not share
mutable storage.

## Inputs and action contract

The semantic compiler converts an official observation and its legal choices into:

- static card prototypes containing identity, evolution, HP, attacks, costs, effects, targets,
  weakness/resistance, and related mechanics;
- dynamic card, event, resource, board, damage, attachment, zone, and public-knowledge fields;
- structured legal-option fields containing action type, source, target, card/effect identity,
  numeric payloads, and conditional allocation information.

The policy scores only currently legal options. Its autoregressive decoder emits an ordered option
sequence followed by STOP. Conditional effects such as damage-counter placement are handled by the
allocation head. Declaring an attack is represented as the terminal action for that turn.

## Network

```text
card prototypes ──> prototype/card encoders ──┐
dynamic state ────> State Transformer ─────────┼─> shared state
legal options ────> Option Transformer ────────┘
                                      ├─> Actor decoder + allocation head
                                      └─> Critic win/prize/meta heads
```

- Semantic width: 320.
- State Transformer: four board layers, eight heads, plus one event layer.
- Option Transformer: two blocks, eight heads.
- Maximum legal options: 128; maximum decoded action steps: 64.
- Actor outputs never consume win value, prize value, opponent-Meta predictions, or hidden exact
  opponent-deck identity.

The public trainer freezes the BC backbone and exposes the final adaptation boundary:

| Trainable component | Parameters | Learning rate |
|---|---:|---:|
| Action Decoder | 1,027,202 | `1e-5` |
| Allocation Head | 621,761 | `1e-5` |
| final Option attention LoRA, rank 16 | 61,440 | `2e-5` |
| final Option FFN LoRA, rank 16 | 40,960 | `3e-5` |
| final Option LayerNorm | 640 | `5e-6` |
| final State board attention LoRA | 30,720 | `2e-5` |
| final State board FFN LoRA | 40,960 | `1.5e-5` |
| final State event attention LoRA | 30,720 | `2e-5` |
| final State family-fusion LoRA | 30,720 | `2e-5` |
| Value head | 3,397,121 | `2e-5` |
| Value adapter | 211,665 | `2e-5` |
| Prize auxiliary head | 103,681 | `2e-5` |

The runtime audit must report exactly 125 trainable tensors and 5,597,590 trainable parameters.

## Training

`pokemon_tcg_ai.training.train` constructs U0 directly from the committed Policy-0814 assets. No
private historical checkpoint is required. U0 is saved as a complete model-only delta and also used
as the immutable reference policy for KL regularization. Optimizer state, scheduler state, RNG state,
and rollout buffers are intentionally excluded from checkpoints.

Each PPO update collects 512 official-engine CUDA games. Deck quotas are fixed at
`001=143, 002=68, 003=48, 007=64, 071=30, 008=14, 009=16, 011=7`; the remaining 122 lanes are
seeded samples from `071/008/009/011`. PPO uses a logical batch of 4096 and physical/probe
microbatches of 256. Every five updates, the saved checkpoint receives the same seeded greedy
Policy-0814 exact-deck CUDA-512 evaluation.

Sampled rollout results are training diagnostics for the policy that generated the batch. Only the
separate fixed greedy evaluation is checkpoint-strength evidence.

## Deployment

Candidate evaluation and export follow `kaggle_fp16_storage_fp32_runtime_v1`: materialize a complete
effective candidate, store floating tensors as FP16, strict-load FP32 runtime tensors, then verify the
deployment-effective hash. The final public router remembers only public, certain opponent Pokémon
identities. Unknown or conflicting evidence falls back to its declared default policy.

The retained CUDA-2048 report records 1,382–666–0 against complete Policy-0809. This is
selection-set experimental evidence, not an independent holdout or automatic promotion decision.

## Code map

- `src/pokemon_tcg_ai/model/`: stable public model API.
- `src/pokemon_tcg_ai/semantic_runtime/`: features, contracts, encoders, and portable deployment.
- `src/pokemon_tcg_ai/policy/`: Actor/Critic ownership, LoRA, decoder, and checkpoint export.
- `src/pokemon_tcg_ai/training/`: canonical PPO runner and checkpoint contracts.
- `src/pokemon_tcg_ai/rollout/`: CUDA official-engine trajectory collection.
- `src/pokemon_tcg_ai/inference/`: stable public inference API.
- `src/pokemon_tcg_ai/evaluation/`: final routing and frozen evaluation contracts.
