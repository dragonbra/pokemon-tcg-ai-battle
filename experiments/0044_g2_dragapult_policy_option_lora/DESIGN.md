# 0044 · G2 Dragapult Policy-only Option LoRA

## Current stage

0044 is a self-contained PPO continuation from immutable promoted Champion-G2
U407. The focal exact deck is fixed to `007` (Dragapult ex). V1–V3 are preserved
historical runs; V3 was stopped after durable U21. The current formal version is
`V6_u4_core16_benchmark_v2`. V6 continues from the durable V5 U4 model-only
checkpoint with a fresh optimizer and fresh on-policy collection; it loads no PFSP
state. V5's old 28-Meta U0 evaluation is retired by the final Core-16 contract.

## Policy identities and decks

- Focal initialization: deployment-effective Champion-G2, loaded from FP16
  storage into FP32 runtime. Training checkpoints remain model-only FP32.
- Focal own-deck input: Own Archetype V2 row for exact deck `007`; only own-deck
  conditioning uses the 29-class embedding.
- Opponent policy pool: singleton immutable complete `Champion-G2`.
- Opponent deck pool: exact numeric IDs `001`–`067`. Every lane binds the exact
  deck, its G2 own-deck embedding row, and G2 effective weights. Opponent weights
  never update and are resident once per rollout collection.
- The frozen 15-way opponent Meta prediction head is unchanged and not expanded.

## Model data flow

The 320-wide Option Encoder has two Transformer blocks. Inputs and block 0 run
once. The final block runs twice:

1. Value branch: immutable final block with no LoRA. Its option tensor feeds the
   Value network, opponent Meta auxiliary prediction, prize Value head, and the
   strategic context used by the policy adapter.
2. Policy branch: the same immutable final block with Q/V LoRA on self-attention
   and cross-attention. This tensor alone feeds the action decoder and allocation
   policy.

The adapter is exactly rank 4, alpha 8, block index 1, eight tensors and 10,240
parameters. A matrices use Kaiming initialization and B matrices start at zero.
Therefore U0 is bit-identical to G2 for both branches, while Value gradients can
never reach LoRA. CUDA Engine 2.0 uses the same project-local dual-option adapter;
it shares Option inputs/block 0 and routes the two final outputs explicitly.

## PPO and optimizer

PPO Protocol V2 remains unchanged: 256 rollout games, three epochs, logical
minibatch 2,048, behavior KL target 0.015, hard guard 0.025, clip 0.10, entropy
coefficient 0.003, and reference KL coefficient 0.02. Reference KL is computed
against an independent, fully frozen complete G2 model, not a decoder-only copy.

V5 learning rates match the non-doubled 0042 V1 actor scale:

- action decoder: `5e-6`
- policy strategy adapter: `5e-6`
- allocation head: `5e-6`
- policy-only Option LoRA: `1e-5` (new in 0044; preserves the approved 2× relative multiplier)
- Value trunk/value adapter/prize auxiliary: `1e-4`
- AdamW weight decay: `0`

## V6 opponent sampling

Every update independently samples all 256 opponent decks uniformly with replacement
from exact IDs `001`–`067`. The opponent policy is always the immutable complete
`Champion-G2`. PFSP and latest-champion branches both have zero lanes; V6 neither
loads, writes nor updates a PFSP state. Seat slots remain exactly 128/128 and all
engine/Search/policy seeds remain deterministic per update. Benchmark evaluation
is isolated from this stochastic training schedule.

## Periodic Benchmark V2 evaluation

Before V6 collects its first rollout, it materializes the inherited U4 checkpoint,
materializes it under `kaggle_fp16_storage_fp32_runtime_v1`, and runs official-engine
Benchmark V2 CUDA-2048 in greedy mode. The result is logged at
`eval/checkpoint_update=4`. The same evaluation repeats after each durable checkpoint
divisible by ten (`U10`, `U20`, ...). Focal deck is `007`; opponent policy is the
complete immutable `Policy-0809`. Games are restricted to Meta IDs `00`–`13`, `17`
and `27`: exactly 128 games per Meta, then balanced across its member decks with count
spread at most one. Deck allocation, engine/Search/policy seeds and coin-winner seeds
are common random numbers independent of focal identity.

Only a PASS report is mirrored to W&B under `eval/*`, including checkpoint update,
W/L/D, overall and seat-split win rates, game count, and throughput. These metrics
are separate from stochastic `rollout/*`, do not enter PPO, and do not affect
training sampling.

## Checkpoint and deployment boundary

All update checkpoints are retained as atomic model-only FP32 files. Periodic
evaluation candidates store the full dual-branch effective policy in FP16,
strict-load every floating tensor to FP32, record the source checkpoint hash,
portable artifact hash, effective tensor hash, exact focal deck hash, and refuse
legacy single-option schemas that would apply LoRA to Value.

## Launch record and version history

The CPU tests, G2 identity audit, U0 parity, gradient isolation, CUDA dual-route
smoke, behavior-logprob parity, periodic candidate round-trip, and official-engine
CUDA rollout gates passed before the initial launch. V1 exposed excessive repeated
reference-model work during PPO and failed before U1; V2 cached the complete G2
reference logits once per rollout batch and durably produced U1, then exposed a
telemetry gate that incorrectly expected three opponent weight loads. V3 corrected
that singleton-resident G2 gate and was stopped after U21.

V4 attempted these declared variables but failed before any Benchmark game or PPO
rollout because the new Benchmark runner lacked the module-inventory adapter for
the complete Policy-0809 runtime. Its used directory and W&B run are retained as a
failed version. V5 is the pristine, same-semantics retry after that adapter fix.

The intended V4/V5 experiment changes only the declared variables: fresh G2
initialization, 100% uniform opponent-deck sampling, the 0042 V1 actor learning-rate
scale, and Benchmark V2 fixed-seed Policy-0809 evaluation including the U0 baseline.
V5 safely stopped at durable U4 when the evaluation domain was revised. V6 preserves
its training semantics and changes the evaluation contract to the final Core-16 scope.
