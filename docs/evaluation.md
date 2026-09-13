# Evaluation and evidence boundaries

## Evidence classes

| Evidence | What it supports | What it does not support |
|---|---|---|
| Teacher-forced / greedy imitation metrics | Action-contract fidelity on held-out demonstrations | Match strength |
| Unit and identity tests | Tensor ownership, strict loading, routing, dtype, deterministic schedules | Match strength |
| Sampled PPO rollout | Training health for the behavior policy that generated the batch | Greedy strength of the post-update checkpoint |
| Fixed greedy official-engine evaluation | Comparable policy strength under the named opponent, seed, deck, and seat contract | Generalization outside that contract |
| Extracted package smoke | Packaging/runtime correctness | Leaderboard performance |
| Kaggle receipt/public score | External competition result for the bound package/submission | Causal attribution to one model change |

## Retained headline result

The 0045 V12 public Meta router completed 2,048/2,048 common-seed CUDA official-engine games against complete Policy-0809 at 1,382 wins, 666 losses, and 0 draws (67.48046875%). Candidate identity, opponent identity, and CUDA runtime audits passed. The same schedule informed checkpoint routing, so this is selection-set experimental evidence rather than an independent holdout or automatic Promote decision.

The detailed report is [V12 public Meta router CUDA-2048](model/evaluation/V12_dragapult_007_public_meta_router_v3_grass_fold_u200_cuda2048.html).

## Invalid historical evidence

An earlier CUDA path accidentally combined a Policy-0809 focal trunk with a Policy-0806 head while naming the result Frozen-0806. Those results remain provenance but are invalid for comparison with complete Frozen-0806. The canonical protocol now requires full effective component identity and hard-fails mismatches before games begin.

## Reproduction levels

1. Run the unit suite to check contracts and deterministic materialization.
2. Validate an extracted archive and confirm its declared hashes and FP32 runtime parameters.
3. Run a small official CPU-engine package smoke for integration.
4. Run the exact frozen CUDA schedule only on compatible NVIDIA hardware with the audited CUDA engine build.

Do not replace level 3 or 4 with mocks when making a strength claim.
