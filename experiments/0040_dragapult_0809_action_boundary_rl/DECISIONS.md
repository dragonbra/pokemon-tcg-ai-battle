# 0040 decisions

## 2026-08-10 · V1 launch contract

- Start a new on-policy run from the paired 0809 policy and Value artifacts; do not continue any 0038 PPO checkpoint.
- Reuse only the pre-PPO Phantom allocation-head BC tensors because 0809 does not contain this new head.
- Preserve the repaired 0038 Action Boundary, PRIZE reward, Frozen Policy-0806 opponent pool, 512-game rollout and fixed 32-step optimizer budget.
- Use the normal base learning rates with no accelerated-transfer multiplier or warmup controller.
- Leave the update limit unset. Stop only on user request at a complete update boundary or on a fail-closed runtime error.
- Frozen strength checks use 2,048 fixed seeds every five updates; first/second is chosen by the toss winner's Agent rather than assigned by the evaluator.
- Guard launch through the completed U5 Frozen evaluation and checkpoint before declaring the run healthy.

## 2026-08-10 · V1 failed before PPO; V2 loader fix

V1 completed the full U0 Frozen-2048, then failed closed in package snapshot parity before update 1. The shared diagnostic reconstructed every training checkpoint with the hard-coded 0038 module, so its strict 0038 validator correctly rejected the new 0040 schema. No PPO update occurred.

The shared diagnostic now takes an explicit validated numbered training-project package. Its default remains 0038; 0040 passes its own self-contained strict loader. Replaying the exact V1 U0 package/checkpoint across all 283 immutable decisions passed with zero observation/mask/greedy divergence, root top-1/top-2 agreement, no Value sign divergence, and maximum CUDA/training Value error `5.364418029785156e-7`. V2 starts fresh rather than appending to V1.
