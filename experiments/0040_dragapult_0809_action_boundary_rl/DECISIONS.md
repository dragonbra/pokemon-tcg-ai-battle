# 0040 decisions

## 2026-08-10 · V1 launch contract

- Start a new on-policy run from the paired 0809 policy and Value artifacts; do not continue any 0038 PPO checkpoint.
- Reuse only the pre-PPO Phantom allocation-head BC tensors because 0809 does not contain this new head.
- Preserve the repaired 0038 Action Boundary, PRIZE reward, Frozen Policy-0806 opponent pool, 512-game rollout and fixed 32-step optimizer budget.
- Use the normal base learning rates with no accelerated-transfer multiplier or warmup controller.
- Leave the update limit unset. Stop only on user request at a complete update boundary or on a fail-closed runtime error.
- Frozen strength checks use 2,048 fixed seeds every five updates; first/second is chosen by the toss winner's Agent rather than assigned by the evaluator.
- Guard launch through the completed U5 Frozen evaluation and checkpoint before declaring the run healthy.
