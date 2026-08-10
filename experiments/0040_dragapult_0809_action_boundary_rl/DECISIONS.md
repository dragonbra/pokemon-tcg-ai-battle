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

## 2026-08-11 · Policy identity invalidation and repair

- Audit found that V2 CUDA resident routing executed `Hybrid(0809 trunk, 0806 head)`: focal 0809 prototype/state/option-input/layer-0 plus 0806 layer-1/final norm/decoder. Stationarity did not make this `Policy-0806`.
- Retain every historical V2 checkpoint and result, but classify all V2 CUDA rollout, `frozen_results`, `champion.json`, `cuda_champion_*`, and related strength summaries as `Hybrid-opponent / invalid-as-Frozen0806`.
- Adopt the repo-level canonical Policy Identity Protocol V1. Full `Policy-0806` and `Policy-0809` are immutable registry entries with complete component/effective hashes. Training, evaluation, and parity share one resolver and fail closed on mismatches.
- Replace automatic champion naming with `candidate_leader.json`, `NOT_PROMOTED`, and `human_decision_required`. Promotion requires separate Full-0806 and Full-0809 reports plus an explicit human `PROMOTE`, `HOLD`, or `REJECT` decision.
- Do not run a replacement CUDA-2048 until the four specified lane-1 Full-0806 CPU/CUDA lockstep seeds pass.
- The repaired Full-0806 lane-1 greedy CPU/CUDA lockstep subsequently passed all four specified seeds with exact full trajectories and no first divergence (181/210/158/213 decisions). This clears the identity/parity prerequisite only; no replacement CUDA-2048 or promotion was performed.
