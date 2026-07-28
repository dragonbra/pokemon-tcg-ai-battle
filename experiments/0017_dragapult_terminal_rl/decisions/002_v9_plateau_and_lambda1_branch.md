# Decision 002: V9 plateau and lambda=1 branch

Date: 2026-07-28

## Evidence

V9 used the corrected frozen-behavior PPO contract, decoder-only actor updates, actor learning
rate `1e-5`, value learning rate `1e-4`, and `gae_lambda=0.95`. It produced a clear gain rather
than a failed trial:

- rolling-2,000 sampled-policy win rate rose from 9.4% at update 2 to 18.9% at update 54;
- update 50, an actually retained checkpoint, had rolling-500 19.4% and rolling-2,000 18.55%;
- all 15,360 episodes through update 60 were valid with zero discards;
- update-60 behavior KL was 0.000152, clip fraction 0.36%, and no guard rejected an update.

The last ten updates did not show a sustained gain. Update 60 ended at rolling-2,000 17.85%,
entropy had fallen from 0.611 at update 45 to 0.579, and rolling seat rates diverged to 20.3%
first versus 15.4% second. This is a plateau and distribution warning, not numerical collapse.

With about 70-80 candidate decisions per game, `gae_lambda=0.95` attenuates direct terminal
credit for early setup decisions. V4's critic explained variance was useful but low, so its
bootstrap may not fully recover that early credit.

## Decision

- Stop V9 after its update-60 model-only checkpoint.
- Use `V9_ppo_lr1e5/checkpoint/update-000050.pt` as the V10 branch point.
- Start `V10_ppo_lambda1` with a fresh optimizer and fresh on-policy episodes.
- Change only `gae_lambda` from `0.95` to `1.0`; retain 12 workers, 256 episodes/update,
  four PPO epochs, batch size 1,024, actor LR `1e-5`, value LR `1e-4`, terminal reward,
  frozen behavior, BC reference, opponent snapshot, and checkpoint cadence.
- Use the new advantage/return/value/explained-variance diagnostics to audit the hypothesis.
- Suppress known third-party worker warnings at process launch; this changes log noise only.

V10 is not an exact resume. It is a new immutable version with fresh optimizer state and rollout
distribution. Training-pool rolling rates remain diagnostics; frozen greedy evaluation is still
required before any promotion claim.
