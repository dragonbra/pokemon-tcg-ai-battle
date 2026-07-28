# Decision 005: Lambda=0.97 actor-LR stabilization

Date: 2026-07-28

## V12 evidence

V12 used the same V9 update-50 branch point and training contract as V11, changing only
`gae_lambda` from 1.0 to 0.97. Its credit diagnostics were substantially better:

- explained variance stayed around 0.42-0.53 through update 10;
- value loss stayed around 0.09-0.12 instead of V11's 0.4-0.5;
- rollout-log-prob MAE remained around `4e-7`;
- rolling-2,000 reached 19.01% at update 6, above the V9 update-50 branch baseline;
- all episodes and PPO guards remained valid.

The strength gain was not stable. Rolling-2,000 declined to 17.8% by update 10, while the
second-seat rate fell from 16.9% at update 6 to 14.3%. The actor's reference surrogate also
trended upward while entropy stayed near 0.60. This is cumulative policy drift, not a critic,
legality, or numerical failure.

## Decision

- Stop V12 after update 10 and retain its update-5 checkpoint as the branch point immediately
  before the observed update-6 strength peak.
- Start `V13_ppo_lambda097_lr5e6` from V12 update 5 with a fresh optimizer and fresh rollouts.
- Halve only actor LR from `1e-5` to `5e-6`.
- Keep lambda 0.97, value LR `1e-4`, 12 workers, 256 episodes/update, four epochs, reference
  coefficient, entropy coefficient, opponent snapshot, terminal reward, and eval-mode contract.
- Judge V13 by rolling-2,000 strength, first/second-seat gap, entropy, reference drift, and the
  already-added critic diagnostics.
