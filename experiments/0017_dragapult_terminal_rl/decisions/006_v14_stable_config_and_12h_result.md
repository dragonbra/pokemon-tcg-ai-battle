# Decision 006: V14 stable configuration and 12-hour result

Date: 2026-07-28

## Evidence

V13 proved that lambda 0.97 and actor LR `5e-6` could reach a rolling-2,000 rate of 20.6%, but
its update-50 window fell to 17.9% and its second-seat rate to 16.1%. The PPO probability,
legality, critic, and numerical contracts remained healthy, so V14 branched from the retained V13
update-25 peak and changed only actor LR to `3e-6`.

V14 completed 30 PPO updates and 7,680 valid official-engine episodes before a planned stop at a
checkpoint boundary. It produced:

- rolling-2,000 rates around 20%-21% throughout the long run;
- a retained balanced checkpoint at update 15: 21.05% overall, 21.3% first, 20.8% second;
- the strongest sampled window at update 29: 22.0% overall and 24.2% rolling-500;
- the strongest retained rolling checkpoint at update 30: 21.25% overall;
- behavior KL around `1e-5`, rollout-log-prob MAE around `4e-7`, finite value diagnostics,
  zero discarded episodes, and zero rollout errors.

The GPU telemetry audit accumulated 43,235.93 active seconds, exceeding the 43,200-second target.
W&B synced successfully under `0017 · dragapult_terminal_rl · V14_ppo_lambda097_lr3e6`.

## Decision

- Select V14 as the stable viable terminal-reward PPO configuration.
- Recommend update 15 as the seat-balanced checkpoint for the first frozen-policy evaluation.
- Retain update 30 as the highest rolling-2,000 checkpoint for a paired comparison under the same
  official-engine opponent, seed, and seat contract.
- Do not start 0018: Dragapult training is viable and materially stronger than the earlier branch.
- Treat the interrupt after update 30 as a planned stop after the 12-hour target, not a training
  failure; optimizer state and raw rollouts remain intentionally unsaved.
