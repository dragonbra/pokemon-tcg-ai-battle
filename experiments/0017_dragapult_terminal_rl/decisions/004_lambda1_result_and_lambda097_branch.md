# Decision 004: Lambda=1 result and lambda=0.97 branch

Date: 2026-07-28

## V11 result

V11 is the valid eval-mode repetition of the V10 hypothesis. Through 20 updates:

- every rollout-log-prob MAE stayed around `4e-7`;
- all 5,120 official-engine episodes were valid;
- behavior KL and clip fraction remained far below their guards;
- explained variance was generally 0.13-0.27, above the original V4 value calibration result;
- rolling-2,000 ended at 17.95%, below the V9 update-50 branch point's 18.55%;
- first/second seat rates ended at 17.7%/18.2%, materially more balanced than late V9;
- entropy remained 0.628 instead of continuing V9's decline.

The evidence supports better credit reach and seat balance, but not a strength gain. Full Monte
Carlo returns also keep value loss around 0.4-0.5 and introduce more variance than lambda 0.95.

## Decision

- Stop V11 after update 20; retain its checkpoint as valid research evidence.
- Start `V12_ppo_lambda097` from the same clean V9 update-50 checkpoint.
- Change only `gae_lambda` to `0.97`; retain the eval-mode rollout fix and all V9/V11 settings.
- Evaluate whether the interpolation recovers V9's sampled strength while preserving V11's seat
  balance and higher entropy.
- Do not create 0018: Dragapult PPO is demonstrably learnable, and the backup trigger is not met.
