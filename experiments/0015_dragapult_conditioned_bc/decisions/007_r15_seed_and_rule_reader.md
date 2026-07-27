# Decision 007: R15 seed variance and typed rule-reader rejection

Date: 2026-07-27

## Decision

Keep `V2_t1_plus_pure_dragapult` as the strongest supported policy. Stop the full-width typed
rule-reader scale, learning-rate and seed sequence after V13. Do not launch the prepared low-rank
rule-reader V14: its predeclared gate required V11 or V12 to exceed V2's 39/200 official-engine
result, and they reached 38/200 and 29/200.

Proceed with the plain-R15 `V14_t1_label_smoothing_005` experiment. It changes only train-time CE
confidence and leaves model input, structure and candidate inference unchanged.

## Evidence

| Version | Contract | Target exact | Official engine |
|---|---|---:|---:|
| V2 | plain R15 T1, seed 20260723 | 58.7% | 39/200 (19.5%) |
| V6 | plain R15 T1, seed 20260727 | 56.2% | 20/200 (10.0%) |
| V7 | fresh full-width rule, scale 0.10 | 60.2% | 25/200 (12.5%) |
| V8 | fresh full-width rule, scale 0.03 | 56.2% | 18/200 (9.0%) |
| V9 | V2 warm-start, update all, LR 1e-4 | 57.9% | 30/200 (15.0%) |
| V10 | V2 frozen, rule-only LR 3e-4 | 61.3% | 29/200 (14.5%) |
| V11 | V2 frozen, rule-only LR 1e-4 | 60.7% | 38/200 (19.0%) |
| V12 | V2 frozen, scale 0.10 | 61.6% | 29/200 (14.5%) |
| V13 | V12 contract, seed 20260727 | 61.5% | 40/200 (20.0%) |

Every formal evaluation completed 200/200 games with zero errors. V12 and V13 have nearly identical
offline exact but differ by 11 engine wins. V13's one-win edge over V2 cannot be treated as an
architectural gain when the same rule contract at V12 is ten wins below V2. The result is further
evidence that label match cannot rank rollout strength and that 200-game point estimates retain
material seed variance.

V9 additionally shows that full-model fine-tuning rapidly damages the inherited V2 policy even at
LR `1e-4`. Freezing the base prevents that damage but the 3,311,360-parameter rule reader overfits
within a few epochs and does not give a stable engine gain.

## Prepared but rejected low-rank reader

The width-80/four-head implementation reduces trainable rule parameters to 637,520 from 3,311,360.
Its five-batch CUDA BF16 smoke, complete validation, portable export and candidate validation pass;
all 271 inherited V2 tensors remain bitwise unchanged. It remains implementation evidence, not a
formal version or candidate claim, because the gate failed. Reusing it requires a new explicit
decision based on new data or a new hypothesis.

Authoritative reports and run IDs are in [`../evaluation/index.html`](../evaluation/index.html).
