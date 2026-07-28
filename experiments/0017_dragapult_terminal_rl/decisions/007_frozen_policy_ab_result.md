# 0017 frozen-policy A/B result

Verdict: **improved**

This is an independently randomized, opponent/seat-balanced A/B evaluation, not a shared-seed paired test. The unmodified official binary uses `random_device` internally.

- Baseline report: `experiments/0017_dragapult_terminal_rl/evaluation/V17_bc_source_eval_10g.html`
- Candidate report: `experiments/0017_dragapult_terminal_rl/evaluation/V18_rl_update15_eval_10g.html`

| Group | BC wins/games | BC rate | RL wins/games | RL rate | RL − BC (95% CI) | p |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 41/260 | 15.77% | 68/260 | 26.15% | 10.38% [3.39%, 17.27%] | 0.003627 |
| 20-opponent train pool | 40/200 | 20.00% | 59/200 | 29.50% | 9.50% [1.03%, 17.80%] | 0.02771 |
| 6-opponent holdout | 1/60 | 1.67% | 9/60 | 15.00% | 13.33% [3.37%, 24.53%] | 0.008234 |
| Candidate first | 24/130 | 18.46% | 33/130 | 25.38% | 6.92% [-3.16%, 16.86%] | 0.1773 |
| Candidate second | 17/130 | 13.08% | 35/130 | 26.92% | 13.85% [4.12%, 23.33%] | 0.005258 |

The complete-pool result is a statistically detectable increase of 27 wins, or `+10.38` absolute
percentage points. The 95% Newcombe interval is `[+3.39, +17.27]` points and the independent
two-proportion score test gives `p=0.00363`. Both policies completed every game with zero errors.
The point estimate improves in both seats, although the 130-game first-seat slice is individually
inconclusive. The six opponents excluded from PPO collection also improve as a group, so this
particular run does not show train-pool-only overfitting.

## Interrupted larger-run evidence

The user reduced the requested scale from 100 to 10 games per opponent. V15/V16 were therefore
stopped before publication and remain marked `interrupted`; their `/tmp` result fragments are not
formal reports. They nevertheless provide a useful independent consistency check on the first four
catalog opponents, for which both arms had completed exactly 100 games:

| Opponent | BC | RL update 15 |
|---|---:|---:|
| `alakazam_dudunsparce_04_sota` | 2/100 | 21/100 |
| `alakazam_dudunsparce_02` | 30/100 | 52/100 |
| `alakazam_dudunsparce_01` | 29/100 | 49/100 |
| `alakazam_dudunsparce_03_bc` | 29/100 | 56/100 |
| Four-opponent total | 90/400 (22.5%) | 178/400 (44.5%) |

This truncated evidence is catalog-order biased and cannot replace the 26-opponent reports. It does,
however, make the formal improvement much less plausibly a lucky ten-game fluctuation in the
Alakazam matchups. Crustle evidence was incomplete across arms and is not included in this table.

## Weight-change audit

The update-15 actor retained all 271 BC actor tensors with matching names and shapes. Exactly 14
tensors changed, all inside the allowed pointer decoder (`pointer_key`, `pointer_query`, `decoder`,
`decoder_init`, `option_bias`, and `stop`); the other 257 actor tensors were bitwise unchanged. The
actor-relative L2 displacement was `0.000585` (`0.0585%`), absolute delta L2 was `1.5676`, and the
largest scalar change was `0.01013`. This confirms a conservative decoder-only policy update rather
than accidental feature-trunk drift. The value head has six separate tensors and is not used by the
exported greedy action policy.

## Per-opponent audit

| Opponent | BC | RL | Difference | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| `alakazam_dudunsparce_01` | 20.00% | 40.00% | 20.00% | [-18.70%, 52.11%] | 0.3291 |
| `alakazam_dudunsparce_02` | 40.00% | 50.00% | 10.00% | [-28.98%, 45.09%] | 0.6531 |
| `alakazam_dudunsparce_03_bc` | 20.00% | 30.00% | 10.00% | [-26.46%, 43.54%] | 0.6056 |
| `alakazam_dudunsparce_04_sota` | 10.00% | 30.00% | 20.00% | [-15.98%, 51.41%] | 0.2636 |
| `crustle_01` | 20.00% | 10.00% | -10.00% | [-42.05%, 23.62%] | 0.5312 |
| `crustle_02` | 0.00% | 40.00% | 40.00% | [3.84%, 68.73%] | 0.02535 |
| `cynthias_garchomp_ex_roserade_01_bc` | 0.00% | 0.00% | 0.00% | [-27.75%, 27.75%] | 1 |
| `cynthias_garchomp_ex_roserade_02_bc` | 0.00% | 30.00% | 30.00% | [-3.76%, 60.32%] | 0.06029 |
| `dragapult_ex_01` | 10.00% | 30.00% | 20.00% | [-15.98%, 51.41%] | 0.2636 |
| `dragapult_ex_02` | 20.00% | 20.00% | 0.00% | [-34.14%, 34.14%] | 1 |
| `ionos_bellibolt_ex_kilowattrel_01` | 30.00% | 20.00% | -10.00% | [-43.54%, 26.46%] | 0.6056 |
| `marnies_grimmsnarl_ex_dudunsparce_01` | 20.00% | 30.00% | 10.00% | [-26.46%, 43.54%] | 0.6056 |
| `marnies_grimmsnarl_ex_froslass_01` | 70.00% | 60.00% | -10.00% | [-44.57%, 28.17%] | 0.6392 |
| `marnies_grimmsnarl_ex_froslass_02` | 30.00% | 20.00% | -10.00% | [-43.54%, 26.46%] | 0.6056 |
| `marnies_grimmsnarl_ex_froslass_03_bc` | 10.00% | 10.00% | 0.00% | [-31.50%, 31.50%] | 1 |
| `marnies_grimmsnarl_ex_froslass_04_bc` | 0.00% | 0.00% | 0.00% | [-27.75%, 27.75%] | 1 |
| `marnies_grimmsnarl_ex_froslass_05_bc` | 0.00% | 20.00% | 20.00% | [-11.24%, 50.98%] | 0.136 |
| `mega_abomasnow_ex_kyogre_01` | 10.00% | 20.00% | 10.00% | [-23.62%, 42.05%] | 0.5312 |
| `mega_lucario_ex_solrock_01` | 20.00% | 10.00% | -10.00% | [-42.05%, 23.62%] | 0.5312 |
| `mega_lucario_ex_solrock_02` | 0.00% | 30.00% | 30.00% | [-3.76%, 60.32%] | 0.06029 |
| `mega_lucario_ex_solrock_03` | 20.00% | 20.00% | 0.00% | [-34.14%, 34.14%] | 1 |
| `mega_lucario_ex_solrock_04` | 20.00% | 20.00% | 0.00% | [-34.14%, 34.14%] | 1 |
| `mega_lucario_ex_solrock_05` | 10.00% | 50.00% | 40.00% | [-0.24%, 67.59%] | 0.05096 |
| `mega_lucario_ex_solrock_06` | 10.00% | 50.00% | 40.00% | [-0.24%, 67.59%] | 0.05096 |
| `mega_lucario_ex_solrock_07_bc` | 0.00% | 10.00% | 10.00% | [-18.94%, 40.42%] | 0.3049 |
| `team_rockets_mewtwo_ex_spidops_01_bc` | 20.00% | 30.00% | 10.00% | [-26.46%, 43.54%] | 0.6056 |

## Decision rule

RL is called improved only when the overall 95% difference interval excludes zero in the positive direction, both arms complete without runtime errors, and the result is not solely attributable to one seat or one matchup. Training rolling-window peaks are not used in this verdict.

The per-opponent p-values above are descriptive and are not corrected for 26 simultaneous tests;
no individual matchup claim is made from ten games. The primary conclusion concerns the equally
weighted frozen 26-opponent pool. Because the official binary internally uses `random_device`, this
is not a shared-seed causal pair and cannot prove that every deck shuffle would improve.
