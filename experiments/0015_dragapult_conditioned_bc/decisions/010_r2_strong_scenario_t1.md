# Decision 010: Keep R15 after the R2 strong-scenario T1 comparison

Date: 2026-07-27

## Decision

Keep `V2_t1_plus_pure_dragapult` as the strongest supported 0015 policy. Do not replace it with
`V18_r2_strong_scenario_t1`, and do not infer that the 0014 Alakazam R2 ranking transfers directly
to Dragapult. V18 is a completed negative architecture comparison, not a failed runtime.

## Controlled contract

V18 changes the T1 model family from the source-conditioned R15 baseline to the complete
`0014/V12_r2_strong_scenario_scalegate` trunk. It retains the same 54,764 training decisions,
1,000-decision target validation split, dataset SHA-256
`32e8c483e5a8f76d20b4c559270349b0bf621b9c2f2bf4c1e3c5e3badc2d2f3d`, source vocabulary,
target persona, seed, optimizer, outcome weights, label smoothing and exact-best selector as V2.
The model is freshly initialized; it does not load the Alakazam V12 checkpoint.

The input contract remains the same 22 actor-visible tensor groups: nine board/legal-option groups
and thirteen zone/deck/ledger/event/opponent-hand groups. A separate `[B]` `source_id` conditions
the declared expert persona. The R2 trunk uses a two-layer scenario encoder, independent Goal-QKV,
state/option FiLM and dynamic per-channel ScaleGates initialized to 1.0 in `(0, 2)`. The unchanged
0015 persona residual is applied afterward at initial scale 0.10. Total parameters are 17,416,642.

## Evidence

Formal CUDA BF16 training stopped after epoch 15. Epoch 10 was selected by target validation
greedy exact at 58.4%; legal action was 100%. W&B online sync completed successfully. The frozen
portable candidate passed package validation and completed the same 20-opponent official-engine
catalog at 10 games per opponent with zero errors.

| Version | Architecture | Target exact | Best epoch | Official engine |
|---|---|---:|---:|---:|
| V2 | source-conditioned R15 | 58.7% | 10 | 39/200 (19.5%) |
| V18 | source-conditioned R2 strong scenario | 58.4% | 10 | 27/200 (13.5%) |

V18 is lower on both axes: -0.3 percentage points target exact and -6.0 percentage points
official-engine win rate. This run therefore provides no evidence that R2's stronger dynamic
scenario path improves the T1 Dragapult policy. R15/V2 remains selected; no R2 sweep is justified
from this result.

The authoritative report is
[`V18_r2_strong_scenario_t1.html`](../evaluation/V18_r2_strong_scenario_t1.html), run
`run-fea4347c833c45d9a44c5c54cbc98040`.
