# Decision 008: Reject label smoothing and checkpoint averaging

Date: 2026-07-27

## Decision

Keep `V2_t1_plus_pure_dragapult` as the selected policy. Stop the label-smoothing direction after
the single predeclared `0.05` run, and do not allocate a formal version for the diagnostic V2
checkpoint average. The available-source campaign is complete; another formal run requires a new
data source or a materially different, predeclared hypothesis.

## Label-smoothing implementation finding

The first two CUDA smoke runs failed with a nonfinite training loss. The pointer decoder masks
illegal classes with `torch.finfo(dtype).min`; ordinary vocabulary-wide smoothing assigns positive
target mass to those masked classes, making the smoothed cross-entropy nonfinite. This was a loss
contract defect rather than GPU instability or corrupt data.

The corrected implementation smooths only over finite legal classes. It excludes padding and mask
sentinels below half the dtype floor, and asserts that every target is a legal class. Validation,
checkpoint selection and inference remain unsmoothed. The failed smoke artifacts remain under
`.tmp/0015_dragapult_conditioned_bc/` as local diagnostic evidence.

## Formal V14 evidence

| Version | Contract | Target exact | Validation loss | Official engine |
|---|---|---:|---:|---:|
| V2 | plain R15 T1, seed 20260723 | 58.7% | — | 39/200 (19.5%) |
| V14 | plain R15 T1, legal-class smoothing 0.05 | 59.0% | 0.584532 | 30/200 (15.0%) |

V14 completed all 200 official-engine games with zero errors. Its small offline increase did not
transfer to rollout strength, so no epsilon sweep is justified.

## Checkpoint-average diagnostic

A read-only GPU diagnostic averaged the V2 epoch 9, 10 and 11 model weights. On the unchanged full
validation split it reached 59.0% greedy exact with loss `0.597316` and 100% legal actions. This is
only +0.3 percentage points over the selected V2 checkpoint and matches the offline level of the
already rejected V14. It therefore does not pass the threshold for a formal version or a 200-game
evaluation.

The authoritative V14 report and run ID are in
[`../evaluation/V14_t1_label_smoothing_005.html`](../evaluation/V14_t1_label_smoothing_005.html),
with the campaign overview in [`../evaluation/index.html`](../evaluation/index.html).
