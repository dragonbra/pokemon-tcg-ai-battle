# Decision 004 — A0 progress telemetry and from-zero long run

## Decision

Run the primary A0 control from epoch 0 for at most 100 epochs, with validation greedy
exact-action early stopping (`patience=12`, `min_delta=0.001`). Keep one complete validation
snapshot per epoch and do not add a train greedy pass.

The active version is `V8_0010_faithful_100epoch_strict_epoch_axis`. Its W&B name is
`0014 · faithful_board_causal_features · V8_0010_faithful_100epoch_strict_epoch_axis`.

## Telemetry contract

- stdout uses a bounded-refresh TQDM bar for train batches with running loss and decisions/s;
- canonical JSONL/TensorBoard metrics include `progress/iteration`, global `progress/fraction`,
  current-epoch `progress/epoch_fraction`, decisions/s, iterations/s, elapsed seconds and stage
  code; W&B Logs show the same bounded-refresh TQDM output;
- batch progress is not mirrored into W&B scalar history. W&B mirrors the epoch-0 start record and
  one complete record per epoch, making internal `_step` equal `trainer/epoch`;
- exact-action is printed in the epoch-complete validation record only.

## Failed attempts retained

V2 and V3 exposed warm-start optimizer-device handling; V4 was manually interrupted after
epoch 8 while comparing warm-start throughput; V5 confirmed the TQDM refresh interval was too
aggressive in non-interactive logs; V6 and V7 diagnosed W&B `_step` inflation caused by batch
progress history. Their version directories and W&B runs remain immutable audit records. V8 does
not warm-start and uses the strict epoch-axis telemetry.
