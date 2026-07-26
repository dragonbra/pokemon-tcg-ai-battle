# 012 - Replace static train evaluation with online optimization metrics

Date: 2026-07-26
Status: accepted

Future 0013 formal BC versions consume the train split exactly once per epoch. The optimization
forward computes teacher-forced logits for the BC loss; token accuracy and teacher exact-action are
aggregated from those same logits under `bc/optimization/*` before backward updates the model. These
values include randomized option permutation, training-mode dropout, and successive parameter
states within the epoch. They are optimization-health diagnostics, not a fixed-model train score and
not policy-strength evidence.

The trainer no longer replays the complete train split after optimization and does not run greedy
full-action decode during the training pass. The epoch-end fixed model still evaluates the
complete validation split every epoch with teacher-forced and greedy full-action metrics.
Checkpoint selection continues to use `bc/validation/loss` and `bc/validation/exact_action`;
official-engine arena results remain the only policy-strength evidence.

This decision supersedes the future-run metric requirement in decisions 003, 006, and 009 without
rewriting those historical records. V5 and V6 were launched under the former complete
train/validation snapshot contract and retain their existing artifacts and already-loaded runtime
behavior. The new training config records `online_teacher_forced_v1`,
`fixed_snapshot_full_action_v1`, and `full_train_evaluation=false`; the first later formal version
that launches from the updated source must use a new immutable repository version and W&B run.

A follow-up runtime audit found that V3's 1,024-row shard padding leaked into 64-row model batches:
train targets had a row-weighted stored width of 13.93, while an eight-shard sample needed only
5.77 steps at the per-batch maximum. Runtime target trimming preserves all valid targets and masks
without changing V3's frozen materializer SHA. On one real M0 batch, BF16 forward/backward improved
from 56.45 ms to 34.21 ms (`1.65x`). BF16 GradScaler was also removed: a real M0 update
benchmark improved from 18.12 to 20.75 batch/s (`1.15x`), while direct loss/gradient finite checks
and clipping remain. BC-only teacher and greedy paths now skip the inactive value MLP; a real M0
forward/backward benchmark improved by another `1.10x` (M5: `1.02x`), while default runtime
`encode()` retains value output. M0/M1 also no longer compute deck features that are only connected
from M2 onward. Full feature-dimension compaction and fused AdamW were benchmarked but not adopted
because they produced only about 3% and no improvement respectively. Batch-size throughput was
measured on real V3 tensors at 64/128/256: M0 reached 1,964/2,555/2,736 decisions/s and M5 reached
844/969/1,085 decisions/s. A 256-row update on the widest measured shard (106 entities and 49
options) remained finite and peaked at 7.23 GiB for M5. By explicit user decision, 256 is therefore
the default for subsequent smoke and formal BC commands. The actual batch size remains recorded in
each run config because this reduces optimizer updates per epoch relative to historical batch-64
runs; V4-V6 retain their original immutable optimization semantics.
