# 010 - Train the complete existing M5 feature ladder before F1

Date: 2026-07-26
Status: accepted

F1 and its richer feature-dataset contract are deferred. The next experiment tests whether the
feature direction already materialized in `V3_model_ready_semantic_v2` produces a meaningful BC
learning change before investing in a new semantic superset dataset.

The formal variant is M5 from the existing M0-M5 ladder. It cumulatively enables entity and option
numeric/64-wide semantic features, registered-deck summary and memory, causal own-resource ledger,
actor-visible event tokens, directed typed entity-relation attention bias, four-role Goal-QKV over
deck+ledger resources, and the finite value interface. Training remains BC-only and the value loss
weight remains zero. Goal queries remain the ladder's pre-Transformer state representation; this
run does not add F1 board-conditioned queries, richer option-conditioned semantics, new event or
relation schemas, or serial-removal changes.

Before allocation, two model-path defects were corrected without changing the accepted V3 feature
dataset: the M2 registered-deck masked summary now accumulates through M3-M5, and M5 relation types
are applied as per-head directed attention bias at their entity source/target pairs instead of being
collapsed to a broadcast scalar. Regression tests bind cumulative deck behavior, relation feature
isolation, and relation endpoint topology.

A real V3 train batch of 64 decisions completed M5 AMP forward, backward, clipping, and optimizer
update with finite loss 1.758117, pre-clip gradient norm 4.259705, 1.804 GiB peak CUDA allocation,
and 2.205 GiB peak reservation while the long M0 run remained active.

The formal version is `V6_m5_full_feature_ladder`, independently initialized with seed 20260726. It
uses the same V3 model-ready dataset, compiler, split, ordered action contract, batch size 64,
learning rate 3e-4, weight decay 1e-2, gradient clipping 1.0, AMP, deterministic settings, complete
per-epoch train/validation evaluation, and a 100-epoch ceiling as the concurrent M0 convergence run.
M1-M4 are not prerequisites for this direction check. Offline metrics remain imitation evidence,
not official-engine policy-strength evidence.
