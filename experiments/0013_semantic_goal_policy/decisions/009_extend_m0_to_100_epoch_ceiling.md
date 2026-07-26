# 009 - Extend reproducible M0 to a monitored 100-epoch ceiling

Date: 2026-07-26
Status: accepted

`V4_m0_bound_seed` completed its three-epoch minimum allocation successfully. Validation loss
improved monotonically from 1.12633 to 1.05466 to 1.00068, while validation exact action improved
from 0.55279 to 0.58960 to 0.61434. This establishes pipeline and reproducibility health but does
not establish convergence.

The next formal version is `V5_m0_100epoch_monitor`. It restarts from the same bound seed and initial
model state rather than resuming V4. Dataset, model variant M0, batch size 64, learning rate 3e-4,
weight decay 1e-2, gradient clipping 1.0, AMP, deterministic settings, compiler, split, and action
contract remain unchanged. The sole training-contract change is the maximum epoch count from 3 to
100.

No automatic early stopping is enabled. Every epoch retains complete train and validation metrics
and an immutable checkpoint. The run will be monitored for sustained validation degradation while
training loss/exact action continues to improve. A manual stop requires evidence across complete
epoch summaries and will be recorded in immutable version status; a single noisy epoch is not
sufficient. This long M0 run remains a within-0013 convergence baseline, not an 0012-matched causal
control.

W&B progress records now retain the canonical `progress/*` namespace and use
`progress/iteration` as their axis. Epoch summaries continue to use `trainer/epoch`.
