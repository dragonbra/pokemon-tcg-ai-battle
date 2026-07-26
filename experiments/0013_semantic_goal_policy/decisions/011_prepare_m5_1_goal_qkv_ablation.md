# 011 - Prepare M5.1 Goal-QKV leave-one-out without launching it

Date: 2026-07-26
Status: prepared; formal allocation not authorized

`M5.1` is registered as the single-module leave-one-out proposed for interpreting M0-to-M5 gains.
It instantiates exactly the same modules, parameter shapes, and initialization order as M5. With an
identical seed, M5 and M5.1 must have identical state-dict tensors and initial model SHA-256.

M5.1 retains M5 entity and option numeric/64-wide semantics, registered-deck masked summary,
causal ledger state tokens, actor-visible event tokens, directed typed entity-relation attention
bias, ordered full-action decoder, and finite uncalibrated value interface. It only disables the
Goal-QKV forward path and omits the four goal tokens from state encoding. Goal-QKV parameters remain
present but disconnected from the policy computation graph; M5.1 returns the standard zero goal
output. BC remains the only objective and value loss remains zero. A temporary real V3 batch-64 AMP
forward/backward/optimizer smoke used the same initial SHA as M5, produced finite loss 1.759639 and
pre-clip gradient norm 4.244364, confirmed all Goal-QKV gradients remained disconnected, and peaked
at 1.742 GiB CUDA allocation. It wrote no run artifacts.

This preparation does not authorize a formal run. The frozen protocol with SHA-256
`da6482cf2d4cdc8d9e56fd4c03431e60dbf63b8327720c83185e907a039d44d3` explicitly allocates M0-M5
and does not include M5.1. Both command entry points reject M5.1 when bound to that protocol. If the
experiment is approved later, it requires a new immutable ablation protocol, the next unused
repository version, a new W&B run, and the same V3 dataset/seed/batch/optimizer/evaluation contract
as M5. No version directory, checkpoint, metrics file, W&B run, or GPU process is created by this
preparation.
