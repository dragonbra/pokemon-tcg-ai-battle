# V2 GPU Forward Stop And V3 Gate

V2 was stopped before epoch 1 with zero checkpoints so GPU-forward optimization could be measured
without mixing implementations in one version.

Profiler evidence shows masked attention already uses fused memory-efficient SDPA. Flash SDPA is
not eligible with the current non-null padding mask, and forcing it raises `No available kernel`.
The profile is dominated by fragmented operator scheduling as well as attention: three train steps
recorded about 685 ms CPU scheduling and 286 ms aggregate CUDA kernel time.

V3 retains only algebraically equivalent common-subexpression elimination: encode each finite
prototype table once per forward, gather it at all state/option sites, and reuse option pointer keys
and bias across teacher-forced action steps. Direct/shared policy logits must remain within `1e-6`
with identical argmax; cached/uncached decoder logits remain bitwise-equal regression contracts.
Fused categorical lookup and fused AdamW were
rejected after failing to improve throughput. `torch.compile` was also rejected because the full
teacher step did not complete reliably at representative batch sizes after toolchain preflight.

After concurrent GPU evaluation exited, the clean matched 100-batch gate measured 468.0
decisions/s for direct prototypes at batch 256, 515.7 for shared prototypes at batch 256, and
609.7 for shared prototypes at batch 512. The selected configuration is 30.3% faster than the
direct baseline and reserves 12.17 GB on the 16.3 GB RTX 5080. This authorizes the fresh formal
version `V3_shared_prototype_batch512`.
