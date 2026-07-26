# 006 - Accept V3 model-ready features for formal M0

Date: 2026-07-26
Status: accepted

`V3_model_ready_semantic_v2` was built from commit `dfbb667` with eight ordered episode-player
workers. It atomically published 143 train and 16 validation tensor shards containing 145,961 and
16,167 decisions. Compiler and materializer start/end digests matched. The dataset content SHA-256
is `a9a7f70190598dbcbaa7b12266e9997f8773fa596401a92f6e03151b9bc442f6`.

Independent validation rehashed and weights-only loaded all 6.2 GiB in 4.90 seconds. A 1,024-record
raw-to-cache parity audit found exact categorical, mask, target, relation, and numeric tensors; the
maximum expected float16 semantic storage error was 0.00021827220916748047. No partial or
undeclared file exists.

At batch size 64, the model-ready input path delivered 5,765 decisions/second while 50 real M0
forward/backward/optimizer steps delivered 638 decisions/second and 9.97 iterations/second with
finite loss and 1.10 GiB peak PyTorch CUDA allocation. The input path is no longer the bottleneck.
A complete 16,167-decision untrained validation pass completed with 100% decoder legality. The
single-decision runtime smoke measured 156.31 decisions/second, resolving the frozen runtime floor
to 10 decisions/second.

V3 is accepted as the only model-ready dataset eligible for the next formal M0 allocation. V2
remains invalid. Formal M0 retains batch size 64, learning rate 3e-4, weight decay 1e-2, seed
20260726, three complete epochs, complete train/validation evaluation every epoch, and online W&B.
W&B must initialize at run start and receive throttled train, train-eval and validation-eval progress
with iterations/second and decisions/second.
