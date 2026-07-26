# 003 - Materialize the shared feature dataset before formal training

Date: 2026-07-26
Status: accepted

## Evidence

Formal `V1_m0_causal_contract` passed all correctness gates and began epoch 1, but its raw-record
loader repeated gzip/JSON parsing, chronological causal-state reconstruction, typed compilation and
collation inside every optimization and evaluation pass. At the last observation it was reading
`train-00014.jsonl.gz`, used about one CPU core and left the GPU near the host graphics baseline.
No complete epoch metric or checkpoint had been written.

## Decision

Stop V1 and preserve it as a failed formal version with failure class
`input_pipeline_throughput_defect`. Do not reuse its artifact, checkpoint, TensorBoard, W&B or
version name.

Publish a separate immutable model-ready feature dataset derived from the raw v3 decisions. It must:

- reconstruct each episode-player causal state exactly once before training;
- compile the common M0-M5 typed and semantic feature families exactly once;
- bind the raw dataset content, action/schema contract, transitive compiler digest and ontology;
- store compact int16 categorical/target tensors, float16 numeric/semantic tensors, masks and sparse
  relation edges in hash-committed PyTorch tensor shards;
- use bounded ordered parallelism across independent episode-player groups while preserving strict
  chronology within each group;
- validate every shard hash, count, field, shape and dtype before formal version allocation;
- make the training hot path only shuffle tensor shards/rows, apply synchronized option permutation,
  reconstruct sparse relations, prefetch pinned batches and transfer them asynchronously.

Replace per-decision free-running metric decoding with an equivalent batched deterministic decoder.
Create W&B at run start and report throttled train, train-eval and validation-eval progress with
iterations/second, decisions/second, fraction and elapsed time. Local JSONL remains canonical and
TensorBoard remains a required audit mirror; W&B is the user-facing live interface and receives
records directly through its SDK rather than by reading TensorBoard.

## M0 boundary

Do not start M1-M5. After model-ready publication, run real tensor parity and throughput smokes,
retain the established baseline optimization hyperparameters, allocate the next strictly increasing
formal version, and complete M0 first.
