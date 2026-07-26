# 002 - Start formal M0 and isolate semantic materialization defect

Date: 2026-07-26
Status: accepted

## Decision

Publish raw causal decision dataset `V1_causal_semantic_v1` as `dataset_reference_v3` with content
SHA-256 `146ac0799b0a23a5089615361e76c33722c08df45b276cfd2b387171c70dc0c7`.
The validated dataset contains 145,961 train and 16,167 validation decisions from 2,110 eligible
complete Yushin-winner episode-player groups.

After full dataset validation, a 256-decision M0 runtime smoke measured
93.91400620694115 decisions/second on the local RTX 5080. The frozen protocol therefore resolves
the runtime floor to 10 decisions/second. A real 64-record forward/backward/optimizer smoke passed
with finite loss and gradients. Allocate and start formal version `V1_m0_causal_contract` for three
complete epochs with W&B online.

The first real semantic-materialization smoke found an out-of-range move semantic bucket at index
64. M0 does not consume semantic vectors by the frozen variant matrix, so V1 M0 explicitly compiles
the M0 baseline with semantic materialization inactive while still validating the bound ontology
and compiler provenance. This defect blocks M1-M5. It must be fixed under a newly bound compiler
version and audited before any M1 formal version is allocated. It must not be patched into the
running immutable V1 contract.

## Consequences

- V1 remains the formal M0 new-contract baseline, not an M1 semantic result.
- Dataset raw records and split remain valid; no partial shard was reused.
- M1 cannot start merely because V1 M0 finishes.
- The M1 gate requires a new compiler digest, corrected 64-wide semantic bucketing, real-record
  full materialization audit, offline/runtime parity, and an explicit version decision.
- No policy-strength claim exists until a self-contained candidate is evaluated with the official
  engine runtime.
