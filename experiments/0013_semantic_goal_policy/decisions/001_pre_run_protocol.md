# 001 — Freeze the pre-run protocol

Date: 2026-07-26
Status: accepted

## Decision

Freeze `train/0013_semantic_goal_policy/configs/pre_run_protocol.json` as the immutable,
canonical contract for every formal dataset build, training, selection, export, and evaluation
command in project `0013_semantic_goal_policy`.

The protocol hash is SHA-256 over UTF-8 JSON serialized with sorted keys, compact separators, no
NaN/Inf, and one trailing newline. A freeze uses exclusive creation and refuses to replace any
existing path. Formal commands must receive the exact canonical hash and fail closed when it is
missing or different. The frozen canonical digest is
`da6482cf2d4cdc8d9e56fd4c03431e60dbf63b8327720c83185e907a039d44d3`.

Ontology identity coverage is measured over the explicit union of registered-deck identities and
observed legal-option identities.

The protocol freezes only the runtime-floor formula
`min(10.0, 0.8 * measured_m0_smoke_decisions_per_second)`. The measured M0 smoke throughput is a
preflight result, not protocol content. Training must receive a finite positive measurement and
resolve the floor before it can proceed; this prevents a result-dependent value from changing the
protocol hash.

## Consequences

Changing any frozen field requires a new protocol rather than overwriting this file. Downstream
formal commands can bind their artifacts to one stable digest. No dataset, training, export, or
formal evaluation result is claimed by this decision.
