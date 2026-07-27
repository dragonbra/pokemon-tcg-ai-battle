# Decision 009: Stop T4 dosage sweep and run one source-scale follow-up

Date: 2026-07-27

## Decision

Keep `V2_t1_plus_pure_dragapult` as the selected supported policy. Stop the Marnie/Munkidori data
cap sweep after V15 and V16. Permit exactly one V17 run on the V16 dataset with source embedding
initial scale `0.03`; keep all other data, model, optimizer, seed, validation and checkpoint
selection contracts unchanged. Do not continue a source-scale sweep after V17.

## Source identity and data audit

The complete Marnie candidate pool contains 4,672 trajectories from 3,488 episodes. The 191 rows
reported before the fix were only the subset whose seven exact Unicode player identities collided
under the old ASCII-normalizing source key; they were not the whole Marnie pool. Dataset generation
now preserves source IDs 1–84 and assigns stable hashed Unicode source IDs 85–91.

V15 uses the deterministic outcome-stratified 535-trajectory cap: 267 wins, 267 losses and one
draw, contributing 48,793 decisions. V16 uses 267 trajectories selected by the same ordering and
is a strict subset of V15, contributing 24,555 decisions. Both retain the unchanged 12-episode,
1,000-decision THIRD PTCG Club validation split.

## Formal evidence

| Version | T4 trajectories | Target exact | Best epoch | Official engine |
|---|---:|---:|---:|---:|
| V2 | 0 | 58.7% | 10 | 39/200 (19.5%) |
| V15 | 535 | 57.5% | 4 | 40/200 (20.0%) |
| V16 | 267 | 60.3% | 7 | 37/200 (18.5%) |
| V17, source scale 0.03 | 267 | 59.9% | 9 | 37/200 (18.5%) |

All formal evaluations completed 200/200 games with zero errors against the same fixed catalog.
V15 is only one win above V2 and V16 is two wins below it. The dosage change strongly moves the
offline metric but does not establish a rollout-strength gain, so more cap tuning is not justified.

A separate read-only V2 loss-best diagnostic scored 26/200 (13.0%) with zero errors. This rejects
validation-loss checkpoint selection for project 0015; formal candidates continue to use the
predeclared best greedy exact checkpoint.

V17 isolates whether the source-conditioning signal begins too strongly by changing only source
initial scale `0.10 → 0.03` on the V16 half-cap data. Its preflight smoke and formal run pass with
finite loss and 100% legal actions. The exact-best epoch reaches 59.9%, but official-engine strength
is again 37/200 (18.5%) with zero errors. Lower source scale therefore does not recover an uplift;
source-scale exploration stops and V2 remains selected.

Authoritative reports are
[`V15_t4_marnie_munkidori.html`](../evaluation/V15_t4_marnie_munkidori.html) and
[`V16_t4_marnie_half_cap.html`](../evaluation/V16_t4_marnie_half_cap.html), and
[`V17_t4_low_source_scale.html`](../evaluation/V17_t4_low_source_scale.html).
