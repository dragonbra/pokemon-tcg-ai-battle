# Initial 0028 Contract

Date: 2026-08-02

## Decision

0028 is the self-contained, readable universal successor to the 0025 V4 canonical semantic
foundation. The project number is intentionally 0028, not 0027, to avoid a collaboration conflict.

The actor semantics remain those proven by 0025 V4:

- official Card, Attack, Skill, and Effect prototype facts;
- actor-visible global state, card instances, exact registered-deck multiset, causal ledger, and
  recent visible events;
- legal options bound to source/target instances and prototype identities;
- ordered legal-option pointer decoding with an explicit STOP action;
- no source, team, expert, or persona identity in actor forward.

The training corpus changes from the single exact James Cox Raging Bolt deck to every audited
unique positive terminal winner perspective in official Episode archives from 2026-07-10 through
the latest locally available archive, currently 2026-08-01. Source identity remains provenance for
audit, split reporting, conflict analysis, and sampling only.

## Implementation Boundary

0028 may copy and then independently maintain stable 0025 V4 concepts and immutable prototype
assets, but it must not import executable code from 0025 or any other numbered project. The rewrite
organizes code so the complete forward path can be read in order:

1. validate the named tensor batch;
2. encode official prototypes;
3. encode full state memory;
4. encode and cross-attend legal options;
5. autoregressively decode ordered option indices and STOP.

## Resource Gate

The formal 0026 RL run remains active. Until it completes, 0028 work is limited to documents,
contracts, unit tests, static assets, and small single-worker smoke tests. Full 23-archive scanning,
dataset materialization, CUDA training, and official-engine evaluation wait for resource release.
