# T4 Marnie/Munkidori dataset audit

Date: 2026-07-27

The complete 0726 catalog contains 4,672 qualified Marnie/Munkidori player trajectories from 3,488
episodes. This is much larger than the 191-trajectory bucket reported by Decision 005: that number
covered only seven non-ASCII team names collapsed by the V1 slugger.

T4 keeps every existing target and pure-Dragapult train trajectory. It selects 535 Marnie
trajectories with the fixed `0015_t4_outcome_cap_v1` SHA-256 order, proportionally allocated by
terminal outcome. The selected set contains 267 wins, 267 losses and one draw across 525 episodes,
60 source personas and eight registered-deck hashes. This matches the existing 535 train
trajectories instead of allowing Marnie to dominate them.

All V1 source IDs 1–84 remain unchanged. The seven exact Unicode names receive stable hashed IDs
85–91. The selected cap happens to contain 21 trajectories from four of those seven identities;
the other three remain audited in the candidate pool and reserved in the vocabulary.

Raw extraction produced 103,557 train decisions: 12,988 target Dragapult/Dusknoir, 41,776 pure
Dragapult and 48,793 Marnie/Munkidori. The original 1,000 target validation decisions are unchanged.
Every materialized decision passes the inherited R15 eligibility contract.

Machine-readable counts, hashes and the exact cap contract are in
[`t4_marnie_v2.json`](t4_marnie_v2.json). The immutable source index and dataset references under
`rl_runs/0015_dragapult_conditioned_bc/dataset/` remain the canonical membership evidence.
