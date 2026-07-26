# 004 - Correct chronological identity and maintain exact hidden-zone resources

Date: 2026-07-26
Status: accepted

## Findings

Real records serialize decision identity as flat fields `date`, `episode_id`, `episode_step`, and
`player_index`. The chronological batcher expected a nested `identity.source`, so every real record
previously entered the isolated fallback instead of reusing its episode-player causal session.

After correcting the key, real multi-decision compilation exposed a second defect: a complete
`select.deck` counter was retained unchanged across later draws and moves. This caused stale hidden
zone identities and resource-conservation failures. A temporary current-decision-only fallback was
rejected because it discarded valid knowledge that can be maintained causally.

## Decision

- Key causal sessions by flat `(date, episode_id, player_index)` while retaining nested source
  compatibility for synthetic/legacy records.
- Treat a deck view as verified complete only when its length equals the actor's current
  `deckCount`; an empty or partial selection list is not exact membership unless the deck count is
  also zero.
- Initialize exact deck identity counts and infer Prize identity counts only when registered-deck,
  serial-deduplicated visible resources, complete deck membership and aggregate Prize count conserve
  the 60-card registration.
- Apply actor-visible official engine transitions using the read-only contracts `LogType::Draw=4`,
  `MoveCard=6`, `MoveCardReverse=7`, `Shuffle=0` and `AreaType::Deck=1`, `Prize=6`.
- Draw and identified MoveCard update the affected exact counters. Shuffle removes order knowledge
  but preserves deck membership. An unidentified reverse move degrades only the hidden zone it
  touches.
- Count Active/Bench cards recursively with `energyCards`, `tools`, and `preEvolution`; also count
  actor-owned Stadium and the resolving `select.contextCard`, deduplicated by physical serial.
- Validate aggregate deck and Prize sizes and per-card conservation every decision. Contradiction
  degrades exact epistemic state rather than emitting stale exact values.

## Evidence and consequence

A real 8-worker smoke compiled all 10,000 decisions in the first train raw shard, covering 135
chronological episode-player groups, in 14.34 seconds without a conservation failure. Compiler v2
uses a transitive digest that includes the chronological batcher and causal knowledge sources, so
this correction necessarily creates a new materialized dataset commitment. All earlier
model-ready staging remained unpublished and was audited before removal.
