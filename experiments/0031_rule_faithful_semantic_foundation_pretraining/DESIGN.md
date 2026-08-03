# 0031 Rule-Faithful Semantic Foundation Pretraining

Status: **training ready; formal dataset materialization and formal training have not started**.

## Evidence boundary

Official general rules follow `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`. Concrete card semantics come from the current unmodified official engine runtime and `official_full_engine_prototypes_v2.json`. Choices about what the actor should learn are project policy, not official rules.

The canonical regression case is Team Rocket's Energy (card 15): engine mask `Psychic | Darkness = 80`, `energy_count = 2`, `only_team_rocket = true`, and skill 8. The model receives those static facts, the physical attached card ID/serial/relation, and the Pokemon's observation-level resolved energy-unit facts. Because the API does not guarantee a per-card partition of the flattened resolved-unit list, a child Energy node's dynamic resolved value is `UNKNOWN`, not a fabricated mapping. It never receives a false Colorless replacement.

## Actor schema

Schema: `0031_rule_faithful_semantic_decision_v1`.

| Family | categorical | numeric | field state | relations |
|---|---:|---:|---:|---|
| global | 11 | 19 | 19 | - |
| card instance | 22 | 10 | 10 | `card_parent` |
| own resource ledger | 4 | 15 | 15 | - |
| bounded event | 16 | 8 | 8 | `event_source`, `event_target` |
| legal option | 13 | 10 | 10 | source, target, skill, effect |

Every physical Pokemon, attached Energy, Tool, pre-evolution card, visible card, stadium, and persistently known opponent-hand card is an independent token. Tokens carry owner/controller, zone, slot, serial, and typed parent edges. The state encoder passes context from parent to child and aggregates children back into the parent, so equal card IDs with different attachments remain distinguishable.

Numeric zero is never used to mean missing. Each numeric field has `PAD`, `PRESENT`, `UNKNOWN`, or `NOT_APPLICABLE`, and each scalar has its own projection. Enums and booleans use independent embeddings. Bitmasks retain the exact raw value and named bits. Target Areas and Target Conditions retain their original ordered values.

## Static prototypes

The v2 asset contains 1,267 cards, 433 skills, 1,556 attacks, and 3,067 flattened ordered effects. The prototype encoder consumes only v2 engine facts for rule semantics. Card, attack, skill, trigger, effect, target, and condition coverage is audited by `contracts/prototype_field_inventory.json`.

Attack inputs include base damage and the ordered required Energy symbols. Effect constants, coefficients, comparators, targets, and ordered areas/conditions are facts. The actor must learn arithmetic and interaction rules itself.

## Explicit exclusions

The actor does not receive typed/total Energy deficit, surplus Energy, preferred removal, attach/remove counterfactuals, `newly_enabled`, post-damage HP, or KO labels. It also does not receive source/team/persona identity. These exclusions prevent feature code from becoming a partial and potentially wrong rules engine.

## Model

Default configuration is `d_model=320`, 8 heads, four state layers, three option cross-attention layers, and an ordered legal-option pointer decoder with STOP. Default capacity is 55,868,802 parameters against the current v2 ontology. Prototype embeddings are shared once per forward.

Data flow is: full-engine prototypes plus instance graph plus causal ledger/events, then state transformer, option-to-instance/prototype binding, option cross-attention, and ordered pointer decoding.

## Training contract

BC uses one optimization pass over train per epoch and a complete teacher-forced plus greedy validation pass per epoch. Checkpoints are model-only. Formal runs use private W&B project `dragon_bra/pokemon-tcg-policy-learning` and strict `V<n>_<tag>` directories. Source provenance remains in manifests/audits and never reaches actor forward.

No formal 0031 version exists yet. A later formal start must first materialize a new immutable 0031 dataset with the v2 prototype commitment, then allocate `V1_rule_faithful_foundation`; it must not reuse a 0025/0028 canonical shard because the actor schema changed.

## Verification

All 38 project unit tests pass. Eight consecutive decisions from a real 0025 raw Episode compile under the 0031 causal ledger; a four-row collated batch produces finite logits. The noncanonical CUDA smoke at `.tmp/0031_rule_faithful_bc_smoke/run-088a7d1cc1` completed two optimizer updates, two validations, TensorBoard logging, and finite model-only checkpoint retention with W&B disabled.
