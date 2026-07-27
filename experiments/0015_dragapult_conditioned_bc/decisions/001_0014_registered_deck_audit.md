# Decision 001: 0014 registered-deck feature audit

**Date:** 2026-07-27
**Decision:** 0015 may reuse the 0014 AC/R1 registered-deck path as its architectural base. A0 is
not eligible. This is a schema/runtime readiness decision, not evidence of cross-deck skill.

## Verdict

| Question | Verdict | Evidence boundary |
|---|---|---|
| Can the cached schema represent an exact 60-card own deck? | Yes | Unique card IDs plus aligned multiplicities reconstruct the complete multiset. |
| Does A0 use this information? | No | Its feature switches are disabled; it is not a 0015 baseline architecture. |
| Do AC and R1 consume it? | Yes | AC builds deck-resource tokens and goal/scenario conditioning; R1 exposes the tokens as scenario memory. |
| Does exported online inference receive the package deck? | Yes | `deck.csv` is read, required to contain 60 cards, and passed to `OnlineCausalEncoder`. |
| Is a real trained checkpoint sensitive to the channel? | Yes | V9 logits changed when only a real one-card multiplicity difference was substituted. |
| Has broad cross-archetype conditioning been proven? | No | 0014 contains only two highly imbalanced, nearly identical deck signatures. |
| Is 0014 sufficient to start 0015? | Yes, with additions | Preserve AC/R1 deck input; add explicit build/source strata, balanced sampling, ablations and grouped tests. |

## End-to-end trace

1. `features/compiler.py::_deck_cards()` expands `deck_manifest.counts`, rejects anything other
   than 60 cards, and emits sorted `registered_card_ids`, aligned
   `registered_multiplicity`, `registered_mask`, and resource-ledger tensors. IDs plus counts are
   lossless for the registered multiset; order is intentionally irrelevant.
2. `training/ac_data.py` keeps these fields in AC/R1 batches and verifies that registered-deck and
   ledger masks align.
3. `ac_model.py::_resource_tokens()` combines card identity embedding, card ontology, deck/Prize
   knowledge, ledger values, multiplicity and token type. `goal_qkv` retrieves over those tokens;
   the resulting scenario/goal context conditions state and legal-option representations.
4. `r1_model.py` makes registered-deck/ledger tokens explicit scenario-memory entries. State and
   legal options cross-attend this memory, with role routing and FiLM conditioning.
5. `online_runtime.py::OnlineCausalEncoder` receives the actual registered deck, enforces length
   60 and emits the same registered IDs, multiplicities, masks and ledger fields. `ac_inference.py`
   passes the package deck to that encoder; AC candidate export copies and reads `deck.csv`.
6. The V9 AC and V10 R1 model contracts declare `registered_deck: true`. This does not apply to A0.

## Empirical audit of the existing 0014 cache

The complete train/validation cache at
`rl_runs/0014_faithful_board_causal_features/dataset/V1_full_feature_superset` contains 162,128
decisions and exactly two registered-deck signatures:

| Signature | Train | Validation | Total | Share |
|---|---:|---:|---:|---:|
| Enhanced Hammer ×4; Nighttime Mine ×2 | 124,850 | 14,048 | 138,898 | 85.7% |
| Enhanced Hammer ×3; Nighttime Mine ×3 | 21,111 | 2,119 | 23,230 | 14.3% |

All other card counts are identical. Therefore 0014 was not literally trained on one exact deck,
but its variation is only a one-card swap inside one expert/archetype and is strongly imbalanced.
It cannot establish component transfer or broad construct-conditioned generalization.

## V9 sensitivity check

On one fixed validation state from the V9 epoch-11 checkpoint, only the registered multiplicities
were changed from Hammer 4 / Mine 2 to Hammer 3 / Mine 3. All other state and legal-option tensors
were held fixed.

- maximum absolute finite-logit change: `0.18785858154296875`
- mean absolute finite-logit change: `0.0823456346988678`

This proves that the trained checkpoint's output is measurably connected to deck multiplicity. It
does not prove that the changed policy choice is strategically correct, nor that it will use large
cross-archetype differences well.

## Required 0015 additions

- Use an AC/R1-style registered-deck reader; never use A0 for the transfer claim.
- Bind every episode to an exact 60-card hash, build family and expert/team source.
- Balance batches by build, source and episode rather than raw decision count; split whole episodes.
- Report validation separately by build/source and by Dragapult/Dusknoir decision slice.
- Add tests that two valid decks produce distinct tensors, offline and online encoders agree, and a
  fixed model responds when only the deck changes.
- Run a same-data no-deck ablation. T0–T3 keep the same complete target pool/split and add every
  qualified auxiliary trajectory at its natural frequency. Record natural counts, updates and wall
  time; do not claim the result isolates data from additional compute.
- Treat official-engine evaluation of frozen target-deck candidates as the strength test.

The current test suite does not provide all of these assertions. That is an acceptance requirement
for implementation, not a reason to rewrite or restore unrelated tests during this documentation
change.
