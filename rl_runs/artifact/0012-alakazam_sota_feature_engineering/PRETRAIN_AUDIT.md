# 0012 pre-training audit

## GPU and storage

- Device: NVIDIA GeForce RTX 5080, 16,303 MiB.
- Allocation-time free repository filesystem space: approximately 886 GiB.
- 0011 `V1_all_rewards_lr4e4` was already using the GPU when 0012 was allocated; 0012
  does not interrupt or overlap that training process.

## Frozen dataset identity

- Source decision dataset SHA-256:
  `36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf`.
- Train: 221,289 decisions / 2,875 episode-player groups.
- Validation: 17,067 decisions / 229 groups.
- Test: 0 decisions.
- 0010 dataset audit SHA-256:
  `43b48079b2016eeb3a6e125cafa86bf3c4265a949b66f3853aaca8da55173fd2`.

## E3 action-field feasibility sample

Read-only audit of the first 200 frozen episode-player groups:

- 15,357 decisions;
- 114,149 legal options;
- option fields with material coverage included `type`, `index`, `area`,
  `inPlayArea`, `inPlayIndex`, `playerIndex`, `attackId`, `number`, `count`,
  `energyIndex`, and `toolIndex`;
- `attackId` appeared on 5,236 sampled options;
- `count` and `energyIndex` appeared on 379 sampled options;
- `toolIndex` appeared on 12 sampled options;
- every select request exposed stable keys for `type`, `context`, `effect`,
  `contextCard`, `minCount`, and `maxCount`, although effect/contextCard values are
  nullable.

The E3 schema therefore encodes attack ID, count, energy/tool indices, effect card,
context card, and min/max count. It deliberately excludes `serial` from learned action
identity because serial identifies a runtime instance rather than a reusable action
primitive.

Full rebuild result over all 3,104 groups:

- identity and action-order differences: 0;
- attack ID nonzero options: 81,217;
- count and energy-index nonzero options: 4,995 each;
- tool-index nonzero options: 51;
- effect-card nonzero decisions: 52,987;
- context-card nonzero decisions: 21,289;
- train/validation counts remain exactly 221,289 / 17,067.

## Verified implementation invariants

- V1 baseline state dict and logits exactly match the 0010 model at equal random seed.
- Candidate permutation remaps every selected index through the inverse permutation and
  preserves the expert multi-selection order.
- The no-position model is permutation equivariant in the deterministic unit fixture.
- CPU training smoke writes best-loss, best-exact, latest, metrics, summary, TensorBoard,
  and the required four-section `ANALYSIS.md`.
- Parameter counts are 7,154,562 (V1), 7,359,362 (V3), and 7,481,666 (V4).
  V4 is 4.57% above V1 and therefore remains inside the precommitted ±5% capacity band.
- A real two-row V4 validation batch produced finite logits with the audited
  `option_primitive_cat [B,M,4]` and `action_context_cat [B,4]` tensors.

## E4 transition-label feasibility

The full 3,104-group audit covers all 238,356 frozen decisions:

- 235,284 decisions (98.71%) have a next decision frame;
- 217,634 available transitions remain in the same engine turn;
- 207,833 same-turn transitions contain at least one audited observable delta;
- 17,252 available transitions have no delta in the first compact state vector;
- the most frequent changed fields are own hand (179,172), own deck (76,941), own
  bench (40,302), own discard (34,143), and own attached energy count (18,224).

E4 is therefore label-feasible. The frozen contract masks resource-delta loss on
cross-turn transitions and predicts turn transition separately. Labels describe the
chosen action plus automatic engine consequences until the next decision frame; they
are not counterfactual labels for unchosen actions.

The V5 policy plus transition heads has 7,502,574 parameters, 4.86% above V1 and still
inside the ±5% capacity band. A real four-row transition validation batch produced finite
pointer logits, 11-dimensional delta predictions, and turn-change logits.

## E5 action-history feasibility

- The history dataset preserves all 221,289 / 17,067 train/validation identities and has
  zero action-order differences.
- Exactly 229 validation rows have empty history, matching one first decision for each
  validation episode-player group; no future decision is inserted.
- Mean validation history length is 14.31 selected-option events; 13,671 rows reach the
  configured 16-event cap.
- V6 has 7,365,122 parameters (2.94% above V1), and a real history batch produced finite
  logits with `action_history_cat [B,H,6]` and its explicit mask.
