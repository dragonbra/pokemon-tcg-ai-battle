# Cynthia c7b3253feaa4 Single-Deck EpochMix v1

Controlled single-exact-list BC agent for local evaluation.

- Exact public deck: `c7b3253feaa4`.
- Strict-win teachers: 1,482 train games and 395 temporal-validation games.
- Encoded decisions: 119,351 train and 30,564 validation.
- Schedule: eight epochs on dual Tesla T4; all train games retained every epoch.
- Best checkpoint: epoch 4, validation loss 0.522619, token accuracy 81.07%, exact-action accuracy 62.62%.
- Architecture held fixed against the multi-deck control: 7.38M-parameter count-aware deck-conditioned ID-only pointer policy.
