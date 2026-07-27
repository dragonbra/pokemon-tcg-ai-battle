# Cynthia Core Mean-Pool EpochMix v1

Local evaluation agent for the dominant public Cynthia's Garchomp ex exact list
`c7b3253feaa4`. The policy is the best-validation-loss checkpoint from the
eight-epoch, dual-Tesla-T4 Cynthia-core BC run.

- Public strict-win teachers: 1,935 games across four exact lists.
- Training decisions: 123,001; temporal validation decisions: 31,442.
- Best checkpoint: epoch 3, validation loss 0.518462.
- Model: 7.38M-parameter count-aware deck-conditioned ID-only pointer policy.
