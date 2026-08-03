# Experiment 0027: Winner-only Semantic Foundation BC

Status: **complete**

0027 trains the complete 0025 `CanonicalSemanticPolicy` architecture from random
initialization on winner-view Top battle records dated 2026-07-15 through
2026-08-01. The actor accepts all 22 canonical features and never receives
team/source identity.

## Kaggle workflow

The four dataset notebooks are independent date partitions so they can run in
parallel and can be mounted selectively for later training:

1. `kaggle/01_days_0715_0719`
2. `kaggle/02_days_0720_0724`
3. `kaggle/03_days_0725_0728`
4. `kaggle/04_days_0729_0801`

`kaggle/05_train_dual_t4` consumes all four outputs. It uses two T4 GPUs through
`torch.nn.DataParallel`, starts from random weights, runs one rolling epoch per
partition, and does not load a pretrained checkpoint or use W&B. The no-W&B
choice was an explicit requirement for this Kaggle BC run; the complete local
training report is retained instead.

## Final model

- architecture: `CanonicalSemanticPolicy`
- trainable parameters: 21,837,082
- actor features: 22/22 canonical features
- winner only: true
- initialization checkpoint: none
- final validation loss: 0.432314
- final validation token accuracy: 86.2273%
- teacher-forced exact action: 74.1182%
- model SHA-256: `274407890d17b70a5a331cac1855a7a2438946e4c5a2989a99345f1ab4af5dfc`

The Git LFS checkpoint and its complete reports are under
`artifacts/final_bc_v3/`.

## Local BC pool check

The exported model was tested with Dragapult and Raging Bolt decks against six
existing BC opponents, 20 games per matchup. Candidate player index was balanced
10/10 in each matchup.

| Deck | W-L | Win rate | Errors | Runtime fallbacks |
|---|---:|---:|---:|---:|
| Dragapult | 59-61 | 49.17% | 0 | 0 |
| Raging Bolt | 27-93 | 22.50% | 0 | 0 |

These are local unseeded research matches, not a formal promotion gate. Raw game
rows and per-matchup reports are retained with the model artifact.
