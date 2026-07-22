# Experiment 0002 Decisions

- Objective: train the pure full-action `Alakazam BC V1 Tried` model from the
  2,021 exact-deck replays already present locally. No further replay download
  is part of this frozen run.
- Leaderboard source: the complete top-100 screening captured at
  `2026-07-22T17:39:24.684756+00:00`; 22 selected submissions matched the
  order-independent 60-card hash
  `3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf`.
- Exact-player identity is fail-closed. A same team name is insufficient:
  every selected player side is also checked against the full deck hash. This
  excluded an older, different-deck Dieter side while retaining the exact
  giacomovin side in episode `86395040`.
- Frozen corpus: 2,021 unique Episode IDs and 2,085 expert trajectories. The
  64 additional trajectories come from games in which both selected expert
  sides are exact. A replay file is stored once and both player trajectories
  share one split.
- Split: monotonic Episode ID order, whole episodes only; 1,616 train, 202
  validation and 203 test episodes. No episode crosses splits.
- Replay alignment: `steps[i].observation` is labeled by
  `steps[i + 1].action`. A next row marked `DONE` still carries a valid label;
  only `action: null` terminal rows are unlabeled and excluded.
- Multi-selection contract: actions are canonicalized as sorted sets; target
  count must equal list length; every target must be distinct, in range and
  legal under the encoded mask; `minCount <= count <= maxCount <= legal
  candidates` is mandatory.
- Pre-training audit passed with no violations across 173,251 unique decision
  keys. It contains 9,829 positive multi-target records, 250 legal empty
  selections and target counts up to 25. Context 8 contributes 2,057 records,
  including 1,998 multi-target records.
- Model/hyperparameters remain directly comparable to V1: `ptcg_features_v6`,
  d_model 256, hidden 512, two Transformer layers, four heads, dropout 0,
  batch 256, AdamW 3e-4, seed 7 and 20 epochs. Training starts from random
  initialization; there is no rule teacher, fallback, reward shaping or old
  checkpoint warm start.
- Final disposition: completed but not promoted. Fixed evaluation is 118 wins,
  47 losses and five opponent-isolated engine errors in 170 games, versus the
  original V1's 122 wins, 42 losses and six opponent-isolated errors. Future
  reward/model work continues from experiment 0001 and `work/alakazam_bc_v1`.
