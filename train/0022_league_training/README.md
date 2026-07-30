# 0022 League Training

This package binds the archived 0019 Universal Winner BC Epoch 13 model as a frozen shared
Foundation and manages exact-deck Decoder/Value plugins. It is self-contained: model and feature
code are frozen under `foundation/`; only the immutable archived `model.pt` is read as data.

Tracked deck identity lives in `deck/<deck_id>/`. Mutable Live weights never live beside source
code; they are atomically created under the allocated run version:

```text
rl_runs/0022_league_training/versions/V<n>_<tag>/
  artifact/training_config.json
  artifact/league_catalog.json
  artifact/status.json
  checkpoint/decks/<deck_id>.pt
  checkpoint/decks/<deck_id>.pt.sha256
```

Frozen zero-shot policies reference the Foundation sentinel and use no duplicate checkpoint.
Live policies receive independent copies of `pointer_key`, `pointer_query`, `option_bias`,
`decoder_init`, `decoder`, `stop`, plus an independent scalar value head. Actor `source_id` is
always neutral `0`; source/team identity remains provenance only.

```bash
python3 -m train.0022_league_training verify-foundation
python3 -m train.0022_league_training validate-decks
python3 -m train.0022_league_training initialize --version V1_initial_league
python3 -m train.0022_league_training audit-version --version V1_initial_league
python3 -m train.0022_league_training smoke-rollout --device cuda:0 --workers 4
python3 -m train.0022_league_training canary-ppo --device cuda:0 --workers 4
python3 -m train.0022_league_training train --version V1_dragapult_focal_20h
```

The catalog contains 48 exact decks. The first formal run schedules 512 games per update across
48 Frozen and 48 Live views with balanced seats, but updates only the `dragapult_ex_001` decoder
and value head. Other Live assets remain at update 0; their simultaneous evolution requires a new
version. Every fifth update runs 96 Frozen greedy games.

Formal training enables W&B online and enforces 100 GiB launch free space, an 80 GiB clean-stop
low-water mark, a 10 GiB version cap, and bounded model-only checkpoint retention. Initialization
alone does not run PPO and is not official-engine strength evidence.
