# 0023 Mega Lopunny ex / Mega Froslass ex League Training

This package binds the archived 0019 Universal Winner BC Epoch 13 model as a frozen shared
Foundation and manages exact-deck Decoder/Value plugins. It is self-contained: model and feature
code are frozen under `foundation/`; only the immutable archived `model.pt` is read as data.

Tracked deck identity lives in `deck/<deck_id>/`. Mutable Live weights never live beside source
code; they are atomically created under the allocated run version:

```text
rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V<n>_<tag>/
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
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training verify-foundation
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training validate-decks
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training initialize --version V1_initial_league
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training audit-version --version V1_initial_league
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training smoke-rollout --device cuda:0 --workers 4
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training canary-ppo --device cuda:0 --workers 4
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training benchmark-workers --device cuda:0 --workers 128 --games 256 --coalesce-ms 5 --output .tmp/evaluation/0023_worker_scaling/workers-128.json
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training train-league --version V1_mega_lopunny_ex_mega_froslass_ex_002_continuous_league --device cuda:0 --workers 128 --coalesce-ms 5 --games-per-update 512
```

The catalog contains 50 exact decks. The formal run schedules 512 games per update across Frozen
and Live views with balanced seats. The focal deck is
`mega_lopunny_ex_mega_froslass_ex_002`; every Live decoder updates only from its own real actor
trajectory. The command has no duration stop and runs until an explicit signal or a fail-closed
safety guard. Every fifth update runs Frozen, Live, and fixed-probe greedy diagnostics.

Formal training enables W&B online and enforces 100 GiB launch free space, an 80 GiB clean-stop
low-water mark, a 10 GiB version cap, and bounded model-only checkpoint retention. Initialization
alone does not run PPO and is not official-engine strength evidence.
