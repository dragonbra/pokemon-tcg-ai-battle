# 0023 Mega Lopunny ex Limitless Focal / Mega Froslass ex League Training

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
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training train-league --version V7_from_v6_limitless_focal_24h --initial-version V6_from_v2_update3_20h_gpu_resume --device cuda:0 --workers 128 --coalesce-ms 5 --games-per-update 512
```

The catalog contains 51 exact decks. The formal run schedules 512 games per update across Frozen
and Live views with balanced seats. The focal deck is the Limitless champion
`mega_lopunny_ex_001` (including Abra, exact deck SHA
`f03203e5bc6fc1cd4c29b4e3e728f360d37abe55aaaf34a73b053085dbe21553`). Every Live decoder
updates only from its own real actor trajectory. V8 inherits all 50 prior deck decoders from V7
update 71 and gives `rmy_teal_mask_ogerpon_001` its own Foundation-default update-0 decoder/value
branch. The foreground supervision wrapper stops a training run only after its configured duration or an
explicit fail-closed alert. Every fifth update runs Frozen, Live, and fixed-probe greedy diagnostics.

For a formal long run, keep the training child and `monitor_training` in one foreground exec
session. Retain that session ID and wait on it repeatedly; do not use a detached watchdog, and do
not package or run competing GPU work while the session is active.

Formal training enables W&B online and enforces 100 GiB launch free space, an 80 GiB clean-stop
low-water mark, a 10 GiB version cap, and bounded model-only checkpoint retention. Initialization
alone does not run PPO and is not official-engine strength evidence.
