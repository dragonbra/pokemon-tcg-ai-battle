# Experiment 0002 Commands

Commands are shown with the reorganized paths and run from the repository root
with Python 3.11.

```bash
PYTHONPATH=. python3.11 -m rl.train.finalize_exact_replay_subset \
  --screening replays/kaggle_top100_exact_deck_20260723/screening.json \
  --deck work/alakazam_v9/deck.csv \
  --replay-root replays/kaggle_yushin_ito_54773249 \
  --replay-root replays/kaggle_top100_exact_deck_20260723/episodes \
  --source-manifest replays/kaggle_yushin_ito_54773249/manifest.json \
  --output rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/data_manifest.json

PYTHONPATH=. python3.11 -m rl.train.build_kaggle_bc_dataset \
  rl/artifact/dataset/0002-alakazam_bc_v1_tried_exact_2021 \
  --manifest rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/data_manifest.json \
  --output rl/artifact/dataset/0002-alakazam_bc_v1_tried_exact_2021/dataset.jsonl \
  --feature-schema ptcg_features_v6

PYTHONPATH=. python3.11 -m rl.train.audit_kaggle_bc_dataset \
  rl/artifact/dataset/0002-alakazam_bc_v1_tried_exact_2021/dataset.jsonl \
  --manifest rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/data_manifest.json \
  --output rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/audit.json

PYTHONPATH=. python3.11 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/0002-alakazam_bc_v1_tried_exact_2021/dataset.jsonl \
  --output rl/_runs/0002-alakazam_bc_v1_tried_exact_2021 \
  --epochs 20 --batch-size 256 --learning-rate 3e-4 --seed 7 \
  --d-model 256 --hidden-dim 512 --num-heads 4 \
  --transformer-layers 2 --dropout 0.0 --device cuda

PYTHONPATH=. python3.11 -m rl.train.build_full_action_submission \
  --checkpoint rl/artifact/checkpoint/0002-alakazam_bc_v1_tried_exact_2021/best_validation.pt \
  --output work/alakazam_bc_v1_tried \
  --source-package work/alakazam_v9 \
  --experiment-id 0002-alakazam_bc_v1_tried_exact_2021 \
  --name 'Alakazam BC v1 Tried'

PYTHONPATH=. python3.11 -m evaluation run \
  --candidate work/alakazam_bc_v1_tried \
  --opponents all --games 10 --no-visualize \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/evaluation
```
