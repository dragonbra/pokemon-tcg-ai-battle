# Experiment 0001 Commands

Commands are shown with the reorganized paths and run from the repository root
with Python 3.11. The frozen replay corpus is identified by `data_manifest.json`.

```bash
PYTHONPATH=. python3.11 -m rl.train.build_kaggle_bc_dataset \
  replays/kaggle_yushin_ito_54773249 \
  --manifest replays/kaggle_yushin_ito_54773249/manifest.json \
  --output rl/artifact/dataset/0001-yushin_ito_exact_bc_baseline/dataset.jsonl \
  --agent-name 'Yushin Ito' \
  --feature-schema ptcg_features_v6

PYTHONPATH=. python3.11 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/0001-yushin_ito_exact_bc_baseline/dataset.jsonl \
  --output rl/_runs/0001-yushin_ito_exact_bc_baseline \
  --device cuda

PYTHONPATH=. python3.11 -m rl.train.build_full_action_submission \
  --checkpoint rl/artifact/checkpoint/0001-yushin_ito_exact_bc_baseline/best_validation.pt \
  --output work/alakazam_bc_v1 \
  --source-package work/alakazam_v9 \
  --experiment-id 0001-yushin_ito_exact_bc_baseline \
  --name 'Alakazam BC v1'

PYTHONPATH=. python3.11 -m evaluation validate work/alakazam_bc_v1

PYTHONPATH=. python3.11 -m evaluation run \
  --candidate work/alakazam_bc_v1 \
  --opponents all --games 10 --no-visualize \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation
```

The package uses only the checkpoint-backed policy. There is no rule teacher,
search, confidence gate, or fallback.
