# Experiment 0003 Commands

Commands run from the repository root with Python 3.11. The source manifest is a copy of the frozen 0001 Yushin Ito corpus.

## Shared dataset

```bash
PYTHONPATH=. python3 -m rl.train.build_kaggle_bc_dataset \
  data/replays/kaggle_yushin_ito_54773249 \
  --manifest rl/_runs/0003-yushin_ito_exact_bc_v2/data_manifest.json \
  --output rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl \
  --agent-name 'Yushin Ito' \
  --feature-schema ptcg_features_universal \
  --storage-path /mnt/c --min-free-gib 10

PYTHONPATH=. python3 -m rl.train.audit_kaggle_bc_dataset \
  rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl \
  --manifest rl/_runs/0003-yushin_ito_exact_bc_v2/data_manifest.json \
  --output rl/_runs/0003-yushin_ito_exact_bc_v2/audit.json
```

## V1 and V2 historical invocation

Both attempts originally used the experiment root as `--output`. After V2 completed, their metrics, TensorBoard events, and checkpoints were separated into `V1_card_token_regression` and `V2_card_token_fix`. V1 was stopped after the semantic token regression was proven; V2 rebuilt the dataset from the same manifest before training from scratch.

```bash
PYTHONPATH=. python3 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl \
  --output rl/_runs/0003-yushin_ito_exact_bc_v2 \
  --epochs 20 --batch-size 256 --learning-rate 3e-4 --seed 7 \
  --d-model 256 --hidden-dim 512 --num-heads 4 \
  --transformer-layers 2 --dropout 0 --device cuda \
  --storage-path /mnt/c --min-free-gib 10
```

## Future attempt contract

The root invocation above is historical and now rejected. Every future attempt must allocate the next unused version and write directly to it:

```bash
PYTHONPATH=. python3 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl \
  --output rl/_runs/0003-yushin_ito_exact_bc_v2/V3_next_hypothesis \
  --epochs 20 --batch-size 256 --learning-rate 3e-4 --seed 7 \
  --d-model 256 --hidden-dim 512 --num-heads 4 \
  --transformer-layers 2 --dropout 0 --device cuda
```

## V2 official-engine evaluation and release

The evaluated candidate was built from the V2 best-validation checkpoint and the already
validated 0001 Alakazam deck/runtime package. The local compatibility runtime only supplies a
new enough `libstdc++`; the packaged official `cg/` runtime is unchanged.

```bash
PYTHONPATH=. python3.11 -m rl.train.build_full_action_submission \
  --checkpoint rl/artifact/checkpoint/0003-yushin_ito_exact_bc_v2/V2_card_token_fix/best_validation.pt \
  --output work/yushin_ito_exact_bc_v2 \
  --source-package work/alakazam_bc_v1 \
  --experiment-id 0003-yushin_ito_exact_bc_v2/V2_card_token_fix \
  --name 'Yushin Ito Exact BC v2'

PYTHONPATH=. python3.11 -m evaluation validate work/yushin_ito_exact_bc_v2

LD_LIBRARY_PATH=/home/dragon_bra/.local/ptcg-cxx-runtime/lib \
  PYTHONPATH=. python3.11 -m evaluation run \
  --candidate work/yushin_ito_exact_bc_v2 \
  --opponents all --games 10 --no-visualize \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/_runs/0003-yushin_ito_exact_bc_v2/evaluation/V2_card_token_fix
```

The completed run is `run-8688bfdaa9c44059a7fec383b5f64f6e`: 180/180 finished, 136 wins,
44 losses, no draws, and zero errors. The evaluated package was then frozen under
`submission/yushin_ito_exact_bc_v2/` and archived without Python cache files.

```bash
tar -C submission/yushin_ito_exact_bc_v2 -czf \
  submission/dist/yushin_ito_exact_bc_v2.tar.gz main.py deck.csv cg strategy

kaggle competitions submit \
  -c pokemon-tcg-ai-battle \
  -f submission/dist/yushin_ito_exact_bc_v2.tar.gz \
  -m '0003 Yushin Ito Exact BC v2 V2_card_token_fix; official-engine eval 136/180 wins, 0 errors'
```

Archive SHA-256: `29d109480cab947422148cb822fccbadb55e77c9e565312f8cd507529cb86fa6`.
The single authorized Kaggle submission is ref `54912599`; it completed with public score
`713.2`. No retry or follow-up submission was made.
