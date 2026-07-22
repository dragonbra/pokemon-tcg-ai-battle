# RL Runs Layout

每个实验使用一个全局编号目录，编号不复用：

```text
rl/runs/0001-experiment_name/
  manifest.json             # objective, git commit, data/model/eval pointers
  data/
    source_manifest.json    # replay IDs, deck hash, split and preprocessing stats
    dataset.jsonl            # episode-grouped records with train/validation/test tags
    dataset.jsonl.summary.json
  source/
    model.py                 # immutable source snapshot used for the run
    features.py
    dataset.py
    train_config.json
  training/
    config.json
    metrics.jsonl
    tensorboard/
    checkpoints/
      latest.pt
      best_validation.pt
    summary.json
  candidate/
    package/                  # pure model candidate, no teacher fallback
    manifest.json
  evaluation/
    run-*/                   # repository evaluation output
    summary.json
  notes/
    decisions.md
```

`manifest.json` is the index for the entire experiment.  A checkpoint is not
considered a result until its model/data configuration, validation metrics and
fixed 17-opponent evaluation are recorded under the same experiment directory.
Generated run data is ignored by Git; long-lived conclusions belong in
`reports/rl/`.

Create a new experiment with:

```bash
PYTHONPATH=. python3.11 -m rl.core.runs create \
  yushin_ito_exact_bc_baseline \
  --objective "Pure full-action BC on Yushin Ito submission 54773249"
```
