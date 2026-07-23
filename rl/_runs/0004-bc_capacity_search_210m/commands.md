# 0004 BC Capacity Search Commands

All trials reuse the frozen 0003 Yushin Ito dataset and start from random initialization.
Search branching consumes train/validation evidence only.

```bash
PYTHONPATH=. python3.11 -m rl.core.runs create bc_capacity_search_210m \
  --objective "Measure BC capacity and evaluation behavior under a 210-minute training budget"

LD_LIBRARY_PATH=/home/dragon_bra/.local/ptcg-cxx-runtime/lib \
PYTHONPATH=. python3.11 -u -m rl.train.bc_capacity_search \
  --experiment-root rl/_runs/0004-bc_capacity_search_210m
```

The campaign runner executes this transaction for every launched `V<n>_<tag>`:

```bash
PYTHONPATH=. python3.11 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl \
  --output rl/_runs/0004-bc_capacity_search_210m/<version> \
  --epochs 20 --batch-size 256 --learning-rate <lr> --seed <seed> \
  --d-model <d_model> --hidden-dim <hidden_dim> --num-heads 4 \
  --transformer-layers <layers> --dropout <dropout> --device cuda \
  --storage-path /mnt/c --min-free-gib 10

PYTHONPATH=. python3.11 -m rl.train.build_full_action_candidate \
  --checkpoint rl/artifact/checkpoint/0004-bc_capacity_search_210m/<version>/best_validation.pt \
  --source-package work/alakazam_bc_v1 \
  --output /tmp/0004-bc_capacity_search_210m-candidates/<version>

python3.11 -m evaluation validate \
  /tmp/0004-bc_capacity_search_210m-candidates/<version>

python3.11 -m evaluation run \
  --candidate /tmp/0004-bc_capacity_search_210m-candidates/<version> \
  --opponents all --games 10 --no-visualize \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/_runs/0004-bc_capacity_search_210m/evaluation/<version>
```

No Kaggle submission, model promotion, or git commit was performed by this campaign.

## User-selected V4 release

After the campaign finished and all results were frozen, the user selected
`V4_s_d192_l2_lr5e4_s7`. The selected checkpoint was converted to the self-contained formal
package; the source package and extracted archive both passed `evaluation validate` plus an
independent import/reset smoke check.

```bash
PYTHONPATH=. python3.11 -m rl.train.build_full_action_submission \
  --checkpoint rl/artifact/checkpoint/0004-bc_capacity_search_210m/V4_s_d192_l2_lr5e4_s7/best_validation.pt \
  --output work/yushin_ito_bc_capacity_v4 \
  --source-package work/alakazam_bc_v1 \
  --experiment-id 0004-bc_capacity_search_210m/V4_s_d192_l2_lr5e4_s7 \
  --name 'Yushin Ito BC Capacity V4'

tar --exclude='__pycache__' --exclude='*.py[co]' \
  -C submission/yushin_ito_bc_capacity_v4 -czf \
  submission/dist/yushin_ito_bc_capacity_v4.tar.gz main.py deck.csv cg strategy

kaggle competitions submit \
  -c pokemon-tcg-ai-battle \
  -f submission/dist/yushin_ito_bc_capacity_v4.tar.gz \
  -m '0004 BC capacity V4 selected; official-engine eval 145/180 wins, 0 errors'
```

Archive SHA-256: `ba8f13683e09c567f501032b46f66011485931764c194da998622237ac5d0490`.
The single authorized Kaggle submission is ref `54916884`; it completed with public score
`727.8`. No retry or follow-up submission was made.
