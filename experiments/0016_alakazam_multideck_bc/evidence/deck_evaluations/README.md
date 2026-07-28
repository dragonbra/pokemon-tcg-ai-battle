# 0016 Interim Deck Evaluations

These immutable HTML copies preserve official-engine evaluations made while formal V1 training was
still in progress. They are exploratory checkpoint evidence, not formal `V<n>_*.html` evaluation
versions and do not update the project evaluation index.

All compared candidates use deployment source ID 0. Within each checkpoint, candidate model hashes
are identical and only the exact 60-card deployment deck changes.

| Checkpoint | Scope | Deck | Result | Report |
| --- | --- | --- | ---: | --- |
| epoch 10 | BC Marnie, 100 games | Yushin | 41-59 | [report](epoch10_bc_marnie_100_yushin.html) |
| epoch 10 | BC Marnie, 100 games | submit | 53-47 | [report](epoch10_bc_marnie_100_submit.html) |
| epoch 12 | BC Marnie, 100 games | Yushin | 42-58 | [report](epoch12_bc_marnie_100_yushin.html) |
| epoch 12 | BC Marnie, 100 games | submit | 53-47 | [report](epoch12_bc_marnie_100_submit.html) |
| epoch 12 | BC Marnie, 100 games | Yushin + 2 Battle Cage | 43-57 | [report](epoch12_bc_marnie_100_yushin_battle_cage_2.html) |
| epoch 12 | full catalog, 20 x 10 games | Yushin | 160-40 | [report](epoch12_full_catalog_yushin.html) |
| epoch 12 | full catalog, 20 x 10 games | submit | 157-43 | [report](epoch12_full_catalog_submit.html) |
| epoch 12 | TUFA BC Marnie, 100 games | Yushin | 48-52 | [report](epoch12_tufa_marnie_100_yushin.html) |
| epoch 12 | TUFA BC Marnie, 100 games | Yushin + 2 Battle Cage | 57-43 | [report](epoch12_tufa_marnie_100_yushin_battle_cage_2.html) |
| epoch 12 | TUFA BC Marnie, 100 games | submit | 64-36 | [report](epoch12_tufa_marnie_100_submit.html) |
| R15 epoch 10 | TUFA BC Marnie 04, 100 games | Yushin | 43-57 | [report](epoch10_r15_tufa_marnie_100_yushin.html) |
| R15 epoch 10 | TUFA BC Marnie 04, 100 games | Yushin + 2 Battle Cage | 53-47 | [report](epoch10_r15_tufa_marnie_100_yushin_battle_cage_2.html) |
| R15 epoch 10 | TUFA BC Marnie 04, 100 games | submit | 67-33 | [report](epoch10_r15_tufa_marnie_100_submit.html) |

See `manifest.json` for checkpoint, deck, model, run and metric-profile commitments.

The TUFA candidate carried an older, incompatible `cg/` copy. Its temporary evaluation package
preserved `main.py`, `idonly_policy.py`, `policy.pt` and `deck.csv` byte-for-byte and replaced only
that runtime copy with the current official evaluation runtime. It remained outside the formal
opponent catalog.

After that historical R2 evaluation, the normalized TUFA package was admitted to the formal arena
as `marnies_grimmsnarl_ex_froslass_04_bc`. The R15 epoch 10 runs use that formal package directly.
