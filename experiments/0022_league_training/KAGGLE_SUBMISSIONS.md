# 0022 Kaggle submissions

This file records the two 0022 Frozen neutral zero-shot packages explicitly authorized for
submission on 2026-07-31. Both use the 0019 Universal Winner BC epoch-13 checkpoint with
`source_id=0` and no RL updates.

## Festival Lead / Dipplin 001

- Competition: `pokemon-tcg-ai-battle`
- Submission ref: `55119231`
- Kaggle timestamp: `2026-07-30 20:56:34.203000` UTC
- Exact deck: `festival_lead_dipplin_001`
- Exact sorted-card hash: `d6c573dd89bd1319494e25d56e66536c0b68c3db2cf4694f4a63f7423ee2f0bd`
- Frozen baseline: 230-70-0, 76.67%
- Package: `archive/submission/dist/0022_frozen_festival_lead_dipplin_001.tar.gz`
- Package SHA-256: `03c452a7af281726a716af39d92f8ec9a44a9edab2bfe4868718313b54c6e754`
- Extracted-package smoke: `run-4fe5e76631b1494d841a60ec183d6166`, 10/10 finished,
  0 errors, 9-1
- Message: `愿裹着蜜糖的裹蜜虫随庆典节拍起舞，让每一回合的鼓点都通向胜利。`
- Final status: `COMPLETE`
- Public score: `600.0`

## Mega Kangaskhan ex / Crustle 004

- Competition: `pokemon-tcg-ai-battle`
- Submission ref: `55119238`
- Kaggle timestamp: `2026-07-30 20:57:08.280000` UTC
- Exact deck: `mega_kangaskhan_ex_crustle_004`
- Exact sorted-card hash: `67cf83ea7e092595551b8cd9668be37903de3beb362b384f21c07b66d0549ff9`
- Frozen baseline: 215-85-0, 71.67%
- Package: `archive/submission/dist/0022_frozen_mega_kangaskhan_ex_crustle_004.tar.gz`
- Package SHA-256: `859214b11075b8af36e057e930905153930de8e96504f1807cfde134aa784d67`
- Extracted-package smoke: `run-8202dcc04ef54a55a5f2183d5cf15f4e`, 10/10 finished,
  0 errors, 10-0
- Message: `愿袋兽守护前路，岩殿居蟹稳住阵地，把每一次重击都化作通往冠军的台阶。`
- Final status: `COMPLETE`
- Public score: `600.0`

Exactly two successful `kaggle competitions submit` commands were issued for this authorization.
An earlier Festival upload was stopped before Kaggle created a submission because the extracted
archive gate detected Python bytecode caches. It consumed no submission quota. The caches were
moved to ignored local staging, both archives were rebuilt, and all final extracted-package gates
passed before refs `55119231` and `55119238` were created. No automatic retry is authorized.
