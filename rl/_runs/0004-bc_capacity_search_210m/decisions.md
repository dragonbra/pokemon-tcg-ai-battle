# BC Capacity Search Decisions

Search decisions below use train/validation evidence only. Test and official-engine
evaluation results were recorded after each trial but never fed into branching.

- Phase A selected lr=0.0005 using validation-only tie rules.
- Phase C: L validation oscillated late; testing adjacent lower LR.
- Phase D candidate: W is within 0.3pp of the highest validation exact at the smallest observed parameter count in that interval.

- Training wall time: 6090.5 seconds.
- Search final status before user review: `complete_waiting_user_selection`.
- The search runner did not use test or official-engine evaluation results for branching and did
  not automatically rank or promote a model.
- After the complete evidence set was frozen, the user selected
  `V4_s_d192_l2_lr5e4_s7` as the current BC model. The decision prioritizes its 145-35
  closed-loop result and smaller 3,539,395-parameter capacity while recognizing that its 80.87%
  validation exact-action rate is not the campaign maximum.
- V4 was released as `yushin_ito_bc_capacity_v4`, archived, and submitted to Kaggle exactly once.
  Submission ref: `54916884`; final status: `COMPLETE`; public score: `727.8`.
