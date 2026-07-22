# Experiment 0003 Decisions

- Goal: validate the universal model/input contract on exactly the frozen 0001 Yushin Ito dataset before any multi-deck BC or RL work.
- Expert constraint: every label comes from team `Yushin Ito`, submission `54773249`; no other team policy is mixed into this BC corpus.
- Dataset identity: 1,000 episodes, 86,875 decisions, and the same train/validation/test episode split and outcomes as 0001.
- New contract: 64 numeric state fields, 40 state cards, 60×18 deck context, 12×28 entities, 32×8 history, one expert ID, and 64×22 candidates.

## V1_card_token_regression

- The first training attempt had correct shapes but a displaced `_card_id` implementation caused all state/entity card IDs and action target IDs to become padding.
- The existing audit did not detect this because it checked dimensions, split leakage, source identity, action legality, and target cardinality rather than semantic token population.
- Training was stopped after epoch 12. Its best validation exact-action rate was 63.92%.
- The event, metrics, checkpoint, config, and shape-only audit are retained under the V1 name. The reproducible 1.7 GB invalid dataset itself is not retained after its checksum and token-population evidence were recorded.

## V2_card_token_fix

- Restored card ID parsing and applied visible candidate-card/target resolution to the universal schema.
- Added regression assertions that state, entity, action, and deck-count fields contain the expected non-padding values.
- Rebuilt all 86,875 records from the same frozen source and reran the fail-closed dataset audit before training.
- Completed 20 epochs. Best epoch 13 reached 80.52% validation exact-action rate; the best checkpoint reached 78.37% test exact-action rate and 100% test legal-action rate.
- Built a self-contained package from the best epoch-13 checkpoint with no rule teacher, search,
  confidence gate, or fallback. Package validation confirmed the exact 60-card deck and official
  `cg/` tree.
- The official-engine closed-loop run completed 180/180 games against the fixed 18-opponent
  catalog: 136 wins, 44 losses, no draws, and zero errors (75.56% win rate). This validates BC
  inference and the action contract in real simulator games; it is not RL training evidence.
- Promoted the evaluated package to `work/yushin_ito_exact_bc_v2`, archived the same payload at
  `submission/yushin_ito_exact_bc_v2`, and produced the 21,229,532-byte dist archive with SHA-256
  `29d109480cab947422148cb822fccbadb55e77c9e565312f8cd507529cb86fa6`.
- Submitted that archive once to Kaggle as ref `54912599`. It completed with public score
  `713.2`; no automatic retry or follow-up submission was performed.
- Rollout collection, reward design, value calibration, and PPO remain unimplemented.

## Versioning decision

- All future attempts in this experiment start at `V3_<tag>` and use matching version names for tracked run files, TensorBoard, checkpoints, and evaluation.
- Failed attempts remain visible. A version number is never reused and an existing run directory is never appended to or overwritten.
