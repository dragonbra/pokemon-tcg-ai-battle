# Experiment 0001 Decisions

- Expert: Yushin Ito, latest exact submission `54773249`, public score `1134.4`
  when selected on 2026-07-22.
- Exact deck: the 60-card multiset in `work/alakazam_v9/deck.csv`.
- Excluded source: submission `54486275` uses three Enhanced Hammer and three
  Battle Cage rather than the exact four/two list.
- Available history: Kaggle's submission episode request has only a
  `submission_id` field and returns the latest 1,000 public episodes; there is
  no page token.  This run freezes that 1,000-episode snapshot.
- Split: complete episodes ordered by creation time, 80% train, 10%
  validation, 10% test.  No episode contributes actions to multiple splits.
- Replay alignment: `steps[i].observation` is labeled by
  `steps[i + 1].action`.  Every label is checked against the earlier
  observation's option list and min/max count.
- Rows marked `DONE` can still carry the submitted action. They are retained;
  only a following row with `action: null` is excluded as unlabeled. This
  yields 86,875 decisions, including effect selections with up to 21 targets.
- Policy: full-action candidate scorer plus selection-count head.  Single
  choices use masked categorical CE; empty/multi choices use candidate BCE;
  cardinality uses masked count CE.  Multi-selection targets are treated as
  sets and emitted in sorted option-index order.
- Runtime: simulator legal-option mask and deck/reset callback only.  No rule
  teacher, confidence fallback, action-type guard, MCTS or search fallback.
- Reward: none.  This run is the frozen behavior-cloning starting point for
  later reward ablations.
- Final checkpoint: epoch 15, validation exact-action rate 78.88%, test
  exact-action rate 77.02%, and test legal-action rate 100%.
- Fixed evaluation: 122 wins, 42 losses, and 6 opponent-isolated engine errors
  in 170 games. No `candidate_error` occurred.
