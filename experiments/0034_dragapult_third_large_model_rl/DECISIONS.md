# 0034 Decisions

## 2026-08-06: Change only the pretrained source and allocate a new project

0034 physically freezes the validated 0033 full-semantic official-CPU PPO implementation. It does not import executable code from 0033 and does not continue a 0033 checkpoint, optimizer, metric curve or W&B run. The focal exact THIRD deck is unchanged at file SHA-256 `5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`.

Initialization changes to `large-model-0806.tar.gz` member `best_validation_loss_0806.pt`, archived as `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/model.pt`. Its SHA-256 is `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`; it is 0031 V2 full-winners, Epoch 11 / step 141878, with validation exact-action 81.4023% and legal-action 100%. Strict loading confirms the same 293-tensor, 56,352,322-parameter SemanticPolicy contract used by 0033.

## 2026-08-06: Reuse the V6 algorithm design, not any V6 weights

0033's comparable zero-error 510-game Frozen51 results were V6 273 wins, V8 252 wins and V9 242 wins. Therefore 0034 V2 uses V6's turn-clock `lambda=0.97`, 204 Episodes/update and Episode-equal decision weighting. It starts from the new 0806 checkpoint with a fresh AdamW optimizer. Reward remains official terminal result only; `gamma=1.0`; all PPO coefficients and the full 39-key semantic actor contract remain unchanged.

V1 first measures the new checkpoint over 510 official-engine games. V2 then runs 200 updates, with every update covering all 51 Frozen 0019 opponents, both focal seats and two seeds per seat. The frozen 102-game greedy probe every ten updates is a trend diagnostic. Sampled rollout rates and 102-game probes do not establish final checkpoint strength; a final claim requires a separate 510-game formal evaluation under the same contract.

## 2026-08-06: Restore full W&B display-name prefixes

W&B stable run IDs and groups already carried the numbered project identity, but copied 0033 runners set the display name to the bare version. The canonical display format is now `number · project_slug · version`. 0034 V2 will use `0034 · dragapult_third_large_model_rl · V2_turn_clock_lambda097_200u`; existing 0033 run display names are renamed in place without changing IDs, URLs, groups, metrics or state.

## 2026-08-06: Fix the 0806 zero-shot baseline at 257-253

V1 completed all 510 official-engine Frozen 0019 games with zero error and zero unfinished games: 257 wins, 253 losses, no draw, or 50.3922%. The authoritative report is `evaluation/V1_large_model_zero_shot_frozen0019.html`, run ID `run-4bf0f4e06b7d4bc88856f1d7ccbf0f07`. This is V2's immutable update-0 comparison point. Its roughly 2.9-point difference from independent 0033 PT0805 evaluations is descriptive, not a paired or statistically established source-model improvement.

## 2026-08-07: Launch V2 for 200 updates

After all unit, package, parity and full 204-Episode PPO gates passed, V2 `V2_turn_clock_lambda097_200u` launched at 2026-08-06 23:59:41 +08:00. It starts from the 0806 source with a fresh optimizer, uses the selected V6 algorithm contract, retains every model-only checkpoint and runs under watchdog/tmux session `0034_v2_turn_clock_200u`. Its W&B run ID is `0034-v2-turn-clock-lambda097-200u` and its full display name includes the restored 0034 prefix.

Update 1 completed from source policy update 0 with 204/204 Episodes, 96-108-0 sampled outcomes, 21,590 focal decisions and zero error. The representation hash remained unchanged, the decoder changed, behavior KL was `2.22e-6`, and checkpoint `update-000001.pt` plus JSONL, TensorBoard and W&B mirrors were persisted. This sampled 47.06% is a training diagnostic, not a frozen checkpoint-strength claim.

## 2026-08-07: Stop V2 at update 1 by user request

The user stopped V2 at 00:06:55 +08:00 to verify another important issue. The tmux session, watchdog, trainer and rollout workers were terminated, and a W&B stop request was issued. The latest complete state is update 1 with 204 Episodes and 21,590 decisions. Model-only checkpoints 0 and 1, canonical JSONL, TensorBoard and W&B history are retained. V2 is `user_stopped`, not completed, and must not be reported as a 200-update result.

## 2026-08-07: Abort V3 and correct the opponent policy

V3 was launched with the 500-update target and fixed 256-game Frozen-0806 schedule, but its collector still routed opponents through the historical Policy-0019 foundation. The user corrected the contract before the first PPO update. V3 was terminated at update 0 with only the initialization checkpoint retained; it is `user_stopped_before_update_1` and must not be compared as a training result. The next version uses an independent frozen copy of the Policy-0806 pretrained checkpoint for all 55 opponents, with per-deck online encoders and no opponent parameter updates.

## 2026-08-07: Launch V4 with Policy-0806 frozen opponents

V4 `V4_policy0806_frozen_256g_500u` launched after a fresh official-engine PPO gate passed full 39-key parity, 4/4 completed Episodes, zero errors, frozen-representation integrity and decoder change. It starts from the immutable Policy-0806 checkpoint with a fresh optimizer; a separate Policy-0806 actor copy remains frozen for every opponent inference. Each update uses the exact 55-deck, 256-slot schedule and V4 targets 500 updates with a full 256-game greedy probe every ten updates.

## 2026-08-07: Add uniform rollout termination contract and launch V5

V4's first 256-game batch reached a nonterminal repeated selection at engine turn 10 and hit the 1,000-selection safety limit. This was an environment termination defect, not evidence about policy strength. The fixed contract now treats the ninth identical selection-chain action within one turn/player as an opponent win, and `ceil(engine_turn / 2) >= 50` as a draw. A synthetic official-worker harness verified both outcomes. V5 `V5_policy0806_frozen_256g_500u_termination_guard` launched with the same Policy-0806 frozen opponents, 256-game schedule and 500-update target after the corrected gate passed.

## 2026-08-07: Stop V5 after actor-strength regression

The user stopped V5 after update 76. All model-only checkpoints 0-76, canonical metrics, TensorBoard and W&B history remain. Sampled rollout win rate stayed near one third without a positive trend, while the periodic greedy probe declined from 127-129-0 (49.61%) at update 10 to 105-150-1 (41.02%) at update 70. PPO reward and policy-loss signs were audited and no reversal defect was found. The critic improved while the actor drifted from its reference, so V5 is retained as negative evidence rather than continued on the assumption that a longer run will reverse the trend. The old greedy probes also changed engine seeds with checkpoint update; they are trend evidence but not strict paired comparisons.

## 2026-08-07: Restart exact 007 with the audited 0023 PPO setting

V6 changes the focal deck to Frozen-0806 number 007, `dragapult_ex_07bedfffbfad`, exact-deck SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`. It restarts from the immutable Policy-0806 checkpoint with a fresh optimizer and does not continue V5 weights.

The optimization setting is restored from 0023 V2: selection clock, `gae_lambda=0.95`, four PPO epochs and minibatch 1024. Actor/value learning rates, clipping, entropy, reference KL, target behavior KL, gamma, Episode-equal decision weighting and gradient clipping already match and remain unchanged. The user-defined rollout unit remains exactly 256 official-engine games from the Frozen-0806 distribution; 1024 is the decision-token optimization minibatch, not the number of games. Policy-0806 opponents, the 55-deck schedule, repeat-forfeit rule and 50-full-round draw rule remain unchanged.

V6 adds a 256-game update-0 greedy baseline. Every periodic greedy probe uses the same engine seeds, seats, opponents, exact decks and order regardless of checkpoint update. This fixes the V5 comparison defect without affecting on-policy training seeds, which continue to vary by source-policy update.

## 2026-08-07: Launch V6 and establish the paired update-0 baseline

V6 `V6_exact007_0023_selection_lambda095` launched with 500 updates, 32 workers, 256 games per update, evaluation every ten updates and W&B run ID `0034-v6-exact007-0023-selection-lambda095`. The update-0 exact-007 greedy baseline completed 256/256 official-engine games with zero error at 146-110-0 (57.03%): 59.38% first-seat and 54.69% second-seat.

Update 1 then completed 256/256 stochastic on-policy games with zero error at 119-137-0 (46.48%). PPO used four epochs and 112 minibatches over 28,517 decisions, preserved the frozen representation hash and saved model-only checkpoint 1. This sampled rollout is not comparable to the greedy update-0 strength result; the first paired strength comparison is the fixed-seed greedy probe at update 10.

## 2026-08-07: Stop V6 manually after update 123

The user manually stopped the V6 training process after update 123. The latest complete local model-only checkpoint and canonical metric row are both update 123, covering 31,488 Episodes and 3,440,480 focal decisions. The W&B remote run is marked `crashed` because the process was terminated manually; its remote history was checked and also reaches `trainer/update=123`, `checkpoint/update=123`, 31,488 Episodes and 3,440,480 decisions. V6 is therefore retained as `user_stopped`, not completed and not an active 500-update run.
