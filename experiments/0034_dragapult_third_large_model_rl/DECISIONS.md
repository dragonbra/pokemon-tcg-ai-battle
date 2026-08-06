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
