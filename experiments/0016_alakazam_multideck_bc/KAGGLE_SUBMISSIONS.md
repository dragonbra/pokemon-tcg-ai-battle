# 0016 Kaggle submissions

本文件记录用户明确授权的 0016 正式 Kaggle 提交。时间采用 Kaggle CLI 返回值；
本地 official-engine 对局用于支持卡表选择和跨卡表部署判断，不代表完整线上环境强度。

## V1 R2 epoch 12 - validation-loss best / full submission deck

- Submission ref: `55036274`
- Kaggle timestamp: `2026-07-27 18:56:06.297000`
- Competition: `pokemon-tcg-ai-battle`
- Archive: `archive/submission/dist/0016_alakazam_multideck_bc_v1_r2_epoch12_loss_best_submit_deck_root.tar.gz`
- Archive SHA-256: `46e5044dec05bb6b114f8a61bad41148b05c114cfe7cedb0f45eaed5ae065263`
- Checkpoint SHA-256: `04f6db8498494cbedccf87bde25e8128a47745e09e457f4839cb38f11ce0f6d0`
- Model SHA-256: `81c5fdf9ba592ab6a7d68edbaea320c597b6cca37da0eba67d197f7ad48eb09f`
- Deck SHA-256: `267ce842b45f960afef843e68f3aad86573bf7d19329ea23e7469ec8265c017b`
- Message: `愿这副胡地把今晚验证的泛化与勇气，稳稳带到赛场。`
- Final queried status: `COMPLETE`
- Public score: `600.0`

Local official-engine evidence against TUFA BC Marnie used 100 games per deck with a fixed
50/50 first-player split: Ito's original deck won `48/100`, Ito plus two Battle Cage won
`57/100`, and the full submission deck won `64/100`. Against the other retained BC Marnie
package, Ito's deck won `42%` and the full submission deck won `53%`. The complete reports are
retained under `experiments/0016_alakazam_multideck_bc/evidence/deck_evaluations/`.

Exactly one `kaggle competitions submit` command was issued for this authorization. No automatic
retry or additional submission was issued while the submission moved from `PENDING` to `COMPLETE`.
