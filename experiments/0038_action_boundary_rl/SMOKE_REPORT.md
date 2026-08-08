# 0038 验收与短 Smoke

日期：2026-08-09。没有启动正式训练。

## Update-0

- Zero-Shot actor SHA256：`0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`
- pre-RL Value SHA256：`e88b2f18089911c4ebd810fa3abfba800a103189184b6265b9d38c03a9e36360`
- 公共 V3 update-0 SHA256：`d572c8673fcf16c39fb17d8d261bec6578885a69c227e579bc4e6f616bb22820`
- PPO updates：0；未加载 V5/RL-updated 权重。

## 验证

- 124/124 unit/property tests 通过。
- Phantom Dive exhaustive official parity：330 allocations，0 failure。
- INTEGRATED 2-game smoke：2 finished、0 error、39 forced shortcuts、5 macros、137 PolicyTransitions。
- smoke rollout/PPO：5.055s / 0.785s；CUDA peak allocated/reserved 1.120/1.726GB；process-tree RSS 2.527GB。
- `||g_prize|| / ||g_win|| = 0.0562`，固定 `alpha_prize=0.10` 满足约 25% 上限。
- root/allocation entropy：0.2683 / 0.0162；clip fraction 0；无 NaN/Inf。

Smoke JSON：`.tmp/evaluation/action_boundary_0038_integrated_smoke/gate.json`。

## Frozen 2,048

V3 使用 8 个互不重叠的 256-game shard，共 2,048 个唯一 seed，先后手各半：

- 1308-739-1，win rate 63.87%，Wilson 95% CI 61.76%–65.92%。
- 0 error，14 safe fallback（12 次 Bench `n>5`，2 次 confused/chance boundary）。
- 57,425 forced shortcuts；6,851 Phantom macros；176,568 PolicyTransitions。
- 1,149.2s，1.78 games/s；峰值 process-tree RSS 22.96GB。
- Meta 全面板指标未发布：当次正常 forward 已产生 logits，但聚合器只接受 Tensor、实际保存为 list。代码已修正；按“不得为统计增加每局 forward”合同，没有仅为该指标重跑面板。
- immediate prize opportunity/chosen-vs-max 同样未从压缩 trace 可靠重建，未追加模型调用。

权威报告：[V3 update-0](evaluation/V3_update0_chance_boundary_fallback.html)。V2 报告保留两局 chance-boundary error，作为修复前失败证据。

新旧 entropy、KL、clip fraction 的分母不同，不可与 V5 历史曲线直接比较。
