# 0012 completed-checkpoint official evaluation

## 评测合同

- 时间：2026-07-25。
- runtime：仓库当前官方 engine runtime；未修改 `engine/source/`。
- opponent：当前 checkout 中全部 20 个启用 package。
- 对局数：每个 opponent 10 局，共 200 局 / checkpoint。
- 并行：`--workers 8 --worker-cpu-threads 1`。
- metric profile：`auto_iteration_v8_setup_relay`（revision 7）。
- checkpoint 选择：每个已完成且编码有效的 run 统一使用 `best_exact.pt`。
- deck SHA-256：
  `0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451`。
- cg tree SHA-256：
  `8a25d7b3497cf4e7c950f3b5038be5debf0f8157fd24abdfd3a730fe863b60ac`。

V4 的 option primitive 编码已确认错配，V6 在首 batch AMP overflow，二者均不进入
候选导出或官方强度比较。

## 结果

| 版本 | validation exact | 胜 | 负 | error | 报告胜率 | 完成率 |
|---|---:|---:|---:|---:|---:|---:|
| V1 baseline control | 0.838929 | 162 | 36 | 2 | 81.0% | 99.0% |
| V2 no position + permutation | 0.837698 | 146 | 52 | 2 | 73.0% | 99.0% |
| V3 role-separated action | 0.843792 | 153 | 46 | 1 | 76.5% | 99.5% |
| V5 action primitive fix | 0.845023 | 155 | 42 | 3 | 77.5% | 98.5% |

报告：

- V1: `rl_runs/evaluation/0012-alakazam_sota_feature_engineering/V1_baseline_control.html`
- V2: `rl_runs/evaluation/0012-alakazam_sota_feature_engineering/V2_no_position_permutation.html`
- V3: `rl_runs/evaluation/0012-alakazam_sota_feature_engineering/V3_role_separated_action.html`
- V5: `rl_runs/evaluation/0012-alakazam_sota_feature_engineering/V5_action_primitive_permutation_fix.html`

## 符合假设的结果

- 四个 checkpoint 都能导出为自包含标准 package，并通过 60-card、deck hash 与 cg
  runtime hash 验证；本轮确实比较了同一卡组、同一 runtime 下的模型差异。
- V3/V5 的候选顺序等变设计没有破坏闭环运行能力；两者均完成至少 197/200 局。
- V5 相对 V3 的离线 exact 小幅提升，在本轮对局中也得到方向一致但很小的
  `+2` 胜局差；这只能作为后续复验信号，不能单独证明提升。

## 不符合或尚未证明假设的结果

- 离线 exact 排序为 V5 > V3 > V1 > V2，本轮报告胜率却为
  V1 > V5 > V3 > V2；更高的 imitation exact 没有单调转化为更高闭环胜率。
- 所有 run 都出现 1--3 个 `game_error`，因此 correctness 护栏没有通过。V1/V3
  的 error evidence 位于 candidate 回合；V2/V5 同时包含 candidate 与 opponent
  回合错误。默认 trace 已按正式 artifact policy 删除，当前报告不足以恢复完整异常文本。
- 四次评测是独立 simulator 随机 run，不是共享随机种子的 paired evaluation；不能仅凭
  当前胜局差宣称 V1 是确定性策略 SOTA。
- V5 `best_exact.pt` 的 validation loss 已明显劣于它的 `best_validation.pt`；本轮没有
  回答 best-loss checkpoint 是否具有更好的闭环策略强度。

## 下一步措施

- 将 error 视为准入阻断，而不是从分母中静默删除；需要时在 `.tmp/evaluation/` 对相关
  opponent 使用 `--keep-temp` 做可删除诊断，区分非法 action sequence 与 engine/opponent
  异常。
- V7/V8 完成后，优先评测其最佳 checkpoint，并补充 V5 `best_validation.pt` 对照。
- 最终晋级判断同时看零 error 的官方对局、离线 exact、validation loss 与候选置换等变，
  不根据单一随机 200 局或单一离线指标决定。

## 实际执行内容

- 导出并验证：`evaluation/arena/candidates/0012_v1_exact_best`、
  `0012_v2_exact_best`、`0012_v3_exact_best`、`0012_v5_exact_best`。
- 每个候选对当前完整 catalog 运行 200 局官方 engine 对战。
- 四份完整 `report.html` 保留在对应训练版本的 `rl_runs/evaluation/` 路径。
- 评测与 V7 GPU 训练并行执行；未发生 GPU OOM，训练 checkpoint 身份未改变。
