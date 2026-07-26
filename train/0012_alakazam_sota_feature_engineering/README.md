# Alakazam Sota Feature Engineering

`0012-alakazam_sota_feature_engineering` 在完全冻结的 0010 Yushin Ito
daily winner-only corpus 上开展纯 Behavior Cloning 输入与网络结构消融。

当前共同合同：

- source dataset SHA-256：
  `36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf`；
- train 221,289 decisions / 2,875 episode-player groups；
- validation 17,067 decisions / 229 groups；没有 test records；
- d_model 320、8 heads、4 Transformer layers、GRU pointer；
- AdamW 3e-4、weight decay 0.02、batch 96；
- 每 epoch 完整 train/validation 评测；validation exact 连续 5 epoch 不创新高时停止；
- 每个版本随机初始化，不覆盖旧 run；每次训练自动生成 `ANALYSIS.md`。

已实现的首批版本：

- `V1_baseline_control`：与 0010 前向路径逐权重、逐 logit 一致的 control；
- `V2_no_position_permutation`：删除 action option position，训练时随机置换候选并重映射标签；
- `V3_role_separated_action`：在 V2 基础上为 source/target 使用独立角色投影。
- `V4_action_primitive_context`：首次增加 primitive 的 run；因 permutation 未同步
  重排 `option_primitive_cat` 而中止，只作失败审计；
- `V5_action_primitive_permutation_fix`：修复后从随机初始化重训
  attack/effect/context/count/energy/tool 语义；
- `V6_chosen_transition_aux`：首次 transition run；因 AMP stop-mask dtype overflow
  在首 batch 失败，只作失败审计；
- `V7_chosen_transition_amp_fix`：修复后在有效 primitive baseline 上增加
  chosen-action next-decision delta 与
  turn-change 辅助监督；跨回合资源 delta 被 mask，不构造未选动作标签。完成 21 个
  epoch 后按用户要求在 epoch 22 中止；best exact `0.843851`，未超过 V5。
- `V8_action_history`：已准备但按用户要求暂停，不启动训练；设计为从 V3 独立分支加入
  最多 16 个过去本方 selected-option
  history token；history 进入 state Transformer K/V，不读取未来动作。
- `V9_live_action_count_contract`：不训练新权重；修复 live inference 将 simulator
  `minCount/maxCount` 静默截断到训练 `max_action_steps=16` 的 action contract 缺陷。
  GRU pointer 在推理时可继续解码到真实 option/count 上限；冻结 dataset 与离线训练
  仍明确不包含 action length >16 的标签。

运行示例：

```bash
python3 -m train.alakazam_sota_feature_engineering train \
  --dataset-root rl_runs/dataset/0010-alakazam_sota_model \
  --output rl_runs/artifact/0012-alakazam_sota_feature_engineering/V1_baseline_control \
  --config train/0012_alakazam_sota_feature_engineering/configs/0012_v1_baseline_control.json
```

本项目不修改 `engine/source/`。离线 BC 指标只证明模仿质量；所有策略强度结论必须来自
官方 engine runtime 的固定 opponent pool 真实对局。
