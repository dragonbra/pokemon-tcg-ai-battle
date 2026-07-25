# 0011 · Alakazam SOTA Reward-Weighted BC

0011 保持 0010 的 ID-only public-state codec、7.15M Transformer + GRU pointer policy、
Yushin Ito winner-only corpus、split 与动作标签，只改变训练时各 decision 对 token CE 的权重。
它仍是离线 Behavior Cloning，不包含 value head、rollout、PPO 或在线 RL。

0010 的 checkpoint A/B 已证明最低 validation loss 比最高 teacher-forced exact 更能对应当前
Kaggle 强度：epoch 9 loss-best 为 `944.2`，epoch 21 exact-best 为 `600.0`。因此 0011 使用
未加权 validation token CE 作为主选择与早停指标，exact 只作辅助观察。

## 数据合同

`build-dataset` 严格按 `(episode_id, player_index, episode_step)` 一一 join：

- `rl_runs/dataset/0010-alakazam_sota_model/*.jsonl.gz`：模型输入与 action label；
- `rl_runs/dataset/0009-reward_weighted_bc/reward_sidecar.*.jsonl`：已审计 reward metrics；
- 输出：`rl_runs/dataset/0011-alakazam_sota_reward_weighted_bc/*.jsonl.gz`。

任何 identity 错位、缺行、多行、schema 变化或 expert 变化都会失败。正式数据仍是单一
`Yushin Ito` policy：221,289 train decisions / 17,067 validation decisions。

## 目标函数

每条 decision 的 scalar reward 由开关组件线性组合并裁剪。使用 dataset-mean baseline 后，
通过 exponential / linear / uniform 配置转成非负 imitation weight。该 weight 乘到这条
decision 的每一个 option / STOP token：

```text
advantage_i = reward_i - mean_train_reward
weight_i = clip(exp(advantage_i / temperature), min_weight, max_weight)
L = sum_i,t weight_i * CE(logits_i,t, target_i,t)
    / sum_i,t weight_i * valid_token_i,t
```

当所有 weight 为 1 时，公式严格退化为 0010 的 token-mean CE。validation 始终不加权。

## 正式入口

```bash
python3 -m train.alakazam_sota_reward_weighted_bc build-dataset \
  --base-root rl_runs/dataset/0010-alakazam_sota_model \
  --reward-root rl_runs/dataset/0009-reward_weighted_bc \
  --output rl_runs/dataset/0011-alakazam_sota_reward_weighted_bc

python3 -m train.alakazam_sota_reward_weighted_bc train \
  --dataset-root rl_runs/dataset/0011-alakazam_sota_reward_weighted_bc \
  --output rl_runs/artifact/0011-alakazam_sota_reward_weighted_bc/V1_all_rewards_lr4e4 \
  --config train/alakazam_sota_reward_weighted_bc/configs/0011_v1_all_rewards.json

python3 -m train.alakazam_sota_reward_weighted_bc export-candidate \
  --checkpoint rl_runs/checkpoint/0011-alakazam_sota_reward_weighted_bc/<version>/best_validation.pt \
  --source-package evaluation/arena/candidates/0010_v4_loss_best \
  --output evaluation/arena/candidates/<candidate-name>
```

正式配置要求 `minimum_epochs_before_stop >= 3`；V1 设置为至少 8 epochs、最多 20 epochs、
validation loss patience 5。每次消融必须分配新的 `V<n>_<tag>`，不得覆盖曲线或 checkpoint。

V1 全奖励在 epoch 11 达到最低未加权 validation loss `0.228867`，对应 exact-action
accuracy `83.6878%`；此后连续五轮未刷新 loss，于 epoch 16 正常早停。相同数据上的 0010
loss-best 是 `0.232557` / `83.0199%`。V2 只关闭覆盖全体 decision 的
`turn_count_penalty`，保持其余奖励、学习率、seed 和训练门槛不变，用于测量该信号的边际作用。

V2 在 epoch 9 达到 loss-best `0.220601` / exact `83.6116%`，随后五轮未刷新并于
epoch 14 早停；相对 V1 loss-best 改善 `0.00827`，因此后续固定关闭
`turn_count_penalty`。V3 在此基础上只额外关闭通用 `non_prize_attack_penalty`，保留
`powerful_hand_non_prize_penalty`，用于隔离两项相关攻击惩罚的边际作用。

V3 loss-best 为 epoch 10 的 `0.223942` / `83.3656%`，劣于 V2，因此通用
`non_prize_attack_penalty` 恢复启用。V4 回到 V2，只关闭 Powerful Hand 专项
non-prize penalty，以判断通用惩罚是否已经足够。

每个已完成版本均以最低 validation loss checkpoint 运行正式官方 engine 20×10 评测：

- V1：150W / 46L / 4 errors，计划总局数胜率 75.0%；
- V2：157W / 40L / 2 errors，胜率 78.5%；
- V3：160W / 39L / 1 error，胜率 80.0%。

V3 的单次官方胜率与二回合 Powerful Hand 率较高，但离线 loss、Dunsparce bridge 和
Post-KO relay 不如 V2 均衡。不同 run 的随机胜负不能验证并行语义一致性，因此当前不以
一次 200 局胜率覆盖离线消融结论。

V4 关闭 Powerful Hand 专项 penalty 后在 epoch 10 达到 loss-best `0.221716` /
`83.5179%`，15 epochs 正常早停；官方 engine 为 149W / 47L / 4 errors（74.5%）。其
离线 loss 略差于 V2，官方胜率和 Post-KO relay 也没有提供支持，因此 V5 恢复该专项
penalty，只关闭与负向 Post-KO relay credit 重叠、且仅影响 591 条 train decision 的
`recoverable_discard_miss_penalty`。

V5 在 epoch 9 达到 loss-best / exact-best `0.221684` / `83.4769%`，14 epochs 正常
早停；官方 engine 为 153W / 40L / 7 errors（76.5%）。它仍未超过 V2 的最低 validation
loss，也没有形成更好的综合官方结果，因此不移除 `recoverable_discard_miss_penalty`。
V1–V5 完成后停止追加训练；本轮推荐 V2 setting，V3 的一次 80.0% 官方胜率保留为需要
重复评测才能确认的观察。

## 最终归档

用户最终按官方对局表现选择 `V3_no_turn_count_or_generic_non_prize` 发布；这项发布决策不改写
上面的消融解释。归档固定使用 epoch 10 的最低 validation loss checkpoint：

- submission source：`submission/alakazam_sota_reward_weighted_bc_v3_loss_best/`；
- archive：`submission/dist/alakazam_sota_reward_weighted_bc_v3_loss_best.tar.gz`；
- archive SHA-256：`b9f9c4e4a5fdb90147e48155cdd72aa3195e37d5aabc4f4f3ad8e3b6acb471ef`；
- exported model SHA-256：`573e0a0263ea199d09fb929dd9d4e0dc68e002d941c750c98deb90b88cdf6601`；
- Kaggle submission ref：`54971599`，提交后首次查询状态为 `PENDING`。

本次授权只执行了这一条 Kaggle CLI submission；后续状态查询不得触发重复提交。

开关、权重、数据流和实验 ledger 见 [`DESIGN.html`](DESIGN.html)。
