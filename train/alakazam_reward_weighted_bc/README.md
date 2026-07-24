# Alakazam Reward-weighted BC

这是胡地项目的阶段二实验 package。它从一个已冻结的 full-action BC checkpoint
继续训练，把可审计的离线 reward 转换成非负 imitation weight，并可同时预训练 value
head。它不是在线 PPO，也不会读取固定 Evaluation 报告作为训练数据。

完整的开关、奖励语义、0009 消融矩阵和实验记录框架见 [`DESIGN.html`](DESIGN.html)。

## 边界

- 复用 `train/alakazam_bc_rl` 的 `ptcg_features_universal`、完整多选动作合同和 checkpoint。
- 数据仍须来自一个明确的 expert team；train/validation/test 按 episode 冻结。
- reward annotator 从原始官方 replay 或冻结 zip 一次性生成轻量 sidecar；训练热路径不反复
  解析 replay，也不复制 4.5 GB encoded dataset。
- 正式大数据训练按 split 顺序流式读取，并用固定 shuffle buffer 打乱；不会把完整 JSONL
  展开到内存。
- 所有实验项由 JSON 配置显式开关；`control.json` 是不使用 reward 和新特征的对照。
- `exploration_template.json` 只默认打开 terminal outcome；其它权重是待验证模板，不是结论。
- test 默认不读取。checkpoint 先按 validation exact-action rate 冻结，再由用户决定何时读 test
  和运行官方固定 Evaluation。

## 已实现开关

| 模块 | 开关/模式 | 默认边界 |
|---|---|---|
| 精确卡牌类别 embedding | `card_category_embedding.enabled` | 关闭；9 类来自冻结常量映射 |
| reward component | `reward.components[*].enabled` | 逐项开关、独立权重与 clip |
| imitation weighting | `uniform` / `linear` / `exponential` | 权重始终非负并限制范围 |
| advantage baseline | `zero` / `dataset_mean` / `value` | value 始终 detach，不反传进权重 |
| value target | `loss.value` | 关闭时保持纯 imitation |
| value warm-up | `value_warmup_epochs` | 可只训练 value head 后再联合训练 |
| value output reset | `reset_value_output` | 显式开关，不静默改变 BC checkpoint |

精确卡牌类别共有九类：Basic/Stage 1/Stage 2 Pokémon，Item、Supporter、Pokémon
Tool、Stadium，Special/Basic Energy。现有 universal schema 已有压缩后的 12 维 card
metadata；新 embedding 是并存的结构实验，不替换原 metadata，也不要求重建 encoded dataset。
`card_categories.py` 已把官方 CSV 中 1,267 个唯一 card ID 固化为
`OFFICIAL_CARD_CATEGORY_BY_ID`。训练和推理只做内存查表，不在运行时读取或解析 CSV；映射版本、
映射 hash 和源 CSV hash 会写入 checkpoint metadata。只有官方卡牌表实际更新时，才需要显式
重新生成并审计这份常量。

## Reward annotation

```bash
python3 -m train.alakazam_reward_weighted_bc.annotate_rewards \
  --dataset <full-action.jsonl> \
  --archive-root <directory-containing-frozen-replay-zips> \
  --output <reward-sidecar.jsonl>
```

annotator 按 `(episode_id, player_index, episode_step)` 保持与 base dataset 相同顺序，只保存：

```text
reward_schema_version
reward_metrics.episode.*
reward_metrics.decision.*
reward_metrics.audit.*
```

同时生成 `.train.jsonl`、`.validation.jsonl`、`.test.jsonl` sidecar 和包含 replay hash、
标签计数、输入/输出 hash 的 `.reward_audit.json`。训练时逐条校验 dataset 与 sidecar identity，
不允许静默错位。`episode_step` 严格解释为 dataset builder 使用的最长 `visualize` trace frame，
动作来自同一 frame 的 `selected`；先后手扫描到首个合法 `firstPlayer` 后再确定 turn 3/4。

当前实现的原始信号：

- terminal outcome；
- 起手可见 Abra 时是否把 Abra 设为 Active；
- 实际先后手、对局最大 engine turn；
- 第二己方回合是否提交 Powerful Hand；
- Dunsparce 特殊路线的逐步组件、第一回合铺 Abra credit、第二回合完整执行 credit；
- 每次攻击的 Prize delta、未拿 Prize 攻击及 Powerful Hand 子集；
- Alakazam 被击倒后的下一己方回合接力 credit；
- 接力失败中的可恢复弃牌区 miss。

0008/0009 冻结 corpus 是 winner-only，因此 `terminal_outcome` 在训练集恒为 `+1`。在这份
数据上，terminal-only 配合 dataset-mean baseline 不会产生样本间权重差异，不能作为有意义的
奖励消融；保留该通用组件只是为了兼容未来同时包含胜负局的数据。

起手可见 Abra 时，Active Abra 记 `+1`，选择其它 Active 记 `-1`；没有 Abra 时记 `0`。
因此 Dunsparce bridge 只处理“起手确实没有 Abra、被迫以 Dunsparce Active 开局”的样本，
不会奖励模型故意把 Dunsparce 放到前场。

Dunsparce 完整路线使用严格观测定义：第一己方回合把 Abra 放到 Bench；第二己方回合
Active Dunsparce 进化为 Dudunsparce，明确选择 `Run Away Draw` Ability，Ability 结算后
Abra 成为 Active，随后进化为 Alakazam，并在实际持有 Psychic Energy 时提交 Powerful
Hand。这里不使用普通 Retreat 作为证据，也不限定 Psychic Energy 在第一还是第二回合附着；
能量必须最终位于发起攻击的 Abra/Alakazam 进化线上。

暂未把 Xerosic 手牌保留、Poffin 目标质量、Hammer/Mist 的反事实机会写成 reward。这些需要
更明确的合法候选和反事实定义，否则容易奖励“结果碰巧正确”而不是正确决策。

## Training

先分配全新的实验和版本，不能写回纯 BC run：

```bash
python3 -m rl_environment.runs create alakazam_reward_weighted_bc \
  --objective "Stage-two reward-weighted full-action BC"

python3 -m train.alakazam_reward_weighted_bc \
  --dataset <full-action.jsonl> \
  --reward-sidecar <reward-sidecar.jsonl> \
  --source-checkpoint <bc-v4-checkpoint-or-model.bin> \
  --output rl_runs/artifact/<experiment>/V1_control_finetune \
  --config train/alakazam_reward_weighted_bc/configs/exploration_template.json
```

每个开关组合必须使用新的 `V<n>_<tag>`。至少保留 terminal-only 对照，再一次只改变一个
feature/reward family。离线 imitation、value MAE 和 effective sample ratio 只用于训练健康；
最终判断仍使用完整固定 Evaluation 的 outcome/correctness 护栏和 setup、relay、attack-quality
过程指标，不能让过程奖励掩盖胜率或错误。
