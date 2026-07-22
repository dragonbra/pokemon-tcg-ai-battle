# Alakazam V6 实验提交

V6 是基于 `alakazam_v5_auto_iter` runtime 的策略重构实验。`deck.csv` 保持与 V5
AutoIter 完全一致，改变只发生在策略状态、回合动作编排和效果选择。

## 本版重点

- 把攻击视为回合终止提交：先完成本回合不可逆的进化、附能、铺场、恢复和必要的 Ability。
- 记录每个己方回合开始时的 Active/Bench 实例、进场回合、进化回合，以及 Supporter、手填 Energy、Retreat 和攻击状态。
- Active Kadabra 带 Psychic Energy 且有 Alakazam 时，先自然进化；随后优先让合法 Bench Abra 进化成 Kadabra 并使用 Psychic Draw。
- Active Alakazam 已经能攻击时，有 Enriching Energy 和 Dudunsparce 则优先转为 Dudunsparce 过牌；其他能量优先补未充能 Abra 线。
- Active 不能 KO 而 Bench 有确定 KO 时使用 Boss，目标按剩余 HP 从高到低选择；没有 Boss KO 且对手 Active 已受伤、手牌至少 6 张时才使用 Xerosic。
- 牌库正好 10 张进入保护；Dudunsparce 的抽牌净消耗按实际附着卡计算；没有明确攻击路线时不使用恢复牌。
- V6 初版禁用 Trading Places，普通 Retreat 只用于把可以在本回合攻击的带能量 Alakazam 交到 Active。

## 固定卡组与验证

```bash
python3 scripts/check_assets.py
python3 -m unittest -v tests.test_alakazam_v6_strategy
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v6
```

策略执行流程见 [`strategy-flow.html`](strategy-flow.html)，设计背景见 [`REFACTOR_DESIGN.md`](REFACTOR_DESIGN.md)。本版本仍需通过 Kaggle replay 校准十张牌库保护线、恢复时机和连续攻击节奏。
