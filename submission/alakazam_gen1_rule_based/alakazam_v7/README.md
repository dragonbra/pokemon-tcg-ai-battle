# Alakazam V7 实验提交

V7 基于 V6 的运行时和固定卡组，只重构策略。卡组文件必须与 V6 完全一致；策略重点
是按真实宝可梦 TCG 回合预算规划进化、附能、过牌、换位和攻击。

## 本版重点

- Active Abra 不攻击；带能量的 Active Abra 保留给完整的 Rare Candy + Alakazam 路线。
- Active 是 Shaymin 等非攻击者时，带 Psychic Energy 的 Bench Abra 保留给 Rare Candy；其它无能量 Bench Abra 先自然进化成 Kadabra。
- 第一只 Alakazam 可以攻击后，Bench Abra 默认自然进化为 Kadabra 并使用 Ability，
  不连续消耗 Rare Candy 制造没有立即收益的第二只 Alakazam。
- 只有观察到对手 `Budew (235)` 使用 `Itchy Pollen (323)` 后，才在我方下一个回合
  进入 Item Lock 路线：所有 Item 暂停，优先自然进化，并把 Enriching Energy 给
  Dunsparce/Dudunsparce 过牌。
- Dudunsparce 的 Run Away Draw 或普通 Retreat 可以在终局闭环中把带 Psychic Energy
  的 Alakazam 送到 Active；Trading Places 永不选择。
- Kadabra 非 KO 攻击仅是最后手段。牌库 10 张是保护线，只有当过牌能在本回合闭合
  最后 Prize 时才允许越过保护线。

## 验证

```bash
python3 scripts/check_assets.py
python3 -m unittest -v tests.test_alakazam_v7_strategy tests.test_alakazam_v6_strategy
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v7
```

V7 的行为验收样例和 replay case 映射见 [`REFACTOR_DESIGN.md`](REFACTOR_DESIGN.md)。
官方 Kaggle replay 才是策略表现依据，本地 battle 只用于合法性、崩溃和状态泄漏检查。
