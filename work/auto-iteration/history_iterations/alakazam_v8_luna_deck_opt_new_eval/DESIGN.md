# Alakazam V8 Luna Deck Opt New Eval：设计记录

## 目标

在保持 V8 已准备好的 60 张卡组不变的前提下，直接优化
`work/alakazam_v8_current/main.py`，让 agent 的实际行为忠实反映
`work/docs/DECK_NOTES.md` 与 V6 规则约束。`work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/`
只保存评测、测试和决策记录，不保存第二份策略源代码。

## 设计边界

- 策略源固定为 `work/alakazam_v8_current/`。
- `deck.csv` 不改；官方 `cg/` runtime 不改；`evaluation/` 和 `engine/source/` 不改。
- 不让 agent 自动读取 `DECK_NOTES.md`；代码中的规则是人工依据该文档实现的确定性行为。
- 不执行 Kaggle submission、git commit 或 git push。
- 所有对外报告使用 `auto_iteration_v8_setup_relay` revision 2；baseline 与 candidate 使用相同的 17×10 评测口径。

## 核心行为模型

将当前回合视为显式的阶段门，而不是全局分数排序：

1. 同步回合开始快照、资源区域、一次性资源和 Item Lock 状态。
2. 处理高价值且不应错过的 Ability、Stadium、合法特殊能量干扰和回收动作。
3. 建立当前攻击者和下一只攻击者：合法进化优先于低价值攻击；Active Abra 的 Rare Candy 路线要保留，Bench Abra 可以先自然进化为 Kadabra 抽牌。
4. 处理实际有收益的搜索、附能、Supporter、Dudunsparce 接力和牌库预算。
5. 最后才检查攻击；攻击提交后本回合结束。永不提交 `Trading Places`，Abra 的非 KO 攻击只作最后手段。

## 本轮优先实现的 deck 语义

- Dudunsparce 同时是抽牌和换位资源：Active Dunsparce 在后场存在已带 Psychic 的 Alakazam、或存在明确的本回合接力路线时，优先寻找/进化 Dudunsparce 并使用 `Run Away Draw`。
- Fezandipiti ex 在上一回合己方 Pokémon 被击倒、Ability 合法可用时，主动使用 `Flip the Script`；不把它误算成 Supporter。
- 对手有合法 Special Energy 目标时，Enhanced Hammer 一定进入可用动作序列；目标优先 Active 的 Mist/Rock 等保护性特殊能量，再按场上目标确定性选择。
- 手里有合法 Nighttime Mine 时立即使用，用来覆盖对方 Stadium；不要求对手已经有 Tera Pokémon。
- Sacred Ash 在合法时尽可能选择最多五只弃牌区 Pokémon，优先补足 Abra/Kadabra/Alakazam 接力资源，再补 Dunsparce 系列和其它宝可梦。
- Lana’s Aid 以恢复 Abra 系列接力为主，能拿多张合法 Pokémon 就不只拿一张；只有攻击路线确实缺 Basic Psychic 时才把 Basic Energy 纳入主要目标。
- Dawn/Hilda/Poké Pad 的检索目标必须服务明确的进化、攻击或 Dudunsparce 接力；第一回合场面已经能铺足两只 Abra 和一只 Dunsparce 时保留 Poké Pad。
- Enriching Energy 不能支付 Psychic 攻击；在 Alakazam 已能攻击且 Dudunsparce draw 路线成立时，用它触发过牌，但仍遵守牌库保护线。

## 验证策略

每个行为按照 RED → GREEN → 回归执行：先在新的
`tests/test_alakazam_v8_sol_deck_opt_strategy.py` 写最小 fixture，确认当前策略没有达到目标；
再修改 `main.py` 的最小决策分支；最后运行定向测试和完整 V8 测试集。

评测结果按以下顺序解释：G0 correctness/agent error 是硬护栏；总体及先后手胜率是结果护栏；
二回合 Powerful Hand 和 Post-KO relay 是主目标；攻击但未拿奖赏率、空 Bench Run Away Draw、
牌库压力和对局长度是辅助审计，不能抵消 correctness 回退。
