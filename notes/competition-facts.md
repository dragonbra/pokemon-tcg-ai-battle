# 比赛事实

信息源优先级：官方 Kaggle 比赛页面、官方 cabt 文档、比赛主办方公告。外部卡牌数据库只能帮助解释卡牌信息；卡池合法性和动作行为以比赛数据及 simulator 为准。

## 已核实事实

- 比赛类型：Simulation Competition。
- 提交物：可执行 agent，不是静态预测 CSV。
- 官方提供 cabt simulator/SDK，用于本地调试和强化学习。
- agent 每回合收到 observation，其中包含游戏日志、当前棋盘状态和合法选项；agent 返回选择项的索引。
- 对手手牌不直接暴露，只暴露手牌数量。
- 比赛数据包含英文/日文卡牌元数据和参考 PDF。
- 提交需要 `.tar.gz`，顶层必须有 `main.py` 和 `deck.csv`。
- 比赛页面给出的提交约束包括：每日最多 5 次提交、最多 2 个近期 active submission、提交包不超过 197.7 MiB；这些约束在正式提交前需要重新核对。
- 每次提交会先进行与自身对战的验证 episode；失败会被标记为 Error。
- 排名基于 agent episode 的技能评分，而不是单局造成的分数差。
- 官方明确说明 simulator 与正式 Pokémon TCG 规则存在差异，比赛中以 simulator 行为为准。
- 官方 simulator 文档：https://matsuoinstitute.github.io/cabt/
- Kaggle Simulation CLI 文档：https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md

## 重要规则差异

主办方当前公开说明包括：

- 某些在正式规则中可以“宣告后失败”的攻击，在 simulator 中可能从一开始就不可选；
- Mega Zygarde ex 的 Nullifying Zero 在 simulator 中自动按从左到右处理硬币和目标顺序；
- 双方同时 Knock Out 时，Prize 处理顺序与正式规则不同；
- 连续/被动效果一般由 simulator 自动处理，不需要 agent 每回合显式激活；显式 activated Ability 才会作为选项出现。

来源：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586

## 研究边界

比赛卡池是 simulator-specific 的封闭世界。不能把现实世界完整 Pokémon TCG 卡池或 Standard 赛制直接当作合法卡池。

## 重要 URL

- Overview：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
- Data：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data
- Code：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code
- Discussion：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion
- Rules：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules
