# Iter-19 Strategy Advisor

## 复核对象

本轮复核 `romanrozen_v9/game_003.json`、shared turn 5 的非终局 KO 前局面：Active
Alakazam 已带 Psychic，Bench 有两只已经带 Psychic 的 Kadabra，手牌有 Alakazam，
对方 Active 可被击倒但 Bench 仍有宝可梦。

## 卡牌与规则核验

- Alakazam 从手牌进化 Bench Kadabra 合法；进化不会消耗本回合手填能量次数，也不会
  把攻击机会提前结束。
- 已带 Psychic 的 Bench Kadabra 是真实的下一只攻击者；Bench 上只有 Dunsparce
  不算 ready attacker。
- 当前 Active 的 `Powerful Hand` 是回合终止动作。若本次 KO 不是最后奖赏闭环，应
  在宣告攻击前完成可见的 Bench Kadabra → Alakazam 接力准备。
- Alakazam 的 `Psychic Draw` 是额外的手牌资源，但是否继续使用要服从牌库保护线；
  本轮没有修改该过牌 gate。

## 边界

本轮只在“Active Alakazam 已带 Psychic + 当前攻击确定 KO + 非终局 + 手牌有 Alakazam
且存在合法、已带 Psychic 的 Bench Kadabra 进化选项”时提升 Bench 进化优先级。没有
可见的完整路线、终局已经闭合、或 Bench Kadabra 没有 Psychic 时，不强行进化。
