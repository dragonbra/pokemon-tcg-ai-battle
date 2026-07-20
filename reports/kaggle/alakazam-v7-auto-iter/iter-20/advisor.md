# Iter-20 Strategy Advisor

## 复核对象

本轮复核 `penguin_915/game_010.json`、shared turn 8 的 Bench 连续性局面：Active
Alakazam 已有 Psychic，Bench 有 Abra，弃牌区有 Basic Psychic，手牌有 Wondrous Patch，
当前攻击可以 KO 但不是最后奖赏闭环。

## 卡牌与规则核验

- Wondrous Patch 可以从弃牌区取 Basic Psychic 并附到符合条件的 Bench Pokémon；它
  不消耗本回合的手填 Energy 次数。
- Abra 本回合即使不能进化，也可以先接受 Psychic Energy；进化时机与附能时机是两个
  独立的规则窗口。
- 因此“手中尚未看到 Kadabra”不能作为拒绝本回合能量准备的理由。下一回合能否自然
  进化仍交给当时实际观察到的手牌和合法 options 判断。
- 终局奖赏闭环仍是例外：如果当前攻击立即拿完最后奖赏，不应为了未来接力使用 Patch。

## 本轮边界

只放宽了“可给 Bench Abra 先充能”的准备条件，没有放宽终局 gate、Item Lock、牌库
保护或进化合法性。实际矩阵结果表明这个 gate 可能仍过宽，下一轮应继续拆分“确实能
形成下一只打手”的资源路径。
