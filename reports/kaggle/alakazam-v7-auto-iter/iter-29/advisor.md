# AutoIter 29 Strategy Advisor

## 本轮假设

iter-28 的接力资格修复改善了 post-KO 指标，但它也会在己方第二回合看到“未充能的
Bench Abra 线 + Poffin”时延迟 Powerful Hand。二回合指标是本项目的重要全局观察量；如果
Bench 已经有 Abra/Kadabra/Alakazam，Poffin 是未来资源，不应覆盖本回合可以完成的攻击。

## 规则边界

- 二回合已有 Abra 线但没有直接附能/Telepath/Patch/回收路线时，保留 Powerful Hand。
- Bench 为空或只有 Dunsparce 时，Poffin/直接放下 Basic 仍属于硬性 Bench insurance。
- 当前可立即附 Psychic、使用 Telepath、Wondrous Patch 或 Lana's Aid 完成接力时，继续
  在攻击前处理该路线；本轮没有放宽这类真实资源转换。
- 终局 KO、Rare Candy、Mist/Rock Fighting Energy 保护和固定 deck 不变。

## 验收 fixture

新增“二回合已有未充能 Bench Kadabra + Poffin → Powerful Hand”的 fixture；iter-28 的
“第三回合同场面 → Poffin”和“无 anchor → 攻击” fixture 继续通过。
