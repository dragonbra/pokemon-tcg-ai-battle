# iteration-005

## Context

- 类型：`strategy`
- control：iteration-004
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- command：`python3.13 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_005 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,crustle_v1,maktha_1084 --games 4 --save-traces --output /tmp/ptcg-v8-iteration-005-focus`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-005-focus --output-dir /tmp/ptcg-v8-iteration-005-analysis --agent-label v8_iteration_005`

## Hypothesis and change

iteration-004 后新增了 Active Kadabra 已不能继续自然进化时，攻击前优先完成合法
Bench Abra→Kadabra 的 must-goal，并新增对应回归测试。本轮只验证该修复是否保持
correctness 和接力机制，不把小样本胜率作为晋级依据。

## Result

- 对手结果：`romanrozen_v9 0/4`、`crustle_v1 4/0`、`maktha_1084 1/3`
- 总体 W/L/D：`5/7/0`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`0/12`
- post-KO：`15` 次；zero-ready `13/15`
- 出现过打手断档的对局：`5/12`
- 分析 case：`36`

## Interpretation and decision

focused correctness 保持为 0 error；新规则没有在这批样本中造成运行时错误，但
post-KO 指标较 iteration-004 变差，且二回合 Powerful Hand 仍为 `0/12`。由于这是
独立随机的 12 局 focused 样本，不能据此判断策略回退或晋级。`observe`。下一步先
完成统一 HTML 归档，再运行 fresh `17×10` full Sample。
