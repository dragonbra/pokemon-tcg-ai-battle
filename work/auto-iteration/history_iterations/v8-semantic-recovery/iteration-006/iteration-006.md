# iteration-006

## Context

- 类型：`strategy`
- control：iteration-005
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- command：`python3.13 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_006 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,crustle_v1,maktha_1084 --games 4 --save-traces --output /tmp/ptcg-v8-iteration-006-focus`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-006-focus --output-dir /tmp/ptcg-v8-iteration-006-analysis --agent-label v8_iteration_006`

## Hypothesis and change

补齐 planner 对“Active Alakazam 尚无 Psychic Energy、手牌有 Hilda”场景的
`supply_evolution_and_energy` purpose，使现有 resources policy 能在攻击前使用 Hilda
补齐当前攻击者路线。新增回归测试
`test_unenergized_active_alakazam_uses_hilda_for_psychic_energy`，先确认 RED，再完成
最小 production 修复。

## Result

- 对手结果：`romanrozen_v9 1/3`、`crustle_v1 3/1`、`maktha_1084 2/2`
- 总体 W/L/D：`6/6/0`
- agent error：`0`
- 胜率：`50.0%`
- Meta 加权胜率：`44.6%`
- 第二回合实际 `attackId=1072`：`1/12 (8.3%)`
- post-KO：`12` 次；zero-ready `6/12 (50.0%)`
- 出现过打手断档的对局：`4/12 (33.3%)`
- 分析 case：`39`

## Interpretation and decision

focused correctness 仍为 0 error；相较 iteration-005，本轮 focused 的胜率、第二回合
Powerful Hand 和 post-KO zero-ready 事件均改善。但 focused 只用于机制诊断，不能替代
17×10 promotion 证据，因此记录为 `observe`。下一步运行 fresh full Sample，检查该
语义修复是否能在统一 170 局口径下继续向 Target 的 `118/50/2`、`69.4%` 靠近。
