# iteration-010

## Context

- 类型：`strategy`
- control：iteration-009
- candidate：`work/alakazam_v8_current`
- 范围：Auto-Iteration Sample 固定 17 个对手 × 10 局，共 170 局 full
- command：`python3.13 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_010 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,pilkwang_v2,kokinn_search,penguin_915,crustle_wall,crustle_v1,kiyotah_lucario,kiyotah_dragapult,kiyotah_iono,kiyotah_abomasnow,kacchan_anti_wall,nursrijan_lucario,yakitori_raging_bolt,zoli_dragapult,sue_alakazam,maktha_1084,yanxiaohan --games 10 --save-traces --output /tmp/ptcg-v8-iteration-010-full`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-010-full --output-dir /tmp/ptcg-v8-iteration-010-analysis --agent-label v8_iteration_010`

## Result

- 总体 W/L/D：`58/112/0`
- agent error：`0`
- 胜率：`34.1%`
- Meta 加权胜率：`32.0%`
- 第二回合实际 `attackId=1072`：`9/170 (5.3%)`
- post-KO：`208` 次；zero-ready `166/208 (79.8%)`
- 出现过打手断档的对局：`79/170 (46.5%)`
- 分析 case：`482`

## Interpretation and decision

当前攻击者先附 Psychic 的语义没有引入错误，full 胜率较 iteration-007 提升，
但仍远低于 Target，且 post-KO 接力断档没有改善。记录为 `observe`，下一轮优先
研究 post-KO 时的恢复、铺场和能量接力。
