# iteration-007

## Context

- 类型：`strategy`
- control：iteration-006
- candidate：`work/alakazam_v8_current`
- 范围：Auto-Iteration Sample 固定 17 个对手 × 10 局，共 170 局 full
- command：`python3.13 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_007 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,pilkwang_v2,kokinn_search,penguin_915,crustle_wall,crustle_v1,kiyotah_lucario,kiyotah_dragapult,kiyotah_iono,kiyotah_abomasnow,kacchan_anti_wall,nursrijan_lucario,yakitori_raging_bolt,zoli_dragapult,sue_alakazam,maktha_1084,yanxiaohan --games 10 --save-traces --output /tmp/ptcg-v8-iteration-007-full`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-007-full --output-dir /tmp/ptcg-v8-iteration-007-analysis --agent-label v8_iteration_007`

## Result

- 总体 W/L/D：`44/126/0`
- agent error：`0`
- 胜率：`25.9%`
- Meta 加权胜率：`23.7%`
- 第二回合实际 `attackId=1072`：`7/170 (4.1%)`
- post-KO：`188` 次；zero-ready `150/188 (79.8%)`
- 出现过打手断档的对局：`73/170 (42.9%)`
- 分析 case：`482`

## Interpretation and decision

Hilda 当前 Alakazam 补能量的修复保持 0 error，但 full 结果仍显著低于 Target；
`post_ko_no_ready_attacker` 是主要失败族。记录为 `observe`，后续转向旧版后场
insurance/handoff 语义。
