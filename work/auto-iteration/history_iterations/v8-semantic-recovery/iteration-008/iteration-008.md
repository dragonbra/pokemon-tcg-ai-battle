# iteration-008

## Context

- 类型：`strategy`
- control：iteration-007
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- command：`python3.13 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_008 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,crustle_v1,maktha_1084 --games 4 --save-traces --output /tmp/ptcg-v8-iteration-008-focus`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-008-focus --output-dir /tmp/ptcg-v8-iteration-008-analysis --agent-label v8_iteration_008`

## Hypothesis and change

加入旧版 bench insurance 的最小 Poffin gate：初始 2 Abra + Dunsparce 不再永久视为
接力完成，Active 已充能且没有可见 ready successor 时允许 Poffin。

## Result

- 总体 W/L/D：`4/8/0`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`0/12`
- post-KO：`14` 次；zero-ready `14/14`
- 出现过打手断档的对局：`7/12`

## Decision

focused 仅作机制观察，结果为 `observe`；post-KO 指标没有在本批次改善，不能晋级。
