# iteration-009

## Context

- 类型：`strategy`
- control：iteration-008
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- command：`python3.13 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_009 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,crustle_v1,maktha_1084 --games 4 --save-traces --output /tmp/ptcg-v8-iteration-009-focus`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-009-focus --output-dir /tmp/ptcg-v8-iteration-009-analysis --agent-label v8_iteration_009`

## Hypothesis and change

在 Poffin gate 之外，当前 Active 缺 Psychic 时优先完成合法 Psychic/Telepath 附能，
避免 setup 阶段先消耗动作窗口。

## Result

- 总体 W/L/D：`4/8/0`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`1/12`
- post-KO：`20` 次；zero-ready `14/20`
- 出现过打手断档的对局：`7/12`

## Decision

focused correctness 保持通过，记录为 `observe`；继续用 full 检查实际收益。
