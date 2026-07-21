# iteration-003

## Context

- 类型：`strategy`
- control：iteration-002
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- command：`python3 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_003 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,crustle_v1,maktha_1084 --games 4 --save-traces --output /tmp/ptcg-v8-iteration-003-focus`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-003-focus --output-dir /tmp/ptcg-v8-iteration-003-analysis --agent-label v8_iteration_003`

## Hypothesis and change

如果 commit 层拒绝 Active Abra 的攻击，策略就不会用 `attackId=1070` 绕过进化路线，
并会保留进化/过牌/接力的动作机会。新增失败测试后，`commit.py` 排除 Active Abra。

## Result

- 对手结果：`romanrozen_v9 1W/3L/0D`、`crustle_v1 1W/3L/0D`、`maktha_1084 0W/4L/0D`
- 总体 W/L/D：`2/10/0`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`1/12`
- post-KO zero-ready：`8/8`
- 出现过打手断档的对局：`4/12`
- 分析 case：`20`

## Interpretation and decision

Abra 1070 误攻的规则缺口已由 focused trace 验证修复，但 agent 在合法进化选项存在
时仍会 END；根因转移到 planner/continuity 没有建立 Abra→Kadabra goal。`observe`。
