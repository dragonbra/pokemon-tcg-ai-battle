# iteration-004

## Context

- 类型：`strategy`
- control：iteration-003
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- command：`python3 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current/main.py --label v8_iteration_004 --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/alakazam_v8_current --opponents romanrozen_v9,crustle_v1,maktha_1084 --games 4 --save-traces --output /tmp/ptcg-v8-iteration-004-focus`
- analyzer：`python3 -m scripts.alakazam_auto_iter analyze --report-dir /tmp/ptcg-v8-iteration-004-focus --output-dir /tmp/ptcg-v8-iteration-004-analysis --agent-label v8_iteration_004`

## Hypothesis and change

如果 planner 为合法 Active Abra→Kadabra 和 Active Alakazam 前 Bench Abra→Kadabra
建立 must-goal，continuity 就会在攻击提交前完成进化，减少击倒后的无打手断档。先写
两个失败回归测试，再修改 `planner.py` 与 `policies/continuity.py`。

## Result

- 对手结果：`romanrozen_v9 0/4`、`crustle_v1 4/0`、`maktha_1084 0/4`
- 总体 W/L/D：`4/8/0`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`0/12`
- post-KO：`7` 次；zero-ready `5/7`
- 出现过打手断档的对局：`3/12`
- 首次出现 Alakazam 的 engine turns：`9, 5, 3, 11, 4, 10`
- 分析 case：`28`，其中 bench insurance `11` 个均为有效进度或后续同回合 anchor 的
  诊断记录，不等同于 11 次失败。

## Interpretation and decision

进化链和 focused correctness 已改善，post-KO zero-ready 从 iteration-003 的 `8/8`
降到 `5/7`，但样本很小且第二回合 Powerful Hand 仍为 `0/12`，尚不能与 Target 或
170 局结果比较。`observe`。下一轮应继续追踪真实起手资源下的二回合进化/附能/攻击，
并区分“击倒时资源不可达”和“更早回合漏建 Bench 接力”。

