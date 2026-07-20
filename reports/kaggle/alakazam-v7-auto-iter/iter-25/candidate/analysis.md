# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：104 / 61 / 5
- 我方错误：0
- 胜率：61.2%
- Meta 加权胜率：63.0%
- 第二回合 Powerful Hand：40/170 (23.5%)
- 我方被击倒事件：170
- 击倒后无 ready attacker：111/170 (65.3%)
- 出现过打手断档的对局：62/170 (36.5%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 365 |
| post_ko_no_ready_attacker | 111 |
| second_turn_powerful_hand_missing | 130 |

## 评测配置

```json
{
  "agent": "/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/submission/alakazam_v7_auto_iter/main.py",
  "command": [
    "/opt/homebrew/opt/python@3.13/bin/python3.13",
    "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py",
    "--agent",
    "/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/submission/alakazam_v7_auto_iter/main.py",
    "--label",
    "alakazam_v7_auto_iter_iter25_bench_stage_attack_priority",
    "--opponents",
    "romanrozen_v9,pilkwang_v2,kokinn_search,penguin_915,crustle_wall,crustle_v1,kiyotah_lucario,kiyotah_dragapult,kiyotah_iono,kiyotah_abomasnow,kacchan_anti_wall,nursrijan_lucario,yakitori_raging_bolt,zoli_dragapult,sue_alakazam,maktha_1084,yanxiaohan",
    "--games",
    "10",
    "--save-traces",
    "--output",
    "/private/tmp/alakazam-v7-auto-iter-25-full.9ANn1x",
    "--cg-path",
    "/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/submission/alakazam_v7_auto_iter"
  ],
  "evaluator_root": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle",
  "games": 10,
  "label": "alakazam_v7_auto_iter_iter25_bench_stage_attack_priority",
  "opponents": [
    "romanrozen_v9",
    "pilkwang_v2",
    "kokinn_search",
    "penguin_915",
    "crustle_wall",
    "crustle_v1",
    "kiyotah_lucario",
    "kiyotah_dragapult",
    "kiyotah_iono",
    "kiyotah_abomasnow",
    "kacchan_anti_wall",
    "nursrijan_lucario",
    "yakitori_raging_bolt",
    "zoli_dragapult",
    "sue_alakazam",
    "maktha_1084",
    "yanxiaohan"
  ],
  "seed_policy": "evaluator_default_independent_randomness",
  "swap": true,
  "trace_mode": "full"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
