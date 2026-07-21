# Start Baseline

## Context

- candidate: `work/alakazam_v8_current`
- control: Target oracle，见 [target-baseline](target-baseline.md)
- protocol: Auto-Iteration Sample，17 个 opponent × 10 局，共 170 局
- evaluator: 隔壁 `eval/alakazam_replay.py`
- trace: `/tmp/ptcg-v8-start-baseline`（仅临时保存，未复制入仓库）

## Result

- W/L/D：`8/128/34`
- 原始胜率：`4.7%`
- agent error：`34`
- 第二回合实际 `attackId=1072`：`0/170`
- post-KO ready attacker：`0/142`

## Root cause

34 个错误均为 `NameError: name 'KADABRA' is not defined`，来源是
`strategy/effects/dispatcher.py` 的 Night Stretcher 分支。无错误局还显示主要语义
缺口：进化、附能和攻击链没有正常串起来，导致击倒后没有 ready attacker。

## Decision

这不是可接受的 control，只作为重构现状的 Start Baseline。先修 correctness，再逐条
恢复旧版 oracle 的语义。

