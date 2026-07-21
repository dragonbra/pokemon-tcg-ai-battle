# Target Baseline

## Context

- candidate/oracle: `submission/alakazam_v8_luna_deck_opt`
- protocol: Auto-Iteration Sample，17 个 opponent × 10 局，共 170 局
- evaluator: 隔壁 `eval/alakazam_replay.py`
- trace: `/tmp/ptcg-v8-target-baseline`（仅临时保存，未复制入仓库）

## Result

- W/L/D：`118/50/2`
- 原始胜率：`69.4%`
- Meta 加权胜率：`70.4%`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`32/170`（`18.8%`）

## Interpretation

这是本轮 semantic recovery 的行为 oracle 和最低目标。后续 candidate 必须先达到
G0 correctness，再以同一 17×10 口径比较总体结果、实际先后手结果、二回合
Powerful Hand 和 post-KO 接力。当前可用记录未保留 Target 的完整 post-KO 分母，
因此不对该项补写推断数字。

