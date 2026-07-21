# Alakazam V8 语义恢复迭代记录

本目录记录 2026-07-21 的 V8 语义恢复阶段。Target 是
`submission/alakazam_v8_luna_deck_opt`，candidate 是
`work/alakazam_v8_current`；所有对局均使用隔壁
`ptcg-agent-kaggle/eval/alakazam_replay.py` 的 Auto-Iteration Sample。

Full Sample 固定为 17 个 opponent × 10 局 = 170 局。先手/后手只统计实际样本，不要求
局序交替。focused 只用于机制诊断，不能替代 full promotion 证据。完整 trace 只保留在
临时目录，本目录只保留轻量结果和 HTML。

## 目录约定

每条记录都是自包含目录，Markdown 与 HTML 使用同名文件：

```text
iteration-023-full/
  iteration-023-full.md
  index.html
  result.json
```

总览入口是 [`index.html`](index.html)，交接说明见 [`HANDOFF.md`](HANDOFF.md)。

## Baseline

| 记录 | 样本 | W/L/D | candidate errors | 关键结果 |
| --- | ---: | ---: | ---: | --- |
| [target-baseline](target-baseline/target-baseline.md) | 170 | 118/50/2 | 0 | Powerful Hand 32/170；Meta 70.4% |
| [start-baseline](start-baseline/start-baseline.md) | 170 | 8/128/34 | 34 | Powerful Hand 0/170；post-KO ready 0/142 |

## 主要迭代

| 记录 | 类型 | W/L/D | 关键结果 |
| --- | --- | ---: | --- |
| [iteration-001](iteration-001/iteration-001.md) | full | 8/162/0 | correctness recovery |
| [iteration-002](iteration-002/iteration-002.md) | focused | 2/10/0 | attack 20；1070=16；1072=4 |
| [iteration-003](iteration-003/iteration-003.md) | focused | 2/10/0 | post-KO zero-ready 8/8 |
| [iteration-004](iteration-004/iteration-004.md) | focused | 4/8/0 | 首次进化链恢复 |
| [iteration-005](iteration-005/iteration-005.md) | focused | 5/7/0 | Kadabra→Bench Abra 规则回归 |
| [iteration-006](iteration-006/iteration-006.md) | focused | 6/6/0 | Powerful Hand 1/12 |
| [iteration-007](iteration-007/iteration-007.md) | full | 44/126/0 | Powerful Hand 7/170 |
| [iteration-008](iteration-008/iteration-008.md) | focused | 4/8/0 | bench insurance gate |
| [iteration-009](iteration-009/iteration-009.md) | focused | 4/8/0 | Powerful Hand 1/12 |
| [iteration-010](iteration-010/iteration-010.md) | full | 58/112/0 | Meta 32.0% |
| [iteration-011](iteration-011/iteration-011.md) | focused | 5/7/0 | Powerful Hand 2/12 |
| [iteration-012](iteration-012/iteration-012.md) | full | 46/123/1 | 外部 evaluator error 1 |
| [iteration-013](iteration-013/iteration-013.md) | focused | 3/9/0 | Active buffer 语义 |
| [iteration-014](iteration-014/iteration-014.md) | focused | 6/6/0 | 打手断档 1/12 |
| [iteration-015](iteration-015/iteration-015.md) | full | 41/129/0 | Powerful Hand 6/170 |
| [iteration-016](iteration-016/iteration-016.md) | focused | 6/6/0 | zero-ready 9/14 |
| [iteration-017](iteration-017/iteration-017.md) | focused | 4/8/0 | Powerful Hand 1/12 |
| [iteration-018](iteration-018/iteration-018.md) | full | 50/120/0 | Powerful Hand 11/170 |
| [iteration-019](iteration-019/iteration-019.md) | focused | 7/5/0 | zero-ready 8/10 |
| [iteration-020](iteration-020/iteration-020.md) | focused | 3/9/0 | zero-ready 14/14 |
| [iteration-020-full](iteration-020-full/iteration-020-full.md) | full | 51/118/1 | Powerful Hand 4/170 |
| [iteration-021](iteration-021/iteration-021.md) | focused | 7/5/0 | Powerful Hand 1/12；zero-ready 2/5 |
| [iteration-021-full](iteration-021-full/iteration-021-full.md) | full | 58/112/0 | Powerful Hand 10/170；zero-ready 152/194 |
| [iteration-022](iteration-022/iteration-022.md) | focused | 6/6/0 | Powerful Hand 1/12；zero-ready 13/13 |
| [iteration-022-full](iteration-022-full/iteration-022-full.md) | full | 53/116/1 | Powerful Hand 6/170；zero-ready 140/176 |
| [iteration-023](iteration-023/iteration-023.md) | focused | 7/5/0 | zero-ready 7/10 |
| [iteration-023-full](iteration-023-full/iteration-023-full.md) | full | 65/105/0 | Powerful Hand 10/170；zero-ready 125/169 |
| [iteration-024](iteration-024/iteration-024.md) | focused | 4/8/0 | Nighttime Mine 实际 6 次 |
| [iteration-024-full](iteration-024-full/iteration-024-full.md) | full | 50/117/3 | candidate errors 0；外部 error 3 |

这些是独立随机批次，不能按局号做 A/B 配对。当前最佳 full 是 `iteration-023-full`，但
仍明显低于 Target；`iteration-024-full` 是 Nighttime Mine 语义修复后的观察结果，没有
晋级为新 baseline。

重新渲染 HTML：

```bash
python3 scripts/render_v8_history.py \
  --history-dir work/auto-iteration/history_iterations/v8-semantic-recovery
```
