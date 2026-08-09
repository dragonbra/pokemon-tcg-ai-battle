# 0038 · 007 CPU/CUDA Post-fix Validation

日期：2026-08-09

runtime repair：`bf37545`

evaluation harness：`e65204a`

## 结论

修复后的 007 在 official CPU 与 CUDA 上没有明显胜率分布异常。

- official CPU 256：`152-104-0`，**59.38%**，Wilson 95% CI **53.26%–65.21%**。
- CUDA 2,048：`1146-902-0`，**55.96%**，Wilson 95% CI **53.80%–58.09%**。
- 点估计差 3.42pp；`z=1.04`、双侧 `p=0.299`，CPU 小样本区间与 CUDA 重叠。
- CPU 59.38% 落在 CUDA 八个 256 shard 的 **51.17%–62.11%** 范围内。
- 同一组 CPU-256 seeds 的补充 CUDA replay 也恰为 `152-104-0`；没有净方向偏差。

因此，Gate A–D 加上这轮真实对局支持：**当前 runtime 在已审计语义范围内一致，可以作为
新 U0/canary 的基础设施；没有证据支持仍存在十个百分点级系统性偏差。** 这不是对所有
卡牌和状态的穷尽证明，也不会让 pre-fix U230 变成有效 checkpoint。

## 合同与结果

共同条件：exact deck 007 `dragapult_ex_07bedfffbfad`、双方 Policy-0806、checkpoint
SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`、
deterministic greedy、strict FP32、Frozen 256-slot 对手频率、seed `341512806`、循环判负
上限 20、engine turn 100 记平局。

| 指标 | official CPU-256 | repaired CUDA-2048 |
|---|---:|---:|
| W-L-D | 152-104-0 | 1146-902-0 |
| 胜率 | 59.38% | 55.96% |
| 先攻 | 79/128 = 61.72% | 605/1024 = 59.08% |
| 后攻 | 73/128 = 57.03% | 541/1024 = 52.83% |
| error / unfinished | 0 / 0 | 0 / 0 |
| turn-limit draw | 0 | 0 |
| wall time | 210.01s | 105.25s |
| 吞吐 | 1.219 games/s | 19.459 games/s |

CUDA 对该 workload 的实测吞吐约为 official CPU harness 的 **15.96×**。CPU 只跑一个
固定 256 诊断单元，CUDA 跑正式八 shard 2,048；两者不是同一规模的正式强度报告。

CUDA 八个 shard 胜率依次为：

```text
59.77%, 52.34%, 56.25%, 53.12%, 62.11%, 54.69%, 58.20%, 51.17%
```

按 CUDA 逐 matchup 胜率折算，CPU-256 的预期胜场为 143.25，实际 152；`z=1.17`、
双侧 `p=0.241`。高频 Marnie matchup 为 CPU 24/51（47.06%）对 CUDA 187/408
（45.83%）；没有显示全局偏差。其余单 matchup 的 CPU 样本过小，不作独立强度结论。

## 同 seed 补充重放

将 CPU-256 的 256 个 opponent/seat/engine seed/search seed 原样交给 CUDA：

| 指标 | 结果 |
|---|---:|
| CPU W-L-D | 152-104-0 |
| CUDA W-L-D | 152-104-0 |
| pairing-key mismatch | 0 |
| exact outcome agreement | 196/256 = 76.56% |
| CPU win → CUDA loss | 30 |
| CPU loss → CUDA win | 30 |

两个 backend 的 RNG 消费和执行实现独立，所以同 seed 不承诺逐局 outcome 全相同；这里最关键
的是两侧 flip 完全对称、总体 W-L-D 相同。逐状态/逐动作的一致性仍由 Gate A–D 提供，而不是
用总体胜率替代。

## 昨天的结果还能否使用

pre-fix CUDA-2048 是 `1174-874-0`（57.32%）；post-fix 同 schedule 是
`1146-902-0`（55.96%），变化 -28 胜/-1.37pp。逐局 outcome agreement 69.14%，
win→loss 330、loss→win 302。

结论不是“昨天所有东西都没用”，而是：

- **不能再用**：pre-fix CUDA 胜率作为 official/Kaggle 强度证据；由该输入分布训练出的
  U215/U230 收益、checkpoint 排名和继续训练依据。
- **仍可使用**：固定 seed/schedule 合同、吞吐/显存测量、规则层 differential fixtures、
  regression 失败证据和问题定位记录。

U230 仍不可 submission-ready，也不应继续训练。下一轮必须从正确 Zero-Shot U0 重新采集
on-policy rollout。

## 证据

- [完整 semantic audit](SEMANTIC_PARITY_AUDIT.md)
- [CUDA-2048 HTML](evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_postfix_bf37545/reports/007_dragapult_ex.html)
- [CUDA-2048 逐局记录](evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_postfix_bf37545/games/007.json)
- [official CPU-256 HTML](.tmp/evaluation/0038_007_cpu_seeded256_postfix_bf37545/run-91e5ee912e874cfab377e22414e0fc9a/report.html)

关键 hashes：

```text
CUDA schedule       98b58bced460c1a2e622ae4b39bf506294fcb0230bb42e4117aaa6efc73c9ce9
CUDA game records   f485e45c2a26599730b64413faf15e9792a7420a2f503ae1c1b5a98bf8dc30d9
CUDA report         9a1a5f3572855fc7738f9e8c32c8d3284c73aad2bd6da2a9453c062d91271078
CPU report          dc0aa2d4d571c5adf7d5ecd4ac8bf0368b92fdd6610ee8fc23025b5fd843169a
CUDA extension      429c3d6638ef0b450b11d5fd43dd2cfa3a5ca28a5e101cfefa18ca8c6a8add45
official CPU source c3f00db927908332490d21e95cc219f2f5ffc54823d125065bbbacadbbf9f9f2
```

本轮未启动 RL、未改 checkpoint 权重、未提交 Kaggle、未修改 `engine/source/`。
