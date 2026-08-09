# 0038 · 007 CPU/CUDA Post-fix Validation

日期：2026-08-09

runtime repair：`bf37545`

evaluation harness：`e65204a`

## 结论

修复后的 Policy-0806/007 在 official CPU 与 CUDA 上没有观察到明显方向性偏差。

同一 Frozen-0806 标准面板：

| runtime | 局数 | W-L-D | 胜率 | Wilson 95% CI | 先攻 / 后攻 | error |
|---|---:|---:|---:|---:|---:|---:|
| official CPU | 2,048 | 1149-899-0 | 56.10% | 53.94%–58.24% | 58.89% / 53.32% | 0 |
| repaired CUDA | 2,048 | 1146-902-0 | 55.96% | 53.80%–58.09% | 59.08% / 52.83% | 0 |

CPU−CUDA 仅 `+0.15pp`。2048 个 opponent/slot/replica/seat mapping 全部通过；逐局 outcome
一致率 76.42%，CPU loss→CUDA win 为 240，CPU win→CUDA loss 为 243，配对差 95% CI
约 `[-1.96pp,+2.25pp]`。这份结果是 shared standard-panel reference，不要求两个独立 RNG
backend 在相同 seed 下逐局一致，也不冒充 independent-seed Gate G。

## Independent-seed Gate G

official CPU 使用 seed `341512806`，CUDA 使用独立 seed `934151280`。两组各 2,048 个
engine seed 均唯一、交集为 0；opponent×seat 分层一致，先后手各 1,024，runtime error 均为 0。

| metric | official CPU | independent CUDA | comparison |
|---|---:|---:|---:|
| win rate | 56.10% | 57.57% | CPU−CUDA `-1.46pp`；95% CI `[-4.50,+1.57]pp` |
| first | 58.89% | 59.47% | `-0.59pp` |
| second | 53.32% | 55.66% | `-2.34pp` |
| mean full rounds | 6.861 | 6.950 | difference `-0.089`；95% CI `[-0.217,+0.039]` |
| errors | 0 | 0 | PASS |

局长通过预注册 `±0.5` round 等价带。胜率差没有显示十个百分点级异常，但区间未完全落入
预注册 `±3pp`，因此 Gate G 仍为 `INCOMPLETE`，不是 FAIL，也不能在看到结果后放宽门槛。
此外 official CPU compact report 缺 final Prize，两边缺 strategic/macro/forced/action-type/
fallback per-game telemetry。

## 吞吐

| run | wall time | games/s | 相对 CPU |
|---|---:|---:|---:|
| official CPU standard 2048 | 1,592.06s | 1.286 | 1.00× |
| repaired CUDA standard 2048 | 105.25s | 19.459 | 15.13× |
| independent CUDA + terminal diagnostics | 129.71s | 15.789 | 12.27× |

新增 CUDA terminal diagnostics 仅序列化 resident scheduler 已收集的终局 turn、selection 和
Prize count，不增加模型 forward 或新的 device hot-path array。

## 昨天的结果还能否使用

pre-fix CUDA-2048 是 `1174-874-0（57.32%）`；post-fix 同 schedule 是
`1146-902-0（55.96%）`，变化 -28 胜/-1.37pp。

- 不能继续用 pre-fix CUDA 胜率作为 official/Kaggle 强度证据，也不能用 U215/U230 的收益、
  checkpoint 排名或 rollout 作为继续训练依据。
- 固定 schedule、吞吐/显存测量、规则 differential fixture、regression failure 和问题定位记录
  仍然有效。

U230 仍不可提交或续训；下一轮必须从正确的 Zero-Shot/Pretrain U0 重新生成 on-policy rollout。

## 证据

- [完整 semantic audit](SEMANTIC_PARITY_AUDIT.md)
- [official CPU-2048 HTML](.tmp/evaluation/0038_007_cpu_seeded2048_conditional_parity/published/reports/007_dragapult_ex.html)
- [CUDA standard-2048 HTML](evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_postfix_bf37545/reports/007_dragapult_ex.html)
- [CUDA standard-2048 games](evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_postfix_bf37545/games/007.json)
- `.tmp/evaluation/0038_semantic_parity_audit/standard_panel_reference/cpu2048_vs_cuda2048.json`
- `.tmp/evaluation/0038_semantic_parity_audit/gate_g/distribution_cpu2048_cuda2048_independent_v2.json`

关键 hashes：

```text
official CPU report       948127e7fb6a99c5dd23f9427bbb3b8892f6449b0d7dc26212cfcb67f5f21853
CUDA standard games       f485e45c2a26599730b64413faf15e9792a7420a2f503ae1c1b5a98bf8dc30d9
CUDA independent evidence aa1b7fea741f1c211fa8a2dfffa07a8e542f01d50bdb4f4794e32bc1b181cfa2
Gate G comparison         f26645c33b72d50ce19b02fe6a218bd655468b5e9c0b590d1d60cd1cd000c698
```

本轮未启动 RL、未改 checkpoint 权重、未提交 Kaggle、未修改 `engine/source/`。
