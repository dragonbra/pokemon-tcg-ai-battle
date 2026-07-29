# Limitless TEF-POR 环境分析

本目录保存 `TEF-POR`（Temporal Forces–Perfect Order）Limitless 赛事环境的可复算聚合事实与生成器。

统一入口：

```bash
python3 -m data.processed.environment_limitless.generate \
  --refresh \
  --output docs/environment/limitless.html
```

- `--refresh` 重新读取 Limitless 主站与 Labs 公开只读端点。
- 原始响应仅缓存到 Git ignore 的 `.tmp/environment_limitless/cache/`。
- `snapshot.json` 保存来源 URL、时间、SHA-256、赛事覆盖、Points Share、实际参赛牌型、逐对局
  聚合矩阵、代表选手和 60 卡构筑。
- `docs/environment/limitless.html` 是面向阅读的权威分析页面。
- Labs ID 从每场赛事详情页发现，不依赖固定赛事映射；无 Labs 链接的赛事保留为缺失证据。
- 细分牌型沿用 Labs 站点标签；代表 exact 60-card deck 只用于关键副轴交叉审计，不外推为
  该标签下所有参赛卡表都具有同一构成。

统计口径：有效胜率为 `(W + 0.5 × D) / n`；每格使用 95% Wilson 区间。`n < 15` 不推断，
`15 <= n < 30` 仅作为方向信号，`n >= 30` 且区间排除 50% 才标记为较可信优势或劣势。
Labs 的 `winner=-1` 是双负/no-result，不进入胜率分母；同型内战只展示决胜、平局与样本量，
不把任意 `player1` 当作有方向的牌型胜率，也不计算 Wilson 或克制等级。
韩国联赛没有公开 Labs pairings，因此进入 8 场总体和主站 Points Share，但不进入逐对局矩阵。
