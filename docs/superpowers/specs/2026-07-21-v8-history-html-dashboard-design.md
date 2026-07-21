# V8 History HTML Dashboard 设计

## 目标

为 `work/auto-iteration/history_iterations/v8-semantic-recovery/` 增加一个统一的、可
直接打开的 HTML 评测历史页面。Target Baseline、Start Baseline 和后续完整
`17×10` iteration 都使用同一套 full-sample 指标区块；focused iteration 仍单独标记
为机制验证，不与 170 局结果直接比较。

## 范围与约束

- 评测来源仍是隔壁 `ptcg-agent-kaggle/eval/alakazam_replay.py` 的 Auto-Iteration
  Sample；不接入当前仓库 `evaluation/`。
- full Sample 固定为 17 个 opponent × 10 局 = 170 局；先后手按实际 trace 统计。
- history 只保存轻量 `summary.json`、Markdown 记录和 HTML；完整 trace 继续留在 `/tmp`。
- 页面不依赖网络、前端构建工具或外部 CDN，使用 Python 标准库生成并内嵌轻量 JSON。

## 页面结构

生成 `history_iterations/v8-semantic-recovery/index.html`，并为每个 record 生成一个
独立报告目录：

```text
v8-semantic-recovery/
├── index.html
├── target-baseline/index.html
├── start-baseline/index.html
├── iteration-001/index.html
└── iteration-002/index.html
```

总览页包含：

1. 顶部目标卡：Target 的 W/L/D、Start 的 W/L/D、当前最佳 full Sample 和当前
   decision。
2. 统一 iteration timeline：每行展示 id、类型、样本规模、W/L/D、胜率、Meta、error、
   二回合 Powerful Hand、post-KO 结果和 decision；full 与 focused 使用明显的样本标签。
3. full-sample 指标对照：对每个有数据的 full record 展示总体/先手/后手胜率、二回合
   `attackId=1072`、error 和 post-KO 接力；缺失字段显示 `—`，不猜测分母。
4. 研究提醒：明确 Target/Start 是可比的 170 局记录，focused 只用于机制定位，不能
   替代 fresh full Sample 的晋级证据。
5. 每行链接到对应 `*.md` 详细记录（若文件存在）。

每个 record 的 `index.html` 包含：

- 该记录的样本类型和评测矩阵；
- 总体/先手/后手 W/L/D、胜率和 error；
- 二回合 `attackId=1072` 及其分母；
- post-KO 接力分子/分母（若该记录提供）；
- 假设、变更、结论和下一步（若有对应 Markdown）；
- 指向跨轮次总览页和原始 Markdown 的链接。

## 数据流

新增 `scripts/render_v8_history.py`，读取指定 history 目录的 `summary.json`，输出总览页
和每个 record 的 `index.html`。脚本只读取 summary 和对应的轻量 Markdown，不扫描
trace。生成结果可重复，且 summary 中新增 record 后重新运行即可刷新页面；缺失的
指标只显示 `—`，不从其他 record 推测。

## 验证

- 用临时 fixture summary 运行渲染脚本，确认输出包含 Target、Start、full/focused 标签、
  `118/50/2`、`8/128/34` 和 `attackId=1072` 等核心字段。
- 运行真实 history 的渲染脚本，确认 HTML 生成成功且 `git diff --check` 通过。
- 不把完整 trace 或 `/tmp` 路径复制到 HTML 的数据区；只允许页面元信息显示 trace 不
  持久化的说明。
