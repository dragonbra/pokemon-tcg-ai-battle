# Pokémon TCG AI Battle Baseline：实现任务书

状态：等待 simulator/SDK 研究完成，尚未开始实现。

## 目标

使用官方 simulator/SDK 实现一个能在本地运行的、可解释的规则型 baseline agent，作为后续概率化搜索、神经网络和 RL 实验的回归基线。

## 为什么先做规则 baseline

它用于验证：

- SDK 安装和导入；
- `Observation`、`current`、`select` 的结构；
- `OptionType` 与合法动作；
- `deck.csv` 和 submission layout；
- 日志、replay 和局面终止条件；
- simulator 与正式规则差异是否被正确遵守。

## 必须产出

- 官方 simulator 依赖安装说明；
- 顶层 `main.py`；
- 一份合法的 `deck.csv`；
- 能处理初始化和 deck-selection 状态的 agent；
- 对每一个非空 `select` 返回合法动作；
- 本地至少完成一局 battle；
- 输出可调试的 JSON battle/replay；
- 打包脚本并列出 archive 内容；
- 中文 baseline 报告，记录命令和结果。

## 验收标准

1. 在干净环境中可以导入 SDK。
2. 使用最小合法卡组启动本地 battle。
3. 所有动作都来自 simulator 提供的合法选项。
4. 不因手牌、牌库、后备位或终止状态导致异常。
5. 至少完成一局本地 battle。
6. 能输出 replay/debug JSON。
7. 生成符合比赛要求的 `.tar.gz`，并确认 `main.py`、`deck.csv` 位于顶层。
8. 不进行 Kaggle 上传，除非用户在当前对话中单独确认。

## 方法边界

第一版只要求规则/启发式策略，不要求：

- GPU 训练；
- 大规模 self-play；
- 纯端到端神经网络；
- MCTS；
- leaderboard 结论。

## 需要记录的指标

- battle 完成率；
- 错误率和异常状态；
- 胜/负/和；
- 平均回合数；
- 每步动作耗时；
- 不同对手和起手状态下的失败案例。

## Claude Code 返回格式

请用中文返回：

- 修改的文件或 commit；
- 实际执行的命令；
- 测试输出；
- 生成的 replay、日志和压缩包绝对路径；
- simulator 规则不确定性；
- 下一步建议。
