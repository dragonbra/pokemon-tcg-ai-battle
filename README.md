# Pokémon TCG AI Battle 研究仓库

这是 Kaggle Pokémon TCG AI Battle Challenge Simulation 的中文研究与协作仓库。

比赛主页：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
模拟器文档：https://matsuoinstitute.github.io/cabt/

## 仓库定位

本仓库是研究和协作层，主要供何瑾雨与 Claude Code 使用，所有面向人的文档统一使用中文。仓库保存：

- 已核实的比赛事实和模拟器行为；
- Kaggle Discussion、公开 Notebook、Replay 的研究记录；
- 卡组、策略和实验协议；
- 交给 Claude Code 或 5080 机器执行的中文实现任务书；
- 不依赖 GPU 的轻量级分析工具。

大规模模拟、向量化环境、MCTS、自博弈和神经网络训练可以在另一台 5080 机器上进行。实现端应读取本仓库的中文任务书，并把代码、命令、指标和失败信息写回仓库。

## 当前研究顺序

1. 先确认 simulator 的 observation、action、deck 和规则差异。
2. 研究热门 Discussion 和公开代码，区分事实、作者推测和我们的假设。
3. 跑通官方示例或最小规则 baseline。
4. 建立可重复的卡组和对手池评估协议。
5. 再比较规则、搜索、神经网络三类方法。
6. 最后考虑神经网络、自博弈或“神经网络 + 有界搜索”。

## 当前状态

Level 1 工具链已经安装到仓库独立的 `.venv` 中。Kaggle 登录、接受规则、下载比赛数据和提交 agent 仍然需要明确的人工操作；仓库不会自动上传 submission。
