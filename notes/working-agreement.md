# 工作约定

## 研究与实现分离

本机 Hermes 负责研究：

- 比赛事实、规则和 simulator 行为核查；
- Kaggle 热门 Discussion、公开 Notebook、Replay 的资料整理；
- 卡组与策略假设；
- baseline 和实验协议设计；
- 写给 Claude Code 的中文实现任务书。

5080 机器上的 Claude Code 负责实现：

- 安装和验证官方 SDK；
- 编写 agent、模拟器适配和实验代码；
- 执行大量本地对局、搜索或训练；
- 输出日志、指标、Replay 和错误信息。

## Claude Code 交接协议

每次交给 Claude Code 的任务书必须写明：

- 目标；
- 相关文件和 URL；
- 环境假设；
- 明确的验收测试；
- 计算预算；
- 预期产物；
- 已知的 simulator 差异和未决问题。

Claude Code 返回时必须记录：

- 修改的文件或 commit；
- 实际执行过的命令；
- 测试和 benchmark 输出；
- 尚未解决的问题；
- 产物的绝对路径。

面向何瑾雨和 Claude Code 的仓库文档统一使用中文。代码中的 API 名称、类名、变量名和官方专有名词可以保留英文。

## 方法路线

不要在没有可靠 baseline 和评估协议之前直接投入大型 RL。推荐顺序：

1. 规则/启发式 baseline；
2. 概率化决策或有限深度搜索；
3. 可选的 MCTS/rollout；
4. 神经网络策略或价值模型；
5. 神经网络与有界搜索的组合；
6. 规模化 self-play 或 offline RL。

## 硬件策略

在 simulator 正确跑通、动作语义明确、baseline 可复现之前，不假设必须使用 GPU。GPU 机器主要用于向量化模拟、大量 self-play、模型训练和大规模 ablation。

## 提交策略

下载、发布和提交属于账号可见或比赛敏感操作。先做只读或 dry-run；提交前必须在当前对话中明确确认比赛、agent 版本、压缩包路径和 submission message。
