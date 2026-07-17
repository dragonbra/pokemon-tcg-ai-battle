# Pokémon TCG AI Battle 研究与实现仓库

这是 Kaggle Pokémon TCG AI Battle Challenge Simulation 的中文研究与协作仓库。

比赛主页：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
模拟器文档：https://matsuoinstitute.github.io/cabt/

## 仓库定位

本仓库同时保存研究资料和第一版可运行提交。所有面向人的文档统一使用中文。仓库保存：

- 已核实的比赛事实和模拟器行为；
- Kaggle Discussion、公开 Notebook、Replay 的研究记录；
- 卡组、策略和实验协议；
- 交给实现机器执行的中文实现任务书；
- 不依赖 GPU 的轻量级分析工具。

## 目录约定

| 目录 | 用途 |
|---|---|
| `data/official/` | 官方下载的卡牌 CSV、Card ID List PDF，只读参考资源，不直接打包提交 |
| `submission/` | 我们实际维护的 agent、卡组和官方 `cg` 模拟器运行时 |
| `scripts/` | 资产校验、提交打包和本地对局 runner |
| `notes/` | 比赛事实、CLI、协作约定和术语等基础资料 |
| `reports/` | 调研结论、卡组分析和实现任务书 |
| `experiments/` | 后续实验协议与结果模板 |

`submission/` 是开发源目录；提交时由脚本把它打包成顶层包含 `main.py`、`deck.csv` 和 `cg/` 的 `.tar.gz`。

大规模模拟、向量化环境、MCTS、自博弈和神经网络训练可以在另一台 5080 机器上进行。实现端应读取本仓库的中文任务书，并把代码、命令、指标和失败信息写回仓库。

## 当前研究顺序

1. 先确认 simulator 的 observation、action、deck 和规则差异。
2. 研究热门 Discussion 和公开代码，区分事实、作者推测和我们的假设。
3. 跑通官方示例或最小规则 baseline。
4. 建立可重复的卡组和对手池评估协议。
5. 再比较规则、搜索、神经网络三类方法。
6. 最后考虑神经网络、自博弈或“神经网络 + 有界搜索”。

## 当前状态

官方卡牌资源和 starter simulator 已经合并。当前 `submission/` 是一个确定性的规则 baseline：初始化返回 60 张卡组，其余步骤只从 simulator 给出的合法选项中选择，并优先攻击、贴能量、进化和出牌。

本机能否直接加载 `submission/cg/libcg.so` 取决于动态库版本。官方二进制要求 `GLIBCXX_3.4.29` 或更新的 `libstdc++.so.6`；如果本机旧于此版本，应在兼容的 Linux/Kaggle 环境运行，不要替换官方模拟器二进制。

## 常用命令

```bash
python scripts/check_assets.py
bash scripts/package_submission.sh
PTCG_CXX_RUNTIME=/path/to/compatible/runtime ./scripts/run_local_battle.sh
```

如果当前环境已经能直接加载官方 `libcg.so`，可以省略 `PTCG_CXX_RUNTIME`；变量指向的目录应包含 `lib/libstdc++.so.6`。

Kaggle 登录、接受规则、下载比赛数据和正式提交仍然需要明确的人工操作；仓库不会自动上传 submission。
