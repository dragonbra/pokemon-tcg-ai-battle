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
| `work/<name>/` | 当前可打包候选，仅包含 agent、卡组和官方 `cg` 模拟器运行时 |
| `submission/<name>/` | 历史版本的完整 agent、卡组和官方 `cg` 模拟器运行时 |
| `submission/dist/` | 打包生成的 `.tar.gz` 提交产物 |
| `engine/` | 官方引擎源码的本地缓存与构建边界，不进入提交包 |
| `rl/` | 模型、训练入口、checkpoint/dataset 边界和可审计实验记录 |
| `scripts/` | 唯一的 TensorBoard 启动入口，不存放训练基建或一次性脚本 |
| `visualization/` | replay 可视化核心、外部 viewer launcher 和使用说明 |
| `notes/` | 比赛事实、CLI、协作约定和术语等基础资料 |
| `docs/reports/` | 调研结论、卡组分析和实现任务书 |
| `experiments/` | 后续实验协议与结果模板 |
| `replays/` | Kaggle 官方 Episode replay 和 agent log；本地临时回放不保存 |

`work/<name>/` 是当前候选源目录，`submission/<name>/` 保存历史完整提交。每套提交都独立包含 `main.py`、`deck.csv` 和 `cg/`；当前训练与评测入口位于 `rl/` 和 `evaluation/`，不再由根目录一次性脚本串联。

大规模模拟、向量化环境、MCTS、自博弈和神经网络训练可以在另一台 5080 机器上进行。实现端应读取本仓库的中文任务书，并把代码、命令、指标和失败信息写回仓库。

## 当前研究顺序

1. 先确认 simulator 的 observation、action、deck 和规则差异。
2. 研究热门 Discussion 和公开代码，区分事实、作者推测和我们的假设。
3. 以 Kaggle 官方 submission/episode/replay 为主要真实对局数据。
4. 从官方对局提取卡组、对手、回合和行动指标，建立复盘协议。
5. 再比较规则、搜索、神经网络三类方法。
6. 最后考虑神经网络、自博弈或“神经网络 + 有界搜索”。

## 当前状态

官方卡牌资源和 starter simulator 已经合并。历史目录中的 `submission/official_water/` 和 `submission/alakazam_v1/` 保留最初 baseline；当前候选位于 `work/alakazam_v8_current/`。所有 agent 都只从 simulator 给出的合法选项中选择。

本机能否直接加载提交目录中的 `cg/libcg.so` 取决于动态库版本。官方二进制要求 `GLIBCXX_3.4.29` 或更新的 `libstdc++.so.6`；如果本机旧于此版本，应在兼容的 Linux/Kaggle 环境运行，不要替换官方模拟器二进制。

## 常用命令

```bash
# 查看全部训练 run；默认 http://127.0.0.1:6006
./scripts/start_tensorboard.sh

# 当前评测入口
python3 -m evaluation list-opponents
python3 -m evaluation validate work/alakazam_bc_v1

# 查看 Kaggle replay 或本地可视化 replay
python3 -m visualization.replay.cli path/to/replay.json
python3 -m visualization.replay.cli path/to/replay.json --no-open
```

TensorBoard 默认读取 `rl/_runs/tensorboard`。需要改变监听地址、端口、日志目录或 Python 解释器时，分别设置 `TENSORBOARD_HOST`、`TENSORBOARD_PORT`、`TENSORBOARD_LOGDIR` 或 `PYTHON`。

Kaggle 登录、接受规则、下载比赛数据和正式提交仍然需要明确的人工操作；仓库不会自动上传 submission。

回放可视化的完整说明见 [`visualization/README.md`](visualization/README.md)。launcher 会在当前标签页完成 POST 跳转，不创建新的浏览器窗口。
