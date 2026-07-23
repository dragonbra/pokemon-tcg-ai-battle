# 通用 RL 基础设施

`rl_environment/` 是与具体卡组、专家来源和实验假设无关的共享层。它负责批处理、通用
policy/value 接口、checkpoint、日志、存储保护、reward 基础类型和不可覆盖的 run 路径分配。
具体 feature schema、网络实现、reward shaping、BC loss、牌组与训练入口不应放在这里。

当前迁移保留了已有训练基础设施，但尚未把官方引擎包装成新的 Gym 风格环境。在线 RL
接入时应在本包新增真实 engine battle adapter，由 adapter 原样返回 simulator 的 observation
和合法 options；具体项目只提供 policy、reward profile 与训练算法。真实能力结论仍必须来自
官方 engine runtime 的完整对局。

## 边界

```text
rl_environment/              通用训练与运行基础设施
train/alakazam_bc_rl/        胡地 feature/model/reward/loss/训练
train/kaggle_bc_top20/       Top-20 数据与 Kaggle worker 归档
rl_runs/                     dataset/checkpoint/TensorBoard/run/evaluation
tests/                       全仓库回归测试
```

分配实验编号：

```bash
PYTHONPATH=. python3.11 -m rl_environment.runs create next_rl \
  --objective "Online RL with the official engine runtime"
```

同一实验的训练版本必须写入 `rl_runs/<experiment>/V<n>_<tag>/`；checkpoint 和
TensorBoard 会分别写入 `rl_runs/checkpoint/` 与 `rl_runs/tensorboard/` 的同名版本。
已有任一路径包含文件时，基础设施会拒绝复用。
