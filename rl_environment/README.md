# 通用 RL 基础设施

`rl_environment/` 是与具体卡组、专家来源和实验假设无关的共享层。它负责批处理、通用
policy/value 接口、checkpoint、日志、存储保护、reward 基础类型和不可覆盖的 run 路径分配。
具体 feature schema、网络实现、reward shaping、BC loss、牌组与训练入口不应放在这里。

当前迁移保留了已有训练基础设施，但尚未把官方引擎包装成新的 Gym 风格环境。在线 RL
接入时应在本包新增真实 engine battle adapter，由 adapter 原样返回 simulator 的 observation
和合法 options；具体项目只提供 policy、reward profile 与训练算法。真实能力结论仍必须来自
官方 engine runtime 的完整对局。

大规模训练语料的通用流式原语位于 `rl_environment.streaming`：JSONL 顺序读取、固定内存
shuffle buffer、batch 分块，以及主数据与轻量 sidecar 的 identity 对齐护栏。共享层只保证
有界内存、确定性和错位时 fail closed；具体训练项目仍负责自己的 schema、动作合法性和
reward 字段语义。

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

同一实验的训练版本必须写入 `rl_runs/artifact/<experiment>/V<n>_<tag>/`；checkpoint 和
TensorBoard 会分别写入 `rl_runs/checkpoint/` 与 `rl_runs/tensorboard/` 的同名版本。
已有任一路径包含文件时，基础设施会拒绝复用。

## W&B 训练曲线镜像

`TrainingLogger` 始终先写 canonical JSONL，再写 TensorBoard，最后按需镜像有限标量到
Weights & Biases。W&B 缺失、未配置、断网或上传失败都不能中断训练，也不能回滚已经落盘的
JSONL。BC 使用 `trainer/epoch` 横轴和 `bc/*` namespace；PPO 使用 `trainer/update` 和
`ppo/*`。不同阶段的 loss 不可直接比较，跨 BC/RL 的能力结论仍只来自同一合同下的官方
engine frozen evaluation。

SDK 在本机安装到宿主机 Python 的用户级环境，不安装到项目 `.venv`：

```bash
python3 -m pip install --user 'wandb>=0.28,<1'
python3 -m wandb --version
```

先运行不会上传数据的离线 smoke：

```bash
python3 -m rl_environment.wandb_smoke \
  --output .tmp/wandb/fake-training-<date> \
  --steps 5 \
  --mode offline
```

三种模式由 `WANDB_MODE` 控制：未设置或 `disabled` 完全关闭，`offline` 只在本地生成可检查
的 W&B run，`online` 才会发送到 W&B Cloud。在线使用前运行
`python3 -m wandb login --relogin`。当前正式 cloud destination 为：

```bash
export WANDB_MODE=online
export WANDB_PROJECT=pokemon-tcg-policy-learning
export WANDB_ENTITY=dragon_bra
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
```

W&B 必须与实际启动训练的解释器处于同一可见环境。本机正式 online run 使用宿主机
`python3`；不要用默认隔离、无法读取宿主 user site-packages 的 `.venv/bin/python` 启动。

2026-07-26 已在该 private project 完成 5 epoch synthetic online smoke：云端回读到 5 行 history
与 6 条指标序列；同步仅包含 4 个 W&B metadata/config 文件，0 media、0 artifact。该 smoke
只证明记录链路，不构成 policy 能力证据。

现有所有使用 `TrainingLogger` 的训练入口都会读取这些环境变量。每个正式
`<experiment>/V<n>_<tag>` 会映射成稳定 W&B run ID；同一 V 的在线断点恢复使用同一 ID，
新训练语义仍必须分配下一个 V。默认只上传 config 和有限标量，不上传 checkpoint、dataset、
optimizer state、trace、replay、observation 或 source patch。

自动镜像只对正式 `rl_runs/artifact/<experiment>/V<n>_<tag>/` 生效，防止持久化的 online
环境变量让单元测试或任意临时 logger 污染云端项目。`rl_environment.wandb_smoke` 会在进程内
显式允许自己的非正式 `.tmp/wandb/` 输出；其他非正式工具不得自行打开该放行开关。

后续正式 BC、value calibration 和 PPO run 默认设置 `WANDB_MODE=online`。每个版本结束时，
在 `status.json`、`training_summary.json` 或等价版本记录中保留 W&B project、稳定 run ID 或
URL、sync 状态和 mirror failure（如有）；W&B 失败时本地 JSONL、TensorBoard、checkpoint 和
版本状态仍必须完整落盘。
