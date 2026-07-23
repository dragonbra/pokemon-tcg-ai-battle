# RL 实验归档规则

实验名称使用全局递增编号：`0001-label`、`0002-label`。编号一旦分配就不复用；同一实验
内部的每次尝试使用严格递增的 `V<n>_<snake_case_tag>`，失败版本也保留且不复用：

```text
rl_runs/dataset/0001-label/       # 仅 BC；Git 忽略
rl_runs/checkpoint/0001-label/V1_initial_contract/ # Git 忽略
rl_runs/tensorboard/0001-label/V1_initial_contract/   # Git 跟踪
rl_runs/0001-label/V1_initial_contract/               # Git 跟踪
rl_runs/0001-label/evaluation/V1_initial_contract/    # Git 跟踪
```

experiment 根目录保存共享的 manifest、data manifest/audit、commands 和 decisions。每个
`V<n>_<tag>` tracked version 至少保存：

- `training_config.json`：模型结构参数、seed、优化器与训练超参数。
- `training_metrics.jsonl`、`training_summary.json`：逐 epoch 指标与最佳结果。
- `status.json` 或版本决策记录：完成/中止/失败状态、原因与下一版本改进。
- 同名 `evaluation/V<n>_<tag>/`：运行 manifest、逐局记录、metrics、case 和报告。

每次启动前必须确认 run、TensorBoard 和 checkpoint 三个同名版本目录均没有文件；训练路径
解析会拒绝 experiment 根目录、非法版本名和任何已使用版本。TensorBoard 新建 event 文件
不代表新逻辑 run，禁止依赖它在同一 logdir 中区分尝试。

模型实现以 `train/alakazam_bc_rl/`、`train/alakazam_bc_rl/training/`、
`rl_environment/` 和 manifest 中的 Git commit 为准，不在 run
内复制 `source/`。工作树未提交而又影响实验时，可以额外保存 `source_revision.patch`。

candidate package 不重复保存在 run 中。晋级候选进入 `work/<name>/`，历史正式提交进入
`submission/<name>/`；run manifest 记录对应路径和 hash。

`rl_runs/INDEX.html` 是所有实验的人工可读索引。每完成一次实验，都应增加结果、关键离线
指标、evaluation outcome、是否晋级和各产物链接。
