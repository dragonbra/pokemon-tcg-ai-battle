# RL 运行时资产

从 `0013` 起，每个项目使用一致的 ID，例如
`0013_alakazam_rollout_value_calibration`。项目目录的职责严格分离：

```text
train/<project_id>/                         # 项目特定训练实现
experiments/<project_id>/                   # 归档；权威 DESIGN 和正式评测
rl_runs/<project_id>/                       # 运行时资产
rl_runs/<project_id>/dataset/               # 仅 BC 数据集；Git 忽略
rl_runs/<project_id>/versions/V1_initial_contract/artifact/
rl_runs/<project_id>/versions/V1_initial_contract/checkpoint/   # Git 忽略
rl_runs/<project_id>/versions/V1_initial_contract/tensorboard/  # Git 忽略
rl_runs/<project_id>/versions/V1_initial_contract/wandb/        # Git 忽略
experiments/<project_id>/evaluation/V1_initial_contract.html     # Git 跟踪
```

`artifact/` 保存被跟踪的 `training_config.json`、`training_metrics.jsonl`、
`training_summary.json`、`status.json` 和 `evaluation.json`。正式评测报告**不**保存于
`rl_runs/`：`artifact/evaluation.json` 将记录 `run_id`、报告 hash，并指向
`experiments/<project_id>/evaluation/<V<n>_<tag>.html` 的权威归档报告。每个项目的
`experiments/<project_id>/DESIGN.html` 与 `DESIGN.md` 是权威设计文档，
`experiments/<project_id>/evaluation/index.html` 汇总版本报告。

版本名必须为严格单调递增的 `V<n>_<snake_case_tag>`；失败版本也保留。每个 epoch 必须对完整
train 与 validation split 运行同口径评测，并按
`training_metrics.jsonl → TensorBoard → W&B` 顺序写入同名标量。启动前必须确认同名的
`artifact/`、`checkpoint/`、`tensorboard/` 和 `wandb/` 均未被使用；已有路径或正式 HTML
不得覆盖，必须分配下一版本。

W&B 是正式训练的镜像而不是事实源：一条版本对应一个稳定 W&B run，真实断点恢复可 resume，
新训练语义必须新建版本。checkpoint、dataset、完整 trace、replay 和 observation 不上传。
策略强度结论只来自同一 official-engine evaluation contract 下的 `eval/*`。

历史项目也按同一项目根目录归档：运行时 artifact、checkpoint、TensorBoard 与 W&B staging 位于
`rl_runs/<project_id>/versions/<V<n>_<tag>/`；权威 HTML 评测位于
`experiments/<project_id>/evaluation/`。不再使用全局的 `rl_runs/artifact/`、
`rl_runs/tensorboard/` 或 `rl_runs/evaluation/` 容器。候选 package 始终位于
`evaluation/arena/candidates/<name>/`；不要写入已废弃的 `work/<name>/`。
