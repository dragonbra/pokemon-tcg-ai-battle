# RL 实验归档规则

实验名称使用全局递增编号：`0001-label`、`0002-label`。编号一旦分配就不复用；同一实验
内部的每次尝试使用严格递增的 `V<n>_<snake_case_tag>`，失败版本也保留且不复用：

```text
rl_runs/dataset/0001-label/       # 仅 BC；Git 忽略
rl_runs/checkpoint/0001-label/V1_initial_contract/ # Git 忽略
rl_runs/tensorboard/0001-label/V1_initial_contract/   # Git 跟踪
rl_runs/wandb/0001-label/V1_initial_contract/         # 本地 W&B staging，Git 忽略
rl_runs/artifact/0001-label/V1_initial_contract/      # Git 跟踪
rl_runs/evaluation/0001-label/index.html              # Git 跟踪，项目总览
rl_runs/evaluation/0001-label/V1_initial_contract.html # Git 跟踪
```

`artifact/<experiment>` 根目录保存共享的 manifest、data manifest/audit、commands 和
decisions。每个
`V<n>_<tag>` tracked version 至少保存：

- `training_config.json`：模型结构参数、seed、优化器与训练超参数。
- `training_metrics.jsonl`、`training_summary.json`：逐 epoch 指标与最佳结果。
- `status.json` 或版本决策记录：完成/中止/失败状态、原因与下一版本改进。
- 同名 `evaluation/V<n>_<tag>.html`：内嵌运行 manifest、逐局记录、metrics 和 case 的报告。
- `evaluation/index.html`：自动汇总同项目全部版本并链接到对应的 `V*.html`。

每个 epoch 必须记录完整 train 与 validation split 的同口径标量，并同步写入 TensorBoard；
禁止通过 `train_eval_interval > 1` 跳过中间 epoch。历史 run 若没有实际执行某个评测，只能保留
缺口并在 status 中说明，不能插值或伪造。

W&B 是正式训练默认启用的远程比较层，但不是新事实源。同一条记录按
`training_metrics.jsonl → TensorBoard → W&B` 的顺序写入；W&B 失败只记录 warning。
BC、value calibration 和 PPO 使用不同 namespace 与语义横轴，只有相同 official-engine
evaluation contract 下的 `eval/*` 才能跨阶段比较策略强度。W&B 本地 staging 写到
`rl_runs/wandb/<experiment>/<version>/` 并保持 Git ignore；checkpoint、完整 trace、replay 和
dataset 不上传。

正式 BC、value calibration 和 RL/PPO 训练使用 private project
`dragon_bra/pokemon-tcg-policy-learning`，并设置 `WANDB_MODE=online`。本机 W&B 运行时来自宿主
`python3` 的用户级环境，不来自项目 `.venv`。一个 `V<n>_<tag>` 只对应一个稳定 W&B run；只有
同一版本的真实断点恢复可以 resume，新的训练语义必须新建版本。

每个版本结束时，`status.json`、`training_summary.json` 或等价版本记录至少应能审计：

- W&B project 与稳定 run ID 或 URL；
- sync 最终状态，以及本地 canonical metrics 是否完整；
- mirror 初始化、上传或 finish 失败的原因（如有）；
- 本次是否仅记录标量，且未上传 checkpoint、dataset、trace、replay 等排除资产。

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
