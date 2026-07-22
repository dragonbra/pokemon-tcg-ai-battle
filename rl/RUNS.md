# RL 实验归档规则

实验名称使用全局递增编号：`0001-label`、`0002-label`。编号一旦分配就不复用，同一个实验
在三棵目录中使用完全相同的名称：

```text
rl/artifact/dataset/0001-label/       # 仅 BC；Git 忽略
rl/artifact/checkpoint/0001-label/    # Git 忽略
rl/_runs/tensorboard/0001-label/      # Git 跟踪
rl/_runs/0001-label/                  # Git 跟踪
```

tracked run record 至少保存：

- `manifest.json`：目标、状态、Git commit、模型/数据/评测指针。
- `data_manifest.json` 与 `data_summary.json`：数据来源、过滤、split 和规模；RL run 可省略。
- `training_config.json`：模型结构参数、seed、优化器与训练超参数。
- `training_metrics.jsonl`、`training_summary.json`：逐 epoch 指标与最佳结果。
- `evaluation/`：运行 manifest、逐局记录、metrics、case 和 Markdown/HTML 报告。
- `commands.md`、`decisions.md`、`environment.txt`：复现命令和关键判断。

模型实现以 `rl/model/`、`rl/train/`、`rl/core/` 和 manifest 中的 Git commit 为准，不在 run
内复制 `source/`。工作树未提交而又影响实验时，可以额外保存 `source_revision.patch`。

candidate package 不重复保存在 run 中。晋级候选进入 `work/<name>/`，历史正式提交进入
`submission/<name>/`；run manifest 记录对应路径和 hash。

`rl/_runs/INDEX.html` 是所有实验的人工可读索引。每完成一次实验，都应增加结果、关键离线
指标、evaluation outcome、是否晋级和各产物链接。
