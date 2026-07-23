# Arena candidate opponents

该目录用于暂存准备加入正式 Evaluation 对手池的标准 package。这里的 candidate 指“候选
opponent”，不是 `evaluation run --candidate` 所指定的被评测卡组。

每个子目录必须先满足标准 package 契约：包含 `main.py`、恰好 60 行的 `deck.csv`，以及与
正式基线 hash 一致、物理复制且非 symlink 的 `cg/`。该目录不会被
`evaluation/configs/opponents.json` 加载，也不参与 `--opponents all`。

完成独立验证和用户确认后，才按关键宝可梦组合分配 `<archetype>_<NN>` 正式名称并移入
`../opponents/`，再为 catalog 补充易读 display name、1–2 个代表宝可梦 card ID 及 evaluation
asset tests。候选目录在获准收编前保持原名；未获得确认时不得直接加入正式评测池。
