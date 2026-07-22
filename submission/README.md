# 独立提交目录

这里的每个子目录是历史版本的一套可以单独提交的完整内容。当前候选位于 `../work/`：

- `official_water/`：官方水系 starter baseline；
- `alakazam_v1/`：胡地 V1 baseline。

每个历史目录都包含自己的 `main.py`、`deck.csv` 和 `cg/`。旧的通用打包 wrapper 已随规则
代理流程一起移除；这些目录现在作为历史源包保留，不再通过根目录 `scripts/` 自动打包。
当前训练产生的候选应进入 `../work/`，并先通过 `python3 -m evaluation validate <package>`。
