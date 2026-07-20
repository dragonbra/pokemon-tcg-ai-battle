# 独立提交目录

这里的每个子目录是历史版本的一套可以单独提交的完整内容。当前候选位于 `../work/`：

- `official_water/`：官方水系 starter baseline；
- `alakazam_v1/`：胡地 V1 baseline。

每个历史目录都包含自己的 `main.py`、`deck.csv` 和 `cg/`。打包时脚本优先读取 `work/<name>/`，找不到时回退到 `submission/<name>/`，产物写入 `submission/dist/`：

```bash
bash scripts/package_submission.sh alakazam_v1
bash scripts/package_submission.sh official_water
```

当前候选示例：

```bash
bash scripts/package_submission.sh alakazam_v8_luna_deck_opt
```
