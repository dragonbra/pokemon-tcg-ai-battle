# 独立提交目录

这里的每个子目录都是一套可以单独提交的完整内容：

- `official_water/`：官方水系 starter baseline；
- `alakazam_v1/`：胡地 V1 baseline。

每个目录都包含自己的 `main.py`、`deck.csv` 和 `cg/`。打包时指定目录名，例如：

```bash
bash scripts/package_submission.sh alakazam_v1
bash scripts/package_submission.sh official_water
```
