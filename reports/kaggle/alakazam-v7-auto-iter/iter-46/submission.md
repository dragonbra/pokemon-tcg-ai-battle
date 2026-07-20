# Iter-46 发布记录

- 发布目录：`submission/alakazam_v7_auto_iter_iter46/`
- 发布含义：保留 iter-46 策略，去掉 iter-47 对“本回合刚放下的 Dunsparce 提前附 Enriching Energy”的放宽规则。
- `deck.csv`：固定不变，SHA-256 为 `208edd2df41ae66d11e81b4394762d1fe1cb8f34452265281e22b6bfe4fb744a`。
- 压缩包：`dist/alakazam_v7_auto_iter_iter46.tar.gz`
- 压缩包 SHA-256：`26cf43ddcb9bbf1d07c4fb2d2e5c6a6bc06b966d2ba30ceb76d5404e73f265d3`
- 本地验证：资产检查通过，105 项回归测试通过，tar 成员完整。

## Kaggle 提交

已完成 Kaggle 实际提交。Kaggle 返回 `Successfully submitted`，提交记录如下：

- submission ref：`54854580`
- 文件：`alakazam_v7_auto_iter_iter46.tar.gz`
- 状态：`PENDING`（等待比赛评测完成）
- 提交时间：`2026-07-20T13:29:20.680000Z`

由于 CLI 的 `competitions submit` 没有自动使用本地 `credentials.json` 刷新的 token，本次提交使用了当前进程内的 `KAGGLE_API_TOKEN`，凭据没有落盘：

```bash
token=$(/opt/homebrew/bin/kaggle auth print-access-token)
KAGGLE_API_TOKEN="$token" /opt/homebrew/bin/kaggle competitions submit \
  pokemon-tcg-ai-battle -f dist/alakazam_v7_auto_iter_iter46.tar.gz \
  -m 'Alakazam V7 AutoIter iteration 46 (without iter-47 Enriching pre-attach rule)'
```

## Trace 清理

已将外部评测目录中约 359 MB 的 `iter-46-discovery-20260720/` 移入 macOS 废纸篓；repo 内的 iter-46 分析文档和摘要保留。最新的 `iter-47-enriching-focus-20260720/` focused trace 仍保留，用于下一轮复盘。
