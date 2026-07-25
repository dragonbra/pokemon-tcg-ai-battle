# 官方卡牌 CSV 纳入普通 Git 设计

**日期：** 2026-07-26  
**状态：** 已批准

## 目标

将以下两个已恢复的官方卡牌 CSV 纳入普通 Git，并同步到 GitHub `main`：

- `data/official/EN_Card_Data.csv`
- `data/official/JP_Card_Data.csv`

## 边界

- 两个 CSV 使用普通 Git，不使用 Git LFS。
- `data/official/` 中除 `README.md` 和这两个 CSV 外的内容继续忽略。
- 在 Git LFS 历史迁移后的干净 clone 中实施和推送。
- 不修改、删除、提交或重置原工作目录中的并发改动。

## 实现

将 `.gitignore` 的 official-data 规则保持为默认全目录忽略，再显式放行：

```gitignore
data/official/*
!data/official/README.md
!data/official/EN_Card_Data.csv
!data/official/JP_Card_Data.csv
```

从原工作目录复制两个 CSV 到干净 clone，校验复制前后 SHA-256 一致。使用 `git check-attr filter` 确认它们没有 `filter=lfs`，使用 `git check-ignore` 确认不再被忽略。

## 验证与发布

1. 验证两个 CSV 可解析、表头非空、包含数据行。
2. 运行完整 `pytest`、`unittest`、`compileall` 和 shell 语法检查。
3. 仅提交 `.gitignore` 与两个 CSV。
4. push `main`，然后核对本地 HEAD、`origin/main` 和 GitHub `refs/heads/main` 完全一致。
