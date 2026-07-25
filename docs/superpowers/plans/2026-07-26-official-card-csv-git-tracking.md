# 官方卡牌 CSV 纳入 Git 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将英文和日文官方卡牌 CSV 纳入普通 Git，并安全同步到历史迁移后的 GitHub `main`。

**Architecture:** 在已验证的 GitHub 干净 clone 中实施，避免旧历史工作目录及并发改动。精确放行两个 CSV，复制后校验哈希，验证非 LFS，运行项目完整测试后提交、push 并核对三个 HEAD。

**Tech Stack:** Git, GitHub, Python 3.11, pytest, unittest, SHA-256

## Global Constraints

- 只纳入 `data/official/EN_Card_Data.csv` 和 `data/official/JP_Card_Data.csv`。
- 两个 CSV 使用普通 Git，不使用 Git LFS。
- `data/official/` 其他内容继续忽略。
- 不修改、删除、提交或重置原工作目录中的并发改动。
- 在 `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/verification-clone` 实施。

---

### Task 1: 精确纳管两个 CSV

**Files:**
- Modify: `.gitignore`
- Create: `data/official/EN_Card_Data.csv`
- Create: `data/official/JP_Card_Data.csv`

**Interfaces:**
- Consumes: 原工作目录的两个已恢复 CSV。
- Produces: 干净 clone 中可由普通 Git 跟踪的同内容 CSV。

- [ ] **Step 1: 确认干净 clone 与远端同步**

```bash
git fetch origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git status --porcelain=v1
```

预期：仅有已提交后的计划文件或无改动，HEAD 等于 `origin/main`；如果远端变化则先 fast-forward。

- [ ] **Step 2: 修改忽略规则**

将规则精确设为：

```gitignore
data/official/*
!data/official/README.md
!data/official/EN_Card_Data.csv
!data/official/JP_Card_Data.csv
```

- [ ] **Step 3: 复制并校验 CSV**

```bash
cp /home/cyd/repos/pokemon-tcg-ai-battle/data/official/EN_Card_Data.csv data/official/
cp /home/cyd/repos/pokemon-tcg-ai-battle/data/official/JP_Card_Data.csv data/official/
sha256sum /home/cyd/repos/pokemon-tcg-ai-battle/data/official/{EN_Card_Data.csv,JP_Card_Data.csv}
sha256sum data/official/{EN_Card_Data.csv,JP_Card_Data.csv}
```

预期：同名文件源和目标哈希一致。

- [ ] **Step 4: 验证普通 Git 属性**

```bash
git check-ignore data/official/EN_Card_Data.csv && exit 1 || true
git check-ignore data/official/JP_Card_Data.csv && exit 1 || true
git check-attr filter -- data/official/EN_Card_Data.csv data/official/JP_Card_Data.csv
```

预期：两文件不再被忽略，`filter: unspecified`，不是 LFS。

### Task 2: 完整验证、提交和同步

**Files:**
- Commit: `.gitignore`, `data/official/EN_Card_Data.csv`, `data/official/JP_Card_Data.csv`

**Interfaces:**
- Consumes: Task 1 的普通 Git 文件。
- Produces: GitHub `main` 上可恢复的官方 CSV 和同步证据。

- [ ] **Step 1: 建立隔离测试环境**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[rl]' pytest
```

预期：依赖安装成功；`.venv` 保持忽略。

- [ ] **Step 2: 运行完整验证**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m compileall -q evaluation visualization rl_environment train
bash -n scripts/start_tensorboard.sh
```

预期：pytest/unittest 全部通过，语法检查退出 0。

- [ ] **Step 3: 精确提交**

```bash
git add .gitignore data/official/EN_Card_Data.csv data/official/JP_Card_Data.csv
git diff --cached --name-only
git -c user.name='dragon_bra' -c user.email='tommy514@foxmail.com' commit -m "data: 纳入官方卡牌 CSV" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

预期：暂存和提交仅包含三个文件。

- [ ] **Step 4: push 并核对同步**

```bash
git push origin main
git fetch origin
LOCAL=$(git rev-parse HEAD)
CACHED=$(git rev-parse origin/main)
REMOTE=$(git ls-remote origin refs/heads/main | cut -f1)
test "$LOCAL" = "$CACHED"
test "$LOCAL" = "$REMOTE"
```

预期：三个 SHA 完全一致。
