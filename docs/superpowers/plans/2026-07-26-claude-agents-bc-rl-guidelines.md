# CLAUDE.md 与 AGENTS.md 统一及 BC + RL 路线优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将根目录项目约定统一到 `CLAUDE.md`，让 `AGENTS.md` 通过相对软链接共享同一内容，并补充当前 BC + RL 路线的硬边界。

**Architecture:** 以现有 `AGENTS.md` 为唯一内容基线，先复制为普通文件 `CLAUDE.md`，再在其中进行局部、可审阅的路线优化，最后将原 `AGENTS.md` 替换为指向 `CLAUDE.md` 的相对软链接。验证采用文件类型、链接目标、内容解析、文本断言和限定范围的 Git diff，不运行与纯文档迁移无关的训练或完整测试。

**Tech Stack:** Markdown、POSIX symbolic link、Git、Bash、Python 3 标准库

## Global Constraints

- `CLAUDE.md` 是仓库根目录唯一真实的项目约定文件。
- `AGENTS.md` 必须是相对软链接，链接文本恰为 `CLAUDE.md`。
- 现有 `AGENTS.md` 的有效硬约束必须完整保留；只进行与当前 BC + RL 路线直接相关的定向优化。
- BC + RL 是当前主要研发路线；规则策略只作为历史资产、分析基线、回归参考或合法性保障。
- 胡地单 deck、单 expert BC 已成功验证，但不得表述为 RL 或跨卡组泛化已经完成。
- 多卡组 BC 默认采用相互隔离的 deck-specific policy；不同 deck、team 或实现逻辑的标签不得无条件混合。
- 正式 BC opponent 必须经过 package 验证、官方 engine runtime 真实对局评测和用户明确确认后才能准入。
- 正式 RL run 必须记录可复现的 opponent 集合；pool 构成、权重或 curriculum 变化必须作为显式实验变量。
- 不修改 `engine/source/`、训练代码、评测代码、实验产物或 opponent catalog。
- 不覆盖或回退用户当前工作区中的其他未提交改动。
- 不执行 `git commit` 或 `git push`。

---

## File Structure

- Create: `CLAUDE.md` — Claude Code 与 Codex 共用的唯一项目级约定正文。
- Replace: `AGENTS.md` — 从普通 Markdown 文件变为相对软链接 `AGENTS.md -> CLAUDE.md`。
- Preserve: `README.md` — 现有 `AGENTS.md` 链接通过软链接继续有效，不需要修改。
- Preserve: `docs/superpowers/specs/2026-07-26-claude-agents-bc-rl-guidelines-design.md` — 已批准设计规格，不在实施阶段重写。

### Task 1: 建立唯一正文并定向优化 BC + RL 约定

**Files:**
- Create: `CLAUDE.md`
- Source: `AGENTS.md`
- Reference: `docs/superpowers/specs/2026-07-26-claude-agents-bc-rl-guidelines-design.md`

**Interfaces:**
- Consumes: 当前 `AGENTS.md` 的完整 Markdown 内容和已批准的设计规格。
- Produces: 普通文件 `CLAUDE.md`；它包含所有原有有效约束，并新增可搜索的“当前 BC + RL 路线”约定。

- [ ] **Step 1: 记录迁移前状态，保护无关工作区改动**

Run:

```bash
git status --short --branch
test -f AGENTS.md
test ! -L AGENTS.md
test ! -e CLAUDE.md
```

Expected:

- Git 状态会显示用户已有的未提交文件以及本次已批准的设计规格。
- `AGENTS.md` 是普通文件。
- `CLAUDE.md` 尚不存在。
- 如果 `CLAUDE.md` 已出现，或 `AGENTS.md` 已不是预期的普通文件，停止并重新检查，不能覆盖未知内容。

- [ ] **Step 2: 从现有约定建立逐字基线副本**

Run:

```bash
cp -- AGENTS.md CLAUDE.md
cmp --silent AGENTS.md CLAUDE.md
```

Expected: `cmp` 退出码为 0，证明优化前 `CLAUDE.md` 完整继承现有约定，没有遗漏。

- [ ] **Step 3: 在 `CLAUDE.md` 定向加入当前路线说明**

在标题 `# Repository Guidelines` 后、`## 项目结构与模块组织` 前加入以下章节；正文可以按相邻文风做最小措辞调整，但不得削弱这些要求：

```markdown
## 当前 BC + RL 路线

- 项目当前全面采用 BC + RL 作为主要研发路线。规则策略保留为历史资产、分析基线、回归参考和必要的合法性保障，不再作为主要扩展方向。
- 胡地卡组的单 deck、单 expert BC 已经成功验证 full-action imitation 与官方引擎闭环；这不代表跨卡组泛化、rollout collection、reward/value calibration 或 RL fine-tuning 已经完成。
- 当前并行推进两条主线：一是建立可审计的 rollout、reward、value calibration 与 RL fine-tuning 闭环；二是使用相近的 BC 结构分别学习常见强力卡组，形成更强、更多样的 arena opponents，为后续 RL 提供 curriculum 和训练环境。
- 多卡组 BC 默认训练相互隔离的 deck-specific policy。每套卡组必须保持明确的 deck、expert/team policy、dataset manifest、experiment/version 和 candidate package 边界；不得将不同 deck、team 或实现逻辑的动作标签无条件混入同一个 policy。
- 未来若采用跨卡组共享模型，必须显式加入可审计的 deck/source conditioning，并为冲突标签以及按 deck、source 分组的独立评测建立明确合同。离线 exact-action 或 legal-action 指标只能证明模仿质量和动作合同，不能单独证明真实对战强度。
- 新 BC policy 必须先形成自包含 candidate package，经过 package 验证和官方 engine runtime 真实对局评测，并取得用户明确确认后，才可进入 `evaluation/arena/opponents/`。不得因训练完成、离线指标较高或单一 matchup 表现良好而自动晋级。
- 每个正式 RL run 必须记录实际 opponent catalog、可复现的 pool snapshot，或足以重建对手集合的 package 标识和版本。opponent 构成、采样权重或 curriculum 阶段变化必须作为显式实验变量，禁止在未记录的情况下跨不同 opponent 池继续同一逻辑版本或直接比较结果。
```

同时检查全文中对规则策略的定位，仅在存在“规则策略仍是当前主要研发方向”的直接冲突表述时做最小修正。保留“宝可梦 TCG 规则学习长期记忆”及其已确认策略语义，因为它们仍用于专家行为分析、数据审计和回归检查。

- [ ] **Step 4: 检查路线文本的必要断言**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("CLAUDE.md").read_text(encoding="utf-8")
required = (
    "## 当前 BC + RL 路线",
    "单 deck、单 expert BC",
    "deck-specific policy",
    "官方 engine runtime 真实对局评测",
    "每个正式 RL run",
    "opponent 构成、采样权重或 curriculum 阶段变化",
)
missing = [item for item in required if item not in text]
assert not missing, f"CLAUDE.md 缺少必要约定: {missing}"
assert "## 官方引擎与评测硬约束" in text
assert "## 宝可梦 TCG 规则学习长期记忆" in text
assert "## 提交与 Pull Request" in text
print("CLAUDE.md route and preserved-section checks passed")
PY
```

Expected: 输出 `CLAUDE.md route and preserved-section checks passed`。

- [ ] **Step 5: 审阅正文差异，确认只有定向优化**

Run:

```bash
git diff --no-index -- AGENTS.md CLAUDE.md || test $? -eq 1
```

Expected:

- 差异主要是新增“当前 BC + RL 路线”。
- 若还修正冲突措辞，每处修改都必须直接服务于已批准路线。
- 原有官方引擎、evaluation、实验版本、规则、测试和提交硬约束不得被删除。

### Task 2: 将 `AGENTS.md` 替换为相对软链接

**Files:**
- Replace: `AGENTS.md`
- Target: `CLAUDE.md`

**Interfaces:**
- Consumes: Task 1 产出的普通文件 `CLAUDE.md`。
- Produces: 相对软链接 `AGENTS.md -> CLAUDE.md`；从任一路径读取时都得到同一正文。

- [ ] **Step 1: 在确认目标正文存在后替换源文件**

Run:

```bash
test -f CLAUDE.md
test ! -L CLAUDE.md
rm -- AGENTS.md
ln -s -- CLAUDE.md AGENTS.md
```

Expected: 命令全部成功；`CLAUDE.md` 保持普通文件，`AGENTS.md` 成为软链接。

- [ ] **Step 2: 验证链接类型和相对目标**

Run:

```bash
test -f CLAUDE.md
test ! -L CLAUDE.md
test -L AGENTS.md
test "$(readlink AGENTS.md)" = "CLAUDE.md"
test "$(readlink -f AGENTS.md)" = "$(readlink -f CLAUDE.md)"
printf 'AGENTS.md -> %s\n' "$(readlink AGENTS.md)"
```

Expected: 输出 `AGENTS.md -> CLAUDE.md`，所有断言退出码为 0。

- [ ] **Step 3: 验证两个入口解析出完全相同的内容**

Run:

```bash
cmp --silent AGENTS.md CLAUDE.md
python3 - <<'PY'
from pathlib import Path

agents = Path("AGENTS.md").read_text(encoding="utf-8")
claude = Path("CLAUDE.md").read_text(encoding="utf-8")
assert agents == claude
assert agents.startswith("# Repository Guidelines\n")
print("AGENTS.md and CLAUDE.md resolve to identical content")
PY
```

Expected: 输出 `AGENTS.md and CLAUDE.md resolve to identical content`。

### Task 3: 完成限定范围验收

**Files:**
- Verify: `CLAUDE.md`
- Verify: `AGENTS.md`
- Verify unchanged: `README.md`
- Verify unchanged: all pre-existing unrelated working-tree paths

**Interfaces:**
- Consumes: Task 1 的统一正文与 Task 2 的软链接。
- Produces: 可审阅的 Git diff 和无格式错误的最终工作区；不创建 commit。

- [ ] **Step 1: 检查 Markdown 和 Git patch 格式**

Run:

```bash
git diff --check -- AGENTS.md CLAUDE.md
```

Expected: 无输出，退出码为 0。

- [ ] **Step 2: 检查 Git 正确记录普通文件和软链接**

Run:

```bash
git status --short -- AGENTS.md CLAUDE.md
git diff --summary -- AGENTS.md CLAUDE.md
git diff -- AGENTS.md CLAUDE.md
```

Expected:

- `CLAUDE.md` 显示为新增普通文件。
- `AGENTS.md` 显示为类型或内容变化，diff 中 symlink mode 为 `120000`，链接内容为 `CLAUDE.md`。
- 正文包含批准的 BC + RL 路线，不包含无关重写。

- [ ] **Step 3: 重新检查整个工作区，确认无关改动仍原样存在**

Run:

```bash
git status --short --branch
```

Expected:

- 本次新增/修改仅包括 `CLAUDE.md`、`AGENTS.md`，以及规划阶段已创建的设计规格和实施计划。
- 用户在任务开始前已有的 `docs/reports/README.md` 修改和其他未跟踪规格/计划/HTML 文件仍存在，但没有被本次实施改写或删除。

- [ ] **Step 4: 给出完成报告，不提交或推送**

报告必须说明：

- `CLAUDE.md` 已成为唯一正文；
- `AGENTS.md -> CLAUDE.md` 已验证；
- 新增了哪些 BC + RL 硬边界；
- 执行了哪些验证及其结果；
- 未运行训练或完整测试，因为变更仅涉及文档和软链接；
- 未执行 `git commit` 或 `git push`；
- 工作区原有的无关未提交改动未被触碰。

不要添加 commit 步骤，也不要把“未 commit”描述为遗漏；这是本计划的明确约束。
