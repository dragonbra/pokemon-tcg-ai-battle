# 仓库职责拆分与工作目录重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 Alakazam 策略行为的前提下，将当前候选、历史提交、长期知识、评测框架和 replay 可视化按职责拆分，并让当前候选从根目录 work/ 直接打包到 submission/dist/。

**Architecture:** 第一阶段建立根目录 work/、codex/index/ 和 submission/dist/，并统一资产检查、打包和本地 battle 的路径解析。第二阶段迁移评测核心到 evaluation/，第三阶段迁移 replay 核心到 visualization/；scripts/ 保留为兼容 CLI 入口。

**Tech Stack:** Python 3.11+、标准库 pathlib/json/tarfile/unittest、Bash set -euo pipefail、官方 cg/ 运行时和 Kaggle replay JSON。

## Global Constraints

- work/ 位于仓库根目录。
- work/ 下的当前候选目录只包含 main.py、deck.csv 和 cg/。
- submission/ 保存历史完整提交；打包产物统一写入 submission/dist/。
- 压缩包顶层必须是 main.py、deck.csv 和 cg/。
- data/official/ 和 engine/source/ 保持只读，本计划不修改策略行为、卡组和官方引擎。
- 详细报告继续放在 docs/reports/，官方 replay 继续放在 replays/，本地临时输出写入 /tmp。
- 除非用户显式要求，不执行 git commit。

## Scope

本计划分成三个可独立验收的阶段：

1. 结构、当前工作包和提交流程。
2. 评测核心和 Auto Iteration 代码。
3. replay 可视化核心和分析入口。

## File Map

第一阶段新增：

- codex/README.md
- codex/index/{README.md,CURRENT_STATE.md,DESIGN_PRINCIPLES.md,RULES.md,EVALUATION.md,CHANGELOG.md}
- codex/design/README.md
- work/README.md
- work/docs/{README.md,DECK_NOTES.md,STRATEGY.md,CURRENT_OBJECTIVES.md}
- work/auto-iteration/{README.md,GOALS.md,METRICS.md,ITERATION_RULES.md,ITERATION_TEMPLATE.md}
- work/auto-iteration/templates/{manifest.json,analysis.md,decision.md,case.json}
- work/alakazam_v8_current/{main.py,deck.csv,cg/}
- scripts/submission_paths.py
- tests/test_submission_paths.py

第一阶段修改：

- scripts/check_assets.py
- scripts/package_submission.sh
- scripts/run_local_battle.py
- scripts/promote_v7_best.sh
- scripts/v7_best_submission.py
- tests/test_v7_best_submission.py
- AGENTS.md
- README.md
- submission/README.md

第二阶段新增或迁移：

- evaluation/__init__.py
- evaluation/README.md
- evaluation/configs/README.md
- evaluation/runner/README.md
- evaluation/adapters/README.md
- evaluation/metrics/README.md
- evaluation/cases/README.md
- evaluation/schemas/auto_iteration_manifest.schema.json
- evaluation/schemas/auto_iteration_case.schema.json
- evaluation/auto_iteration.py
- evaluation/analyze_alakazam_replays.py
- evaluation/analyze_kaggle_alakazam_replays.py
- tests/test_evaluation_layout.py

第二阶段保留兼容包装：

- scripts/alakazam_auto_iter.py
- scripts/analyze_alakazam_replays.py
- scripts/analyze_kaggle_alakazam_replays.py

第三阶段新增或迁移：

- visualization/{__init__.py,README.md}
- visualization/replay/{__init__.py,core.py}
- visualization/analysis/README.md
- visualization/templates/replay-analysis.md
- tests/test_visualization_layout.py

第三阶段保留兼容包装：

- scripts/replay_visualizer.py
- scripts/visualize_replay.py

---

### Task 1: 建立统一路径契约

**Files:**

- Create: scripts/submission_paths.py
- Create: tests/test_submission_paths.py
- Modify: scripts/check_assets.py
- Modify: scripts/run_local_battle.py

**Interfaces:**

- is_complete_submission(path: Path) -> bool
- historical_submission_dirs(root: Path) -> list[Path]
- work_submission_dirs(root: Path) -> list[Path]
- resolve_submission(name: str, root: Path, prefer_work: bool = True) -> Path

- [ ] Step 1: 先写测试，覆盖 work 候选发现、docs/ 和 auto-iteration/ 排除、work 优先级和 submission/dist/ 排除。

- [ ] Step 2: 运行失败测试。

    python3 -m unittest tests.test_submission_paths -v

    Expected: 因 scripts.submission_paths 尚不存在而失败。

- [ ] Step 3: 实现 pathlib 路径模块。is_complete_submission 必须检查 main.py、deck.csv 和 cg/；historical_submission_dirs 忽略 dist/；work_submission_dirs 只扫描 work/ 的直接子目录；resolve_submission 按 work/ 再 submission/ 顺序查找，不存在时抛出 ValueError。

- [ ] Step 4: 修改 check_assets.py 扫描历史提交和 work 候选；修改 run_local_battle.py 的加载、sys.path 和可用 agent 列表使用统一解析。

- [ ] Step 5: 验证。

    python3 -m unittest tests.test_submission_paths -v
    python3 scripts/check_assets.py

    Expected: 所有历史提交通过，work 候选建立后也通过，submission/dist/ 不被当作 agent。

---

### Task 2: 建立 Codex 索引和 work 文档骨架

**Files:**

- Create: codex/、work/ 及 File Map 中列出的 Markdown、JSON、.gitkeep 文件。

- [ ] Step 1: 创建目录。

    mkdir -p codex/index codex/design work/docs/decisions work/auto-iteration/templates

- [ ] Step 2: 编写 codex/index/。RULES.md 链接 docs/rules/；EVALUATION.md 链接 docs/history/kaggle/；CHANGELOG.md 按 V1 至 V8 记录卡组变化、策略变化、评测结论和来源；CURRENT_STATE.md 记录当前候选、基线和下一目标。

- [ ] Step 3: 编写 work/docs/。DECK_NOTES.md 和 STRATEGY.md 只描述当前候选的设计，CURRENT_OBJECTIVES.md 记录本轮目标；这些文件不进入打包包体。

- [ ] Step 4: 编写 work/auto-iteration/。固定目标、指标、correctness gate、case resolution、outcome guardrail、control/candidate、focused 和 promotion 规则。模板必须包含 manifest、analysis、decision、case 的既有字段。

- [ ] Step 5: 验证。

    python3 -m json.tool work/auto-iteration/templates/manifest.json >/dev/null
    python3 -m json.tool work/auto-iteration/templates/case.json >/dev/null
    rg -n 'docs/reports/|submission/|control|candidate|case_status' codex/index work/auto-iteration
    git diff --check

---

### Task 3: 创建纯净当前候选并迁移 dist/

**Files:**

- Create: work/alakazam_v8_current/main.py
- Create: work/alakazam_v8_current/deck.csv
- Create: work/alakazam_v8_current/cg/
- Move: dist/ to submission/dist/

- [ ] Step 1: 只复制现有 V8 的运行时文件。

    mkdir -p work/alakazam_v8_current
    cp submission/alakazam_v8/main.py work/alakazam_v8_current/main.py
    cp submission/alakazam_v8/deck.csv work/alakazam_v8_current/deck.csv
    cp -R submission/alakazam_v8/cg work/alakazam_v8_current/cg
    find work/alakazam_v8_current -type d -name __pycache__ -prune -exec rm -rf {} +
    find work/alakazam_v8_current -type f \( -name '*.pyc' -o -name '*.json' -o -name '*.md' \) -delete

- [ ] Step 2: 确认 submission/dist/ 没有冲突后移动现有产物。

    git mv dist submission/dist

- [ ] Step 3: 验证纯净候选和资产。

    find work/alakazam_v8_current -maxdepth 1 -mindepth 1 -print | sort
    python3 scripts/check_assets.py
    python3 -m py_compile work/alakazam_v8_current/main.py work/alakazam_v8_current/cg/*.py

    Expected: 候选顶层只有 main.py、deck.csv、cg/，卡组和 simulator 检查通过。

---

### Task 4: 更新打包、best artifact 和本地运行路径

**Files:**

- Modify: scripts/package_submission.sh
- Modify: scripts/promote_v7_best.sh
- Modify: scripts/v7_best_submission.py
- Modify: tests/test_v7_best_submission.py
- Modify: AGENTS.md、README.md、submission/README.md

**Interfaces:**

    bash scripts/package_submission.sh alakazam_v8_current
    tar -tzf submission/dist/alakazam_v8_current.tar.gz

- [ ] Step 1: 先更新 tests/test_v7_best_submission.py，临时 archive 使用 root/submission/dist/，并增加根目录 dist/ 被拒绝的测试。

- [ ] Step 2: 修改 package_submission.sh。优先读取 work/<name>/；不存在时兼容读取 submission/<name>/；输出目录固定为 root/submission/dist/；保留 main.py、deck.csv、cg/ 的 archive 顶层结构。

- [ ] Step 3: 修改 v7_best_submission.py 的 artifact 边界为 repo_root/submission/dist/，修改 promote_v7_best.sh 的输出为 submission/dist/<name>.tar.gz；同步更新 BEST_STRATEGY.json 的相对路径。

- [ ] Step 4: 更新 AGENTS.md、README.md 和 submission/README.md 的目录表和命令。当前命令使用 work/alakazam_v8_current 和 submission/dist/；历史报告中的旧命令作为历史记录保留，不作为当前操作入口。

- [ ] Step 5: 验证。

    python3 -m unittest tests.test_submission_paths tests.test_v7_best_submission -v
    python3 scripts/check_assets.py
    bash scripts/package_submission.sh alakazam_v8_current
    tar -tzf submission/dist/alakazam_v8_current.tar.gz

    Expected: archive 只有顶层 main.py、deck.csv、cg/；新脚本不创建根目录 dist/。

---

### Task 5: 迁移评测核心并保留 scripts 兼容入口

**Files:**

- Move: scripts/alakazam_auto_iter.py to evaluation/auto_iteration.py
- Move: scripts/analyze_alakazam_replays.py to evaluation/analyze_alakazam_replays.py
- Move: scripts/analyze_kaggle_alakazam_replays.py to evaluation/analyze_kaggle_alakazam_replays.py
- Create: evaluation/ 的 README、子目录 README、schema 和 tests/test_evaluation_layout.py
- Modify: 原三个 scripts/ 文件为 re-export 和 CLI wrapper

**Interfaces:**

- evaluation.auto_iteration 保留 EvaluationMetrics、CaseRecord、AnalysisResult、PromotionDecision、analyze_records、analyze_report、compare_reports 和 main。
- scripts/alakazam_auto_iter.py 继续支持 from scripts.alakazam_auto_iter import ...。
- schema 固定 manifest 的 control、candidate、deck、opponents、games、swap、randomness 字段，以及 case 的 case_id、source、state_summary、legal_options、control_action、candidate_action、expected_action、expected_reason、case_status 字段。

- [ ] Step 1: 先写新模块和旧 wrapper 的对象一致性测试。

- [ ] Step 2: 使用 git mv 移动实现；原 scripts 文件只 re-export 新模块并保留 if __name__ == '__main__' 的 CLI 调用。

- [ ] Step 3: 编写 evaluation/README.md，明确评测代码在本仓库、详细结果在 docs/reports/、原始 trace 不进入当前候选。

- [ ] Step 4: 验证。

    python3 -m unittest tests.test_evaluation_layout tests.test_alakazam_auto_iter -v
    python3 -m evaluation.auto_iteration --help
    python3 scripts/alakazam_auto_iter.py --help

    Expected: 新旧 CLI 都可运行，现有 AutoIter 测试结果不变。

---

### Task 6: 迁移 replay 可视化核心并保留 CLI 入口

**Files:**

- Move: scripts/replay_visualizer.py to visualization/replay/core.py
- Create: visualization/、visualization/replay/、visualization/analysis/ 和 templates/ 入口文件
- Modify: scripts/replay_visualizer.py、scripts/visualize_replay.py、tests/test_replay_visualization.py、docs/replay-visualization.md

**Interfaces:**

- visualization.replay.core 保留 ReplayFormatError、NormalizedReplay、ReplayLaunch、extract_visualize_frames、load_replay、attach_trace_metadata、create_viewer_launcher、show_replay。
- scripts.replay_visualizer 继续 re-export 上述 API。
- CLI 继续使用 python3 scripts/visualize_replay.py replay.json --no-open。

- [ ] Step 1: 先写新核心模块和旧 wrapper 的对象一致性测试。

- [ ] Step 2: 使用 git mv 移动实现，原脚本只 re-export；CLI 改为从 visualization.replay.core 导入。

- [ ] Step 3: 保持 visualize、visualize_frames、Kaggle steps[*][*].visualize 的提取优先级，以及旧 trace 缺帧时的明确错误。

- [ ] Step 4: 验证。

    python3 -m unittest tests.test_visualization_layout tests.test_replay_visualization -v
    python3 scripts/visualize_replay.py replays/kaggle_v6_54829503/episode-86876715-replay.json --no-open

    Expected: 测试通过，CLI 生成临时 launcher，不修改 replays/。

---

### Task 7: 更新全局路径并完成验收

**Files:**

- Modify: AGENTS.md、README.md、docs/reports/README.md、docs/replay-visualization.md
- Move: docs/superpowers/specs/ to codex/design/specs/；docs/superpowers/plans/ to codex/design/plans/，仅在 live links 更新后执行

- [ ] Step 1: 用 rg 检查当前入口中的旧路径。

    rg -n 'dist/|docs/superpowers|scripts/replay_visualizer|scripts/alakazam_auto_iter' AGENTS.md README.md submission reports notes docs scripts tests codex work evaluation visualization

- [ ] Step 2: 更新有效链接和当前命令；历史报告可以保留旧命令文本，但不能让当前 README、AGENTS 或测试指向不存在的路径。

- [ ] Step 3: 运行最终矩阵。

    python3 scripts/check_assets.py
    python3 -m unittest discover -s tests -v
    python3 -m compileall -q scripts submission work evaluation visualization
    bash scripts/package_submission.sh alakazam_v8_current
    tar -tzf submission/dist/alakazam_v8_current.tar.gz
    python3 scripts/visualize_replay.py replays/kaggle_v6_54829503/episode-86876715-replay.json --no-open
    git diff --check
    git status --short

- [ ] Step 4: 确认策略、卡组和官方引擎无非计划修改。

    git diff -- data/official engine/source submission/alakazam_v8/main.py submission/alakazam_v8/deck.csv

    Expected: 没有策略行为、卡组或官方引擎的非计划变化。

## Execution Notes

- 阶段 1 完成后独立 review，再开始阶段 2；阶段 2 和阶段 3 可以分开执行。
- 不删除历史 submission、reports 或 replays。
- 不把 submission/dist/ 解压内容写入 work/。
- 本计划不包含 commit；执行时每个阶段只报告 diff 和验证结果，commit 需用户另行授权。
