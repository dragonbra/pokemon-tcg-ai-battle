# W&B Formal Training Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the verified W&B scalar mirror the standard remote record for future formal BC, value-calibration, and RL training while preserving repository-local audit artifacts.

**Architecture:** Formal trainers continue writing `training_metrics.jsonl` first, TensorBoard second, and a failure-isolated W&B mirror last through `TrainingLogger`. Repository experiment/version identity determines the W&B group, name, and stable run ID; local artifacts remain authoritative, while W&B provides remote curves and comparisons.

**Tech Stack:** Python 3.11+, `wandb` 0.28.x in the host user environment, canonical JSONL, TensorBoard, W&B Cloud, standard-library `unittest`, HTML project documentation.

## Global Constraints

- Do not modify `engine/source/`.
- Use W&B project `dragon_bra/pokemon-tcg-policy-learning`, verified private on 2026-07-26.
- Install and invoke W&B through host `python3`, not the repository `.venv`.
- Keep `training_metrics.jsonl` authoritative and flush it before TensorBoard and W&B.
- Restrict automatic online tracking to `rl_runs/artifact/<experiment>/V<n>_<tag>/`.
- One repository version maps to one stable W&B run; only the same version may resume it.
- Never upload checkpoints, datasets, optimizer state, traces, replays, observations, or source patches by default.
- Compare BC and RL policy strength only with the same official-engine evaluation contract.
- Preserve unrelated worktree changes and do not commit unless the user explicitly requests it.

---

### Task 1: Promote the verified behavior into repository policy

**Files:**
- Modify: `CLAUDE.md` (the project convention is exposed through the `AGENTS.md` symlink)

**Interfaces:**
- Consumes: the existing immutable `V<n>_<tag>` run policy and `TrainingLogger` behavior.
- Produces: a mandatory checklist for formal BC, value-calibration, and PPO runs.

- [x] **Step 1: Add the formal W&B recording contract**

Document the required project/entity, online mode, stable identity, write order, metric namespaces, host-Python environment, failure behavior, and upload exclusions.

- [x] **Step 2: Align the existing epoch-recording rule**

Change the formal epoch rule from JSONL plus TensorBoard to JSONL plus TensorBoard plus W&B when online tracking is enabled, without making W&B the canonical source.

- [x] **Step 3: Review the policy through both entry paths**

Run:

```bash
rg -n "W&B 正式训练记录约定|pokemon-tcg-policy-learning" AGENTS.md CLAUDE.md
```

Expected: both paths expose the same policy text through the existing symlink.

### Task 2: Update operator documentation to the real host environment

**Files:**
- Modify: `rl_environment/README.md`
- Modify: `rl_runs/README.md`

**Interfaces:**
- Consumes: `python3 -m rl_environment.wandb_smoke`, `WANDB_MODE`, `WANDB_ENTITY`, `WANDB_PROJECT`, and formal run paths.
- Produces: copyable login, offline smoke, online smoke, and formal-run commands plus an archival checklist.

- [x] **Step 1: Replace virtual-environment commands**

Use `python3 -m pip install --user 'wandb>=0.28,<1'`, `python3 -m wandb login --relogin`, and host `python3` smoke commands. State that `.venv` is not the W&B runtime on this machine.

- [x] **Step 2: Record the verified cloud destination**

Document `WANDB_ENTITY=dragon_bra`, `WANDB_PROJECT=pokemon-tcg-policy-learning`, private visibility, and the 2026-07-26 five-epoch online smoke result.

- [x] **Step 3: Add the formal-run archival checklist**

Require the version status or summary to retain the W&B project, stable run ID or URL, sync outcome, and any mirror failure while keeping local metrics complete.

### Task 3: Synchronize the live project design and research report

**Files:**
- Modify: `train/alakazam_bc_rl/DESIGN.html`
- Modify: `docs/training/rl/wandb-bc-rl-experiment-tracking-design-20260725.html`

**Interfaces:**
- Consumes: the completed online run and API verification.
- Produces: current-state HTML that distinguishes implemented scalar tracking from future official-evaluation export.

- [x] **Step 1: Promote online scalar tracking to verified**

Record that the cloud smoke finished with five history rows, six metric series, private project access, four W&B metadata files, and zero artifacts/media.

- [x] **Step 2: Preserve the remaining boundary**

Keep automatic `eval/*` summary and per-opponent W&B Table export explicitly marked as not implemented; do not imply that scalar mirroring proves policy strength.

- [x] **Step 3: Parse both HTML files**

Run:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

for name in (
    "train/alakazam_bc_rl/DESIGN.html",
    "docs/training/rl/wandb-bc-rl-experiment-tracking-design-20260725.html",
):
    parser = HTMLParser()
    parser.feed(Path(name).read_text(encoding="utf-8"))
    parser.close()
PY
```

Expected: exit code 0.

### Task 4: Verify the persisted implementation and documentation

**Files:**
- Test: `tests/test_rl_wandb_identity.py`
- Test: `tests/test_rl_wandb_logging.py`
- Test: `tests/test_rl_wandb_smoke.py`

**Interfaces:**
- Consumes: all W&B implementation and policy changes.
- Produces: evidence that local logging remains canonical and the repository is regression-free.

- [x] **Step 1: Run focused W&B tests**

Run:

```bash
python3 -m unittest -v \
  tests.test_rl_wandb_identity \
  tests.test_rl_wandb_logging \
  tests.test_rl_wandb_smoke
```

Expected: all focused tests pass without a cloud upload.

- [x] **Step 2: Run the full test suite**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests pass.

- [x] **Step 3: Run syntax and whitespace validation**

Run:

```bash
python3 -m compileall -q evaluation visualization rl_environment train
git diff --check
```

Expected: both commands exit successfully.
