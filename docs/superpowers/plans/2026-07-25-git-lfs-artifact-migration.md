# Git LFS 全历史制品迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库全部历史中的模型权重与正式 submission 压缩包迁移到 GitHub Git LFS，在仓库外保留可验证回滚备份，并完成受保护的 force-push 和干净 clone 恢复演练。

**Architecture:** 先冻结并记录远端引用及当前制品哈希，在仓库外建立完整 Git bundle 和未跟踪文件保护副本；随后安装 Git LFS，在临时迁移 clone 中重写所有 refs，而不是直接冒险改写唯一工作目录。验证通过后，将迁移结果同步回主工作目录、上传 LFS 对象、以审计时的远端 SHA 作为 lease force-push，最后从 GitHub 进行全新 clone 验证。

**Tech Stack:** Git, Git LFS, GitHub HTTPS remote, SHA-256 (`sha256sum`), Git bundle, Bash coreutils

## Global Constraints

- 迁移范围严格为 `*.bin`、`*.pt`、`*.ckpt`、`submission/dist/*.tar.gz`。
- 不迁移 dataset、official data、replay、trace、cache 或未被 Git 跟踪的 checkpoint。
- 不得修改、删除或提交 `docs/superpowers/plans/2026-07-25-alakazam-rwbc-system-environment.md`。
- 不得修改、删除或提交 `docs/superpowers/specs/2026-07-25-alakazam-rwbc-training-environment-design.md`。
- force-push 前必须建立并验证仓库外完整备份，且本次不得删除该备份。
- 任一 refs、bundle、Git、LFS、哈希或远端 lease 校验失败时，停止发布。
- 用户已明确授权在所有门禁通过后 force-push 现有远端分支和 tags。
- Git 提交身份使用近期仓库提交身份 `dragon_bra <tommy514@foxmail.com>`，通过单次命令的 `git -c` 参数传入，不修改全局配置。

---

## File Structure

- Create: `.gitattributes` — 定义当前和未来二进制制品的 Git LFS 路径规则；由历史迁移写入所有相关 refs。
- Create outside repo: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/` — 保存 bundle、refs、哈希、审计日志和未跟踪文件保护副本。
- Create outside repo: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/migration-clone/` — 从 bundle 恢复出的隔离迁移 clone。
- Create outside repo: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/verification-clone/` — force-push 后从 GitHub 创建的恢复演练 clone。
- Modify through history rewrite: all refs containing `*.bin`, `*.pt`, `*.ckpt`, or `submission/dist/*.tar.gz` — 将普通 Git blob 替换为 LFS pointer。
- Preserve unchanged and untracked: the two pre-existing Alakazam design/plan files listed under Global Constraints.

### Task 1: Freeze Remote State and Build Recovery Backup

**Files:**
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/pre-migration.bundle`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/refs-before.txt`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/remote-refs-before.txt`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/artifacts-before.sha256`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/untracked-before.sha256`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/protected-untracked/`

**Interfaces:**
- Consumes: current repository at `/home/cyd/repos/pokemon-tcg-ai-battle`, remote `origin`.
- Produces: verified immutable recovery inputs and exact expected remote SHA values for protected force-push leases.

- [ ] **Step 1: Verify repository identity, connectivity, and clean tracked state**

Run:

```bash
git remote get-url origin
git status --porcelain=v1
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*'
git ls-remote --heads --tags origin
```

Expected: origin is `https://github.com/dragonbra/pokemon-tcg-ai-battle.git`; status contains only the two protected untracked files; fetch and `ls-remote` both succeed. If TLS fails again, stop and diagnose connectivity before any migration.

- [ ] **Step 2: Create the external backup directory and protect untracked files**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
mkdir -p "$BACKUP/protected-untracked/docs/superpowers/plans" "$BACKUP/protected-untracked/docs/superpowers/specs"
cp --preserve=mode,timestamps docs/superpowers/plans/2026-07-25-alakazam-rwbc-system-environment.md "$BACKUP/protected-untracked/docs/superpowers/plans/"
cp --preserve=mode,timestamps docs/superpowers/specs/2026-07-25-alakazam-rwbc-training-environment-design.md "$BACKUP/protected-untracked/docs/superpowers/specs/"
sha256sum docs/superpowers/plans/2026-07-25-alakazam-rwbc-system-environment.md docs/superpowers/specs/2026-07-25-alakazam-rwbc-training-environment-design.md > "$BACKUP/untracked-before.sha256"
```

Expected: both files exist in the external protection directory and the manifest has two entries.

- [ ] **Step 3: Record refs and artifact hashes**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git show-ref > "$BACKUP/refs-before.txt"
git ls-remote --heads --tags origin > "$BACKUP/remote-refs-before.txt"
git rev-parse HEAD > "$BACKUP/head-before.txt"
git remote -v > "$BACKUP/remotes-before.txt"
find . -type f \( -name '*.bin' -o -name '*.pt' -o -name '*.ckpt' -o -path './submission/dist/*.tar.gz' \) -not -path './.git/*' -print0 | sort -z | xargs -0 -r sha256sum > "$BACKUP/artifacts-before.sha256"
```

Expected: all files are populated; artifact manifest paths are repository-relative with `./` prefixes.

- [ ] **Step 4: Create and verify a complete Git bundle**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git bundle create "$BACKUP/pre-migration.bundle" --all
git bundle verify "$BACKUP/pre-migration.bundle" | tee "$BACKUP/bundle-verify.txt"
```

Expected: verification reports a complete history and exits zero.

- [ ] **Step 5: Prove the bundle can restore refs**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
rm -rf "$BACKUP/bundle-smoke-clone"
git clone "$BACKUP/pre-migration.bundle" "$BACKUP/bundle-smoke-clone"
git -C "$BACKUP/bundle-smoke-clone" fsck --full
```

Expected: clone and fsck exit zero. `bundle-smoke-clone` remains as backup evidence.

### Task 2: Install and Validate Git LFS

**Files:**
- Modify system package state: install `git-lfs` using the host package manager.
- Create: user Git LFS filter configuration via `git lfs install`.

**Interfaces:**
- Consumes: working network/package manager and Git installation.
- Produces: executable `git-lfs` and functioning clean/smudge filters required by migration and recovery.

- [ ] **Step 1: Diagnose the broken command shim**

Run:

```bash
command -v git-lfs || true
type -a git-lfs || true
git --exec-path
find "$(git --exec-path)" -maxdepth 1 -name 'git-lfs*' -ls
```

Expected: evidence identifies whether `git-lfs` is missing or a broken executable/link. Do not alter Git configuration based on a guess.

- [ ] **Step 2: Install Git LFS from the platform package manager**

Run on this Ubuntu/WSL host:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
```

Expected: package installation exits zero. If sudo requires interactive authentication, the user runs `! sudo apt-get update && sudo apt-get install -y git-lfs` in this session.

- [ ] **Step 3: Initialize and verify Git LFS**

Run:

```bash
git lfs version
git lfs install
git lfs env
```

Expected: a concrete Git LFS version is printed and LFS filter/process configuration points to the installed executable.

### Task 3: Rewrite History in an Isolated Migration Clone

**Files:**
- Create/modify: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/migration-clone/.gitattributes`
- Rewrite in isolated clone: all refs containing matching artifact paths.

**Interfaces:**
- Consumes: verified `pre-migration.bundle`, artifact hash manifest, working Git LFS.
- Produces: isolated rewritten refs, populated local LFS object store, and migration audit files.

- [ ] **Step 1: Create a migration clone from the verified bundle**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
rm -rf "$BACKUP/migration-clone"
git clone "$BACKUP/pre-migration.bundle" "$BACKUP/migration-clone"
git -C "$BACKUP/migration-clone" remote set-url origin https://github.com/dragonbra/pokemon-tcg-ai-battle.git
git -C "$BACKUP/migration-clone" fetch origin '+refs/heads/*:refs/remotes/origin/*'
```

Expected: migration clone has the original repository history and expected remote-tracking refs.

- [ ] **Step 2: Estimate LFS migration size**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git -C "$BACKUP/migration-clone" lfs migrate info --everything --include='*.bin,*.pt,*.ckpt,submission/dist/*.tar.gz' | tee "$BACKUP/lfs-migrate-info.txt"
```

Expected: command identifies only intended extensions/path. Record reported object count and total size; stop if it is unexpectedly near or above the account's acceptable LFS quota.

- [ ] **Step 3: Rewrite all refs**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git -C "$BACKUP/migration-clone" lfs migrate import --everything --include='*.bin,*.pt,*.ckpt,submission/dist/*.tar.gz'
```

Expected: command exits zero and writes standard LFS rules to `.gitattributes` in rewritten history.

- [ ] **Step 4: Verify exact attribute rules at migrated HEAD**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
printf '%s\n' \
  '*.bin filter=lfs diff=lfs merge=lfs -text' \
  '*.pt filter=lfs diff=lfs merge=lfs -text' \
  '*.ckpt filter=lfs diff=lfs merge=lfs -text' \
  'submission/dist/*.tar.gz filter=lfs diff=lfs merge=lfs -text' \
  > "$BACKUP/expected-gitattributes.txt"
git -C "$BACKUP/migration-clone" show HEAD:.gitattributes > "$BACKUP/actual-gitattributes.txt"
diff -u "$BACKUP/expected-gitattributes.txt" "$BACKUP/actual-gitattributes.txt"
```

Expected: no diff. If migration emits equivalent but differently ordered rules, normalize only after confirming all four exact semantics; do not broaden `*.tar.gz`.

### Task 4: Validate Rewritten History and Artifact Integrity

**Files:**
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/refs-after.txt`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/artifacts-after.sha256`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/lfs-files-all.txt`

**Interfaces:**
- Consumes: isolated rewritten migration clone and pre-migration manifests.
- Produces: evidence that refs are preserved by name, target blobs are LFS pointers, and current artifact bytes are unchanged.

- [ ] **Step 1: Run Git and LFS integrity checks**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git -C "$BACKUP/migration-clone" fsck --full | tee "$BACKUP/git-fsck-after.txt"
git -C "$BACKUP/migration-clone" lfs fsck | tee "$BACKUP/lfs-fsck-after.txt"
git -C "$BACKUP/migration-clone" lfs ls-files --all --long > "$BACKUP/lfs-files-all.txt"
```

Expected: both fsck commands exit zero and LFS inventory is non-empty.

- [ ] **Step 2: Verify all matching historical blobs are LFS pointers**

Run a Python one-off audit from the migration clone that enumerates every commit and matching path, reads each blob with `git show <commit>:<path>`, and requires the first line to equal `version https://git-lfs.github.com/spec/v1`. Save its checked count to `historical-pointer-audit.txt`; exit nonzero on any mismatch.

Expected: zero mismatches and a positive checked count.

- [ ] **Step 3: Compare pre/post current artifact hashes**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
cd "$BACKUP/migration-clone"
find . -type f \( -name '*.bin' -o -name '*.pt' -o -name '*.ckpt' -o -path './submission/dist/*.tar.gz' \) -not -path './.git/*' -print0 | sort -z | xargs -0 -r sha256sum > "$BACKUP/artifacts-after.sha256"
diff -u "$BACKUP/artifacts-before.sha256" "$BACKUP/artifacts-after.sha256"
```

Expected: no diff.

- [ ] **Step 4: Verify ref names are preserved**

Run a comparison that strips SHA columns from `refs-before.txt` and newly generated `refs-after.txt`, excluding migration-created internal refs if Git LFS creates any.

Expected: all original local branches, remote-tracking branches, and tags remain present by name; SHA changes are expected.

- [ ] **Step 5: Verify protected files in the original worktree**

Run from the original repository:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
sha256sum -c "$BACKUP/untracked-before.sha256"
git status --porcelain=v1
```

Expected: both hashes report `OK`; status still lists only the two protected untracked files.

### Task 5: Publish Rewritten Refs with Explicit Leases

**Files:**
- Update remotely: `refs/heads/main`, `refs/heads/to_better_opponents`, and any tags recorded before migration.
- Upload remotely: all referenced LFS objects.

**Interfaces:**
- Consumes: verified migration clone and `remote-refs-before.txt` lease values.
- Produces: GitHub repository whose public refs point to rewritten LFS history without overwriting unexpected concurrent changes.

- [ ] **Step 1: Recheck remote identity and frozen refs**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git -C "$BACKUP/migration-clone" remote get-url origin
git -C "$BACKUP/migration-clone" ls-remote --heads --tags origin > "$BACKUP/remote-refs-pre-push.txt"
diff -u "$BACKUP/remote-refs-before.txt" "$BACKUP/remote-refs-pre-push.txt"
```

Expected: exact match. Any change stops publication.

- [ ] **Step 2: Upload all LFS objects first**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git -C "$BACKUP/migration-clone" lfs push --all origin
```

Expected: every LFS object uploads successfully; quota/authentication failures stop publication before Git refs change.

- [ ] **Step 3: Force-push branches using explicit expected SHA leases**

Construct one `--force-with-lease=refs/heads/<name>:<old-sha>` argument per remote branch from `remote-refs-before.txt`, then push each rewritten local branch explicitly to the same remote branch. Do not use unguarded `--force`.

Expected: `main` and `to_better_opponents` update; a changed remote SHA rejects the push instead of overwriting it.

- [ ] **Step 4: Force-push tags with equivalent frozen-ref protection**

If `remote-refs-before.txt` contains tags, recheck exact remote tag refs immediately before pushing, then push the rewritten tags. If no tags existed, record `no pre-migration tags` and skip this step.

Expected: all pre-existing tag names point to rewritten objects; no unrecorded tag is deleted.

- [ ] **Step 5: Record published refs**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
git -C "$BACKUP/migration-clone" ls-remote --heads --tags origin > "$BACKUP/remote-refs-after.txt"
```

Expected: remote branch/tag names match pre-migration names and their SHAs match migration clone refs.

### Task 6: Align the Original Worktree and Perform Clean Recovery Test

**Files:**
- Rewrite local tracked Git state: `/home/cyd/repos/pokemon-tcg-ai-battle/.git/` refs and objects, while preserving the two protected untracked files.
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/verification-clone/`
- Create: `/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/final-report.txt`

**Interfaces:**
- Consumes: published rewritten GitHub refs, protected untracked backup, pre/post hash manifests.
- Produces: usable original worktree on rewritten `main`, independent successful recovery clone, and final evidence report.

- [ ] **Step 1: Copy rewritten refs into the original worktree safely**

Before changing the original worktree, reverify protected hashes. Fetch rewritten remote refs, then reset tracked `main` to `origin/main`; because this changes only tracked files, do not run `git clean` and do not remove untracked files.

Expected: original `main` equals rewritten `origin/main`; the two protected files remain untracked and hash-identical.

- [ ] **Step 2: Verify original worktree LFS checkout**

Run:

```bash
git lfs pull
git lfs fsck
sha256sum -c /home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25/untracked-before.sha256
git status --porcelain=v1
```

Expected: LFS succeeds; protected files remain the only untracked paths.

- [ ] **Step 3: Clone from GitHub into a fresh recovery directory**

Run:

```bash
BACKUP=/home/cyd/backups/pokemon-tcg-ai-battle-lfs-migration-2026-07-25
rm -rf "$BACKUP/verification-clone"
git clone https://github.com/dragonbra/pokemon-tcg-ai-battle.git "$BACKUP/verification-clone"
git -C "$BACKUP/verification-clone" lfs pull
git -C "$BACKUP/verification-clone" fsck --full
git -C "$BACKUP/verification-clone" lfs fsck
```

Expected: all commands exit zero without relying on the original repository's object store.

- [ ] **Step 4: Validate recovered artifacts and refs**

Generate `artifacts-verification-clone.sha256` with the same deterministic `find | sort -z | sha256sum` command and diff it against `artifacts-before.sha256`. Compare GitHub remote branch/tag names and SHAs with the clean clone.

Expected: artifact hashes match pre-migration bytes; clean clone refs match published refs.

- [ ] **Step 5: Write final evidence report**

Record in `final-report.txt`:

```text
backup_directory=<absolute path>
old_head=<pre-migration SHA>
new_head=<rewritten SHA>
lfs_object_count=<count>
lfs_total_bytes=<sum if reported>
remote_branches=<names>
remote_tags=<names or none>
bundle_verify=PASS
git_fsck=PASS
lfs_fsck=PASS
historical_pointer_audit=PASS
artifact_hash_comparison=PASS
protected_untracked_hashes=PASS
clean_clone_recovery=PASS
```

Expected: every required result is `PASS`; do not claim completion if any result is missing or failed.

- [ ] **Step 6: Commit only if a post-migration tracked documentation adjustment is required**

No routine post-migration commit is expected: `.gitattributes` is introduced by the history rewrite. If an evidence pointer is intentionally added to the repository, stage only that explicit file and commit with:

```bash
git -c user.name='dragon_bra' -c user.email='tommy514@foxmail.com' commit -m "docs: 记录 Git LFS 迁移结果" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Expected: protected untracked files are never staged.
