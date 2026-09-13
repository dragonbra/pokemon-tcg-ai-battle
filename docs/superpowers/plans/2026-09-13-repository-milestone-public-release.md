# Repository Milestone And Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the complete current repository state with its five final submission archives stored through Git LFS, then create a clean branch that presents the final BC + RL system as a coherent open-source project.

**Architecture:** Preserve the historical repository exactly through one immutable annotated tag before any cleanup. On the public branch, treat `0047_meta_routed_moe_rl` as the final integration point, retain only its runtime dependencies and auditable provenance, and replace numbered-project navigation with a public-facing model/training/evaluation narrative.

**Tech Stack:** Git, Git LFS, Python 3.11, PyTorch, pytest, Markdown, official Pokemon TCG engine runtime.

## Global Constraints

- Never modify `engine/source/`.
- The archive tag must point at a clean commit containing all current tracked and untracked work intended for the milestone.
- Every final `.tar.gz` must be represented by a Git LFS pointer and pass gzip/tar integrity checks.
- The public branch must preserve complete effective policy identity, independent focal/opponent weights, and `kaggle_fp16_storage_fp32_runtime_v1` deployment semantics.
- Historical numbered projects may disappear from the public branch only after the archive tag exists; their history remains recoverable from that tag.
- No remote push, release publication, or GitHub mutation is part of this plan.

---

### Task 1: Freeze The Current Repository

**Files:**
- Modify: `.gitattributes`
- Move: `0045-003-FINAL_DANCE_FIN-MEGA_LOPUNNY_EX.tar.gz` to `archive/submission/dist/0045-003-FINAL_DANCE_FIN-MEGA_LOPUNNY_EX.tar.gz`
- Move: `0045-003-PINHAOTU_FIN-MEGA_LOPUNNY_EX.tar.gz` to `archive/submission/dist/0045-003-PINHAOTU_FIN-MEGA_LOPUNNY_EX.tar.gz`
- Move: `0045-007-PINHAOLONG_V2-DRAGAPULT_EX.tar.gz` to `archive/submission/dist/0045-007-PINHAOLONG_V2-DRAGAPULT_EX.tar.gz`
- Move: `0045-007-PINHAOLONG_V3-DRAGAPULT_EX.tar.gz` to `archive/submission/dist/0045-007-PINHAOLONG_V3-DRAGAPULT_EX.tar.gz`
- Move: `0045-007-PINHAOLONG_V4-DRAGAPULT_EX.tar.gz` to `archive/submission/dist/0045-007-PINHAOLONG_V4-DRAGAPULT_EX.tar.gz`
- Create: `archive/submission/dist/2026-09-13-final-packages.sha256`

**Interfaces:**
- Consumes: the five visible untracked archives and all existing working-tree changes.
- Produces: one clean milestone commit and one annotated archive tag.

- [ ] **Step 1: Validate archive integrity and inventory**

Run `gzip -t` and `tar -tzf` for every archive, then compute SHA-256 checksums. Reject an archive that is corrupt or empty.

- [ ] **Step 2: Move archives under the existing LFS contract**

Move the five archives into `archive/submission/dist/`, where `.gitattributes` already declares `filter=lfs diff=lfs merge=lfs -text`.

- [ ] **Step 3: Record checksums**

Write deterministic `sha256sum` output for the five archived package paths to `archive/submission/dist/2026-09-13-final-packages.sha256`.

- [ ] **Step 4: Run pre-commit validation**

Run `git diff --check`, the tests touched by the outstanding 0045 and environment-report changes, and `git lfs status`. Confirm that staged archive blobs begin with `version https://git-lfs.github.com/spec/v1`.

- [ ] **Step 5: Commit the complete milestone state**

Stage all current repository changes, inspect the staged name/status and diff summary, then commit with `chore: freeze final competition repository`.

- [ ] **Step 6: Create the immutable archive tag**

Create annotated tag `archive/final-competition-repo-2026-09-13` on the milestone commit and verify the tag target, clean worktree, archive pointers, and five checksums.

### Task 2: Establish The Public Branch And Dependency Boundary

**Files:**
- Create: `docs/public-release/inventory.md`
- Create: `docs/public-release/provenance.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: tagged milestone commit and `train/0047_meta_routed_moe_rl` dependency graph.
- Produces: branch `public/bc-rl-writeup` plus an explicit keep/remove inventory and provenance map.

- [ ] **Step 1: Create the public branch from the archive tag**

Run `git switch -c public/bc-rl-writeup archive/final-competition-repo-2026-09-13`.

- [ ] **Step 2: Audit executable dependencies**

Enumerate imports, file reads, checkpoints, manifests, decks, engine/runtime calls, and evaluation assets reachable from the final model. Record every dependency that is retained or replaced.

- [ ] **Step 3: Define the public tree**

Document a compact tree centered on the final BC foundation, PPO/MoE policy, official-engine integration, evaluation, reproducibility manifests, and write-up evidence.

- [ ] **Step 4: Commit the boundary document**

Commit the inventory before deleting or renaming history so reviewers can audit the cleanup decisions independently.

### Task 3: Curate The Final BC + RL Implementation

**Files:**
- Move: `train/0047_meta_routed_moe_rl/` to `src/pokemon_tcg_ai/`
- Move: `experiments/0047_meta_routed_moe_rl/` to `docs/model/`
- Modify: Python imports, manifests, CLIs, tests, and documentation that refer to the numbered path.
- Delete on public branch: superseded numbered `train/`, `experiments/`, and historical-only artifacts not required by the final runtime or evidence bundle.

**Interfaces:**
- Consumes: the dependency inventory from Task 2.
- Produces: a self-contained importable package `pokemon_tcg_ai` with no executable dependency on another numbered training project.

- [ ] **Step 1: Add a clean import-contract test**

Test that importing `pokemon_tcg_ai`, its policy, deployment materializer, and evaluator does not import any `train.00xx_*` module.

- [ ] **Step 2: Move the final implementation and update imports**

Rename the integration project into `src/pokemon_tcg_ai/`, replace absolute numbered imports, and preserve all immutable hash/provenance declarations.

- [ ] **Step 3: Keep the final model assets and required shared infrastructure**

Retain the exact decks, policy definitions, schemas, official read-only data, legal-action/runtime glue, and evaluation contracts required to reproduce the final model.

- [ ] **Step 4: Remove historical numbered projects from the public branch**

Delete superseded numbered experiments, run-specific diagnostics, and redundant archive packages only after confirming each removed path remains reachable from the archive tag.

- [ ] **Step 5: Verify policy and deployment identity**

Run policy-identity, strict-load, FP16-storage/FP32-runtime, focal/opponent independence, package export, and import-boundary tests.

- [ ] **Step 6: Commit the curated implementation**

Commit the independently testable final package and its retained evidence.

### Task 4: Build The Open-Source Entry Points

**Files:**
- Rewrite: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/reproducing.md`
- Create: `docs/evaluation.md`
- Create: `docs/data-and-model-cards.md`
- Create: `LICENSE` or `LICENSES/` only after confirming the repository's distributable licenses.
- Modify: packaging/test configuration as required by the curated tree.

**Interfaces:**
- Consumes: the curated `pokemon_tcg_ai` package and evidence inventory.
- Produces: install, test, evaluation, architecture, limitations, and attribution entry points suitable for a public GitHub link.

- [ ] **Step 1: Write the public README**

Explain the problem, final BC + RL method, repository map, quick-start commands, headline evidence, limitations, and links to the longer write-up materials.

- [ ] **Step 2: Document the architecture and policy contract**

Describe semantic inputs, autoregressive actions, BC initialization, PPO/MoE routing, focal/opponent identity, and deployment materialization without exposing historical experiment-number clutter.

- [ ] **Step 3: Document reproducibility and evidence boundaries**

Separate official-engine match evidence from offline imitation metrics and training diagnostics; provide exact commands and expected artifacts.

- [ ] **Step 4: Audit secrets, licenses, and oversized files**

Scan tracked files for credentials and personal paths, inventory third-party code/data/model licenses, and list remaining Git/LFS object sizes. Fail closed on assets whose redistribution rights are unclear.

- [ ] **Step 5: Run the public-repository smoke suite**

Test from a clean checkout/worktree with LFS materialized, run package import and focused unit tests, validate final candidate packages, and confirm `engine/source/` matches the archive tag.

- [ ] **Step 6: Commit the public entry points**

Commit documentation and packaging only after every link and command resolves in the curated tree.

### Task 5: Draft The GitHub Write-Up Package

**Files:**
- Create: `docs/writeup/outline.md`
- Create: `docs/writeup/evidence-map.md`
- Create: `docs/writeup/figures/README.md`

**Interfaces:**
- Consumes: final architecture, evaluation reports, training metrics, and repository paths.
- Produces: a claim-to-evidence write-up outline whose links point only to public-branch paths.

- [ ] **Step 1: Define the write-up thesis and section order**

Use the arc: environment and action complexity, audited behavior cloning, official-engine rollout, PPO specialist learning, public-information routing/MoE, deployment identity, results, failures, and lessons.

- [ ] **Step 2: Build a claim-to-evidence matrix**

For every numeric or architectural claim, record the source report/manifest, evaluation contract, sample count, and whether it is offline, diagnostic, or official-engine evidence.

- [ ] **Step 3: Select reproducible figures and tables**

Specify the exact source data and generation command for architecture, learning curves, matchup results, router behavior, and ablation/failure figures.

- [ ] **Step 4: Run link and evidence checks**

Ensure the outline contains no references to removed numbered paths and no strength claim unsupported by a retained public artifact.

- [ ] **Step 5: Commit the write-up package**

Commit the outline and evidence map as the handoff for drafting the final GitHub write-up.
