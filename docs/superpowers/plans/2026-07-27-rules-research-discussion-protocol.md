# Rules Research Discussion Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require evidence-based reading of the project's detailed Pokémon TCG rules research before deep design discussions, with escalation to the official rulebook when needed.

**Architecture:** Add one repository-level discussion protocol beside the existing long-term rules memory. It points to the existing research document as the first source, distinguishes game rules from project strategy assumptions, and defines when the original official rulebook must be consulted.

**Tech Stack:** Markdown documentation; repository instruction file.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve the detailed rules research document as the primary project-level reference.
- Official rulebook consultation is required only for questions needing authoritative wording or not covered by the research document.
- Do not alter unrelated worktree changes.

---

### Task 1: Add the rules-evidence discussion protocol

**Files:**
- Modify: `AGENTS.md` (symlinked to `CLAUDE.md`), immediately after the detailed rules-research reference

**Interfaces:**
- Consumes: `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`
- Produces: an instruction for future discussions of game design, feature schemas, model architecture, action contracts, loss, reward, value, BC, and RL.

- [ ] **Step 1: Add the protocol text**

Add a bullet that requires reading the detailed research before deep discussion of game design or model design; require separating official rules from project strategy assumptions; require consulting the linked official rulebook when exact rules wording, current-card interaction, or an uncovered issue is material.

- [ ] **Step 2: Verify the instruction is discoverable and the link remains valid**

Run: `rg -n -C 2 '讨论前规则证据流程|pokemon-tcg-par-rulebook-and-v6-rules' AGENTS.md && test -f docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md && git diff --check`

Expected: the new protocol and existing research link are present; the document exists; no whitespace errors.
