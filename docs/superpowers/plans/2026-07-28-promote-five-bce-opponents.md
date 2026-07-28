# Promote Five BC Opponents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Evaluate each requested candidate against the current formal arena pool, then promote the five validated packages as anonymized BC opponents.

**Architecture:** The existing catalog remains the evaluation source for all five isolated candidate reports. After all reports complete, packages move unchanged into `evaluation/arena/opponents/`; the catalog gains one BC-tagged, key-Pokemon-derived entry per package. Combat Matrix assets remain untouched.

**Tech Stack:** Python 3.11, standard-library `unittest`, official engine runtime, `python3 -m evaluation`.

## Global Constraints

- Do not modify `engine/source/`.
- Use the current enabled Opponents catalog and at least 10 games per opponent.
- Write temporary reports below `.tmp/evaluation/`.
- Do not run or update `evaluation/arena/combat_mat/`.
- Promoted packages remain self-contained standard packages and catalog entries only reference `arena/opponents/`.
- Use ASCII snake_case opponent names, with actual key Pokemon in deck order and a BC marker.

---

### Task 1: Audit packages and derive identities

**Files:**
- Read: `evaluation/arena/candidates/{five requested candidates}/`
- Read: `evaluation/configs/opponents.json`

- [x] **Step 1: Validate every candidate package**

Run:

```bash
python3 -m evaluation validate evaluation/arena/candidates/<candidate>
```

Expected: each prints a 60-card-valid package and compatible cg tree hash.

- [x] **Step 2: Derive deck-order key Pokemon and available two-digit sequence**

Run:

```bash
python3 -m evaluation list-opponents
```

Expected: identities are based on actual deck cards and do not collide with catalog entries.

### Task 2: Produce independent pre-promotion reports

**Files:**
- Create: `.tmp/evaluation/promote_bc_<candidate>/run-*/report.html`

**Interfaces:**
- Consumes: each validated candidate package and the unmodified enabled catalog.
- Produces: one complete official-engine report per candidate.

- [x] **Step 1: Run full-pool report for each candidate**

```bash
python3 -m evaluation run --candidate evaluation/arena/candidates/<candidate> --opponents all --games 10 --workers 8 --worker-cpu-threads 1 --output .tmp/evaluation/promote_bc_<candidate>
```

Expected: 20 opponents x 10 games, no worker errors, with report.html.

- [x] **Step 2: Inspect report outcomes and manifest**

Run:

```bash
rg -n '"errors"|"completed"|"total_games"' .tmp/evaluation/promote_bc_<candidate>/run-*/report.html
```

Expected: complete coverage and an auditable embedded result payload.

### Task 3: Promote packages and update catalog

**Files:**
- Move: `evaluation/arena/candidates/<candidate>/` to `evaluation/arena/opponents/<anonymized_key_pokemon>_<NN>_bc/`
- Modify: `evaluation/configs/opponents.json`

**Interfaces:**
- Consumes: package directories, deck-derived representative Pokemon IDs, validation reports.
- Produces: five enabled catalog entries with `[BC]` display labels and a `bc_agent` tag.

- [x] **Step 1: Move each complete package without changing its runtime contents**

```bash
mv evaluation/arena/candidates/<candidate> evaluation/arena/opponents/<derived_name>
```

Expected: old candidate path is absent and new package root remains self-contained.

- [x] **Step 2: Add catalog entries in stable key-Pokemon order**

```json
{
  "name": "<derived_name>",
  "package": "arena/opponents/<derived_name>",
  "display_name": "[BC] <Key Pokemon> NN",
  "representative_card_ids": [<deck Pokemon ID>],
  "enabled": true,
  "tags": ["<archetype>", "bc_agent"]
}
```

Expected: every entry references one or two actual Pokemon in its own deck.

### Task 4: Verify the promoted pool

**Files:**
- Test: `tests/test_evaluation_assets.py`

- [x] **Step 1: Validate all five promoted packages**

```bash
python3 -m evaluation validate evaluation/arena/opponents/<derived_name>
```

Expected: each prints `60-card valid: true`.

- [x] **Step 2: Validate formal arena assets**

```bash
python3 -m unittest -v tests.test_evaluation_assets
```

Expected: PASS.

- [x] **Step 3: Confirm no Combat Matrix asset changed**

```bash
git diff -- evaluation/arena/combat_mat
```

Expected: no output.
