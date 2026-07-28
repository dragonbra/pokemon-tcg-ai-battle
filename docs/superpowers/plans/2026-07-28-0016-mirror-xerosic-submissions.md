# 0016 Mirror Xerosic Submissions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit one new R2 package and one new R15 package with the user-authorized 4-4-4 to 4-4-3 plus Xerosic/Hammer deck change.

**Architecture:** Copy each archived, self-contained submission package into a distinct immutable payload. Change only its 60-line `deck.csv`, validate the archive layout and card counts, then submit R2 once followed by R15 once.

**Tech Stack:** Python 3.11, `tar`, repository package validation, Kaggle CLI.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve the original R2/R15 package and archive unchanged.
- The requested deck change is `Alakazam 4->3`, `Xerosic's Machinations 1->2`, `Wondrous Patch 1->0`, and `Enhanced Hammer 2->3`.
- Make exactly one Kaggle submission for R2, then exactly one for R15; do not retry either.
- Retain a receipt with archive hash, deck hash, command result, and Kaggle submission reference.

---

### Task 1: Prepare and Validate R2 Variant

**Files:**
- Create: `archive/submission/0016_alakazam_multideck_bc_v1_r2_epoch12_mirror_xerosic/`
- Create: `archive/submission/dist/0016_alakazam_multideck_bc_v1_r2_epoch12_mirror_xerosic_root.tar.gz`

- [ ] Copy the complete archived R2 package to the new payload path.
- [ ] Apply only the four requested count changes to `deck.csv`.
- [ ] Verify 60 cards, the expected counts, and package self-containment.
- [ ] Package the payload and record SHA-256 before submitting it once.

### Task 2: Submit and Record R2

**Files:**
- Create: `experiments/0016_alakazam_multideck_bc/submission_receipt_v1_r2_epoch12_mirror_xerosic.json`

- [ ] Check the remaining team submission quota.
- [ ] Submit the R2 archive once with the user-authorized message.
- [ ] Record the returned submission reference and status without retrying.

### Task 3: Prepare, Validate, and Submit R15

**Files:**
- Create: `archive/submission/0016_alakazam_multideck_bc_v2_r15_epoch10_mirror_xerosic/`
- Create: `archive/submission/dist/0016_alakazam_multideck_bc_v2_r15_epoch10_mirror_xerosic_root.tar.gz`
- Create: `experiments/0016_alakazam_multideck_bc/submission_receipt_v2_r15_epoch10_mirror_xerosic.json`

- [ ] Repeat Task 1 using the archived R15 package.
- [ ] Re-check quota after the R2 submission.
- [ ] Submit the R15 archive once with the user-authorized message.
- [ ] Record the returned submission reference and status without retrying.
