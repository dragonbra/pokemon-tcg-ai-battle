# Kaggle Confidentiality Agreement HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (required). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a formal Chinese HTML signing draft for a Kaggle competition team's confidentiality and competition-compliance agreement.

**Architecture:** A standalone, print-friendly HTML document with numbered clauses, defined terms, competition-specific controls, signature blocks, and an explicit legal-review notice. It will distinguish binding draft language from fields and rules that the team must confirm on Kaggle before signing.

**Tech Stack:** Semantic HTML5, CSS, vanilla JavaScript for print and checklist controls only.

## Global Constraints

- Do not claim to be a lawyer or present this as final legal advice.
- Do not invent Kaggle deadlines, quotas, prize terms, or rule wording; mark unverified items for confirmation.
- Protect user worktree changes and do not modify `engine/source/`.
- Keep the HTML directly openable and printable without a build step.

---

### Task 1: Draft agreement text

**Files:**
- Create: `docs/reports/2026-08-02-kaggle-team-confidentiality-agreement.html`

- [ ] Define parties, purpose, effective period, confidential information, permitted use, and exclusions.
- [ ] Add Kaggle-specific no-cross-team-sharing and no-unauthorized-disclosure clauses.
- [ ] Add security, incident reporting, return/deletion, publication, ownership, breach, withdrawal, and dispute placeholders.
- [ ] Add a competition compliance appendix with fields for the verified Rules/Team URL, deadline, quota, and confirmation owner.

### Task 2: Formal document UI

**Files:**
- Modify: `docs/reports/2026-08-02-kaggle-team-confidentiality-agreement.html`

- [ ] Add document header, version/effective-date fields, clause numbering, callout notices, signature tables, and print CSS.
- [ ] Add a small pre-signing checklist with checkboxes that remain usable on screen and print clearly.
- [ ] Add a source/assumption footer distinguishing repository evidence from Kaggle terms that require rechecking.

### Task 3: Verification

**Files:**
- Verify: `docs/reports/2026-08-02-kaggle-team-confidentiality-agreement.html`

- [ ] Validate HTML parsing, unique IDs, local links, required clauses, signature fields, and no placeholder omissions.
- [ ] Check desktop/mobile layout and print rendering using a local browser.
