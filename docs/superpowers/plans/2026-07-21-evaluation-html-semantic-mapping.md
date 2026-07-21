# Evaluation HTML Semantic Mapping Implementation Plan

> **For agentic workers:** Execute this plan inline in the current session; the user has already approved the architecture and requested direct implementation.

**Goal:** Make the Auto-Iteration evaluation HTML report map technical metric IDs to the documented three-stage strategy semantics without changing machine-readable metric IDs or adding promotion decisions.

**Architecture:** Store semantic groups and metric display metadata on `MetricProfile`, serialize it through the existing manifest/report payload, and render the Auto-Iteration profile through a semantic metric table. Resolve display values from metric payloads when the profile defines a semantic event rate, while retaining raw numerator/denominator and metric ID for auditability. Core profile reports keep their existing generic behavior.

**Tech Stack:** Python 3.11+, standard-library `dataclasses`, `unittest`, existing standalone HTML renderer.

## Global Constraints

- Only modify local `evaluation/` framework code, its focused tests, and evaluation documentation/plan artifacts.
- Do not change engine/source, strategy code, opponent packages, or add promotion/decision logic.
- Preserve raw metric IDs and existing JSON schemas; additive profile metadata is allowed.
- Use the Auto-Iteration document's definitions for stages, denominators, and metric direction.
- HTML must escape dynamic values and remain standalone with no external assets.

### Task 1: Semantic profile contract

**Files:**
- Modify: `evaluation/metrics/profiles.py`
- Test: `tests/test_evaluation_profiles.py`

- [x] Add immutable semantic group and metric display metadata with stage, title, role, direction, and tracking description.
- [x] Define Auto-Iteration groups for result/correctness guardrails, Stage 1 setup, Stage 2 post-KO relay, Stage 3 attack quality, and auxiliary health/audit metrics.
- [x] Include metadata in `MetricProfile.manifest()` while preserving revision and metric IDs.
- [x] Test exact stage ordering, semantic names, directions, and the post-KO success-rate display source.

### Task 2: Semantic HTML rendering

**Files:**
- Modify: `evaluation/reporting/html.py`
- Test: `tests/test_evaluation_reporting.py`

- [x] Render profile semantic groups in order before the generic raw metric table.
- [x] Render each metric with semantic name, `metric_id`, role, direction, numerator/denominator, semantic value, and tracking target.
- [x] Prefer payload fields such as `success_rate`, `non_prize_attacks.rate`, and reached-second-turn rates where declared; fall back to aggregate value for generic metrics.
- [x] Keep generic rendering for profiles without semantic metadata and keep embedded JSON additive and safe.
- [x] Test that HTML exposes all three stages, preserves raw IDs, and does not mislabel the old post-KO failure value as success rate.

### Task 3: Documentation and sample

**Files:**
- Modify: `evaluation/README.md`
- Regenerate: `work/auto-iteration/history_iterations/evaluation-sample/<run>/report.html`

- [x] Document the semantic report sections and the distinction between raw metric payloads and displayed semantic values.
- [x] Re-render the existing sample using the updated framework without changing its run data.

### Task 4: Verification and review

- [x] Run focused profile/report tests and inspect generated HTML section order and values.
- [x] Run all evaluation tests, asset checks, and evaluation compilation.
- [x] Review the diff for scope leaks, stale labels, incorrect denominator semantics, and accidental promotion logic.
