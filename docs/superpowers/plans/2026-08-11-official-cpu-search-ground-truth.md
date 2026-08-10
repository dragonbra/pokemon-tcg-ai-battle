# Official CPU Search Ground Truth Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use systematic-debugging to trace and verify each boundary before drawing conclusions. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the official CPU engine search API's exact implementation, inputs, outputs, mutation semantics, and basic execution boundary using source evidence and reproducible runtime experiments.

**Architecture:** Trace the exported API down through `Api.h` into the official `Search` implementation and the normal CPU transition machinery. Build a standalone diagnostic outside `engine/source/` that drives the official runtime, captures canonical state snapshots before and after search, and records deterministic action traces without changing production inference or CUDA code.

**Tech Stack:** C++ official engine headers/export, Python 3.11 ctypes/runtime harness where already established, JSON state snapshots, pytest or direct CLI assertions.

## Global Constraints

- Do not modify `engine/source/`.
- Do not inspect or use CUDA/GPU engine behavior as search ground truth.
- Do not modify production inference or implement Value Search.
- Do not classify actions into search allow/deny lists.
- Ground every conclusion in official CPU source plus a runnable experiment.

---

### Task 1: Trace the canonical CPU search implementation

**Files:**
- Read: `engine/source/ptcgProgram 22/Search.h`
- Read: `engine/source/ptcgProgram 22/Api.h`
- Read: `engine/source/ptcgProgram 22/Export.cpp`
- Read: official state-transition headers reached by the call chain

**Interfaces:**
- Consumes: exported official CPU API calls and internal engine types.
- Produces: exact signatures, type relationships, and source line evidence.

- [x] Read each implementation in full and record the export-to-transition call chain.
- [x] Distinguish search simulation from normal step/action submission and unrelated card-text searches.
- [x] Identify ownership, cloning, RNG, state, and serialization behavior from code.

### Task 2: Build a minimal reproducible official runtime probe

**Files:**
- Create: `tests/official_cpu_search_probe.py`

**Interfaces:**
- Consumes: the repository's existing official CPU engine build/runtime loader and real legal actions.
- Produces: stable JSON traces and assertions for original-state immutability, object identity, RNG-visible behavior, actors, legal actions, hand, and board.

- [x] Reuse the official CPU C ABI without relying on wrapper semantics as evidence.
- [x] Construct or advance real games deterministically until simple legal actions are available.
- [x] Snapshot the original game before and after each search and compare canonical dumps/hashes.
- [x] Capture the returned search state and trace one-action execution boundaries for three simple actions.
- [x] Make failures explicit when a required action cannot be obtained reproducibly.

### Task 3: Execute, cross-check, and report

**Files:**
- Create: `docs/reports/OFFICIAL_CPU_SEARCH_GROUND_TRUTH.md`

**Interfaces:**
- Consumes: Task 1 source evidence and Task 2 runtime output.
- Produces: the requested seven-section ground-truth report with runnable command and observed results.

- [x] Build the official CPU runtime from the unmodified official source.
- [x] Run the probe and retain the observed output needed for audit.
- [x] Cross-check every report statement against source lines and runtime evidence.
- [x] Run the probe's assertions and a final diff/status check confirming no protected or unrelated files were changed.
