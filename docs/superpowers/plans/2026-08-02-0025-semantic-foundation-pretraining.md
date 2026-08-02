# 0025 Semantic Foundation Pretraining Implementation Plan

**Goal:** Establish a self-contained, correctness-first foundation BC project that preserves the 0019 actor contract while adding typed official card/action semantics, lossless dynamic references, and separately queryable model memories.

**Project ID:** `0025_semantic_foundation_pretraining`

**Scope boundary:** This iteration implements and validates the feature/model/data framework. It does not materialize the complete archive and does not start formal BC or RL training.

## Evidence Boundary

- General turn and zone semantics follow the official-rule audit in `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`.
- Card, skill, attack, energy, target, and effect facts come from the read-only official engine/runtime tables.
- Resource ledgers, direct action deltas, and memory routing are project modelling decisions and must be labelled as such.

## Implementation Tasks

1. Freeze the relevant 0019 raw-row, split, causal-ledger, and full-action contracts inside `train/0025_semantic_foundation_pretraining/`; record source paths and hashes. No runtime import from another numbered project is allowed.
2. Export a versioned static prototype sidecar from the official runtime API. Preserve card/skill/attack identity, ordered attack energy requirements, numeric values, text/effect evidence, and explicit present/unknown/not-applicable states. Never modify `engine/source/`.
3. Define a typed dynamic feature schema that retains the legacy observation and 12-field option codec, then adds attack/skill/card binding, presence masks, typed energy gaps, target HP/KO facts, zone transitions, turn-budget consumption, and attack termination.
4. Materialize compact ragged decisions: static prototypes are stored once; each decision stores dynamic values and prototype references only. Add a streaming shard writer suitable for 2026-07-10 onward without loading the corpus into RAM.
5. Build a multi-memory policy skeleton with board, prototype, deck/ledger, event/known-hand, and turn-budget memories. Options independently cross-attend to each memory, then combine contexts through zero-initialized gated residuals. Preserve ordered legal-option indices plus STOP as the output contract.
6. Add contract tests for semantic identity, zero/unknown/N/A/PAD separation, mask invariance, ledger retention, actor provenance exclusion, Crispin/Ogerpon typed-energy facts, shape validity, finite logits, and numbered-project import isolation.
7. Run a single-process, single-thread smoke on a tiny contiguous sample from the 0019 raw corpus. Do not scan or build the complete archive.
8. Benchmark legacy and 0025 encoding latency and serialized bytes per decision on the same sample. Report dynamic bytes separately from the amortized static prototype sidecar, sample/warmup sizes, and peak RSS where available.
9. Estimate complete 2026-07-10 through 2026-08-02 costs from observed 0019 decisions/day and measured 0025 costs. Clearly separate observed data through 2026-07-28, locally available archives through 2026-07-29, and the unobserved date projection.
10. Publish synchronized `experiments/0025_semantic_foundation_pretraining/DESIGN.md` and `DESIGN.html`, a lineage manifest, decisions, and V1 smoke/benchmark status under `rl_runs/.../versions/V1_semantic_contract_smoke/artifact/`.

## Verification Commands

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover \
  -s train/0025_semantic_foundation_pretraining/tests -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 \
  -m train.0025_semantic_foundation_pretraining.tools.smoke_and_benchmark \
  --sample-decisions 32 --warmup 4
```

## Completion Criteria

- The public prototype sidecar has deterministic hashes and fail-closed referential checks.
- A real raw 0019 decision encodes and runs through the 0025 model with finite logits and correct option/STOP shapes.
- Semantic regression tests demonstrate that distinct attacks and field states remain distinguishable.
- Benchmark artifacts provide measured old/new bytes and latency plus an explicitly labelled full-period projection.
- No full data build, formal training, W&B run, engine-source edit, or cross-numbered runtime dependency occurs.
