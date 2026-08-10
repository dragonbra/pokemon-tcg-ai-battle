# Semantic0031 Signed Negative HP Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CUDA semantic0031 card observation preserve canonical CPU signed negative HP and verify exact trajectory parity on the requested seeds.

**Architecture:** Keep the CUDA engine transition and semantic history unchanged. Change only the derived `hp` expression in the semantic0031 card encoder, protect its signed arithmetic with a focused source contract test, rebuild the existing staged PyTorch extension, and validate through the existing Full-0806 lockstep harness.

**Tech Stack:** C++20/CUDA, PyTorch CUDA extension, Python 3.11 `unittest`, official CPU seeded runtime, CUDA resident official runtime.

## Global Constraints

- Do not modify `engine/source/` or canonical CPU semantics.
- Do not modify game rules, policy checkpoints, opponent model, router, training, or sampling.
- Run only seed `1142670790` first; only after it fully matches, run the original four seeds.
- Do not run large-scale RL, retraining, or large benchmarks.
- Audit other semantic sanitization sites read-only and do not repair unproven issues.

---

### Task 1: Signed HP Regression Contract

**Files:**
- Create: `engine_cuda/tests/test_semantic0031_signed_hp.py`
- Modify: `engine_cuda/src/official_engine_kernels.cu:1129-1135`

**Interfaces:**
- Consumes: semantic0031 card entity encoder locals `max_hp`, `damage`, and `hp`.
- Produces: signed `float card_num[..., 0]` equal to canonical `max_hp - damage`.

- [ ] **Step 1: Add a failing source contract test**

```python
def test_card_hp_preserves_signed_canonical_difference(self):
    source = KERNEL.read_text(encoding="utf-8")
    self.assertIn("const std::int32_t hp = max_hp - damage;", source)
    self.assertNotIn("const std::int32_t hp = max_hp > damage ? max_hp - damage : 0;", source)
```

- [ ] **Step 2: Run the focused test and verify the current clamp fails**

Run: `python3 -m unittest -v engine_cuda.tests.test_semantic0031_signed_hp`

Expected: FAIL because the signed subtraction expression is absent.

- [ ] **Step 3: Replace only the HP clamp expression**

```cpp
const std::int32_t hp = max_hp - damage;
```

- [ ] **Step 4: Run the focused contract and existing semantic scaffold tests**

Run: `python3 -m unittest -v engine_cuda.tests.test_semantic0031_signed_hp engine_cuda.tests.test_semantic0031_parity_scaffold`

Expected: PASS, including signed type and float output checks.

### Task 2: Rebuild and Single-Seed Full Lockstep

**Files:**
- Rebuild: `.tmp/engine_cuda_benchmark/build_sm120_staged/_ptcg_cuda.so`
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/postfix_seed_1142670790.json`

**Interfaces:**
- Consumes: rebuilt `_ptcg_cuda.so`, U270 focal checkpoint, Full-0806 epoch 11 opponent.
- Produces: decision-155 observation equality and terminal exact trajectory verdict.

- [ ] **Step 1: Rebuild the existing staged extension**

Run: `cmake --build .tmp/engine_cuda_benchmark/build_sm120_staged --target _ptcg_cuda -j2`

Expected: `_ptcg_cuda.so` relinks successfully from the one-line kernel change.

- [ ] **Step 2: Run only seed 1142670790 through terminal lockstep**

Run the isolated Full-0806 parity harness with `--seeds 1142670790 --focal-first false` and a new postfix output.

Expected: decision 155 CPU/CUDA `card_num[42][0] == -10`, no divergence, terminal `PASS`.

### Task 3: Original Four-Seed Regression

**Files:**
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/postfix_four_seeds.json`

**Interfaces:**
- Consumes: the same rebuilt extension and immutable policies used in Task 2.
- Produces: four exact full-trajectory verdicts in the requested seed order.

- [ ] **Step 1: Run the original four seeds**

Run the Full-0806 lockstep harness with seeds `2078498379,1145059092,1142670790,899381985`, `--focal-first false`, and continuation enabled only to collect all four outcomes.

Expected: all four games reach terminal with no divergence; `4/4 exact trajectory match`.

### Task 4: Read-Only Semantic Sanitization Audit

**Files:**
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/observation_encoder_static_audit.md`

**Interfaces:**
- Consumes: CUDA semantic0031 observation encoder and canonical CPU feature/compiler behavior.
- Produces: one row per clamp/max/min/saturate/default-zero/conditional-zero/unsigned conversion site, classified `MATCH` or `POTENTIAL MISMATCH` without code changes.

- [ ] **Step 1: Search the encoder for sanitization patterns**

Run: `rg -n "clamp|max|min|saturat|\\?[^:]*:[^;]*0|uint|unsigned|< 0|> 0" engine_cuda/src/official_engine_kernels.cu`

Expected: a bounded list of sites inside the semantic0031 encoder functions.

- [ ] **Step 2: Compare each site to canonical CPU construction**

Inspect `train/0040_dragapult_0809_action_boundary_rl/semantic_policy/features/compiler.py`, its field helpers, and official JSON behavior for the corresponding field.

Expected: audit entries state CUDA behavior, CPU behavior, and `MATCH` or `POTENTIAL MISMATCH`; no additional implementation edits.
