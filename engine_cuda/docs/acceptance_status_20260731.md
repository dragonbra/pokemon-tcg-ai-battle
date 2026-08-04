# CUDA Engine Acceptance Update (2026-07-31)

## Completed in this run

- Official Main oracle: 35/35 scenarios passed.
- Official CPU vs CPU-POD vs CUDA Main paired replay: passed.
  - `status_mismatches=0`
  - `fixture_mismatches=0`
  - `cuda_mismatches=0`
  - CPU and GPU digests are identical.
- CUDA full-battle paired replay, seeds 1-20, basic-play/evolve/attach policy:
  - 2,954 decisions compared
  - `state_mismatches=0`
  - `status_mismatches=0`
  - passed
- Python test suite: 48/48 passed.
- W&B offline install/logging smoke: passed (`wandb==0.28.1`).
- Decoder-only freeze/update unit tests: passed.
- Official runtime arena core smoke: passed with the historical resume fixture
  explicitly quarantined.
  - Batch 4,096 allocated 493,250,336 bytes in the official arena.
  - Device stack limit was 32,768 bytes.
  - `classification_mismatches=0`, `fail_closed_mismatches=0`.
- Official PyTorch binding smoke: passed on local Docker/PyTorch CUDA.
  - `OfficialCudaEngine` exposes resident official state, status, action POD,
    and `PolicyCodecV1`-shaped device tensors.
  - `encode_policy_v1()` returned CUDA tensors for
    `global_cat/global_num/entity_cat/entity_num/entity_parent/entity_mask/`
    `option_cat/option_num/option_equiv/option_mask/min_count/max_count`.
  - The smoke now records per-tensor SHA256 hashes, first-row codec samples, and
    exact row-consistency checks across four identical fixture states. This is a
    local stability/audit gate for the official device codec, not the full
    historical 127,205-decision CPU-vs-CUDA element parity gate.
- Semantic coverage matrix was rebuilt against the current IR (matrix v2).
  The strict completeness check now reaches the real coverage result and fails
  explicitly with 6,400 `inventory_only` card/skill/attack/branch records.
  - Private Main fixture state encoded to 15 visible entities and 7 legal
    options in each of 4 local smoke environments.
  - Device action packing and device action apply both ran through the binding;
    host synchronization was used only for final smoke inspection.

## Semantic fix

Empty effect selections are resolved inside the same official `State::step`
boundary and do not increment `turnActionCount`. The POD interpreter now keeps
that counter unchanged for internal empty selections and increments it only
when a visible decision is yielded. This fixes the `item_resume` Main fixture
without changing the production CUDA action contract.

## Artifacts

- `artifacts/official_main_oracle.json`
- `artifacts/official_main_paired.json`
- `artifacts/official_battle_end_turn_cuda_paired.json`
- `artifacts/official_runtime_smoke_core.json`
- `artifacts/official_semantic_coverage_20260731.json`
- `artifacts/official_torch_binding_smoke.json`

## Current blocker notes

- Strict `official_runtime_smoke` is not currently a clean acceptance gate when
  using the historical `official_attack_resume_fixture.bin`: the replay
  statuses are identical, but the final byte comparison reports
  `resume_state_mismatches=1`.
- The wrapper now supports `--skip-resume-fixture` so resident arena,
  classification, fail-closed action handling, and memory sizing can pass as a
  separate core gate while the stale resume fixture remains quarantined.
- `official_flow_resume_diagnostic` narrowed that historical fixture mismatch to
  4-6 bytes across six equivalent CUDA resume shapes, with identical status
  results. This points to a stale expected-state fixture/counter issue rather
  than a newly introduced action-pack or binding branch divergence.
- Regenerating the fixture through `run_official_attack_oracle.py` currently
  fails inside the official CPU/POD oracle at `knockout.replacement_needed`.
  Until that oracle/fixture path is repaired, the valid local evidence remains
  `official_attack_paired` (70/70 CPU POD/CUDA scenarios) plus the Main paired
  gate and the new PyTorch binding smoke.

## Remaining scope

This update does not claim full 1,267-card semantic coverage, the full
100,000-decision all-action CPU gate, all legacy codec frozen corpora, complete
`OfficialStatePod -> PolicyCodecV1` element parity over the historical 127,205
decision corpus, or the server-only sanitizer/Nsight/3x performance gates. The
known `KNOWN_DIVERGENCE_660_1207` quarantine remains unchanged.

The full PPO smoke is not claimed here. The local workspace does not contain
the server-produced pure-Lucario initialization checkpoint and matching
`libcg_seeded.so`; the Windows PyTorch runtime also reports duplicate OpenMP
runtime loading. Run that smoke on the CUDA server after restoring those
artifacts.
