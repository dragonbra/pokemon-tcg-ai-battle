# 0043 Champion League RL Design

Status: **CPU implementation and synthetic CUDA forward smoke PASS; `BLOCKED_FORMAL_TRAINING_PREFLIGHT`**

## Purpose and evidence boundary

0043 changes opponent evolution, asset routing, curriculum, telemetry, and Promotion governance. It does not change the approved model architecture, PPO loss, optimizer budget, rollout size, reward, action contract, or official-engine semantics.

The governing documents are `docs/rl/0043_Project_Charter_Codex_Handoff.md`, `docs/rl/0043_Promote_Champion_V2_Protocol.md`, and the canonical `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`. General Pokémon rules, current card/runtime facts, and project strategy assumptions remain distinct evidence layers. The rules relevant to this unchanged action contract include attack as a turn-ending commitment, evolution timing, once-per-turn Supporter/hand Energy/Retreat/Stadium budgets, and the official victory/deck-out semantics.

## Asset data flow

```text
one-time approved historical source
  -> SHA-256 verification
  -> 0043 semantic directory (`001`, `policy_0809`, `champion_g001`)
  -> immutable project-relative registry
  -> fail-closed loader
  -> policy identity resolution
  -> only then lane routing and batching
```

There are three independent domains:

- Training Deck Pool: initially the 55 exact meta decks; append-only through new versioned assets.
- Active Policy Pool: Policy-0809 historical anchor and frozen Champion-G1.
- Frozen Evaluation Pool: FrozenMeta256-V1, the unchanged 55-deck/256-game integer-frequency composition using independently resolved Policy-0809.

Within 0043, the sole canonical deck identity is the zero-padded numeric `deck_id`: the initial immutable set is `001`–`055`, and future decks append from `056`. Archetype names, exact-deck hashes, and historical semantic IDs remain audit metadata and never replace this identity. Filesystem directories likewise carry stable semantic names rather than hashes: decks live under `definitions/<deck_id>`, while policies live under `definitions/policy_0809`, `definitions/champion_g001`, and future equivalent generation names. SHA-256 remains mandatory in manifests for integrity but has no directory-name semantics.

## Model and training contract

The inherited actor uses the 0031 rule-faithful semantic observation/action schemas and the 0042 strategy-conditioned readout. The trainable boundary remains:

- action decoder;
- policy strategy adapter;
- value query/blocks/final norm/value head;
- value adapter;
- allocation head;
- prize auxiliary when FULL_MODEL enables it.

Prototype, state, and option representation weights remain frozen. PPO remains FP32 with one 256-game on-policy rollout per update, `gamma=1`, turn-clock GAE `0.95`, terminal outcome/value objective, directional prize auxiliary, three complete shuffled data passes at most, and the frozen learning-rate/loss/KL values in `active_training_config.json`.

The inference source required to reconstruct both policy families is frozen below `train/0043_champion_league_rl/semantic_runtime/` with a per-file manifest and tree hash. Runtime code imports only this project-local package. Policy-0809 strict-loads its complete checkpoint; Champion-G1 strict-loads its portable FP16 artifact into FP32 runtime. The fixed synthetic contract batch has no source/team/persona tensor and exercises actor logits or Champion value/strategy decode on CPU and CUDA without changing model semantics.

## League schedule

```text
outer focal-deck phase
  -> 128 PFSP lanes: independent deck and policy weakness draws
  ->  64 uniform lanes: independent Training Deck / Active Policy draws
  ->  64 latest lanes: uniform Training Deck + immutable Champion-G1
  -> seeded shuffle preserving lane seat/seed identities
  -> 256 official-engine games
```

PFSP uses Beta(1,1) smoothing and produces a new immutable curriculum manifest every ten updates. Its probability floor `0.001`, cap `1.0`, refresh interval and exact `128/64/64` quotas live in `league/config.json`; changing them is a new versioned experiment variable. Deck/policy joint statistics are diagnostic only; V2.0 has no compatibility whitelist.

## Telemetry and strength claims

Every completed rollout will aggregate raw WR, curriculum identity, deck/policy distributions, terminal prize margin, loss-only focal prizes, win-only opponent prizes, win/loss turns, PFSP entropy/difficulty, p10 tail WR, and red counts from the same 256 games. Existing PPO and runtime health metrics remain. Sampled rollout metrics are training diagnostics, not frozen greedy checkpoint strength.

Formal Kaggle-strength evidence materializes a complete candidate as FP16 storage and strict-loads that artifact into FP32 runtime. Candidate and opponent identities are independently audited. CPU-256 and CUDA-2048 remain separate reports with all games terminal and zero error, unfinished, or semantic fallback.

Frozen schedule materialization is identity-bound to focal deployment hash, complete opponent hash, exact numeric deck slot, replica and namespace. CPU-256 is replica 0; CUDA-2048 is eight immutable replicas, and its first 256 jobs equal the CPU frequency unit. Seeded toss identifies the winning Agent, which must process official context 41 and choose first or second; the harness never assigns the seat directly.

## CUDA Engine 2.0 backend

`engine_cuda_2_0/` is the required shared GPU engine implementation for 0043. The project-local `cuda_engine_2/` adapter pins the actual tracked source bytes, official state/rule ABI, private rule-pack hash, SM120 capability, native binary, extension, and complete policy/deck identities before creating lanes. The current RTX 5080 build produced the production static library, native runtime smoke binary, and PyTorch extension; official rule upload plus a two-lane device reset/classify smoke passed with ABI 7 and rule ABI 1. A real `001` versus `048` / reverse-seat smoke then materialized the full `semantic0031_v2` tensors on-device and routed every decision through independent complete Policy-0809 and Champion-G1 FP32 models. Both games reached terminal after 163 total decisions with engine turns 16/12, zero engine error, unfinished game, or CPU fallback. The existing upstream old-55 acceptance evidence covers 1,100 battles and 215,937 compared decisions with zero state/status/outcome mismatch, but 0043 still requires its own complete observation/action first-divergence gate before formal PPO.

CUDA 12.8 compilation of the optional `official_continuation_dispatch_smoke` translation unit reached roughly 30 GiB RSS by itself. The self-contained 0043 build entrypoint therefore uses `--parallel 1` and builds only the production dependencies `ptcg_cuda_smoke` and `_ptcg_cuda`; it never invokes the CMake `all` target. This is a build-resource constraint, not a runtime or policy-semantic change.

## Current and next stage

- Complete: project-local assets/runtime, complete policy materialization and isolation, league sampler, PFSP persistence, one-pass telemetry, frozen schedule materialization, candidate evaluation gates, and explicit human-decision Promote workflow.
- Complete diagnostic: real CPU forward plus small RTX 5080 FP32 forward parity for Policy-0809 and Champion-G1; greedy actions matched, with maximum absolute errors below `6e-5`. CUDA Engine 2.0 native runtime and PyTorch official-arena reset/classify smokes also pass. These are implementation evidence only, not policy strength evidence.
- Remaining before formal training: complete official-game CPU/CUDA observation/action first-divergence audit and allocation of a fresh formal `V<n>_<tag>` with W&B online preflight.

No formal 0043 training version exists yet, and `formal_training_authorized` remains false. Research smoke output must remain under `.tmp/`; formal versions will use `rl_runs/0043_champion_league_rl/versions/V<n>_<tag>/` and W&B private project `dragon_bra/pokemon-tcg-policy-learning`.
