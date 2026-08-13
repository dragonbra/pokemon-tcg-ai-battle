# 0043 Champion League RL Design

Status: **V2 stopped by user after source update 206 / checkpoint U207; selected Policy-0809 CUDA-2048 evidence complete**

## Own Archetype Taxonomy V2 (2026-08-13)

Own strategy and opponent Meta are separate schemas. Opponent Meta remains the frozen 15-way pretrained head and still contributes 15 probabilities to the Strategy Adapter. Own strategy now uses the append-only `own_archetypes_v2` registry: 29 rows × 16 dimensions in both `ValueResidualAdapter.own_embedding` and `PolicyStrategyAdapter.own_embedding`. The row-count increase changes only each adapter embedding table; it does not alter either downstream MLP input/output width, and source/team identity remains absent from actor inputs.

Known decks resolve by `(deck_id, exact content SHA-256)` through `deck_own_archetype_mapping_v2.json`. Trigger-card priority is compatibility fallback only and is never source of truth for 001–067. IDs 0–14 retain their exact V1 meanings; IDs 15–28 are append-only strategic splits. `other` (14) remains the unknown/unmodeled fallback, and no known deck in 001–067 maps to it.

Champion-G1 remains byte-for-byte frozen with V1 tensors `[15,16]`. V1-Focal-Seed is a mutable, unpromoted run-local initialization with V2 tensors `[29,16]`: rows 0–14 are exact copies, and each appended row is copied from its declared G1 parent. The FP32 model-only training seed and portable FP16 artifact were migrated independently; no optimizer state exists or was migrated. A fresh optimizer includes both expanded own embeddings under the unchanged approved parameter groups and learning rates.

Zero-update parity exercised all 15 historical rows through complete portable inference and obtained exact policy logits, action probabilities, greedy actions, value outputs, and both adapter outputs. It separately verified all 55 historical exact decks resolve to their G1 row and all 67 decks resolve through V2 exact mapping. This is schema-migration correctness evidence, not official-engine policy-strength evidence.

The canonical human-readable catalog is published inside the asset tree at `train/0043_champion_league_rl/assets/decks/index.html`, backed by image-rich `definitions/<deck_id>/index.html` pages and deterministic text records `text/001.txt` through `text/067.txt`. Each record contains the complete counted 60-card list, official card names/types, exact hashes, V2 mapping, strategic characteristics, and source provenance. Deck 062's 0043 display name is content-corrected to `Mega Starmie ex / Mega Froslass ex`; its upstream Dusknoir label remains provenance only. User-approved exact lists 066/067 append Dragapult/Dusknoir control and ordinary Dragapult/Munkidori control without changing the 29-row vocabulary.

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

- Training Deck Pool: 67 exact decks, numbered `001`–`067`; append-only through new versioned assets.
- Active Policy Pool: Policy-0809 historical anchor and frozen Champion-G1.
- Frozen Evaluation Pool: FrozenMeta256-V1, the unchanged 55-deck/256-game integer-frequency composition using independently resolved Policy-0809.

Within 0043, the sole canonical deck identity is the zero-padded numeric `deck_id`: the initial immutable set was `001`–`055`, the frozen-pool extension is `056`–`065`, and user-approved Dragapult exact lists are appended as `066`–`067`. Archetype names, exact-deck hashes, and historical semantic IDs remain audit metadata and never replace this identity. Decks 056–067 are training-only; FrozenMeta256-V1 still resolves exactly `001`–`055`, so this is not an implicit Frozen67 benchmark migration. Filesystem directories likewise carry stable semantic names rather than hashes: decks live under `definitions/<deck_id>`, while policies live under `definitions/policy_0809`, `definitions/champion_g001`, and future equivalent generation names. SHA-256 remains mandatory in manifests for integrity but has no directory-name semantics.

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
outer opponent-policy phase (focal deck remains a per-lane feature)
  -> 128 PFSP lanes: independent deck and policy weakness draws
  ->  64 uniform lanes: independent Training Deck / Active Policy draws
  ->  64 latest lanes: uniform Training Deck + immutable Champion-G1
  -> seeded shuffle preserving lane seat/seed identities
  -> 256 official-engine games
```

PFSP uses Beta(1,1) smoothing and produces a new immutable curriculum manifest every ten updates. Its probability floor `0.001`, cap `1.0`, refresh interval and exact `128/64/64` quotas live in `league/config.json`; changing them is a new versioned experiment variable. Deck/policy joint statistics are diagnostic only; V2.0 has no compatibility whitelist.

## Telemetry and strength claims

Every completed rollout will aggregate raw WR, curriculum identity, deck/policy distributions, terminal prize margin, loss-only focal prizes, win-only opponent prizes, win/loss turns, PFSP entropy/difficulty, p10 tail WR, and red counts from the same 256 games. Existing PPO and runtime health metrics remain. Sampled rollout metrics are training diagnostics, not frozen greedy checkpoint strength. Formal W&B mirrors rolling 100/500/2000 windows; per-focal-deck, per-opponent-policy, per-opponent-deck and per-branch win/prize strata; source-policy/checkpoint chronology; CUDA games/s and strategic decisions/s; resident-load/cache counts; feature D2H bytes; routing failures; PPO behavior/reference KL, clipping, losses and optimizer health. `training_metrics.jsonl` remains canonical and is flushed before TensorBoard and W&B.

Formal Kaggle-strength evidence materializes a complete candidate as FP16 storage and strict-loads that artifact into FP32 runtime. Candidate and opponent identities are independently audited. CPU-256 and CUDA-2048 remain separate reports with all games terminal and zero error, unfinished, or semantic fallback.

Frozen schedule materialization is identity-bound to focal deployment hash, complete opponent hash, exact numeric deck slot, replica and namespace. CPU-256 is replica 0; CUDA-2048 is eight immutable replicas, and its first 256 jobs equal the CPU frequency unit. Seeded toss identifies the winning Agent, which must process official context 41 and choose first or second; the harness never assigns the seat directly.

## CUDA Engine 2.0 backend

`engine_cuda_2_0/` is the required shared GPU engine implementation for 0043. The project-local `cuda_engine_2/` adapter pins the actual tracked source bytes, official state/rule ABI, private rule-pack hash, SM120 capability, native binary, extension, and complete policy/deck identities before creating lanes. The current RTX 5080 build produced the production static library, native runtime smoke binary, and PyTorch extension; official rule upload plus a two-lane device reset/classify smoke passed with ABI 7 and rule ABI 1. A real `001` versus `048` / reverse-seat smoke then materialized the full `semantic0031_v2` tensors on-device and routed every decision through independent complete Policy-0809 and Champion-G1 FP32 models. Both games reached terminal after 163 total decisions with engine turns 16/12, zero engine error, unfinished game, or CPU fallback. The existing upstream old-55 acceptance evidence covers 1,100 battles and 215,937 compared decisions with zero state/status/outcome mismatch, but 0043 still requires its own complete observation/action first-divergence gate before formal PPO.

CUDA 12.8 compilation of the optional `official_continuation_dispatch_smoke` translation unit reached roughly 30 GiB RSS by itself. The self-contained 0043 build entrypoint therefore uses `--parallel 1` and builds only the production dependencies `ptcg_cuda_smoke` and `_ptcg_cuda`; it never invokes the CMake `all` target. This is a build-resource constraint, not a runtime or policy-semantic change.

## Mixed-focal cohort optimization

CUDA resident collection now groups the 256 jobs only by immutable opponent policy identity. Each focal row receives its resident `job_index`, which selects that job's exact `focal_deck_id` and 29-way own-archetype embedding; exact-deck static features remain lane-bound and audited. Policy-0809 and Champion-G1 still use separate complete, immutable resident models, audits, and collector calls. This changes batching only: it does not share weights or caches across policy identities and does not change the 15-way opponent Meta head.

On the same 99 U20 Policy-0809 jobs, old two-focal-cohort execution ran at 2.96 games/s and the mixed-focal cohort at 4.47 games/s (1.51×); outcome, terminal turn, error, and routing status matched for every game. A separate 64-game mixed 002/007 CUDA smoke completed 64/64 with zero error, zero routing failure, and zero feature D2H. V2 starts from the immutable V1 update-21 model-only checkpoint with a fresh optimizer, carries forward the C002 PFSP state, and retains the original G1/update-0 reference-KL anchor.

## Current and next stage

- Complete: project-local assets/runtime, complete policy materialization and isolation, league sampler, PFSP persistence, one-pass telemetry, frozen schedule materialization, candidate evaluation gates, and explicit human-decision Promote workflow.
- Complete diagnostic: real CPU forward plus small RTX 5080 FP32 forward parity for Policy-0809 and Champion-G1; greedy actions matched, with maximum absolute errors below `6e-5`. CUDA Engine 2.0 native runtime and PyTorch official-arena reset/classify smokes also pass. These are implementation evidence only, not policy strength evidence.
- Complete: `V1_focal_002_007` reached checkpoint 21; its source-update-20 rollout used curriculum C002 and is frozen as the planned optimization boundary. U20 Frozen Policy-0809 CUDA-2048 reports are complete for focal decks 002 and 007.
- Complete and stopped: `V2_mixed_focal_cohorts` resumed logical source update 21 with a fresh optimizer, retained C002 PFSP state and the original G1 reference anchor, and completed source updates 21–206 before the user-requested stop. Its final emitted model-only checkpoint is U207; no training process remains.
- Complete Frozen selection audit: rollout landmarks U57 (early deck-002 peak), U103 (deck-007 point peak), U136 (best joint rolling balance), and U190 (late stable region) were each evaluated for focal decks 002 and 007 against independently materialized Policy-0809 with CUDA Engine 2.0, 2,048 official-engine games apiece. All eight reports passed candidate/opponent/deck identity, terminality, routing and zero-feature-D2H gates.
- Result: U190 is the best single shared checkpoint among the tested V2 points (mean deck-002/007 win rate 60.74%, worst-deck 59.13%). Per deck, U190 leads deck 002 at 59.13%; U57 and U136 tie deck 007 at 63.53%, with U136 the preferred descriptive candidate because its actual-first/actual-second split is more balanced. None of the tested changes versus U20 is statistically decisive at 2,048 games, so this ranks candidates but does not establish Promotion or Kaggle strength superiority.

V1 and V2 are immutable stopped runs. Formal selected-checkpoint reports are indexed under `experiments/0043_champion_league_rl/evaluation/`; any resumed training must allocate a new strictly increasing repository version and a new W&B run rather than append to V2.
