# 0043 Champion League RL Design

Status: **Phase 0 PASS; Phase 1 asset layer implemented; `BLOCKED_PRETRAINING_CONTRACT`**

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

## League schedule

```text
outer focal-deck phase
  -> 128 PFSP lanes: independent deck and policy weakness draws
  ->  64 uniform lanes: independent Training Deck / Active Policy draws
  ->  64 latest lanes: uniform Training Deck + immutable Champion-G1
  -> seeded shuffle preserving lane seat/seed identities
  -> 256 official-engine games
```

PFSP uses Beta(1,1) smoothing and produces a new immutable curriculum manifest every ten updates. Deck/policy joint statistics are diagnostic only; V2.0 has no compatibility whitelist.

## Telemetry and strength claims

Every completed rollout will aggregate raw WR, curriculum identity, deck/policy distributions, terminal prize margin, loss-only focal prizes, win-only opponent prizes, win/loss turns, PFSP entropy/difficulty, p10 tail WR, and red counts from the same 256 games. Existing PPO and runtime health metrics remain. Sampled rollout metrics are training diagnostics, not frozen greedy checkpoint strength.

Formal Kaggle-strength evidence materializes a complete candidate as FP16 storage and strict-loads that artifact into FP32 runtime. Candidate and opponent identities are independently audited. CPU-256 and CUDA-2048 remain separate reports with all games terminal and zero error, unfinished, or semantic fallback.

## Current and next stage

- Complete: Phase 0 audit and Phase 1 project-local asset registries/import/integrity validation.
- Next: complete Policy-0809 and Champion-G1 materializers plus focal/opponent storage-isolation and no-hybrid tests.
- Then: sampler, PFSP persistence, PPO integration/regression, telemetry, FrozenMeta parity, and manual Promote workflow.

No formal 0043 training version exists yet. Research smoke output must remain under `.tmp/`; formal versions will use `rl_runs/0043_champion_league_rl/versions/V<n>_<tag>/` and W&B private project `dragon_bra/pokemon-tcg-policy-learning`.
