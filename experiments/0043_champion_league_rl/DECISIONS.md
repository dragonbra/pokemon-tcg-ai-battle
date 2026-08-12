# 0043 Decisions

## 2026-08-12: establish the self-contained Champion League boundary

- Name the numbered project `0043_champion_league_rl`.
- Freeze the emitted V22 training config as the regression baseline; do not use old parser defaults to reconstruct approved hyperparameters.
- Treat the existing `Champion-G1` archive as a user-designated generation anchor. Its import is asset migration, not a new automatic Promotion.
- Separate immutable deck identity, policy identity, and evaluation composition. The historical string `0806` may remain only in provenance; it does not define the 0043 Frozen opponent.
- Register the existing 55 exact decks in both the initial Training Deck Pool and FrozenMeta256-V1, while retaining separate pool manifests so later training additions cannot mutate evaluation.
- Use `001`–`055` as the sole canonical identities of the initial exact decks and reserve monotonically appended `056` onward for future decks. Names, archetypes, historical IDs and hashes are metadata only.
- Store Policy-0809 and Champion-G1 under human-readable project-local directories `policy_0809` and `champion_g001`. Hashes validate content but never act as semantic folder names. Runtime code must not fall back to imported historical paths.
- Keep formal training blocked until complete G1/0809 materialization, no-hybrid, sampler, PFSP, telemetry, PPO regression, and Frozen parity gates pass.
