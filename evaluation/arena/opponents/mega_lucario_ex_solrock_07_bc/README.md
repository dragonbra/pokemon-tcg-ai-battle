# Lucario Arch-scoped v11

This is an isolated challenger derived from
`agent_lucario_b1cb1f7b_stall_bench_v1_20260710`. The deck is unchanged and
the incumbent directory is not modified.

The new planner activates only after public Duraludon `169` or Archaludon ex
`190` is visible. It adds:

- Full Metal Lab and Metal Defender damage accounting;
- exact Premium Power Pro counts for an immediate KO;
- Boss only for an immediate bench KO or a game-winning KO;
- a KO-able Relicanth engine target unless the active Archaludon can be KO'd.

Cinderace `666` alone is deliberately not an archetype marker because some
Starmie lists also contain it. Non-Arch states preserve incumbent behavior.

Validation:

- fast-Cinder five-policy gate, 500 paired games: incumbent 38.2%, v11 55.6%
  (`+17.4pp`);
- refined Top50 scope gate, 1,000 pairs: 998 identical, v11-only 2,
  incumbent-only 0;
- focused regression tests: 27 passed;
- package smoke: 4-0, with no errors or max-step games.

Kaggle:

- ref: `54724932`;
- submitted: `2026-07-15 20:21` Asia/Shanghai;
- package SHA256:
  `982DE37805351AFD6730DD016E93F33457029E8105BF57DD29EE19B4191795E8`;
- observed score at `2026-07-15 20:29` Asia/Shanghai: `746.1`;
- status: pending the roughly 24-hour convergence window; do not promote over
  ref `54523823` (`953.1`) yet.

The full diagnosis and rejected probes are recorded in
`docs/104_lucario_archaludon_scoped_repair_20260715_zh.md`.
