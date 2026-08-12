# Engine CUDA Provenance

This subtree was refreshed on 2026-08-04 from the `cuda_engine/` working-tree
snapshot on branch `codex/0027-semantic-foundation` (base commit
`b8cd9d0b1e6b0c21ba53604b743348faf80b9eea`). The destination was based on
`dev/cyd_main` commit `652c3205c83da384be2411d9f7254b891aa32f3f`.

The snapshot includes the working-tree CUDA semantic/parity changes and the
new focused coverage tests present at import time. Therefore the destination
commit, rather than the source base commit alone, is the canonical identity of
the published snapshot.

The 0022 catalog under `fixtures/0022_deck40/` contains 40 exact 60-card public
deck fixtures used by the recorded parity matrix. Machine-local replay paths
were replaced by stable `kaggle_episode:<id>` provenance identifiers. Card
contents and declared SHA-256 values were not changed.

Excluded from the import:

- build directories and compiled binaries;
- benchmark artifacts and logs;
- model/checkpoint/W&B data;
- competition-private generated rule packs and official generated fixtures;
- credentials and machine-specific runtime state.

The official CPU source is not duplicated into this subtree. Tools use the
repository's read-only `engine/source/ptcgProgram 22/` tree as the oracle.
