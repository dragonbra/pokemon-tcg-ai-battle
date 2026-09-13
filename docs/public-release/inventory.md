# Public release inventory

This branch presents the final competition system as one BC + RL codebase instead of a sequence of numbered experiments. The complete research workspace remains immutable at tag `archive/final-competition-repo-2026-09-13`.

## Public core

| Area | Public path | Reason retained |
|---|---|---|
| Model and training | `src/pokemon_tcg_ai/` | Final 0045 semantic BC foundation, Actor/Critic split, PPO, LoRA adaptation, routing, export, and tests |
| Model design | `docs/model/` | Authoritative architecture, training stages, identity contracts, decisions, and formal reports |
| Policy protocol | `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md` | Canonical policy identity and FP16-storage/FP32-runtime contract |
| Rules evidence | `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md` | Separates official rules, engine behavior, and policy assumptions |
| Official runtime | `engine/` | Read-only official engine source and local build interface used for real match evaluation |
| Evaluation runtime | `evaluation/` | Isolated official-engine package loading, seeded schedules, workers, and reports |
| Training logging | `rl_environment/logging.py` | Canonical local/TensorBoard/W&B metric ordering used by the trainer |
| Static game data | `data/official/` | Card and engine-facing reference data used by the semantic compiler/runtime |
| Final packages | `archive/submission/` | Git-LFS-backed competition packages and immutable package manifests |

## Historical material removed from this branch

- Numbered training projects other than the final 0045 integration.
- Numbered experiment directories and run records not needed to explain or execute the final system.
- The unfinished 0047 Meta-routed MoE continuation. Its design records multiple failed/stopped versions and no committed final checkpoint; it is research provenance, not the final submitted implementation.
- Daily leaderboard snapshots, historical combat matrices, exploratory reports, one-off plans, and local debugging assets.
- Superseded pretrained archives already copied and hash-bound inside the final project's self-contained policy registry.

Nothing in this cleanup rewrites history. Every removed path is recoverable with:

```bash
git show archive/final-competition-repo-2026-09-13:<path>
```

## Dependency boundary

The final Python package may import only:

- its own `pokemon_tcg_ai` modules;
- Python/PyTorch dependencies declared by the repository;
- `evaluation` for official-engine evaluation;
- `rl_environment.logging` for metric mirroring.

It must not import executable code from a numbered `train/00xx_*` project. Historical source paths may remain only as provenance strings in immutable manifests.

## Known archive-state test boundary

At the archive tag, `train/0045_single_deck_expert_minimal_lora/tests` passes 95 tests. The separately added Top-500 report test has four failures because its generator freezes the 001–070 catalog while the referenced live 0045 registry has since become contiguous 001–071. The public branch drops that historical leaderboard generator rather than silently rewriting the published snapshot.
