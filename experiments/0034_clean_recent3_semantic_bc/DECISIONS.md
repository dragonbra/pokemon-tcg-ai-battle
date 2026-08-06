# 0032 Decisions

## 2026-08-04: effective, truthful, sufficient, nonredundant input

- Fork 0032 from the complete 0031 implementation, while keeping the project fully self-contained and retaining the immutable full-engine v2 prototype asset as authority.
- Keep every visible physical card as an instance token and keep `card_parent`, `option_source`, `option_target`, `event_source`, and `event_target` as explicit relations.
- Replace the lossy resolved-Energy total/mask pair with `resolved_energy_histogram[12]`, directly counted from the official observation's `energies` values. This preserves EnergyTypeIndex multiplicity.
- Keep Energy card prototype `energy_type_mask` as nine independent bits and `engine_energy_count` as a numeric fact. Conditional special Energy remains identified by its card/skill prototype and target relation; no counterfactual attach result is fabricated.
- Keep card-to-attack, card-to-skill, skill area, trigger type, and the complete static trigger subject target specification.
- Do not add runtime `trigger_subject` or `trigger_target`: the official actor observation does not expose the engine's runtime `TriggerInfo.subject/object`, so doing so would break train/inference parity.
- Remove the entire effect-node actor graph. The policy never directly selects an effect node, and the graph duplicated option/card/skill semantics at high cost.
- Remove counts and masks exactly derivable from retained relations or histograms, including physical attached count, resolved total units, and resolved type mask.
- Exclude Energy deficits, attach-provides answers, newly enabled attacks, retreat deficits, damage results, and KO answers because they require a counterfactual rules-engine execution.
- Treat present zero, unknown, not applicable, and padding as distinct numeric states. Encode a Pokemon with no special condition as an explicit `status_bits=1`; reserve zero for non-applicable/padding.
- Require categorical ranges, finite numeric values, valid missingness, legal one-based relations, and legal targets at the `DecisionBatch` boundary.
- Make card zones owner-neutral and use `relative_owner` as the sole ownership channel.
- Keep card/attack IDs as prototype lookup keys but skip separate dynamic categorical identity embeddings, eliminating a second learned encoding of the same identity.
- Add functional tests that perturb every retained categorical and numeric column, every missingness family, and every explicit relation, and require model logits to change.
- Remove `deck_order_known` from actor input because the actor-visible knowledge implementation never has authority to set it true.
- Encode owner and area as unknown when an option has no real source/target; bind Attack and Retreat to the own Active instance instead of fabricating a generic own-player source.
- Remove option-skill relations because official options do not expose `skillId` and the previous relation was reconstructed solely from source/context/effect card IDs. Keep those four card roles distinct with role-specific projections.
- Remove reverse attack/skill-to-card relations; the retained card-to-attack/card-to-skill tables reconstruct them exactly.
- Preserve event direction with separate source/target instance projections and separate projections
  for source-card, target-card, Active, Bench, before, and after card prototypes.
- Mark resolved-Energy histogram fields `NOT_APPLICABLE` on physical Energy-card child tokens; the
  histogram describes the parent Pokemon's engine-resolved state, while the child card's own Energy
  semantics come from its prototype and `card_parent` relation.

Formal data materialization and local acceptance are complete. The formal 0032 dataset contains
2,048,069 winner-only decisions from 2026-07-28 through 2026-08-01: 1,851,008 train and 197,061
validation records. The formal model is not claimed until the running five-epoch training job and
downloaded checkpoint pass their completion gates.

Verification on 2026-08-04: 56/56 project tests, Ruff, compileall, JSON validation, and the reusable
raw-replay feature audit pass. Official 2026-08-01 Episode 89228732 produced 68 winner decisions and
all 68 compiled; all 680 in-play Pokemon histograms matched the engine observation, all 94 Attach
options had source/target relations, all 520 attached child tokens had a parent, and an eight-row
CUDA batch produced finite teacher logits with shape `[8, 4, 11]`.

The fail-closed audit reconstructs every visible card instance in observation order and checks
identity, owner, zone, slot, serial lookup, and exact parent. It then recomputes every option and
event relation from source addresses. Duplicate-card regressions prove that two identical Energy
cards or Pokemon cannot pass by matching only `cardId`: Attach source/target and child-parent
relations must point to the exact slot. Episode 89228732 passed 575 exact option relations and 1,999
exact event relations.

The production gate ran the same source-observation audit for every canonical decision before
shard write, then verifies every shard hash, reloads every record through `DecisionBatch`, and runs an
exact-policy forward. A two-Episode notebook smoke produced 136 audited records with 68 train and 68
validation decisions, nonzero values for all five explicit relation families, and finite model
logits. Verification is now 56/56 tests. Four independent Kaggle kernels produced five daily
partitions as version 3. All five outputs were downloaded and every shard byte count, SHA256, record
count, date, split, relation, winner-only flag, `DecisionBatch`, and exact-policy forward passed local
acceptance. The five canonical manifest hashes are committed in `manifest.json`.

Kaggle versions 1 and 2 exposed environment-only failures before any dataset was accepted: input
mounts are symlinks, and Kaggle starts with about 20 GiB free rather than the server guard's 50 GiB
floor. Version 3 discovers numeric Episode JSON files through the mounted path and keeps a fail-closed
18 GiB dataset limit plus 1 GiB free-space floor. The feature and model acceptance gates are unchanged.

The dual-T4 training notebook was pushed as Kaggle version 1 only after all four data kernels passed.
One
epoch is defined as one exact pass over all five daily partitions; the formal run requests exactly
five train passes and five full validation passes. It uses random initialization, FP16 GradScaler,
two-T4 `DataParallel`, no W&B, no pretrained input, and saves only ordinary `SemanticPolicy`
`best_model.pt` and `last_model.pt` artifacts without optimizer or resume state. A local five-partition
single-GPU smoke completed exact train/validation coverage and produced a loadable state dict without
`module.` prefixes. Version 1 is currently running; no epoch or checkpoint result is recorded until
the downloaded artifacts prove it.

## 2026-08-07: minimal recent-three-day cleaning, merged epochs, and formal evaluation

- Materialize a dedicated 0034 actor schema rather than claiming that the old 0032 label is the
  final contract. The accepted corpus covers 2026-08-01 through 2026-08-03 and contains 1,204,592
  winner-only decisions: 1,081,527 train and 123,065 validation.
- Keep cleaning deliberately narrow. Ordinary actions use weight 1.0, low-confidence special-Energy
  Attach actions use 0.95, and high-confidence Attach actions use 1.15. Compact event history and
  recompute relation metadata after compaction instead of preserving stale event-source/target counts.
- Define one epoch as one complete pass over the merged three-day train split, followed by one
  complete pass over the merged validation split. Do not rotate dates across epochs.
- Train from random initialization for three full passes with weighted cross-entropy, FP16, dual T4
  `DataParallel`, no pretrained input, no W&B, and model-only checkpoints. Preserve the prototype
  assets, actor contract, loader, and usage documentation in the package.
- Accept version 2 after local package validation proved 22,595,202 parameters, exact three-pass
  coverage, ordinary state-dict keys, package-only loading, and a finite `[8, 6, 8]` teacher forward.
  The best checkpoint is 90,610,211 bytes with SHA-256
  `6bb23f0b23c04c478c7c0926310967c73325685b0308959f147240219e3538ff`.
- Record the final validation metrics without treating BC labels as ground truth: weighted loss
  0.436949, token accuracy 0.850293, teacher exact 0.703661, action type 0.932346, Attach target
  0.497538, and Attack 0.691584. Attach remains a weak action family and should drive later targeted
  cleaning or supervision work.
- Evaluate the Marnie deck for exactly 50 games against each fixed six-opponent BC suite on CUDA.
  The run completed 300 games with zero errors and zero fallbacks: 169 wins, 131 losses, and 56.33%
  aggregate win rate. Per-opponent rates were Lucario 60%, Cynthia 42%, Kangaskhan/Crustle 44%,
  Marnie 44%, Alakazam 56%, and Dragapult 92%.
- Treat the arena result as protocol-specific evidence. It does not establish a general metagame
  ranking, and the wide opponent variance argues against summarizing this model by one win-rate number.
