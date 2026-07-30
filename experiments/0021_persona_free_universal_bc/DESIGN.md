# 0021 Corrected Persona-Free Universal BC

## Current Status

0021 is a non-trainable scaffold. The project is self-contained; the R15 Actor no longer accepts
`source_id` or persona conditioning; relative-seat, option-set, termination-target, and
decision-balanced loss corrections are implemented and covered by focused tests. Corrected dataset
materialization and IID/deck-OOD split audits still block V1 training.

Formal training is blocked until all V1 release gates below pass. There is no 0021 checkpoint,
training metric, W&B run, candidate package, or strength claim yet.

## Intended V1 Policy

The policy is:

```text
pi(action | actor-relative causal state, exact own 60-card deck,
            causal visible history, unordered legal-option set)
```

Teacher/team identity is metadata only. It may support provenance and grouped audits, but is absent
from Actor batches, model parameters, loss weighting, inference, and deployment.

Winner-side actions are the only BC targets. The project does not use loser actions as imitation
labels.

## Model Data Flow

```text
actor-visible observation + exact own deck + legal options
  -> corrected actor-relative codec
  -> board/entity encoder + causal resource/event memory
  -> unordered option-content encoder
  -> R15 scenario/goal integration
  -> gradual-option pointer decoder
  -> ordered legal action, with learned STOP only when optional
```

The current scaffold directly instantiates `R15DeterministicGradualOptionPolicy`; the 0019
`SourceConditionedR15Policy`, source embedding, source state residual, and source option residual are
not present in 0021.

The current corrected R15 has 17,346,562 parameters. Removing the 129 x 320 option-position
embedding accounts for the 41,280-parameter reduction from the old source-free backbone count.

## Required Input Contract

The final tensor shapes remain to be frozen after corrected materialization. The semantic fields are:

- actor-relative global categorical/numeric state;
- public zone inventories ordered as `[self, opponent]`, never absolute player 0/1;
- public card/entity categorical and numeric state with masks;
- exact registered own-deck card IDs, multiplicities, and masks;
- causal own-resource ledger and event-memory tensors;
- known/unknown opponent-hand evidence;
- content-only legal-option tensors and masks;
- ordered action targets, target mask, min/max count, and termination code.

No source, persona, team, player name, submission ID, absolute player index, or legal-option list
position may appear in Actor input.

## Action and Loss Contract

Legal options are a set. Permuting them must only permute option logits and decoded indices after
inverse remapping.

`optional_stop` receives a learned STOP target after the selected sequence. `forced_max` receives no
STOP target because runtime termination is automatic.

Training minimizes mean per-decision sequence NLL: token NLL is averaged within each decision, then
decision losses are averaged across the batch. Token accuracy and exact action remain diagnostics.

## Data and Validation

0021 freezes the accepted 0019 winner Episode/player membership but does not reuse the old
model-ready shards. New shards must be rematerialized from committed raw rows under the corrected
schema.

Validation must publish separate Episode-IID and deck-OOD namespaces. A deck-OOD split is valid only
when held-out deck hashes do not occur in train and have enough Episode support for an uncertainty
statement. Offline imitation metrics do not prove official-engine strength.

The verified preflight split freezes 89,503 winner Episode/player trajectories. It assigns 76,453
Episodes to train, 8,280 to IID validation, and 4,770 to deck-OOD validation. The OOD set contains
20 deterministically ranked exact deck hashes selected from 84 decks with 50-1000 Episodes each;
every held-out deck therefore has at least 50 Episodes and is absent from train by construction.

## V1 Release Gates

1. Relative-seat tests pass and no absolute player identity reaches the Actor.
2. Randomized option-permutation property tests pass.
3. Optional and forced termination target tests pass.
4. Decision-balanced loss tests pass.
5. New shard and split commitments pass the full data audit.
6. Rematerialized shard shapes and the measured 17,346,562 parameter count agree with code and HTML.
7. Exact epoch-boundary recovery checkpoint, TensorBoard, and W&B contracts pass smoke.
8. Only then may `V1_corrected_persona_free_r15` start.

## Checkpoint and Resume Contract

0021 V1 is an explicit user-authorized exception to the model-only checkpoint default. Every
completed epoch publishes one atomic `0021_resumable_bc_checkpoint_v1` payload containing:

- model and AdamW optimizer state;
- scheduler and GradScaler state when those components exist (V1 BF16 currently uses neither);
- completed epoch, global step, best-selection metrics, early-stopping counter, progress counter,
  and metric history;
- Python, NumPy, Torch CPU, and all Torch CUDA RNG states;
- project/version, dataset, feature compiler, model-config, and training-config commitments.

Loading verifies all identity commitments before restoring state and may only append to the same
repository version and stable W&B run. V1 saves after the complete train, IID-validation, and
deck-OOD-validation passes; shuffling is derived from `(seed, epoch)`. The guarantee is therefore
exact recovery from a completed epoch, not from the middle of a DataLoader pass. Named criteria
retain at most eight full recovery files and report retained bytes.

W&B receives canonical per-epoch optimization and both validation namespaces. It also receives
every-100-batch progress snapshots containing running loss, decisions/iterations per second,
gradient norm, learning rate, elapsed time, and allocated/reserved GPU memory, plus checkpoint byte
usage after every epoch.

Formal V1 sets `minimum_gpu_training_seconds=43200`. Early stopping requires both the configured
IID-loss patience and at least 12 accumulated hours of complete CUDA epoch work. The cumulative
counter is checkpointed and restored, so an interruption cannot reset the minimum-duration gate.

## Later Research Boundary

Both players' visible states may support a separately versioned representation or transition
objective. Such a version must prevent future/private information leakage and must never turn loser
actions into BC labels. It is not part of V1.
