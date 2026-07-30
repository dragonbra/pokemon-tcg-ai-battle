# Decision 001: Corrected Persona-Free Pretraining Contract

Date: 2026-07-30

Status: accepted for scaffold; formal training remains blocked until implementation and data audits
pass.

## Decision

0021 is not a source-free-only ablation of 0019. It is a corrected universal BC pretraining
project. It freezes the accepted 0019 winner Episode/player membership and winner-only BC action
rule, but rematerializes features and targets under a new schema.

Teacher/team identity remains available only as provenance and grouped audit metadata. It never
enters Actor input, representation, option encoding, decoder state, sampling weights, loss weights,
checkpoint deployment metadata, or runtime inference.

The corrected V1 contract also requires:

- actor-relative first-player encoding instead of absolute player index;
- no engine option-list position embedding;
- permutation equivariance over legal options;
- STOP supervision only for `optional_stop`, never for `forced_max`;
- mean per-decision sequence NLL rather than token-balanced cross entropy;
- separate Episode-IID and meaningful deck-OOD validation reports.

## Preserved User Constraints

Winner-side actions remain the only BC action labels. Winner and loser visible states may be
considered later for separately versioned representation or transition objectives, but loser
actions must not become BC targets.

## Evidence

The 0019 source-persona residual changed both state and option encodings. Epoch 13 validation exact
action was 81.29% with source conditioning and 74.84% at neutral source ID 0, so persona was not a
mere reporting label.

The 0019 codec encoded absolute `firstPlayer` without encoding `yourIndex`, so it did not directly
represent whether the Actor itself went first.

The 0019 codec encoded `option_index + 1`. Reversing physically identical legal options and
inverse-remapping indices flipped the first decoded action on 21.88% of a 256-decision audit; mean
absolute logit change was 0.513 and p95 was 1.987.

The 0019 materializer appended STOP after every ordered action, including `forced_max` decisions
where deployment terminates automatically. Its cross entropy was averaged across action tokens, so
longer multi-select actions received more total influence.

The old validation split was not a meaningful deck generalization benchmark: 208 of 214 validation
decks also appeared in train, while only six validation-only decks remained and each was represented
by one Episode.

## Consequences

The 0019 model-ready tensor shards are provenance, not trainable 0021 input. Reusing them would retain
the invalid feature and target semantics. 0021 must build new shards from committed raw rows and
publish new content hashes before smoke or formal training.

The initial copied 0021 code is deliberately marked non-trainable until feature, target, loss, split,
and documentation tests pass. Any auxiliary representation/transition objective requires a later
version and a separate decision record.

## Checkpoint Exception Accepted 2026-07-30

The user explicitly requires optimizer-backed interruption recovery for this pretraining run. 0021
V1 therefore overrides the repository's model-only default and saves bounded, atomic recovery
checkpoints after each fully completed epoch. Each checkpoint contains model, AdamW optimizer,
optional scheduler/GradScaler state, trainer counters/history, and Python/NumPy/Torch CPU/Torch CUDA
RNG state.

Resume is fail-closed and permitted only inside the same repository version and stable W&B run when
dataset content, feature compiler, model configuration, and training configuration commitments all
match. This is an exact epoch-boundary contract, not a claim of mid-epoch DataLoader cursor recovery.
At most eight full recovery checkpoints are retained through named selection criteria.
