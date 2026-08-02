# 0026 Raging Bolt Canonical Decoder RL

## Question

Can the 0025 canonical card/action representation support policy improvement when only the
ordered action decoder is updated? The focal policy is the exact James Cox / James Cox & Henry
Chao Raging Bolt deck. Its opponents are all 51 exact-deck Frozen 0019 Foundation policies.

## Evidence Boundary

- **Official rules:** turns, legal actions, attack termination, prizes, zones, and win conditions
  follow the official rules summarized in the repository rules evidence document.
- **Runtime/card facts:** legality, card effects, state transitions, and terminal results come
  only from the unmodified official engine runtime and its current card data.
- **Project hypothesis:** explicit prototype/action semantics reduce avoidable ambiguity enough
  that a small decoder-only PPO update can learn useful Raging Bolt decisions. A win-rate gain
  supports usability, not semantic completeness or broad generalization.

## Actor Input Contract

The online causal encoder retains the original observation-derived game state and enriches card
IDs by prototype lookup. It creates typed variable-length memories rather than one flat vector.

| Memory | Categorical fields | Numeric fields | Relation fields |
|---|---:|---:|---|
| Global | 11 | 17 | selection bounds |
| Card instance | 5 | 7 | parent/evolution |
| Resource ledger | 4 | 15 | card identity/prototype |
| Recent event | 8 | 4 | actor, zones, target |
| Legal option | 14 | 16 | source, target, skills, effects |

Categorical values use typed embeddings. Real quantities such as HP, damage, energy deficits,
deck counts, and selection counts use numeric projection. Padding has an explicit mask and never
becomes an attendable token.

The option identity includes `action_type`, source/target card and area, `attack_id`, selected
energy type, context/effect card IDs, current/base damage and KO facts, typed and total energy
deficits, plus variable-length skill/effect prototype relations. This is the action-semantic
contract that the legacy state/option representation lacked.

## Model Data Flow

```text
RAW official observation
  -> OnlineCausalEncoder + card/attack/skill/effect prototype lookup
  -> typed global/card/ledger/event tokens
  -> CanonicalStateEncoder (4 Transformer blocks)              [FROZEN]
  -> state token memory + summary

RAW legal option list from the same observation
  -> typed option fields + exact source/target relations
  -> attack/skill/effect prototype relations
  -> CanonicalOptionEncoder (3 cross-attention blocks)          [FROZEN]
  -> one contextual token per legal option

state summary + legal-option tokens + prior selected options
  -> OrderedOptionDecoder (unique ordered choices + STOP)       [TRAINABLE]
  -> official-engine selection indices

state summary
  -> LayerNorm -> Linear -> GELU -> Linear -> tanh              [TRAINABLE VALUE]
  -> actor-relative V(s) in [-1, +1]
```

Residual connections exist inside the Transformer blocks, but 0026 does not depend on a single
compressed summary alone: the option encoder cross-attends to the full typed state token memory,
and each option directly receives source/target/prototype relations before the final decoder.

## Parameters And Freezing

| Component | Parameters | 0026 status |
|---|---:|---|
| 0025 canonical actor | 21,837,082 | loaded from best exact |
| Frozen representation | 20,809,880 | immutable |
| Ordered action decoder | 1,027,202 | trainable |
| New value head | 103,681 | trainable |
| Actor-critic total | 21,940,763 | 1,130,883 trainable |

The frozen representation SHA-256 is
`22004a14c08a9640a0f48e30fe715a310e849b892d85400f57ac44b7dd4b9587`.
Every PPO update recomputes it and fails if it changes.

## Heterogeneous League

Each game runs in an isolated official-engine worker. The parent process routes focal observations
through the canonical encoder and stochastic decoder, while opponent observations use a project-
local frozen copy of the 0019 Foundation encoder and greedy policy. Opponent/source identity never
enters focal actor features.

One PPO update collects 512 games across all 51 identities with balanced focal seats. Only focal
decisions enter trajectories. The terminal result is `+1/0/-1`; `gamma=1.0` and GAE lambda is
`0.95`. Episodes receive equal total weight regardless of decision count.

## PPO Objective

The optimized objective combines clipped policy loss (`0.10`), value MSE (`0.5`), entropy bonus
(`0.01`), and a squared log-probability anchor to the immutable initial best-exact decoder
(`0.02`). Actor LR is `1e-5`, value LR is `1e-4`, target behavior KL is `0.02`, and gradient norm
is capped at `0.5`.

Chronology is explicit:

```text
checkpoint k-1 -> stochastic 512-game rollout -> PPO update -> checkpoint k
                     source_policy_update=k-1

checkpoint k -> separate deterministic 102-game Frozen evaluation
                eval/checkpoint_update=k
```

Rollout win rate is a sampling diagnostic. Checkpoint strength claims require the separate greedy
suite with the same 51 opponents, seeds, and balanced seats.

## Storage And Stage

Checkpoints contain only action-decoder/value tensors and identity metadata. Optimizer, scheduler,
RNG, rollout, replay, and engine state are forbidden. V1 is the official-engine smoke gate. V2 may
start only after zero engine/encoder/decode errors, finite PPO metrics, a changed decoder hash, and
an unchanged representation hash are demonstrated.

## V2 Stop, Selection, And Formal Evaluation

The user stopped V2 while update 10 rollout collection was in progress. Update 10 did not produce
a checkpoint; update 9 is the last completed PPO update. This is an intentional interruption, not
a completed training run. The sampled rollout that was in progress is not checkpoint-strength
evidence.

Checkpoint selection used the separate deterministic 102-game Frozen suite. `update-0005.pt` was
the best observed checkpoint under that contract at `32-69-1` (`31.37%`), compared with update 0 at
`31-71-0` (`30.39%`). Its SHA-256 is
`228db480c405f199dad0ca3447633d8ee10608d0e31487fd6455336b8ba5551e`.

The selected checkpoint was then evaluated through the unmodified official engine against all 51
Frozen opponents for 10 games each:

| Candidate | W-L-D | Win rate | Complete / errors |
|---|---:|---:|---:|
| 0026 V2 update 5 | **156-354-0** | **30.59%** | 510/510 / 0 |
| 0025 V4 best-greedy reference | 153-357-0 | 30.00% | 510/510 / 0 |

The 0026 result is three additional wins in 510 games. This is effectively tied at this sample
size and is not meaningful evidence that decoder-only PPO improved the policy. It also explains
why the 102-game `31.37%` selection estimate must not be reported as the final policy strength.
The authoritative report is
[`evaluation/V2_best_exact_decoder_only_frozen51.html`](evaluation/V2_best_exact_decoder_only_frozen51.html),
run `run-9da854485e114ad88f8ef10a7a89156d`.
