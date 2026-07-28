# Universal BC Pre-training and Deck-specific Adaptation

Status: **forward-looking architecture decision and experiment contract**

Date: 2026-07-29
Scope: the long-term BC + RL path toward a shared policy foundation with small deck-specific heads

## 1. Decision summary

The project should move toward a three-stage policy architecture:

```text
high-quality, multi-deck official demonstrations
                    |
                    v
       Universal BC representation pre-training
       shared card / board / resource / event Encoder
                    |
          explicit deck + source conditioning
                    |
                    v
         Deck-specific Adapter and Decoder
       exact construction and game-plan specialization
                    |
                    v
          terminal-reward deck-specific RL
        official-engine games against Arena pool
```

The central hypothesis is that most expensive knowledge is shared: card semantics, legal action
meaning, public and hidden-information boundaries, evolution timing, once-per-turn resources,
attack commitment, Prize pressure, deck exhaustion, target relations and multi-step action syntax.
BC can learn this representation from many strong winning trajectories. A much smaller
deck-specific module can then learn which of several legal and plausible plans matters for one
exact 60-card construction. Terminal-reward RL finally calibrates those choices against actual
opponents rather than teacher agreement.

This is the target architecture, not a claim that the current Encoder is already universal. The
0017 result is encouraging evidence for the decomposition: with the R15 representation frozen,
updating only 1,027,202 Decoder parameters (5.90% of the 17,416,642-parameter actor) improved the
same 26-opponent, seat-balanced official-engine evaluation from 41/260 (15.77%) to 68/260
(26.15%). It does not yet prove cross-archetype transfer or identify the optimal module boundary.

## 2. Evidence boundary

### Official rules

The official rules define the common game dynamics that a transferable representation must encode:

- draw, then main actions in any order, then attack as the turn-ending commitment;
- evolution timing, including the fact that Rare Candy does not bypass turn timing;
- one Supporter, one manual Energy attachment, one Retreat and one Stadium per turn;
- victory by taking all Prizes, leaving no opposing battle Pokemon, or an opponent failing the
  mandatory draw at the start of their turn;
- separate hand, deck, Prize, Active, Bench and discard resource zones.

These are universal game facts. They do not prescribe a network architecture, an optimal policy or
a reward function. The detailed evidence is
[`docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`](docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md).

### Current card and runtime facts

Exact card effects, Card IDs, legal option payloads, registered decks and terminal results come
from the current official card data and the unmodified official engine runtime. The engine supplies
legal options; the policy must learn their long-term value. Model features must never replace the
runtime as the legality authority.

### Project hypotheses

- High-rating, winner-perspective official trajectories are a practical denoising method for
  pre-training, not proof that every demonstrated action is optimal.
- A multi-deck Encoder can learn more transferable game concepts than an equally sized single-deck
  Encoder when deck and expert identity are explicitly conditioned.
- Exact construction changes can often be handled by a small Adapter/Decoder without relearning
  card semantics and rules from scratch.
- Terminal win/loss RL can improve deck-specific selection after BC, but may still overfit the
  Arena distribution or destroy useful BC behavior if updates are too broad.

## 3. What is universal and what is deck-specific

| Layer of knowledge | Default owner | Examples |
|---|---|---|
| Card identity and text-derived capability | Shared Encoder | evolution relation, search, draw, damage placement, switching |
| Board and resource state | Shared Encoder | zones, damage, Energy, Prize, deck pressure, known hand boundary |
| Rule and temporal semantics | Shared Encoder | evolution clock, turn budgets, attack ends turn |
| Legal-option semantics | Shared option Encoder | source, target, action type, selected-card relation |
| Exact registered construction | Shared conditioning path | the complete 60-card multiset, multiplicities, cards absent from deck |
| Expert/source style | Source conditioning path | conflicting labels from different teams or agents |
| Deck game plan | Deck-specific Adapter | preferred setup line, attacker chain, damage-routing plan |
| Ordered full-action choice | Deck-specific Decoder | which legal options to choose, their order and STOP timing |
| Arena calibration | Deck-specific RL | choices that increase terminal win rate against the target pool |

The boundary is deliberately permeable through a controlled unfreezing ladder. It is not correct to
assume that everything before the final Decoder is universal. The project must measure where
deck-specific information becomes necessary.

## 4. Mandatory conditioning contract

Different decks and different experts may choose different actions from visually similar boards.
They must never be merged into an unconditional BC target distribution.

Every training decision must carry:

1. `registered_card_ids`, `registered_multiplicity` and `registered_mask`, representing the exact
   registered 60-card multiset;
2. a stable `deck_id` derived from the canonical exact-deck hash;
3. a coarser `archetype_id` for grouped reporting, never as a substitute for the exact deck;
4. a stable `source_id` for the team/expert policy that produced the action;
5. Episode ID, player index, outcome, rating evidence, data snapshot and payload hash;
6. a whole-Episode train/validation split assignment.

The exact deck must condition both the global state and every legal option representation. This is
what lets the policy distinguish, for example, a pure Dragapult construction from a Dragapult +
Dusknoir construction even when the current public board happens to look the same. `source_id`
handles residual differences in expert style; it must have an explicit neutral deployment value and
must not leak the target action.

Dataset manifests must report decisions and Episodes by exact deck, archetype, source, rating band,
outcome and split. Conflicting-label audits must group identical actor-visible state signatures and
show action distributions by deck and source.

## 5. Universal BC data policy

The first practical corpus should use naturally available official data rather than artificial
50/50 balancing:

- prefer complete winning trajectories from high-rating players or explicitly approved experts;
- use every validated useful Episode available within each approved source/deck slice;
- keep naturally occurring frequency as the default, but prevent a very common archetype from
  numerically erasing rare decks through capped sampling or per-archetype loss normalization;
- do not silently duplicate rare Episodes to manufacture evidence;
- retain loss-side trajectories in the raw audit so they can later support value learning or
  controlled BC ablations, but do not mix them into the first winner-perspective BC objective;
- fail closed on missing exact deck, ambiguous player identity, incomplete action sequence,
  feature-schema mismatch or official-runtime incompatibility.

Data quality has two independent axes: player/action quality and state coverage. Rating filters help
the first; many decks, matchups, seats and game phases help the second. Winner-only filtering reduces
some noise but also biases the state distribution toward successful games. The bias must be measured,
not treated as a universal truth.

## 6. Model contract

### 6.1 Shared Encoder

The initial implementation should inherit the strongest audited 0014/0016 feature contract:

```text
observation
  -> card/entity embeddings and relation-aware board encoder
  -> exact registered-deck reader and causal resource ledger
  -> public event memory and known/unknown hand boundary
  -> turn/resource/global encoder
  -> legal-option semantic encoder
  -> goal/scenario reasoning blocks
  -> shared state representation + shared option representations
```

The shared representation must remain causal: no future action, final result, hidden opponent card or
post-decision event may enter actor-visible tensors. The terminal outcome is training metadata, not
an inference feature.

### 6.2 Deck-specific Adapter

The preferred Adapter is a small residual modulation inserted after shared scenario reasoning and
before the autoregressive action Decoder. It consumes the exact-deck representation and optionally
the neutral/source persona representation:

```text
shared_state, shared_options, deck_context
  -> LayerNorm
  -> bottleneck down projection
  -> activation
  -> up projection initialized near zero
  -> gated residual into state and options
```

The first Adapter budget should be approximately 1-5% of the shared actor. Near-zero initialization
must preserve the pretrained policy at step zero. Separate Adapter weights are maintained per target
deck or exact-deck family; the shared Encoder checkpoint is immutable and content-addressed.

### 6.3 Deck-specific Decoder

The Decoder remains the ordered legal-option pointer:

- query/key projections over state and current legal options;
- autoregressive recurrent state over already selected options;
- legal and not-yet-selected masking;
- explicit STOP after `minCount` and forced completion at `maxCount`;
- identical token log-probability contract for BC, sampled rollout and PPO replay.

The Decoder is the narrowest specialization point and therefore the first RL update target. Adapter
and deeper blocks are opened only when frozen evaluation shows that Decoder-only learning has
plateaued rather than merely fluctuated.

## 7. Training stages

### Stage 1: Universal BC pre-training

Train the shared Encoder, conditioning paths and a general Decoder on the approved multi-deck,
multi-source winner-perspective corpus. Optimize full-action token cross-entropy. Every epoch keeps
the repository BC contract: one train update pass and one complete fixed-model validation pass with
teacher-forced and greedy full-action metrics.

Primary diagnostics are validation loss, token accuracy, exact action, legal action, per-deck and
per-source metrics, conflict buckets and rare-action coverage. They measure imitation and data
health, not playing strength.

### Stage 2: Deck-specific BC adaptation

For a target exact deck, initialize from the universal checkpoint, freeze the shared Encoder and
train a fresh Adapter + Decoder on all approved target-deck demonstrations. Compare against:

- training the same target from a fresh seed;
- fine-tuning the entire universal model;
- Decoder-only adaptation;
- Adapter + Decoder adaptation.

This stage establishes whether the universal representation reduces target data needs and whether
the Adapter carries meaningful deck-specific strategy.

### Stage 3: Deck-specific terminal-reward RL

Start from the best official-engine BC-adapted candidate. Use complete official-engine Episodes and
only terminal reward: win `+1`, loss `-1`, draw `0`. Initially update the value head and Decoder,
while keeping the shared Encoder fixed and maintaining a frozen-BC reference policy. Expand the
trainable boundary only in a new immutable experiment version.

### Stage 4: serving and scalable self-play infrastructure

Run official engines in isolated CPU workers. Route policy requests to a centralized GPU inference
service that batches by compatible model family and `policy_id`. Keep one shared Encoder resident
and load multiple small Adapter/Decoder states rather than launching one CUDA context per opponent.
Rule-based opponents may remain on CPU.

## 8. Controlled unfreezing ladder

| Level | Trainable actor modules | Purpose |
|---|---|---|
| L0 | Decoder only | safest terminal-reward calibration |
| L1 | Decoder + deck Adapter | learn exact-construction game plan |
| L2 | L1 + option ScaleGate/FiLM | change how scenario evidence modulates choices |
| L3 | L2 + option scenario attention / goal router | revise task-specific option reasoning |
| L4 | L3 + final scenario Encoder block | adapt higher-level state abstraction |
| L5 | full actor | last resort with strongest BC/KL protection |

Movement to the next level requires a plateau on periodic frozen greedy official-engine evaluation,
stable legality, bounded KL/reference drift and no material regression across the Arena catalog.
Training-rollout win rate alone cannot open a deeper level.

## 9. The minimum transfer experiment

The first convincing test should be small enough to run tomorrow and strong enough to falsify the
claim.

### Pre-training comparison

1. `Single-Deck`: train only Alakazam demonstrations.
2. `Universal-BC`: jointly train approved Alakazam and Azumarill demonstrations with exact-deck and
   source conditioning.
3. Keep feature schema, parameter budget, optimizer contract, train decisions seen and target-deck
   validation protocol comparable.

### Transfer comparison

Freeze each Encoder and adapt an equal-sized fresh Decoder or Adapter + Decoder to the same held-out
target deck using identical target decisions. Evaluate data-efficiency curves at several natural
target-data checkpoints rather than claiming success from one final point.

### Strength comparison

Evaluate frozen greedy policies through real official-engine games against the complete frozen Arena
catalog, balanced by opponent and seat. Record the exact catalog/package snapshot. The primary
strength metric is all-pool average win rate with W/L/D counts and Wilson intervals; per-opponent and
per-archetype results diagnose transfer. There is no special six-opponent holdout: Arena is the
competition target distribution.

The hypothesis is supported only if Universal-BC improves either target data efficiency or final
official-engine strength without merely shifting performance to one overrepresented matchup.

## 10. Ablations that answer real design questions

Run these only after a stable baseline exists:

| Ablation | Question answered |
|---|---|
| remove exact-deck conditioning | Does the model collapse conflicting deck plans? |
| remove source conditioning | Are expert-policy conflicts materially harmful? |
| Decoder-only vs Adapter + Decoder | Is final action calibration sufficient? |
| freeze vs unfreeze final scenario block | Is the current Encoder boundary too restrictive? |
| winner-only vs controlled loss-side inclusion | Does broader state coverage outweigh action noise? |
| natural sampling vs capped dominant archetype | Is common-deck frequency erasing rare-deck learning? |
| shared Encoder vs fresh target model | Is transfer real rather than extra compute or parameters? |

Each ablation gets its own strictly increasing project version. Failed or interrupted runs remain
auditable and are never overwritten.

## 11. Metrics and promotion contract

### BC metrics

- overall and per-deck/source validation loss, token accuracy and exact full-action accuracy;
- legal action rate, action-length accuracy and STOP/count validity;
- registered-deck/source conditioning sensitivity;
- conflicting-state action entropy and rare-action coverage.

### RL metrics

- rolling sampled-policy W/L/D over 100, 500 and 2,000 valid Episodes;
- cumulative and seat-balanced win rate by opponent and archetype;
- PPO policy/value loss, entropy, behavior/reference KL, clip fraction and gradient/update norm;
- Episode and decision throughput, inference batching, worker errors and GPU utilization;
- periodic frozen greedy full-Arena win rate under a fixed snapshot.

### Promotion rule

Offline BC agreement never promotes a policy by itself. Sampled rollout curves never promote a
policy by themselves. A candidate must be self-contained, package-validated and stronger in a
controlled official-engine Arena comparison. Admission into `evaluation/arena/opponents/` still
requires explicit user confirmation.

## 12. Failure modes and defenses

| Failure mode | Required defense |
|---|---|
| dominant deck overwhelms rare decks | capped sampler or normalized loss plus per-deck metrics |
| conflicting experts create averaged behavior | explicit `source_id` and conflict audit |
| same board needs different plans | exact registered-deck conditioning and Adapter |
| Encoder learns deck shortcuts instead of rules | held-out-deck adaptation and conditioning ablations |
| winner-only corpus misses recovery states | measure coverage; later controlled loss-side experiment |
| RL overfits Arena packages | retain exact pool snapshot; inspect per-opponent movement and broaden pool |
| RL destroys BC knowledge | frozen reference KL, conservative LR and staged unfreezing |
| rollout curve looks better through sampling noise | periodic frozen greedy full-pool evaluation |
| opponent inference starves GPU training | centralized batched GPU inference service |

## 13. Immediate bridge to 0018 Alakazam RL

0018 is the next deck-specific RL experiment, not the Universal-BC proof itself. It should reuse the
validated 0017 terminal-reward algorithm as a self-contained copy and initialize the actor from the
0016 R15 epoch-10 validation-loss-best Alakazam checkpoint. Its first branch should be L0
Decoder-only PPO against the complete current Arena pool, with no dedicated holdout subset.

The purpose of 0018 is to test whether the 0017 mechanism transfers from a low-data Dragapult policy
to a stronger Alakazam BC policy. It must compare the exact frozen pre-RL actor with selected RL
checkpoints using the same official engine, full opponent snapshot, games per opponent and balanced
seat schedule. A smaller gain than 0017 is expected because the starting Alakazam policy is already
stronger; lack of rapid gain is not by itself evidence that RL failed.

If L0 plateaus, 0018 should branch in this order: L1 Adapter + Decoder, then L2 option modulation.
It should not jump directly to full-actor PPO. This sequence both protects the useful BC
representation and produces the evidence needed to decide how the future Universal BC checkpoint
should expose its specialization boundary.

## 14. Artifact boundary

The future pre-training project and every deck-specific adaptation/RL project remain separate,
self-contained numbered implementation units. They may share `rl_environment/`, `evaluation/`,
official data and the official engine, but cannot import executable code from another numbered
training project. Cross-project checkpoints are immutable provenance inputs identified by SHA-256.

The shared Encoder, Adapter, Decoder, source/deck vocabularies and feature schema each need explicit
versioned manifests. A deployable package must contain everything needed for inference with one
exact 60-card deck and must not depend on another project directory at runtime.

## 15. Open decisions after the first transfer result

- Whether the universal unit should end before or after the scenario blocks.
- Whether one Adapter should represent an exact deck, an archetype family or a learned mixture.
- Whether source conditioning remains useful after deck-specific RL.
- Whether a loss-side auxiliary value objective improves representation without degrading BC.
- How much dominant-archetype capping is needed once the corpus expands substantially.
- Whether shared-Encoder GPU serving yields enough throughput to replace independently loaded
  opponent packages during RL.

These decisions are intentionally deferred until the minimum transfer experiment and 0018 provide
real official-engine evidence.
