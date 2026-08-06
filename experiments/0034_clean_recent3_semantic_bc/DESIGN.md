# 0034 Effect-Summary Semantic Foundation Prototype

Status: **recent-three-day cleaned dataset accepted; dual-T4 training and fixed six-opponent evaluation complete**.

0034 is a model-side extension of 0032. It keeps the audited actor batch contract and
legal-option decoder, but makes card prototype vectors more separable before the decoder sees them.
The goal is to express "what this card can do" without changing per-card tensor width or leaking
rule-execution answers.

## Research inputs

- Local paper reviewed: research/papers/PTCG-Bench_arXiv-2605.29653v1.pdf.
- PTCG-Bench takeaway: structured observation, legal action exposure/masking, and history/context
  management are harness-level variables that materially affect gameplay strength. This supports
  keeping 0032's legal-option pointer decoder and structured actor facts.
- RLCard takeaway: card-game environments should expose both encoded observation and legal actions,
  with raw observation/action available for faithful game semantics.
- Pok茅Champ takeaway: strong Pok茅mon agents use history and human knowledge to reduce action
  sampling/search burden under partial observability.
- Hearthstone agent work takeaway: collectible-card-game agents benefit from compact but complete
  state summaries over all elements currently in play, not only one-card identities.

## Design change from 0032

0032 already keeps:

- card type, Pokemon type, evolution type, evolves-from links
- HP, retreat, weak/resist masks, Energy type mask and engine Energy count
- card-to-attack and card-to-skill relations
- attack cost slots, base damage, attack rule bits
- skill type, area mask, skill flags, and static trigger subject rules
- dynamic card/event/option relations, including card_parent, option_source, and option_target

0034 adds three model-side prototype encoding layers:

1. **Shared type space**
   - A shared type_identity table covers Colorless plus the nine primitive Energy/Pokemon types.
   - Pokemon type, Energy-providing mask, Weakness, Resistance, and attack Energy costs project
     through this shared space with role-specific projections.
   - This makes "Fire Pokemon", "Fire Energy", "Fire attack cost", and "Fire weakness" related
     but not collapsed.

2. **Card-type-specific projection**
   - Card prototype vectors pass through a projection selected by card_type.
   - Pokemon, Trainer, Tool/Stadium/Supporter-like cards, and Energy cards still share the same
     output width, but their compressed vectors can occupy different regions before state encoding.

3. **Static effect summary slots**
   - Full-engine attack pre_effects/post_effects and skill effects are encoded directly from
     structured engine metadata, not from card text keywords.
   - EFFECT_SLOTS=12 covers the current full-engine asset without truncation: attacks max at 7
     effects and skills max at 11 effects.
   - Each effect slot contains:
     - effect_type, select_type, select_context
     - target player, not_me, skip_enemy_target
     - 25-bit target-area mask
     - condition_type, comparator_type
     - selection/control flags such as enemy/random/each-selected, keep-target-list,
       coin-head count, active/bench effect target, seeing-deck, separator, and fail-skip
     - numeric select_count, loop_count, priority, values[0], values[1]

## What 0034 still does not encode

0034 still avoids dynamic rule-answer leakage:

- no attach-provides counterfactual
- no attack/retreat deficit before/after
- no newly-enabled attack answer
- no post-damage HP or KO answer
- no runtime trigger subject/object when it is not actor-visible

The effect summary is static prototype information. It tells the model the structure of a card's
possible effects, but it does not run the effect against the current board.

## Dataset cleaning

Kaggle cleaning kernel `horizen12/ptcg-0034-01-clean-recent3-data` version 5 produced the accepted
2026-08-01 through 2026-08-03 winner-only corpus. Full local validation accepted 1,204,592 decisions:
1,081,527 train and 123,065 validation. Each shard byte count and SHA-256, day commitment, split,
sample weight, external/embedded action audit, prototype asset, `DecisionBatch`, and finite model
forward passed. The minimal cleaning policy retains ordinary actions at weight 1.0, downweights
low-confidence special-Energy Attach actions to 0.95, and upweights high-confidence Attach actions
to 1.15. Compacted history intentionally contains no exact event-source or event-target relations.

Every training epoch merges all three accepted days. No epoch is assigned to only one day.

## Formal training

Kaggle kernel `horizen12/ptcg-0034-02-train-recent3-cleaned-bc` version 2 completed three exact full
train passes and three exact full validation passes. It used random initialization, weighted
cross-entropy, FP16 GradScaler, dual Tesla T4 `DataParallel`, no pretrained checkpoint, and no W&B.
The final architecture is `d_model=320`, four state layers, three option cross-attention layers,
FFN multiplier 3, eight heads, and 22,595,202 parameters.

The 90,610,211-byte best model is a model-only checkpoint with SHA-256
`6bb23f0b23c04c478c7c0926310967c73325685b0308959f147240219e3538ff`. Package-only loading into the
exact 0034 `SemanticPolicy` produced finite logits with shape `[8, 6, 8]`.

Final merged validation metrics:

- weighted loss: 0.436949
- token accuracy: 0.850293
- teacher exact action: 0.703661
- action type accuracy: 0.932346
- Attach target exact/token accuracy: 0.497538
- Attack exact/token accuracy: 0.691584

## Fixed six-opponent evaluation

The trained policy was evaluated with the Marnie deck for 50 games against each fixed BC opponent,
using CUDA inference and the committed evaluation protocol. All 300 games completed with zero
errors, zero draws, and zero fallback decisions. The aggregate result was 169 wins and 131 losses,
or 56.33%.

| Opponent | Wins | Losses | Win rate |
|---|---:|---:|---:|
| Lucario | 30 | 20 | 60% |
| Cynthia | 21 | 29 | 42% |
| Kangaskhan/Crustle | 22 | 28 | 44% |
| Marnie prize control | 22 | 28 | 44% |
| Alakazam | 28 | 22 | 56% |
| Dragapult | 46 | 4 | 92% |

These results show useful overall play strength, but the 49.75% Attach-target validation accuracy
and the losses to Cynthia, Kangaskhan/Crustle, and Marnie remain clear targets for better cleaning
and action-specific supervision. The evaluation is evidence for this exact deck/opponent protocol,
not a general metagame win-rate claim.

## Verification

Local checks run in D:\w33:

- python -m compileall -q train/0034_clean_recent3_semantic_bc
- python -m pytest train/0034_clean_recent3_semantic_bc/tests -q

Result: **60 passed**.

Additional 0034-specific tests prove:

- effect slots cover the full-engine attack/skill asset
- attack effect categorical fields change embeddings
- skill effect categorical and numeric fields change embeddings
- shared type embeddings change card and attack encodings
- card-type-specific projections change card encodings

Default parameter count: **22,595,202**. The accepted dataset, training metrics, checkpoint identity,
package validation, and arena results are committed in `manifest.json` and `evaluation/`.
