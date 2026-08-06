# 0033 Effect-Summary Semantic Foundation Prototype

Status: **model-structure prototype verified locally; no 0033 data materialization or training claimed**.

0033 is a model-side extension of 0032. It keeps the audited actor batch contract and
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
- PokéChamp takeaway: strong Pokémon agents use history and human knowledge to reduce action
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

0033 adds three model-side prototype encoding layers:

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

## What 0033 still does not encode

0033 still avoids dynamic rule-answer leakage:

- no attach-provides counterfactual
- no attack/retreat deficit before/after
- no newly-enabled attack answer
- no post-damage HP or KO answer
- no runtime trigger subject/object when it is not actor-visible

The effect summary is static prototype information. It tells the model the structure of a card's
possible effects, but it does not run the effect against the current board.

## Verification

Local checks run in D:\w33:

- python -m compileall -q train/0033_effect_summary_semantic_foundation_pretraining
- python -m pytest train/0033_effect_summary_semantic_foundation_pretraining/tests -q

Result: **60 passed**.

Additional 0033-specific tests prove:

- effect slots cover the full-engine attack/skill asset
- attack effect categorical fields change embeddings
- skill effect categorical and numeric fields change embeddings
- shared type embeddings change card and attack encodings
- card-type-specific projections change card encodings

Default parameter count: **22,595,202**.

## Next gate before training

Before pushing any Kaggle 0033 training:

1. Decide whether 0033 should reuse the accepted 0032 materialized dynamic records or regenerate
   records with a new actor schema label.
2. Validate one real episode end-to-end with the 0033 model payload and exact DecisionBatch.
3. Add a model contract field for effect_summary_semantic_decision_v1.
4. Run a short BC smoke train before launching a full dual-T4 run.
