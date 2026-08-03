# Model Reading Order

Read the 0028 policy in this order:

1. `config.py`: capacity and finite identity limits.
2. `prototype_encoder.py`: official Card, Attack, Skill, and Effect facts.
3. `state_encoder.py`: global/card/resource/event tokens and full state memory.
4. `option_encoder.py`: option facts, explicit relations, and cross-attention to state.
5. `action_decoder.py`: ordered legal-option pointer and STOP.
6. `policy.py`: short top-level forward assembly.

The core forward is intentionally visible in `SemanticPolicy.forward`:

```python
batch, state, options = self.encode(batch)
decoder_state = self.action_decoder.initialize(batch, state.summary)
decoder_state = self.action_decoder.consume_prefix(options, decoder_state, selected_prefix)
return self.decode_next(batch, options, decoder_state)
```

No feature inference, replay parsing, source conditioning, or checkpoint compatibility logic belongs
in these modules.
