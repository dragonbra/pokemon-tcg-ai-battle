# 0042 Decisions

## 2026-08-11: canonical model boundary

- Freeze PrototypeEncoder, StateEncoder, and OptionEncoder.
- Remove 0040 Option Q/V LoRA and local Option LayerNorm tuning from the 0042 runtime.
- Keep the pretrained q1 15-class Meta head frozen while retaining its forward autograd graph.
- Add a q0 Value Adapter and a decoder readout-only Policy Strategy Adapter, both scalar
  zero-gated with normally initialized residual MLPs.
- Detach q1, Meta logits, and adapted V at the Value-to-Policy boundary.
- Use separate `E_V_own` and `E_pi_own` embedding tables.
- Treat registered own archetype and opponent Meta target as different semantic types even while
  both use the initial 15-class taxonomy.
- Preserve PPO, GAE, reward, action-boundary, deployment precision, opponent, and Frozen protocol
  semantics.
- Use the exact 55 numbered decks from `0806_kaggle_top100_plus_v1`; do not retain the copied 0040
  named deck directory layout.
- Do not launch training or formal evaluation as part of architecture implementation.

## Evidence

- `train/0042_full_model_design/tests/test_strategy_architecture.py`
- `.tmp/strategy_adapter_v2_audit/0042_runtime_audit.json`
- `.tmp/strategy_adapter_v2_audit/0042_value_meta_baseline.json`
- `docs/rl/0042_full_model_design.md`
