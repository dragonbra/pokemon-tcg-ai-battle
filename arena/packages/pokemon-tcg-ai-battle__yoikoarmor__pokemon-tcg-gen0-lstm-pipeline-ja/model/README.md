# Model

`policy.pt` is the Generation 0 Residual LSTM checkpoint:

```text
checkpoints_selfplay/generation0_h384_value_head_only_20k_best_mse_seed101.pt
```

The policy parameters are the stable Gen0 policy. The tactical overlay is implemented in code and only adjusts selected logits for public wall/Dragapult signals.
