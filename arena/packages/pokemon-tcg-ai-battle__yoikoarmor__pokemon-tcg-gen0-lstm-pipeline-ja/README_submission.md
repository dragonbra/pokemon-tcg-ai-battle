# Lucario Tactical Overlay Submission - v61 Deck + ver58 Fallback Branch

Entrypoint:

```text
main.agent
```

Primary method:

```text
Generation 0 Residual LSTM Policy
```

Narrow overlay:

```text
If the opponent publicly reveals Crustle wall line or Dragapult line,
Ver64 tactical scores may add a small logit bonus to one legal action.
All other single-select decisions stay on the original Gen0 neural policy.
```

Fallback:

```text
Ver58 rule fallback for unsupported multi-select contexts.
First legal action only if every safer path fails.
```

This package does not use MCTS, Kaggle API, internet access, external drives, or GPU-only inference.

Deck branch:

```text
lucario_v61_rank2
```

This is the same Generation 0 policy and tactical overlay as the accepted v1 package,
with the Lucario deck list changed from the Hariyama-heavy v63 patch to v61,
and unsupported-context fallback changed from ver61 to ver58.
