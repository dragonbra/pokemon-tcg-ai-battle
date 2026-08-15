# Policy-0814 · 0031 Friend GSB V6 + Value V10

This directory is the self-contained internal pretrained release built from `n-m-0814.zip`.
The supplied model-only files are preserved byte-for-byte as `model.pt` and `value_head.pt`.
Neither file contains optimizer, scheduler, scaler, RNG, rollout, replay, or exact-resume state.

`model.pt` is immutable actor identity `Policy-0814`: a 0031 exact-deck-conditioned,
source-identity-invisible `SemanticPolicy`. Its state keys, tensor shapes, dtypes, model config,
action contract, and 56,352,322-parameter instantiated architecture exactly match the internal
`Policy-0809` release. This proves architecture compatibility, not equal weights or equal strength.

`value_head.pt` is the paired 0036 latent-query critic with 3,407,389 trainable Value parameters.
Its embedded `source_checkpoint_sha256` exactly matches this release's `model.pt`. It emits a win
logit plus 15-way opponent-archetype and 13-way final-Prize-difference auxiliary logits; its
PPO-style scalar is `2 * sigmoid(value_logit) - 1`. The critic is a companion artifact and is not
part of the actor's logits or actor effective-policy hash.

Verify and load the release from this directory:

```bash
python3 verify_archive.py
```

```python
from loader import load_policy, load_value_network

policy = load_policy("cpu")
critic = load_value_network("cpu")
```

Release identities:

- actor: `Policy-0814`, 0031 V6 epoch 23, global step 143198, validation loss
  `0.2500847065524083`, SHA-256
  `d7921f420c8f12155119d6caa0fef414f51c0a8368cd5ecf07cd7d71312c897b`;
- Value: 0036 V10 epoch 13, validation Value loss `0.39862922933317185`, SHA-256
  `0ad6f57a32d37942cccdc78a8a9c8ef2f6f8784d08ea8ed77ca1513628c77d2d`;
- actor effective identity:
  `476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580`.

The release is an internal immutable source asset. A numbered RL project must copy/register it in
that project's self-contained resolver and pass the canonical identity gates before using it as a
focal policy, opponent, or Frozen benchmark. This publication does not modify or supersede
`Policy-0809`, and it does not promote `Policy-0814` into any existing opponent pool.

Offline validation metrics describe imitation quality and Value calibration only. Policy strength
still requires a same-contract official-engine evaluation.
