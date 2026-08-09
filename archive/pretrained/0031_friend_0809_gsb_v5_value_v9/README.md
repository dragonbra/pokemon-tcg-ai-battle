# 0031 Friend 0809 GSB V5 + Value V9

This directory is the self-contained pretrained release of both model-only checkpoints supplied in
`new-model-0809.tar.gz`. The files are preserved byte-for-byte as `model.pt` and `value_head.pt`;
neither checkpoint contains optimizer, scheduler, scaler, RNG, rollout, replay, or exact-resume
state.

`model.pt` is a 0031 exact-deck-conditioned, source-identity-invisible `SemanticPolicy`. It uses the
same 0031 architecture and implementation identity as the 0806 pretrained release, while its
embedded training identity is `V5_medal_zone_gsb_all_days`, epoch 20. The checkpoint proves a
distinct all-days dataset identity, but it does not embed the train-decision count needed to
quantify the claim that it used more training data than 0806.

`value_head.pt` is a separate 0036 latent-query critic, not an actor-visible head inside
`SemanticPolicy`. Its embedded `source_checkpoint_sha256` exactly matches this release's
`model.pt`, so the two files form one compatible frozen-encoder policy/Value pair. The critic emits
a win-probability logit plus opponent-archetype and final-Prize-difference auxiliary logits; its
PPO-style scalar is `2 * sigmoid(value_logit) - 1`.

Verify and load the release from this directory:

```bash
python3 verify_archive.py
```

```python
from loader import load_policy, load_value_network

policy = load_policy("cpu")
critic = load_value_network("cpu")
```

The policy's offline validation metrics describe imitation quality, and the critic's validation
loss describes offline return calibration. Neither is official-engine policy-strength evidence.
Any strength comparison still requires the project's frozen official-engine evaluation contract.

Release identities:

- policy: 0031 V5, epoch 20, global step 115820, validation loss `0.24142944799295105`, SHA-256
  `926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f`;
- Value: 0036 V9, epoch 9, validation Value loss `0.39974688940408915`, SHA-256
  `f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486`.
