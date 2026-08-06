# 0031 Friend 0806 Epoch 11 Best Validation Loss

This directory is the self-contained pretrained release of the model-only checkpoint supplied in
`large-model-0806.tar.gz`. The checkpoint is preserved byte-for-byte as `model.pt`; it is not an
exact-resume asset and contains no optimizer, scheduler, scaler, RNG, rollout, or replay state.

The model is an exact-deck-conditioned, source-identity-invisible `SemanticPolicy`. Its offline
validation metrics describe imitation quality only. Policy strength must be measured with the
unmodified official engine and a recorded opponent snapshot.

Verify and load it from this directory:

```bash
python3 verify_archive.py
```

```python
from loader import load_model

model = load_model("cpu")
```

Release identity: epoch 11, global step 141878, best validation loss 0.25827361053759834,
checkpoint SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`.
