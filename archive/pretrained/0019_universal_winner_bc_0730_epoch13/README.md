# 0019 Universal Winner BC 0730 Epoch 13

This is the immutable, model-only foundation archive for asset `0019-0730-epoch13`.
It is the selected Epoch 13 checkpoint from `V1_universal_winner_r15`, trained on
winner-perspective decisions from the 2026-07-10 through 2026-07-28 official snapshots.

## Intended Use

Use this asset as the shared BC initialization for deck-specific decoder, adapter, or RL
fine-tuning. A downstream training run must copy or explicitly bind this archive identity in its
own manifest and must start a new version with a newly initialized optimizer. Do not append new
training metrics to the original 0019 run.

The archive is not a claim that the neutral policy is already a final universal game-playing
agent. Validation exact action is 81.2871% with the recorded source persona and 74.8431% with
neutral `source_id=0`. Official-engine evaluation remains required for policy-strength claims.

## Identity

- Asset: `0019-0730-epoch13`
- Model: `SourceConditionedR15Policy`
- Parameters: 17,756,162
- Epoch/global step: 13 / 337,194
- Weight SHA-256: `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`
- Dataset SHA-256: `141aa82ac8a162cec5e6e57970d4020024fc0c03ef408b89c4dfa538c870b724`
- Ontology SHA-256: `8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d`

`model.pt` is tracked by Git LFS. It contains model weights and identity metadata only; it does
not contain optimizer, scheduler, scaler, RNG, DataLoader, replay, or rollout-buffer state.

## Bound Contracts

The weights are strictly bound to the frozen files in this directory:

- `model_source/`: exact model and feature-compiler source copied from project 0019.
- `contracts/model_contract.json`: architecture, input keys, feature switches, and parameter count.
- `contracts/training_config.json`: exact R15 and source-persona dimensions.
- `contracts/feature_dataset_reference.json`: feature schema and the immutable 509-source mapping.
- `contracts/card_ontology.json`: card semantics used to construct persistent model buffers.
- `checkpoint_metadata.json`: original model-only checkpoint sidecar and selection metrics.

Changing entity/option limits, card ID space, feature widths, ontology, source vocabulary,
scenario readers, R15 ScaleGate, or pointer decoder breaks the strict weight contract. A changed
contract requires a new pretrained asset version; do not overwrite this directory.

## Load And Verify

From this directory, run:

```bash
python3 verify_archive.py
```

For programmatic loading, add this directory to `sys.path` and call the strict loader:

```python
from pathlib import Path
import sys

asset = Path("archive/pretrained/0019_universal_winner_bc_0730_epoch13").resolve()
sys.path.insert(0, str(asset))

from loader import load_model, neutral_source_id

model = load_model("cuda")
batch["source_id"] = neutral_source_id(batch_size=len(batch["global_cat"]), device="cuda")
```

Use source ID 0 for a new or deliberately neutral deck policy. Reusing a nonzero source ID is only
valid when it is resolved through the frozen source mapping in
`contracts/feature_dataset_reference.json`. For a new deck-specific policy, record whether the
source residual is retained, neutralized, replaced, or distilled before RL begins.

The authoritative training interpretation and evidence boundaries remain in
`experiments/0019_universal_winner_bc/DESIGN.md` and its synchronized HTML counterpart.
