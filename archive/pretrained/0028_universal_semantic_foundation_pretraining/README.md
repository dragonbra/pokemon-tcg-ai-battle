# 0028 Universal Semantic Foundation Pretraining

This directory is the mutable current-release slot for the self-contained project 0028 pretrained
model. `manifest.json` and `checkpoint_metadata.json` are the authoritative identity of whichever
weight is currently published. After the authorized V3 run finishes, `model.pt` may be replaced by
the selected final checkpoint only together with refreshed metadata, metrics, byte counts, and
SHA-256 commitments.

## Initial Release Record

The first real publication on 2026-08-03 used the Epoch 1 `best_validation_loss` checkpoint as an
early zero-shot sample:

- Initial stage: `epoch1_sample`
- Model: `SemanticPolicy`
- Parameters: 21,837,082
- Actor input: typed state, exact-deck causal ledger, visible events, legal-option semantics, and
  finite Card/Attack/Skill/Effect prototypes
- Source/team identity: not actor-visible
- Checkpoint: model-only; no optimizer, scheduler, scaler, RNG, dataset, replay, or rollout state
- Initial validation loss: 0.4295406839
- Initial validation exact action: 74.7020%
- Initial validation legal action: 100%

Offline imitation metrics do not establish official-engine playing strength. Zero-shot testing
must use the official runtime before making policy-strength claims.

## Verify And Load

Run from this directory:

```bash
python3 verify_archive.py
```

Programmatic loading:

```python
from pathlib import Path
import sys

asset = Path("archive/pretrained/0028_universal_semantic_foundation_pretraining").resolve()
sys.path.insert(0, str(asset))

from loader import load_model
from zero_shot import ZeroShotPolicy

model = load_model("cuda")
policy = ZeroShotPolicy(deck_ids, device="cuda", model=model)
action = policy.select(observation)
```

`deck_ids` must be the registered exact 60-card deck. Call `policy.reset()` at the start of every
new game. `select()` is strict and surfaces compiler/runtime errors; `select_or_fallback()` resets
causal state and returns the first structurally required legal options when strict inference fails.

## Frozen Contents

- `model.pt`: current model-only release weight
- `model_source/`: exact inference-side 0028 contracts, prototype/domain code, feature compiler,
  causal knowledge ledger, model, and ordered decoder
- `assets/`: public and full-engine prototype registries used by this checkpoint
- `contracts/`: model, training, and dataset identity documents
- `loader.py`: strict reconstruction and checkpoint loading
- `zero_shot.py`: chronological observation-to-action adapter
- `verify_archive.py`: hash, payload, parameter-count, and forward-smoke validation

The authoritative project design remains in
`experiments/0028_universal_semantic_foundation_pretraining/DESIGN.md` and `DESIGN.html`.
