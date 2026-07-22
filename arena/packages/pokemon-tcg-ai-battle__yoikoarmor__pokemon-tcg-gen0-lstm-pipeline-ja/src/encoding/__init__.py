"""State / action encoding for the Ver2 neural agent.

`encode(obs)` turns an `Observation` into:
  - a fixed-length global feature vector (board summary, both players),
  - a per-option feature matrix (one row per legal option in `select.option`),
  - a legality mask.

The option list returned by the engine *is* the legal action set, so the mask
simply marks real vs padded slots; the network never scores an illegal action.
"""
from .features import (
    encode,
    global_dim,
    option_dim,
    MAX_OPTIONS,
    EncodedDecision,
)

__all__ = ["encode", "global_dim", "option_dim", "MAX_OPTIONS", "EncodedDecision"]
