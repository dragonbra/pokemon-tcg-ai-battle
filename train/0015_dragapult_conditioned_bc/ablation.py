"""Safe initial-deck ablation that retains actor-visible live resource evidence."""

from __future__ import annotations

import torch
from torch import Tensor


def remove_initial_deck_information(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a copy with hidden registered composition removed.

    A row remains live only after that card identity has appeared in an actor-visible own zone.
    Initial multiplicity, hidden deck/prize bounds and ledger categorical bound states are removed.
    At states with no visible own card, one zero sentinel remains valid to prevent all-masked
    attention. Board, event, public inventory and source-persona tensors are untouched.
    """

    required = {
        "registered_card_ids",
        "registered_multiplicity",
        "registered_mask",
        "ledger_cat",
        "ledger_num",
        "ledger_mask",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"deck ablation is missing tensors: {sorted(missing)}")
    result = dict(batch)
    ids = batch["registered_card_ids"].clone()
    multiplicity = torch.zeros_like(batch["registered_multiplicity"])
    ledger_cat = batch["ledger_cat"].clone()
    ledger_num = batch["ledger_num"].clone()
    original_mask = batch["registered_mask"]

    # ledger_num columns 1:7 are current actor-visible own zones. All other count/bound fields
    # can reveal the exact registered list or multiplicity and are removed in this arm.
    visible = ledger_num[..., 1:7].abs().sum(-1).gt(0) & original_mask
    safe_mask = visible.clone()
    empty = ~safe_mask.any(1)
    if safe_mask.size(1) < 1:
        raise ValueError("registered resource width must be at least one")
    safe_mask[empty, 0] = True

    ids[~visible] = 0
    ledger_cat[~visible] = 0
    # Preserve the observed card identity for visible rows but remove bound-state categories.
    ledger_cat[..., 1:] = 0
    kept_live = torch.zeros_like(ledger_num)
    kept_live[..., 1:7] = ledger_num[..., 1:7]
    kept_live[..., 13:15] = ledger_num[..., 13:15]
    kept_live[~visible] = 0

    result.update(
        {
            "registered_card_ids": ids,
            "registered_multiplicity": multiplicity,
            "registered_mask": safe_mask,
            "ledger_cat": ledger_cat,
            "ledger_num": kept_live,
            "ledger_mask": safe_mask.clone(),
        }
    )
    return result


__all__ = ["remove_initial_deck_information"]
