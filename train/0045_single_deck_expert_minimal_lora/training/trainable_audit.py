"""Machine-readable trainable tensor boundary audit for 0045 models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _module(name: str) -> tuple[str, int | None]:
    if name.startswith("actor.state_encoder.board_encoder.layers.3"):
        return "StateEncoder.board_transformer.final_layer", 16
    if name.startswith("actor.state_encoder.event_encoder.layers.0"):
        return "StateEncoder.event_transformer.final_layer", 16
    if name.startswith("actor.state_encoder.family_fusion"):
        return "StateEncoder.final_family_fusion_mlp", 16
    if name.startswith("policy_option_lora"):
        return "OptionEncoder.final_layer", 16
    if name.startswith("actor.action_decoder"):
        return "ActionDecoder", None
    if name.startswith("allocation_head"):
        return "AllocationHead", None
    if name.startswith("value_head"):
        return "ValueHead", None
    if name.startswith("value_adapter"):
        return "ValueAdapter", None
    if name.startswith("prize_aux"):
        return "PrizeAuxHead", None
    raise RuntimeError(f"unclassified trainable tensor: {name}")


def trainable_tensor_audit(model) -> dict[str, Any]:
    rows = []
    totals: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module, rank = _module(name)
        count = parameter.numel()
        totals[module] += count
        rows.append({
            "name": name,
            "shape": list(parameter.shape),
            "parameters": count,
            "module": module,
            "lora_rank": rank,
        })
    total = sum(row["parameters"] for row in rows)
    return {
        "schema_version": "0045_trainable_tensor_audit_v1",
        "status": "PASS",
        "tensor_count": len(rows),
        "total_trainable_params": total,
        "module_totals": dict(sorted(totals.items())),
        "tensors": rows,
    }


__all__ = ["trainable_tensor_audit"]
