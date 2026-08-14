from __future__ import annotations

import importlib

import torch


def _module():
    return importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.policy.strategy_adapters"
    )


def test_meta_actor_residual_is_small_zero_delta_and_meta_specific() -> None:
    module = _module()
    residual = module.MetaActorResidual(width=320, rank=4, archetype_classes=29)
    assert sum(value.numel() for value in residual.parameters()) == 74_240
    hidden = torch.randn(3, 320)
    meta = torch.tensor([2, 2, 7])
    output, delta = residual(hidden, meta)
    assert torch.equal(output, hidden)
    assert torch.count_nonzero(delta) == 0

    with torch.no_grad():
        residual.up[2].fill_(0.1)
    changed, _ = residual(hidden, meta)
    assert not torch.equal(changed[:2], hidden[:2])
    assert torch.equal(changed[2], hidden[2])


def test_meta_actor_residual_rejects_invalid_ids() -> None:
    module = _module()
    residual = module.MetaActorResidual(width=320, rank=4, archetype_classes=29)
    hidden = torch.randn(2, 320)
    for ids in (torch.tensor([0]), torch.tensor([0, 29])):
        try:
            residual(hidden, ids)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Meta Actor Residual IDs were accepted")
