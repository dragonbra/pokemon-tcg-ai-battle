from __future__ import annotations

import importlib


runtime = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.runtime")


def test_runtime_tree_and_both_policy_loaders_execute() -> None:
    assert len(runtime.audit_runtime_tree()) == 64
    anchor = runtime.forward_parity("Policy-0809", deck_id="001", gpu=False)
    champion = runtime.forward_parity("Champion-G1", deck_id="048", gpu=False)
    g2 = runtime.forward_parity("Champion-G2", deck_id="048", gpu=False)
    assert anchor.cpu_shape == (2, 6)
    assert champion.cpu_shape == (2,)
    assert g2.cpu_shape == (2,)


def test_g2_uses_exact_new_own_class_while_g1_uses_parent_row() -> None:
    g1 = runtime.load_policy("Champion-G1", deck_id="057")
    g2 = runtime.load_policy("Champion-G2", deck_id="057")
    assert g1.metadata["own_archetype_id"] == 0
    assert g2.metadata["own_archetype_id"] == 15
    assert g1.value_adapter.own_embedding.num_embeddings == 15
    assert g2.value_adapter.own_embedding.num_embeddings == 29


def test_synthetic_batch_excludes_source_identity() -> None:
    batch = runtime.synthetic_batch()
    assert "source_id" not in batch
    assert "team_name" not in batch
