import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runtime = importlib.import_module("train.0043_champion_league_rl.runtime")


def test_v1_and_v2_runtime_strict_load():
    g1 = runtime.load_policy("Champion-G1", deck_id="007")
    g2_old = runtime.load_focal_seed(deck_id="007")
    g2_new = runtime.load_focal_seed(deck_id="043")
    g2_user_dusknoir = runtime.load_focal_seed(deck_id="066")
    g2_user_normal = runtime.load_focal_seed(deck_id="067")
    assert g1.value_adapter.own_embedding.num_embeddings == 15
    assert g2_old.value_adapter.own_embedding.num_embeddings == 29
    assert g2_new.policy_strategy_adapter.own_embedding.num_embeddings == 29
    assert g1.metadata["own_archetype_id"] == g2_old.metadata["own_archetype_id"] == 0
    assert g2_new.metadata["own_archetype_id"] == 15
    assert g2_user_dusknoir.metadata["own_archetype_id"] == 15
    assert g2_user_normal.metadata["own_archetype_id"] == 0


def test_focal_seed_is_not_a_registered_policy_asset():
    registry = json.loads((ROOT / "assets/policies/registry.json").read_text())
    assert registry["latest_champion_policy_id"] == "Champion-G1"
    assert registry["active_policy_pool"] == ["Policy-0809", "Champion-G1"]
    assert {row["policy_id"] for row in registry["policies"]} == {
        "Policy-0809", "Champion-G1",
    }
