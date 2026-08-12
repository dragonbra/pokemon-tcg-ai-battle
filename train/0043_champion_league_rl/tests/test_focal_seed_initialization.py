import hashlib
import importlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
assets = importlib.import_module("train.0043_champion_league_rl.assets")
taxonomy = importlib.import_module("train.0043_champion_league_rl.own_archetype")


def test_g1_is_unchanged_and_g2_rows_are_parent_copies():
    g1_root = ROOT / "assets/policies/definitions/champion_g001"
    assert assets.sha256_file(g1_root / "source_update_000010.pt") == "aea408e8e140ceb28d23962cf45abac76a29be01888953bbbc5253ff6d120fe3"
    assert assets.sha256_file(g1_root / "model.bin") == "cd5c05741aadf16db8bb8ca4f527e02eae431db75c33eba30f291fe89f57de84"
    v2 = taxonomy.OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    g1 = torch.load(g1_root / "source_update_000010.pt", map_location="cpu", weights_only=True)
    g2 = torch.load(ROOT.parents[1] / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/checkpoint/update-000000.pt", map_location="cpu", weights_only=True)
    keys = ("value_adapter.own_embedding.weight", "policy_strategy_adapter.own_embedding.weight")
    for key in keys:
        old, new = g1["state_dict"][key], g2["state_dict"][key]
        assert old.shape == (15, 16) and new.shape == (29, 16)
        assert torch.equal(new[:15], old)
        for row in v2.classes[15:]:
            assert torch.equal(new[row.archetype_id], old[row.embedding_init_from])


def test_no_unrelated_fp32_tensor_changed_and_no_optimizer_state():
    g1 = torch.load(ROOT / "assets/policies/definitions/champion_g001/source_update_000010.pt", map_location="cpu", weights_only=True)
    g2 = torch.load(ROOT.parents[1] / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/checkpoint/update-000000.pt", map_location="cpu", weights_only=True)
    embeddings = {"value_adapter.own_embedding.weight", "policy_strategy_adapter.own_embedding.weight"}
    assert set(g1["state_dict"]) == set(g2["state_dict"])
    for name in set(g1["state_dict"]) - embeddings:
        assert torch.equal(g1["state_dict"][name], g2["state_dict"][name]), name
    forbidden = {"optimizer", "optimizer_state", "scheduler", "rng_state", "grad_scaler"}
    assert forbidden.isdisjoint(g2)
    assert g2["metadata"]["optimizer_state_migrated"] is False


def test_portable_parent_copy_and_manifest_identity():
    v2 = taxonomy.OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    g1 = torch.load(ROOT / "assets/policies/definitions/champion_g001/model.bin", map_location="cpu", weights_only=True)
    g2 = torch.load(ROOT.parents[1] / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/artifact/focal_seed/model.bin", map_location="cpu", weights_only=True)
    for field in ("value_adapter_state_dict", "policy_strategy_adapter_state_dict"):
        old, new = g1[field]["own_embedding.weight"], g2[field]["own_embedding.weight"]
        assert old.dtype == new.dtype == torch.float16
        assert torch.equal(new[:15], old)
        for row in v2.classes[15:]:
            assert torch.equal(new[row.archetype_id], old[row.embedding_init_from])
    manifest = json.loads((ROOT.parents[1] / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/artifact/focal_seed/manifest.json").read_text())
    assert manifest["parent_policy_id"] == "Champion-G1"
    assert manifest["taxonomy"]["class_count"] == 29
    assert manifest["optimizer"]["state_migrated"] is False
