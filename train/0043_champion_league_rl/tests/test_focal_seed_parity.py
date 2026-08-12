import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parents[1] / "experiments/0043_champion_league_rl"


def test_recorded_zero_step_parity_is_complete():
    report = json.loads((EXPERIMENT / "focal_seed_zero_step_parity.json").read_text())
    assert report["status"] == "PASS"
    assert report["old_decks_tested"] == 55
    assert len(report["rows"]) == 15
    assert len(report["old_deck_resolution"]) == 55
    for field in (
        "policy_logits", "action_probabilities", "greedy_action", "value",
        "policy_adapter_output", "value_adapter_output",
    ):
        assert report[field] == "EXACT"
    assert report["fresh_optimizer"] == {
        "optimizer_state_migrated": False,
        "opponent_meta_excluded": True,
        "own_embeddings_included": True,
        "state_entries": 0,
    }


def test_active_runtime_has_no_fixed_own_embedding_constructor():
    source = (ROOT / "semantic_runtime/deployment/compound_inference.py").read_text()
    assert "nn.Embedding(15" not in source
    assert "own_archetype_classes: int = 15" not in source
