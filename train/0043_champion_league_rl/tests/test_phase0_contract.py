from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/0043_champion_league_rl"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase0_snapshot_freezes_the_approved_training_contract() -> None:
    config_path = EXPERIMENT / "active_training_config.json"
    manifest = _json(EXPERIMENT / "manifest.json")
    config = _json(config_path)

    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == manifest[
        "active_training_config_sha256"
    ]
    assert config["games_per_update"] == 256
    assert config["rollout_batch_size"] == 256
    assert config["trajectory_games_per_update"] == 256
    assert config["ppo_epochs"] == 3
    assert config["ppo_minibatch_size"] == 2048
    assert config["ppo"]["decoder_learning_rate"] == 2e-5
    assert config["ppo"]["policy_adapter_learning_rate"] == 4e-5
    assert config["ppo"]["allocation_learning_rate"] == 2e-5
    assert config["ppo"]["value_learning_rate"] == 1e-4
    assert config["ppo"]["prize_learning_rate"] == 1e-4
    assert config["ppo"]["gae_lambda"] == 0.95
    assert config["ppo"]["reference_kl_coefficient"] == 0.02
    assert config["trainable_contract"] == manifest["approved_trainable_contract"]


def test_phase0_documents_exist_in_both_authoritative_formats() -> None:
    assert (EXPERIMENT / "PHASE0_AUDIT.md").is_file()
    assert (EXPERIMENT / "DESIGN.md").is_file()
    assert (EXPERIMENT / "DESIGN.html").is_file()
    assert (EXPERIMENT / "DECISIONS.md").is_file()


def test_experiment_manifest_hashes_every_published_registry() -> None:
    manifest = _json(EXPERIMENT / "manifest.json")
    for identity in manifest["asset_registries"].values():
        path = ROOT / identity["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
    assert manifest["formal_training_authorized"] is False
    assert manifest["status"] == "BLOCKED_FORMAL_TRAINING_PREFLIGHT"
