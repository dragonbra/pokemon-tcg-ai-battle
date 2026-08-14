from __future__ import annotations

import importlib
import inspect
from pathlib import Path

def test_run_supports_an_explicit_single_focal_deck() -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    parameters = inspect.signature(runner.run).parameters
    assert parameters["focal_deck_id"].default == runner.FOCAL_DECK_ID
    checkpoint_parameters = inspect.signature(runner._checkpoint).parameters
    assert "focal_deck_id" in checkpoint_parameters
    jobs_parameters = inspect.signature(runner._jobs).parameters
    assert "focal_deck_id" in jobs_parameters


def test_model_loader_resolves_the_explicit_deck003_identity() -> None:
    assets = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.assets")
    policy = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.policy")
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    registry = assets.AssetRegistry.load(runner.PROJECT_ROOT)
    model, _ = policy.load_actor_critic(
        deck=runner._cards(registry, "003"), deck_id="003",
    )
    assert model.default_own_archetype_id.item() == 1


def test_candidate_actor_state_keeps_one_canonical_prototype_copy() -> None:
    candidate = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.evaluation.candidate"
    )
    state = {
        "prototype_encoder.weight": object(),
        "state_encoder.prototypes.weight": object(),
        "option_encoder.prototypes.weight": object(),
        "action_decoder.weight": object(),
    }
    compact = candidate._portable_actor_state_dict(state)
    assert set(compact) == {"prototype_encoder.weight", "action_decoder.weight"}
    expanded = candidate._expanded_actor_state_dict(compact)
    assert set(expanded) == set(state)


def test_v8_restarts_v6_u33_as_local_u0_deck003(tmp_path: Path) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v8"
    )
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["version"] == "V8_u33_deck003_core16_benchmark_v2"
    assert result["parent"] == {
        "version": "V6_u4_core16_benchmark_v2",
        "source_checkpoint_update": 33,
        "local_start_update": 0,
        "checkpoint_sha256": "a7bc5a488ad031c07be96576113762b40cb75c51d46370d1e7eace22df818cbf",
        "optimizer": "fresh",
        "pfsp_state": None,
    }
    assert result["training"]["focal_deck_ids"] == ["003"]
    assert result["training"]["sampling"] == {
        "mode": "uniform_001_067", "pfsp": 0, "uniform": 256,
    }
    assert result["benchmark_v2"]["baseline_checkpoint_update"] == 0
    assert result["benchmark_v2"]["focal_deck_id"] == "003"
    assert result["benchmark_v2"]["selected_meta_ids"] == list(range(14)) + [17, 27]
    assert result["benchmark_v2"]["games"] == 2048
