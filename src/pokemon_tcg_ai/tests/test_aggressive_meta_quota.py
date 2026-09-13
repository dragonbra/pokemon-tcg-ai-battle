from __future__ import annotations

from collections import Counter, defaultdict
import importlib
from pathlib import Path


PKG = "pokemon_tcg_ai"
PROJECT = Path(__file__).resolve().parents[1]


def _mappings():
    module = importlib.import_module(f"{PKG}.own_archetype")
    return module.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT
    ).mappings


def test_aggressive_schedule_has_exact_fixed_and_uniform_remaining_quotas():
    module = importlib.import_module(f"{PKG}.league.aggressive_meta_quota")
    rows = module.aggressive_meta_quota_schedule(
        quota_seed=945_275,
        shuffle_seed=945_276,
        mappings=_mappings(),
        lanes=512,
        fixed_meta_quotas={2: 50, 3: 200, 5: 200},
    )
    counts = Counter(row.archetype_id for row in rows)
    assert len(rows) == 512
    assert {meta_id: counts[meta_id] for meta_id in (2, 3, 5)} == {
        2: 50, 3: 200, 5: 200,
    }
    remaining = [value for key, value in counts.items() if key not in {2, 3, 5}]
    assert sum(remaining) == 62
    assert max(remaining) - min(remaining) <= 1
    by_meta_deck = defaultdict(Counter)
    for row in rows:
        by_meta_deck[row.archetype_id][row.deck_id] += 1
    assert all(
        max(deck_counts.values()) - min(deck_counts.values()) <= 1
        for deck_counts in by_meta_deck.values()
    )
    again = module.aggressive_meta_quota_schedule(
        quota_seed=945_275,
        shuffle_seed=945_276,
        mappings=_mappings(),
        lanes=512,
        fixed_meta_quotas={2: 50, 3: 200, 5: 200},
    )
    assert rows == again


def test_periodic_cadence_supports_every_update_without_changing_default():
    periodic = importlib.import_module(f"{PKG}.training.periodic_evaluation")
    assert [u for u in range(1, 6) if periodic.is_due(u)] == [5]
    assert [
        u for u in range(1, 6)
        if periodic.is_due(u, interval_updates=1)
    ] == [1, 2, 3, 4, 5]


def test_generic_jobs_lock_policy0809_and_realize_exact_quotas():
    assets = importlib.import_module(f"{PKG}.assets")
    runner = importlib.import_module(f"{PKG}.training.run_v1")
    registry = assets.AssetRegistry.load(PROJECT)
    jobs, _, policy_weights, curriculum = runner._jobs(
        275,
        registry,
        focal_deck_ids=("007",),
        focal_deck_id="007",
        focal_schedule_mode="fixed",
        opponent_sampling_mode="aggressive_meta_quota_training_pool",
        opponent_policy_ids=("Policy-0809",),
        latest_champion_policy_id="Policy-0809",
        rollout_games=512,
        opponent_meta_quotas={2: 50, 3: 200, 5: 200},
    )
    by_deck = {row.deck_id: row.archetype_id for row in _mappings()}
    counts = Counter(by_deck[job.opponent_id] for job in jobs)
    assert len(jobs) == 512
    assert {job.opponent_policy_id for job in jobs} == {"Policy-0809"}
    assert policy_weights == {"Policy-0809": 1.0}
    assert counts[2] == 50 and counts[3] == 200 and counts[5] == 200
    assert sum(value for key, value in counts.items() if key not in {2, 3, 5}) == 62
    assert curriculum == "aggressive_meta_quota-u000275"


def test_v6_u275_aggressive_experiment_contract():
    module = importlib.import_module(
        f"{PKG}.training.run_v6_u275_aggressive_meta_quota"
    )
    assert module.START_UPDATE == 275
    assert module.ROLLOUT_GAMES == 512
    assert module.OPPONENT_POLICY_IDS == ("Policy-0809",)
    assert module.OPPONENT_META_QUOTAS == {2: 50, 3: 200, 5: 200}
    assert module.PERIODIC_EVALUATION_INTERVAL_UPDATES == 1
    assert module.PERIODIC_EVALUATION_PROFILE == "policy0809_three_pool_cuda512"


def test_rollout_telemetry_accepts_aggressive_quota_mode():
    telemetry = importlib.import_module(f"{PKG}.telemetry")
    metrics = telemetry.aggregate_rollout(
        [{
            "result": "win",
            "focal_prizes_taken": 6,
            "opponent_prizes_taken": 3,
            "full_turns": 8,
            "opponent_deck_id": "002",
            "opponent_policy_id": "Policy-0809",
            "branch": "aggressive_meta_quota",
        }],
        curriculum_version="aggressive_meta_quota-u000275",
        deck_weights={"002": 1.0},
        policy_weights={"Policy-0809": 1.0},
        sampling_mode="aggressive_meta_quota_training_pool",
        expected_games=1,
    )
    assert metrics["sampling/mode_meta_balanced"] == 1.0
    assert metrics["sampling/mode_aggressive_meta_quota"] == 1.0


def test_v7_continues_exact_durable_v6_u276_with_same_experiment():
    module = importlib.import_module(
        f"{PKG}.training.run_v7_u276_aggressive_meta_quota"
    )
    assert module.START_UPDATE == 276
    assert module.OPPONENT_POLICY_IDS == ("Policy-0809",)
    assert module.OPPONENT_META_QUOTAS == {2: 50, 3: 200, 5: 200}
    assert module.PERIODIC_EVALUATION_INTERVAL_UPDATES == 1
    assert module.BASELINE_EVALUATION_CHECKPOINT == 276


def test_v8_continues_v7_u282_and_evaluates_every_two_updates():
    module = importlib.import_module(
        f"{PKG}.training.run_v8_u282_aggressive_meta_quota_eval2"
    )
    periodic = importlib.import_module(f"{PKG}.training.periodic_evaluation")
    assert module.START_UPDATE == 282
    assert module.BASELINE_EVALUATION_CHECKPOINT == 282
    assert module.PARENT_VERSION == (
        "V7_dragapult_007_u276_aggressive_meta_quota_policy0809"
    )
    assert module.PARENT_CHECKPOINT_SHA256 == (
        "25f312fa1d2c41a7ddbec2e3d4631c0437013781ed797a0d98b0c9f635d32d5c"
    )
    assert module.OPPONENT_POLICY_IDS == ("Policy-0809",)
    assert module.OPPONENT_META_QUOTAS == {2: 50, 3: 200, 5: 200}
    assert module.PERIODIC_EVALUATION_INTERVAL_UPDATES == 2
    assert module.PERIODIC_EVALUATION_PROFILE == "policy0809_three_pool_cuda512"
    assert module.EXPERT_COLD_START_LR_PROFILE.metadata()["profile_id"] == (
        "0045_expert_cold_start_lr_v1"
    )
    assert not periodic.is_due(283, interval_updates=2)
    assert periodic.is_due(284, interval_updates=2)
    assert not periodic.is_due(285, interval_updates=2)
    assert periodic.is_due(286, interval_updates=2)
