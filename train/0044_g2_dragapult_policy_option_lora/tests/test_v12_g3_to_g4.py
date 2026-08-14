from __future__ import annotations

from collections import Counter, defaultdict
import importlib

import torch


def _modules():
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v12"
    )
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    return runner, common


def test_v12_jobs_are_512_meta_balanced_g3_only_and_agent_owned_toss() -> None:
    runner, common = _modules()
    registry = common.AssetRegistry.load(common.PROJECT_ROOT)
    jobs, deck_weights, policy_weights, version = common._jobs(
        0, registry,
        focal_deck_ids=runner.FOCAL_DECK_IDS,
        focal_deck_id=runner.INITIALIZATION_DECK_ID,
        focal_schedule_mode=runner.FOCAL_SCHEDULE,
        opponent_sampling_mode=runner.OPPONENT_SCHEDULE,
        opponent_policy_ids=(runner.OPPONENT_POLICY_ID,),
        latest_champion_policy_id=runner.OPPONENT_POLICY_ID,
        rollout_games=runner.ROLLOUT_GAMES,
    )
    assert len(jobs) == 512
    assert {job.opponent_policy_id for job in jobs} == {"Champion-G3"}
    assert policy_weights == {"Champion-G3": 1.0}
    assert len(deck_weights) == 67
    assert version.startswith("meta-balanced-")
    assert all(job.focal_first is job.focal_won_toss for job in jobs)
    assert len({job.coin_winner_seed for job in jobs}) == 512

    own = common.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=common.PROJECT_ROOT
    )
    meta_by_deck = {row.deck_id: row.archetype_id for row in own.mappings}
    focal_meta = Counter(meta_by_deck[job.focal_deck_id] for job in jobs)
    opponent_meta = Counter(meta_by_deck[job.opponent_id] for job in jobs)
    assert focal_meta == opponent_meta
    assert set(focal_meta.values()) == {18, 19}
    assert sum(job.focal_deck_id == job.opponent_id for job in jobs) < 64
    for side in ("focal", "opponent"):
        by_meta = defaultdict(Counter)
        for job in jobs:
            deck_id = job.focal_deck_id if side == "focal" else job.opponent_id
            by_meta[meta_by_deck[deck_id]][deck_id] += 1
        assert all(max(c.values()) - min(c.values()) <= 1 for c in by_meta.values())


def test_v12_ppo_contract_has_approved_capacity_and_learning_rates() -> None:
    runner, common = _modules()
    config = common._config(
        actor_learning_rate_scale=runner.ACTOR_LEARNING_RATE_SCALE,
        batch_size=runner.PPO_MINIBATCH_SIZE,
        value_learning_rate=runner.VALUE_LEARNING_RATE,
        prize_learning_rate=runner.VALUE_LEARNING_RATE,
        meta_actor_residual_learning_rate=runner.META_RESIDUAL_LEARNING_RATE,
    )
    assert config.epochs == 3
    assert config.batch_size == 4096
    assert config.forward_microbatch_size == 1024
    assert config.decoder_learning_rate == 5e-6
    assert config.policy_adapter_learning_rate == 5e-6
    assert config.allocation_learning_rate == 5e-6
    assert config.option_lora_learning_rate == 1e-5
    assert config.meta_actor_residual_learning_rate == 5e-6
    assert config.value_learning_rate == 2e-5
    assert config.prize_learning_rate == 2e-5


def test_complete_compound_policy_device_inventory_includes_g3_adapters() -> None:
    inference = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.inference"
    )
    class Policy:
        pass
    policy = Policy()
    for name in (
        "actor", "value_head", "allocation_head", "value_adapter",
        "policy_strategy_adapter", "policy_option_lora", "meta_actor_residual",
    ):
        setattr(policy, name, torch.nn.Linear(1, 1))
    modules = inference._modules(policy)
    assert len(modules) == 7
    assert policy.policy_option_lora in modules
    assert policy.meta_actor_residual in modules


def test_context41_is_evidence_but_not_a_semantic_ppo_transition() -> None:
    collector = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.rollout.cuda_collector"
    )
    focal = torch.tensor([True, True, False, True])
    global_cat = torch.zeros(4, 12, dtype=torch.long)
    global_cat[:, 2] = torch.tensor([0, 1, 0, 2])
    route = collector._post_seat_focal_route(focal, global_cat)
    assert route.tolist() == [False, True, False, True]
