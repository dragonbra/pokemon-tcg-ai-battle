"""One-update official-engine PPO canary with immutable encoder proofs."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch

from . import FOCAL_DECK_ID
from .decoder import extract_decoder_state, save_decoder_checkpoint
from .decks import load_deck_plugins
from .league import DEFAULT_DECK_ROOT
from .policy import load_league_actor_critic
from .policy.action_distribution import greedy_actions
from .policy.batching import collate_feature_batches, move_batch
from .rollout import LeaguePolicyView, LeagueRolloutCollector, RolloutJob
from .storage import preflight_storage, runtime_storage
from .training import PPOConfig, PPOTrainer, prepare_episodes
from .training.ppo import frozen_reference


def _l2_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return float(sum((after[key].float() - before[key].float()).square().sum() for key in before).sqrt())


def run_ppo_canary(*, device: str = "cuda:0", workers: int = 4,
                   deck_root: Path = DEFAULT_DECK_ROOT) -> dict[str, object]:
    output = Path(".tmp/evaluation/0024_ppo_canary").resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight_storage(output)
    plugins = {item.deck_id: item for item in load_deck_plugins(deck_root)}
    focal = plugins[FOCAL_DECK_ID]
    opponent_ids = ("festival_lead_dipplin_002", "mega_kangaskhan_ex_crustle_004")
    runtime = sorted(Path("evaluation/arena/opponents").glob("*/cg/game.py"))[0].parents[1].resolve()
    torch_device = torch.device(device)
    if torch_device.type == "cuda": torch.cuda.init(); torch.cuda.reset_peak_memory_stats(torch_device)
    model, identity = load_league_actor_critic(device)
    encoder_before = model.encoder_sha256()
    decoder_before = extract_decoder_state(model.actor, model.value_head)
    jobs = []
    for index, opponent_id in enumerate(opponent_ids):
        opponent = plugins[opponent_id]
        for focal_first in (True, False):
            jobs.append(RolloutJob(
                f"canary-{index}-{int(focal_first)}", focal.deck_id, opponent.deck_id,
                LeaguePolicyView.FROZEN, focal_first, 22100 + index * 2 + int(focal_first), 0,
                focal.deck, opponent.deck, runtime,
            ))
    collector = LeagueRolloutCollector(model, device=torch_device, workers=workers, mode="sample")
    started = time.perf_counter(); episodes = collector.collect(jobs); rollout_seconds = time.perf_counter() - started
    if sum(item.valid for item in episodes) != len(jobs):
        raise RuntimeError(f"canary rollout errors: {[item.error for item in episodes if not item.valid]}")
    batch = prepare_episodes(episodes, policy_deck_id=focal.deck_id, gamma=1.0, gae_lambda=0.95)
    reference = frozen_reference(model, torch_device)
    trainer = PPOTrainer(model, reference, device=torch_device, config=PPOConfig(
        epochs=2, batch_size=128, actor_learning_rate=1e-5, value_learning_rate=1e-4,
    ))
    metrics = trainer.update(batch)
    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise FloatingPointError("canary produced nonfinite metrics")
    encoder_after = model.encoder_sha256()
    if encoder_after != encoder_before:
        raise RuntimeError("frozen encoder changed during PPO canary")
    decoder_after = extract_decoder_state(model.actor, model.value_head)
    delta = _l2_delta(decoder_before, decoder_after)
    if delta <= 0:
        raise RuntimeError("PPO canary did not change decoder/value parameters")
    checkpoint = output / "update-000001.pt"
    checkpoint.unlink(missing_ok=True); checkpoint.with_suffix(".pt.sha256").unlink(missing_ok=True)
    digest = save_decoder_checkpoint(
        checkpoint, model.actor, model.value_head, foundation_sha256=identity.weights_sha256,
        deck_id=focal.deck_id, deck_sha256=focal.deck_sha256, policy_role="live",
        policy_version="canary_not_formal", update=1,
    )
    probe = move_batch(collate_feature_batches([episodes[0].decisions[0].features]), torch_device)
    expected = greedy_actions(model.eval(), probe)[0].indices
    reloaded, _ = load_league_actor_critic(device, decoder_checkpoint=checkpoint, deck_id=focal.deck_id, deck_sha256=focal.deck_sha256)
    actual = greedy_actions(reloaded.eval(), probe)[0].indices
    if actual != expected:
        raise RuntimeError("decoder checkpoint reload changed greedy action")
    storage = runtime_storage(output)
    result = {
        "gate": "0024_official_engine_ppo_canary_v1", "games": len(episodes),
        "finished": sum(item.valid for item in episodes), "errors": sum(not item.valid for item in episodes),
        "decisions": batch.decisions, "source_policy_update": batch.source_policy_update,
        "rollout_seconds": rollout_seconds, "decoder_value_l2_delta": delta,
        "encoder_sha256_before": encoder_before, "encoder_sha256_after": encoder_after,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "reload_greedy_parity": True, **metrics, **collector.metrics(), **storage.metrics(),
    }
    if torch_device.type == "cuda": result["cuda/peak_allocated_bytes"] = torch.cuda.max_memory_allocated(torch_device)
    return result


__all__ = ["run_ppo_canary"]
