"""Batched two-sided GPU inference over isolated official-engine games."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Literal

import torch

from ..decoder import DECODER_COMPONENTS
from ..policy.action_distribution import greedy_actions_encoded, sample_actions_encoded
from ..policy.actor_critic import LeagueActorCritic, load_league_actor_critic
from ..policy.league_pool import LeaguePolicyPool
from ..policy.batching import collate_feature_batches, cpu_batch, move_batch
from ..policy.online_runtime import OnlineCausalEncoder
from .protocol import EpisodeTrajectory, LeaguePolicyView, RolloutJob, TrajectoryDecision
from .worker import run_engine_episode


@dataclass
class _Live:
    job: RolloutJob
    connection: Connection
    process: mp.Process
    focal_encoder: OnlineCausalEncoder
    opponent_encoder: OnlineCausalEncoder
    decisions: list[TrajectoryDecision] = field(default_factory=list)


def _slice(batch: dict[str, torch.Tensor], indices: list[int]) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in batch.items()}


class LeagueRolloutCollector:
    def __init__(self, model: LeagueActorCritic, *, device: torch.device, workers: int = 8,
                 mode: Literal["sample", "greedy"] = "sample", coalesce_ms: float = 0.5,
                 timeout_seconds: float = 60.0,
                 policy_pool: LeaguePolicyPool | None = None) -> None:
        if workers < 1 or coalesce_ms < 0:
            raise ValueError("invalid collector concurrency configuration")
        self.model = model.eval()
        self.policy_pool = policy_pool
        self.device = device
        self.workers = workers
        self.mode = mode
        self.coalesce_seconds = coalesce_ms / 1000.0
        self.timeout_seconds = timeout_seconds
        self.context = mp.get_context("spawn")
        self.frozen_head, _ = load_league_actor_critic("cpu")
        self.frozen_head.eval()
        for component in DECODER_COMPONENTS:
            getattr(self.frozen_head.actor, component).to(device)
        self.inference_batches = 0
        self.inference_requests = 0
        self.max_batch_size = 0
        self.inference_seconds = 0.0

    def _start(self, job: RolloutJob) -> _Live:
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(target=run_engine_episode, args=(child, job), name=f"0024-rollout-{job.game_id}")
        process.start()
        child.close()
        focal_actor = 0 if job.focal_first else 1
        return _Live(
            job, parent, process,
            OnlineCausalEncoder(focal_actor, job.focal_deck, self.model.actor.config),
            OnlineCausalEncoder(1 - focal_actor, job.opponent_deck, self.model.actor.config),
        )

    @staticmethod
    def _finish(live: _Live, message: dict) -> EpisodeTrajectory:
        live.connection.close()
        live.process.join(timeout=5)
        if live.process.is_alive():
            live.process.terminate(); live.process.join(timeout=2)
        episode = EpisodeTrajectory(live.job, decisions=live.decisions)
        if message.get("valid") and message.get("reward") is not None:
            episode.finish(float(message["reward"]), int(message.get("final_turn") or 0))
        else:
            episode.error = str(message.get("error") or message.get("status"))
        episode.diagnostics = {
            "engine_selections": int(message.get("engine_selections", 0)),
            "source_policy_update": live.job.source_policy_update,
        }
        return episode

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        pending = iter(jobs); live: dict[Connection, _Live] = {}; results = []
        def fill() -> None:
            while len(live) < self.workers:
                try: item = self._start(next(pending))
                except StopIteration: return
                live[item.connection] = item
        fill()
        try:
            while live:
                ready = list(wait(list(live), timeout=self.timeout_seconds))
                if not ready: raise TimeoutError("official-engine League workers stopped responding")
                deadline = time.perf_counter() + self.coalesce_seconds
                while len(ready) < len(live) and time.perf_counter() < deadline:
                    more = wait([c for c in live if c not in ready], timeout=max(0.0, deadline-time.perf_counter()))
                    if not more: break
                    ready.extend(more)
                requests: list[tuple[_Live, str, dict]] = []
                for connection in ready:
                    item = live[connection]
                    try: message = connection.recv()
                    except EOFError: message = {"kind": "result", "valid": False, "error": "worker EOF"}
                    if message.get("kind") == "decision":
                        requests.append((item, str(message["role"]), message["observation"]))
                    elif message.get("kind") == "result":
                        results.append(self._finish(item, message)); del live[connection]
                    else: raise RuntimeError("unknown League worker message")
                if requests:
                    rows = [
                        (item.focal_encoder if role == "focal" else item.opponent_encoder).encode(observation)
                        for item, role, observation in requests
                    ]
                    batch = move_batch(collate_feature_batches(rows), self.device)
                    started = time.perf_counter()
                    with torch.no_grad():
                        encoder = self.policy_pool.shared_actor if self.policy_pool is not None else self.model.actor
                        state, options = encoder.encode(batch)
                    actions: dict[int, object] = {}
                    route_groups: dict[tuple[str, str], list[int]] = {}
                    for index, (item, role, _) in enumerate(requests):
                        if role == "focal":
                            route = item.job.focal_deck_id
                        elif item.job.opponent_view is LeaguePolicyView.LIVE:
                            route = item.job.opponent_deck_id
                        else:
                            route = "__frozen__"
                        route_groups.setdefault((route, role), []).append(index)
                    for (route, role), route_indices in route_groups.items():
                        sub = _slice(batch, route_indices)
                        s = state[route_indices]; o = options[route_indices]
                        policy = self.frozen_head if route == "__frozen__" else (
                            self.policy_pool.policy(route) if self.policy_pool is not None else self.model
                        )
                        values = (
                            torch.zeros(len(route_indices), device=self.device)
                            if route == "__frozen__"
                            else policy.value_head(s).squeeze(-1)
                        )
                        sampled = role == "focal"
                        decoded = (
                            sample_actions_encoded(policy, sub, s, o, values)
                            if sampled and self.mode == "sample"
                            else greedy_actions_encoded(policy, sub, s, o, values)
                        )
                        actions.update(zip(route_indices, decoded, strict=True))
                    elapsed = time.perf_counter() - started
                    self.inference_batches += 1; self.inference_requests += len(requests)
                    self.max_batch_size = max(self.max_batch_size, len(requests)); self.inference_seconds += elapsed
                    for index, (item, role, observation) in enumerate(requests):
                        action = actions[index]
                        if role == "focal" or item.job.opponent_view is LeaguePolicyView.LIVE:
                            policy_deck_id = item.job.focal_deck_id if role == "focal" else item.job.opponent_deck_id
                            item.decisions.append(TrajectoryDecision(
                                cpu_batch(rows[index]), tuple(action.indices), bool(action.stopped),
                                float(action.log_prob), float(action.entropy), float(action.value),
                                policy_deck_id, item.job.source_policy_update,
                                1 if role == "focal" else -1,
                            ))
                        item.connection.send({"kind": "action", "action": list(action.indices)})
                fill()
        finally:
            for item in live.values():
                item.connection.close()
                if item.process.is_alive(): item.process.terminate()
                item.process.join(timeout=2)
        return sorted(results, key=lambda episode: episode.job.game_id)

    def metrics(self) -> dict[str, float]:
        return {
            "rollout/inference_batches": float(self.inference_batches),
            "rollout/inference_requests": float(self.inference_requests),
            "rollout/max_batch_size": float(self.max_batch_size),
            "rollout/inference_seconds": self.inference_seconds,
        }


__all__ = ["LeagueRolloutCollector"]
