"""Official-engine rollout with one shared 0028 representation pass per ready batch."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any, Literal

import torch

from ..foundation.online_runtime import OnlineCausalEncoder
from ..legacy_foundation import LegacyFoundationService
from ..policy.action_distribution import cached_features, infer_actions
from ..policy.actor_critic import SemanticActorCritic
from ..policy.batching import collate_feature_batches, cpu_batch, move_batch
from .protocol import EpisodeTrajectory, RolloutJob, TrajectoryDecision
from .worker import run_engine_episode


@dataclass
class _Live:
    job: RolloutJob
    connection: Connection
    process: mp.Process
    focal_encoder: Any
    opponent_encoder: Any
    decisions: list[TrajectoryDecision] = field(default_factory=list)


class HeterogeneousRolloutCollector:
    def __init__(
        self,
        model: SemanticActorCritic,
        *,
        device: torch.device,
        workers: int = 8,
        mode: Literal["sample", "greedy"] = "sample",
        coalesce_ms: float = 0.5,
        timeout_seconds: float = 60.0,
        opponent_foundation: Literal["0028", "0019", "mixed"] = "0028",
        legacy_service: LegacyFoundationService | None = None,
    ) -> None:
        if workers < 1 or coalesce_ms < 0 or timeout_seconds <= 0:
            raise ValueError("invalid collector configuration")
        self.model = model.eval()
        self.model.prepare_inference_cache()
        self.device = device
        self.workers = workers
        self.mode = mode
        self.coalesce_seconds = coalesce_ms / 1000.0
        self.timeout_seconds = timeout_seconds
        if opponent_foundation not in {"0028", "0019", "mixed"}:
            raise ValueError("unknown opponent foundation")
        if opponent_foundation in {"0019", "mixed"} and legacy_service is None:
            raise ValueError("legacy opponent foundation requires a verified service")
        self.opponent_foundation = opponent_foundation
        self.legacy_service = legacy_service
        self.context = mp.get_context("spawn")
        self.prototypes = model.runtime_prototypes
        self.inference_batches = 0
        self.inference_requests = 0
        self.focal_requests = 0
        self.opponent_requests = 0
        self.opponent_0019_requests = 0
        self.opponent_0028_requests = 0
        self.max_batch_size = 0
        self.inference_seconds = 0.0

    def _start(self, job: RolloutJob) -> _Live:
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=run_engine_episode,
            args=(child, job),
            name=f"0030-rollout-{job.game_id}",
        )
        process.start()
        child.close()
        focal_actor = 0 if job.focal_first else 1
        opponent_actor = 1 - focal_actor
        foundation = self._job_foundation(job)
        opponent_encoder = (
            self.legacy_service.new_encoder(opponent_actor, job.opponent_deck)
            if foundation == "0019"
            else OnlineCausalEncoder(
                opponent_actor, job.opponent_deck, self.model.actor.config, self.prototypes
            )
        )
        return _Live(
            job,
            parent,
            process,
            OnlineCausalEncoder(focal_actor, job.focal_deck, self.model.actor.config, self.prototypes),
            opponent_encoder,
        )

    def _job_foundation(self, job: RolloutJob) -> str:
        foundation = job.opponent_foundation
        if foundation not in {"0019", "0028"}:
            raise ValueError(f"job has unknown opponent foundation: {foundation}")
        if self.opponent_foundation != "mixed" and (
            foundation != self.opponent_foundation
        ):
            raise ValueError(
                "job opponent foundation disagrees with collector configuration"
            )
        return foundation

    @staticmethod
    def _finish(item: _Live, message: dict) -> EpisodeTrajectory:
        item.connection.close()
        item.process.join(timeout=5)
        if item.process.is_alive():
            item.process.terminate()
            item.process.join(timeout=2)
        episode = EpisodeTrajectory(item.job, decisions=item.decisions)
        if message.get("valid") and message.get("reward") is not None:
            episode.finish(float(message["reward"]), int(message.get("final_turn") or 0))
        else:
            episode.error = str(message.get("error") or message.get("status"))
        episode.diagnostics = {
            "engine_selections": int(message.get("engine_selections", 0)),
            "source_policy_update": item.job.source_policy_update,
        }
        return episode

    def _infer(self, requests: list[tuple[_Live, dict, bool]]):
        modern_indices = [
            index
            for index, (item, _, focal) in enumerate(requests)
            if focal or self._job_foundation(item.job) == "0028"
        ]
        legacy_indices = [
            index
            for index, (item, _, focal) in enumerate(requests)
            if not focal and self._job_foundation(item.job) == "0019"
        ]
        routed: list[tuple[int, ...] | None] = [None] * len(requests)
        if modern_indices:
            modern_rows = [
                (
                    requests[index][0].focal_encoder
                    if requests[index][2]
                    else requests[index][0].opponent_encoder
                ).encode(requests[index][1])
                for index in modern_indices
            ]
            batch = move_batch(collate_feature_batches(modern_rows), self.device)
            with torch.no_grad():
                validated, summary, options = self.model.encode(batch)
                route = torch.tensor(
                    [requests[index][2] for index in modern_indices],
                    device=self.device,
                )
                actions = infer_actions(
                    self.model,
                    validated,
                    summary,
                    options,
                    focal=route,
                    greedy=self.mode == "greedy",
                )
            for local_index, request_index in enumerate(modern_indices):
                item, _, focal = requests[request_index]
                action = actions[local_index]
                routed[request_index] = tuple(action.indices)
                if not focal:
                    continue
                full = cached_features(validated, summary, options)
                option_count = int(validated.option_mask[local_index].sum().item())
                single = {}
                for key, value in full.items():
                    row = value[local_index:local_index + 1]
                    if key in {"options", "option_mask"}:
                        row = row[:, :option_count]
                    if key in {"summary", "options"}:
                        row = row.to(dtype=torch.float16)
                    single[key] = row
                item.decisions.append(TrajectoryDecision(
                    cpu_batch(single), tuple(action.indices), bool(action.stopped),
                    float(action.log_prob), float(action.entropy), float(action.value),
                    item.job.source_policy_update,
                ))
        if legacy_indices:
            if self.legacy_service is None:
                raise RuntimeError("legacy service disappeared during inference")
            legacy_rows = [
                requests[index][0].opponent_encoder.encode(requests[index][1])
                for index in legacy_indices
            ]
            legacy_actions = self.legacy_service.greedy(legacy_rows)
            for request_index, action in zip(
                legacy_indices, legacy_actions, strict=True
            ):
                routed[request_index] = action
        if any(action is None for action in routed):
            raise RuntimeError("an inference request was not routed")
        return [action for action in routed if action is not None]

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        pending = iter(jobs)
        live: dict[Connection, _Live] = {}
        results: list[EpisodeTrajectory] = []

        def fill() -> None:
            while len(live) < self.workers:
                try:
                    item = self._start(next(pending))
                except StopIteration:
                    return
                live[item.connection] = item

        fill()
        try:
            while live:
                ready = list(wait(list(live), timeout=self.timeout_seconds))
                if not ready:
                    raise TimeoutError("official-engine workers stopped responding")
                deadline = time.perf_counter() + self.coalesce_seconds
                while len(ready) < len(live) and time.perf_counter() < deadline:
                    more = wait(
                        [connection for connection in live if connection not in ready],
                        timeout=max(0.0, deadline - time.perf_counter()),
                    )
                    if not more:
                        break
                    ready.extend(more)
                requests: list[tuple[_Live, dict, bool]] = []
                for connection in ready:
                    item = live[connection]
                    try:
                        message = connection.recv()
                    except EOFError:
                        message = {"kind": "result", "valid": False, "error": "worker EOF"}
                    if message.get("kind") == "decision":
                        requests.append((item, message["observation"], message.get("role") == "focal"))
                    elif message.get("kind") == "result":
                        results.append(self._finish(item, message))
                        del live[connection]
                    else:
                        raise RuntimeError("unknown official-engine worker message")
                started = time.perf_counter()
                routed = []
                if requests:
                    actions = self._infer(requests)
                    routed = [
                        (item, action)
                        for (item, _, _), action in zip(requests, actions, strict=True)
                    ]
                if routed:
                    elapsed = time.perf_counter() - started
                    focal = [request for request in requests if request[2]]
                    opponent = [request for request in requests if not request[2]]
                    self.inference_batches += 1
                    self.inference_requests += len(routed)
                    self.focal_requests += len(focal)
                    self.opponent_requests += len(opponent)
                    self.opponent_0019_requests += sum(
                        not focal and self._job_foundation(item.job) == "0019"
                        for item, _, focal in requests
                    )
                    self.opponent_0028_requests += sum(
                        not focal and self._job_foundation(item.job) == "0028"
                        for item, _, focal in requests
                    )
                    self.max_batch_size = max(self.max_batch_size, len(requests))
                    self.inference_seconds += elapsed
                    for item, action in routed:
                        item.connection.send({"kind": "action", "action": list(action)})
                fill()
        finally:
            for item in live.values():
                item.connection.close()
                if item.process.is_alive():
                    item.process.terminate()
                item.process.join(timeout=2)
        return sorted(results, key=lambda episode: episode.job.game_id)

    def metrics(self) -> dict[str, float]:
        return {
            "rollout/inference_batches": float(self.inference_batches),
            "rollout/inference_requests": float(self.inference_requests),
            "rollout/focal_requests": float(self.focal_requests),
            "rollout/opponent_requests": float(self.opponent_requests),
            "rollout/opponent_0019_requests": float(
                self.opponent_0019_requests
            ),
            "rollout/opponent_0028_requests": float(
                self.opponent_0028_requests
            ),
            "rollout/max_batch_size": float(self.max_batch_size),
            "rollout/inference_seconds": self.inference_seconds,
        }


__all__ = ["HeterogeneousRolloutCollector"]
