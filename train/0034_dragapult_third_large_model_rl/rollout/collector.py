"""Batch full Large Model 0806 inference around isolated official CPU engine workers."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Literal

import torch

from ..opponents import load_foundation
from ..opponents.online_runtime import OnlineCausalEncoder as OpponentEncoder
from ..policy.action_distribution import greedy_actions, sample_actions
from ..policy.actor_critic import SemanticActorCritic
from ..policy.batching import collate_feature_batches, cpu_batch, move_batch
from ..semantic_policy.deployment.online_runtime import OnlineCausalEncoder as FocalEncoder
from .protocol import EpisodeTrajectory, RolloutJob, TrajectoryDecision
from .worker import run_engine_episode


@dataclass
class _Live:
    job: RolloutJob
    connection: Connection
    process: mp.Process
    focal_encoder: FocalEncoder
    opponent_encoder: OpponentEncoder
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    last_decision: dict[str, object] | None = None
    last_action: tuple[int, ...] | None = None


class FullSemanticRolloutCollector:
    def __init__(
        self,
        model: SemanticActorCritic,
        *,
        device: torch.device,
        workers: int = 8,
        mode: Literal["sample", "greedy"] = "sample",
        coalesce_ms: float = 0.5,
        timeout_seconds: float = 120.0,
    ) -> None:
        if workers < 1 or coalesce_ms < 0 or timeout_seconds <= 0:
            raise ValueError("invalid collector configuration")
        self.model = model.eval()
        self.device = device
        self.workers = workers
        self.mode = mode
        self.coalesce_seconds = coalesce_ms / 1000.0
        self.timeout_seconds = timeout_seconds
        self.context = mp.get_context("spawn")
        self.opponent, self.opponent_identity = load_foundation(device, eval_mode=True)
        self.inference_batches = 0
        self.inference_requests = 0
        self.focal_requests = 0
        self.opponent_requests = 0
        self.max_batch_size = 0
        self.inference_seconds = 0.0
        self.focal_encode_seconds = 0.0
        self.focal_collate_move_seconds = 0.0
        self.focal_model_seconds = 0.0
        self.opponent_encode_seconds = 0.0
        self.opponent_collate_move_seconds = 0.0
        self.opponent_model_seconds = 0.0

    def _start(self, job: RolloutJob) -> _Live:
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=run_engine_episode,
            args=(child, job),
            name=f"0034-full-semantic-{job.game_id}",
        )
        process.start()
        child.close()
        focal_actor = 0 if job.focal_first else 1
        return _Live(
            job=job,
            connection=parent,
            process=process,
            focal_encoder=FocalEncoder(focal_actor, job.focal_deck, self.model.actor.config),
            opponent_encoder=OpponentEncoder(
                1 - focal_actor, job.opponent_deck, self.opponent.config
            ),
        )

    @staticmethod
    def _finish(item: _Live, message: dict) -> EpisodeTrajectory:
        item.connection.close()
        item.process.join(timeout=5)
        if item.process.is_alive():
            item.process.terminate()
            item.process.join(timeout=2)
        exitcode = item.process.exitcode
        episode = EpisodeTrajectory(item.job, decisions=item.decisions)
        if message.get("valid") and message.get("reward") is not None:
            episode.finish(float(message["reward"]), int(message.get("final_turn") or 0))
        else:
            detail = message.get("error") or message.get("status")
            episode.error = (
                f"{detail}; game_id={item.job.game_id}; opponent={item.job.opponent_id}; "
                f"focal_first={item.job.focal_first}; seed={item.job.seed}; "
                f"worker_exitcode={exitcode}; last_decision={item.last_decision!r}; "
                f"last_action={item.last_action!r}"
            )
        episode.diagnostics = {
            "engine_selections": int(message.get("engine_selections", 0)),
            "source_policy_update": item.job.source_policy_update,
            "worker_exitcode": exitcode,
            "last_decision": item.last_decision,
            "last_action": item.last_action,
        }
        return episode

    def _infer_focal(self, requests: list[tuple[_Live, dict]]) -> list[object]:
        started = time.perf_counter()
        rows = [item.focal_encoder.encode(observation) for item, observation in requests]
        encoded = time.perf_counter()
        batch = move_batch(collate_feature_batches(rows), self.device)
        prepared = time.perf_counter()
        actions = sample_actions(self.model, batch) if self.mode == "sample" else greedy_actions(self.model, batch)
        inferred = time.perf_counter()
        self.focal_encode_seconds += encoded - started
        self.focal_collate_move_seconds += prepared - encoded
        self.focal_model_seconds += inferred - prepared
        for index, (item, observation) in enumerate(requests):
            action = actions[index]
            turn = (observation.get("current") or {}).get("turn")
            if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
                raise RuntimeError(f"official observation has invalid turn: {turn!r}")
            item.decisions.append(
                TrajectoryDecision(
                    cpu_batch(rows[index]),
                    tuple(action.indices),
                    bool(action.stopped),
                    float(action.log_prob),
                    float(action.entropy),
                    float(action.value),
                    item.job.source_policy_update,
                    turn,
                )
            )
        return actions

    def _infer_opponent(self, requests: list[tuple[_Live, dict]]) -> list[tuple[int, ...]]:
        started = time.perf_counter()
        rows = [item.opponent_encoder.encode(observation) for item, observation in requests]
        encoded = time.perf_counter()
        batch = move_batch(collate_feature_batches(rows), self.device)
        prepared = time.perf_counter()
        with torch.inference_mode():
            decoded = self.opponent.deterministic_action_tensors(batch)
        if not bool(decoded.legal.all()):
            raise RuntimeError("Frozen 0019 opponent produced an illegal action")
        actions = [
            tuple(int(value) for value in decoded.sequences[index, : decoded.lengths[index]].tolist())
            for index in range(len(requests))
        ]
        inferred = time.perf_counter()
        self.opponent_encode_seconds += encoded - started
        self.opponent_collate_move_seconds += prepared - encoded
        self.opponent_model_seconds += inferred - prepared
        return actions

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
                    raise TimeoutError("official CPU engine workers stopped responding")
                deadline = time.perf_counter() + self.coalesce_seconds
                while len(ready) < len(live) and time.perf_counter() < deadline:
                    more = wait(
                        [connection for connection in live if connection not in ready],
                        timeout=max(0.0, deadline - time.perf_counter()),
                    )
                    if not more:
                        break
                    ready.extend(more)
                focal: list[tuple[_Live, dict]] = []
                opponent: list[tuple[_Live, dict]] = []
                for connection in ready:
                    item = live[connection]
                    try:
                        message = connection.recv()
                    except EOFError:
                        message = {"kind": "result", "valid": False, "error": "worker EOF"}
                    if message.get("kind") == "decision":
                        observation = message["observation"]
                        current = observation.get("current") or {}
                        selection = observation.get("select") or {}
                        options = selection.get("option") or []
                        item.last_decision = {
                            "selection_index": int(message.get("selection_index", -1)),
                            "role": message.get("role"),
                            "actor": message.get("actor"),
                            "turn": current.get("turn"),
                            "select_type": selection.get("type"),
                            "select_context": selection.get("context"),
                            "option_count": len(options),
                            "min_count": selection.get("minCount"),
                            "max_count": selection.get("maxCount"),
                        }
                        request = (item, observation)
                        (focal if message.get("role") == "focal" else opponent).append(request)
                    elif message.get("kind") == "result":
                        results.append(self._finish(item, message))
                        del live[connection]
                    else:
                        raise RuntimeError("unknown official CPU worker message")
                started = time.perf_counter()
                routed: list[tuple[_Live, tuple[int, ...]]] = []
                if focal:
                    routed.extend(
                        (item, tuple(action.indices))
                        for (item, _), action in zip(focal, self._infer_focal(focal), strict=True)
                    )
                if opponent:
                    routed.extend(
                        (item, action)
                        for (item, _), action in zip(opponent, self._infer_opponent(opponent), strict=True)
                    )
                if routed:
                    elapsed = time.perf_counter() - started
                    self.inference_batches += int(bool(focal)) + int(bool(opponent))
                    self.inference_requests += len(routed)
                    self.focal_requests += len(focal)
                    self.opponent_requests += len(opponent)
                    self.max_batch_size = max(self.max_batch_size, len(focal), len(opponent))
                    self.inference_seconds += elapsed
                    for item, action in routed:
                        item.last_action = action
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
            "rollout/max_batch_size": float(self.max_batch_size),
            "rollout/inference_seconds": self.inference_seconds,
            "rollout/focal_encode_seconds": self.focal_encode_seconds,
            "rollout/focal_collate_move_seconds": self.focal_collate_move_seconds,
            "rollout/focal_model_seconds": self.focal_model_seconds,
            "rollout/opponent_encode_seconds": self.opponent_encode_seconds,
            "rollout/opponent_collate_move_seconds": self.opponent_collate_move_seconds,
            "rollout/opponent_model_seconds": self.opponent_model_seconds,
        }


__all__ = ["FullSemanticRolloutCollector"]
