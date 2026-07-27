from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from collections.abc import Callable
from typing import Literal

import torch

from ..constants import TARGET_DECK, TARGET_SOURCE_ID
from ..policy.action_distribution import greedy_actions, sample_actions
from ..policy.actor_critic import DragapultActorCritic
from ..policy.batching import collate_feature_batches, cpu_batch, move_batch
from ..policy.online_runtime import OnlineCausalEncoder
from .protocol import Episode, RolloutJob, TrajectoryDecision
from .worker import run_engine_episode


@dataclass
class _LiveEpisode:
    job: RolloutJob
    connection: Connection
    process: mp.Process
    encoder: OnlineCausalEncoder
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)


def _selected_options(observation: dict, action: tuple[int, ...]) -> tuple[dict, ...]:
    options = ((observation.get("select") or {}).get("option") or [])
    return tuple(
        dict(options[index])
        for index in action
        if 0 <= index < len(options) and isinstance(options[index], dict)
    )


def _update_diagnostics(target: dict[str, float], selected: tuple[dict, ...]) -> None:
    target["decisions"] = target.get("decisions", 0.0) + 1.0
    for option in selected:
        card_id = int(option.get("cardId", 0) or 0)
        if card_id in (119, 120, 121, 131, 132, 133):
            key = f"selected_card_{card_id}"
            target[key] = target.get(key, 0.0) + 1.0
        if card_id == 121 and int(option.get("attackId", 0) or 0) > 0:
            target["dragapult_attack_submissions"] = (
                target.get("dragapult_attack_submissions", 0.0) + 1.0
            )
        if card_id in (132, 133):
            target["cursed_blast_submissions"] = (
                target.get("cursed_blast_submissions", 0.0) + 1.0
            )


class RolloutCollector:
    """Bounded multi-process official-engine collector with centralized inference."""

    def __init__(
        self,
        model: DragapultActorCritic,
        *,
        device: torch.device,
        workers: int = 8,
        mode: Literal["sample", "greedy"] = "sample",
        request_timeout_seconds: float = 60.0,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.model = model
        self.device = device
        self.workers = workers
        self.mode = mode
        self.request_timeout_seconds = request_timeout_seconds
        self._context = mp.get_context("spawn")

    def _start(self, job: RolloutJob) -> _LiveEpisode:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=run_engine_episode,
            args=(child, job),
            name=f"0017-rollout-{job.episode_id}",
        )
        process.start()
        child.close()
        candidate_index = 0 if job.candidate_first else 1
        return _LiveEpisode(
            job,
            parent,
            process,
            OnlineCausalEncoder(candidate_index, TARGET_DECK, self.model.actor.config),
        )

    @staticmethod
    def _finish(live: _LiveEpisode, message: dict) -> Episode:
        live.connection.close()
        live.process.join(timeout=5.0)
        if live.process.is_alive():
            live.process.terminate()
            live.process.join(timeout=2.0)
        return Episode(
            episode_id=live.job.episode_id,
            opponent=live.job.opponent.name,
            opponent_hash=live.job.opponent.package_hash,
            candidate_first=live.job.candidate_first,
            seed=live.job.seed,
            valid=bool(message.get("valid")),
            reward=message.get("reward"),
            winner=message.get("winner"),
            status=str(message.get("status", "worker_error")),
            error=message.get("error"),
            engine_selections=int(message.get("engine_selections", 0)),
            final_turn=message.get("final_turn"),
            complete_rounds=message.get("complete_rounds"),
            decisions=live.decisions,
            diagnostics=live.diagnostics,
        )

    def collect(
        self,
        jobs: list[RolloutJob],
        *,
        on_episode: Callable[[Episode], None] | None = None,
    ) -> list[Episode]:
        if not jobs:
            return []
        pending = iter(jobs)
        live_by_connection: dict[Connection, _LiveEpisode] = {}
        results: list[Episode] = []

        def fill_workers() -> None:
            while len(live_by_connection) < self.workers:
                try:
                    job = next(pending)
                except StopIteration:
                    return
                live = self._start(job)
                live_by_connection[live.connection] = live

        fill_workers()
        try:
            while live_by_connection:
                ready = wait(
                    list(live_by_connection), timeout=self.request_timeout_seconds
                )
                if not ready:
                    raise TimeoutError("official-engine rollout workers stopped responding")
                decision_requests: list[tuple[_LiveEpisode, dict]] = []
                for connection in ready:
                    live = live_by_connection[connection]
                    try:
                        message = connection.recv()
                    except EOFError:
                        message = {
                            "kind": "result",
                            "valid": False,
                            "status": "worker_eof",
                            "error": "worker connection closed without a result",
                        }
                    if message.get("kind") == "decision":
                        decision_requests.append((live, message["observation"]))
                    elif message.get("kind") == "result":
                        episode = self._finish(live, message)
                        results.append(episode)
                        if on_episode is not None:
                            on_episode(episode)
                        del live_by_connection[connection]
                    else:
                        raise RuntimeError("rollout worker sent an unknown message")
                if decision_requests:
                    encoded = []
                    for live, observation in decision_requests:
                        item = live.encoder.encode(observation)
                        item["source_id"] = torch.tensor(
                            [TARGET_SOURCE_ID], dtype=torch.long
                        )
                        encoded.append(item)
                    inference_batch = move_batch(
                        collate_feature_batches(encoded), self.device
                    )
                    started = time.perf_counter()
                    actions = (
                        sample_actions(self.model, inference_batch)
                        if self.mode == "sample"
                        else greedy_actions(self.model, inference_batch)
                    )
                    latency_ms = (time.perf_counter() - started) * 1_000.0
                    for (live, observation), features, sampled in zip(
                        decision_requests, encoded, actions, strict=True
                    ):
                        selected = _selected_options(observation, sampled.indices)
                        _update_diagnostics(live.diagnostics, selected)
                        live.diagnostics["inference_ms_sum"] = (
                            live.diagnostics.get("inference_ms_sum", 0.0)
                            + latency_ms / len(actions)
                        )
                        live.diagnostics["inference_requests"] = (
                            live.diagnostics.get("inference_requests", 0.0) + 1.0
                        )
                        live.diagnostics["inference_batch_size_max"] = max(
                            live.diagnostics.get("inference_batch_size_max", 0.0),
                            float(len(actions)),
                        )
                        current = observation.get("current") or {}
                        live.decisions.append(
                            TrajectoryDecision(
                                features=cpu_batch(features),
                                action=sampled.indices,
                                stopped=sampled.stopped,
                                old_log_prob=sampled.log_prob,
                                old_value=sampled.value,
                                entropy=sampled.entropy,
                                turn=int(current.get("turn", 0)),
                                inference_ms=latency_ms / len(actions),
                                selected_options=selected,
                            )
                        )
                        live.connection.send(
                            {"kind": "action", "action": list(sampled.indices)}
                        )
                fill_workers()
        finally:
            for live in live_by_connection.values():
                live.connection.close()
                if live.process.is_alive():
                    live.process.terminate()
                live.process.join(timeout=2.0)
        return sorted(results, key=lambda episode: episode.episode_id)


__all__ = ["RolloutCollector"]
