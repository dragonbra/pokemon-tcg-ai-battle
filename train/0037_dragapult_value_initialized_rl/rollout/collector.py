"""Central CUDA batching around persistent multi-battle official-engine workers."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Literal

import torch
from torch import nn

from ..policy.action_distribution import greedy_actions, sample_actions
from ..policy.actor_critic import SemanticActorCritic
from ..policy.batching import cpu_batch, move_batch
from ..semantic_policy.features.collate import collate_canonical_records
from .protocol import EpisodeTrajectory, RolloutJob, TrajectoryDecision
from .pool_worker import run_engine_pool


@dataclass
class _Session:
    job: RolloutJob
    policy_generator: torch.Generator
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    last_decision: dict[str, object] | None = None
    last_action: tuple[int, ...] | None = None
    recent_actions: list[dict[str, object]] = field(default_factory=list)


@dataclass
class _PoolProcess:
    process: mp.Process
    result_connection: Connection
    inference_connections: list[Connection]
    jobs: list[RolloutJob]


class FullSemanticRolloutCollector:
    def __init__(
        self,
        model: SemanticActorCritic,
        opponent: nn.Module,
        *,
        device: torch.device,
        workers: int | None = None,
        worker_processes: int | None = None,
        engines_per_worker: int = 1,
        inference_channels_per_role: int = 1,
        mode: Literal["sample", "greedy"] = "sample",
        coalesce_ms: float = 0.5,
        timeout_seconds: float = 120.0,
    ) -> None:
        if worker_processes is None:
            worker_processes = workers if workers is not None else 8
        if (
            worker_processes < 1
            or engines_per_worker < 1
            or inference_channels_per_role < 1
            or inference_channels_per_role > engines_per_worker
            or coalesce_ms < 0
            or timeout_seconds <= 0
        ):
            raise ValueError("invalid collector configuration")
        self.model = model.eval()
        self.opponent = opponent.eval().requires_grad_(False)
        self.device = device
        self.worker_processes = worker_processes
        self.engines_per_worker = engines_per_worker
        self.inference_channels_per_role = inference_channels_per_role
        self.mode = mode
        self.coalesce_seconds = coalesce_ms / 1000.0
        self.timeout_seconds = timeout_seconds
        self.context = mp.get_context("spawn")
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
        self.peak_process_tree_rss_bytes = 0
        self._focal_prototype_start = self.model.actor.prototype_cache_stats()
        self._opponent_prototype_start = self.opponent.prototype_cache_stats()

    def _session(self, job: RolloutJob) -> _Session:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(job.policy_seed or job.seed)
        return _Session(
            job=job,
            policy_generator=generator,
        )

    def _start_pools(self, jobs: list[RolloutJob]) -> list[_PoolProcess]:
        shard_count = min(self.worker_processes, len(jobs))
        shards = [jobs[index::shard_count] for index in range(shard_count)]
        pools: list[_PoolProcess] = []
        for worker_index, shard in enumerate(shards):
            result_parent, result_child = self.context.Pipe(duplex=False)
            focal_pairs = [self.context.Pipe(duplex=True) for _ in range(
                self.inference_channels_per_role
            )]
            opponent_pairs = [self.context.Pipe(duplex=True) for _ in range(
                self.inference_channels_per_role
            )]
            focal_parents = [pair[0] for pair in focal_pairs]
            focal_children = [pair[1] for pair in focal_pairs]
            opponent_parents = [pair[0] for pair in opponent_pairs]
            opponent_children = [pair[1] for pair in opponent_pairs]
            process = self.context.Process(
                target=run_engine_pool,
                args=(
                    result_child,
                    focal_children,
                    opponent_children,
                    shard,
                    self.engines_per_worker,
                ),
                name=f"0037-value-rl-pool-{worker_index:02d}",
            )
            process.start()
            result_child.close()
            for connection in focal_children + opponent_children:
                connection.close()
            pools.append(_PoolProcess(
                process=process,
                result_connection=result_parent,
                inference_connections=focal_parents + opponent_parents,
                jobs=shard,
            ))
        return pools

    @staticmethod
    def _finish(item: _Session, message: dict, worker_exitcode: int | None) -> EpisodeTrajectory:
        episode = EpisodeTrajectory(item.job, decisions=item.decisions)
        if message.get("valid") and message.get("reward") is not None:
            episode.finish(float(message["reward"]), int(message.get("final_turn") or 0))
        else:
            detail = message.get("error") or message.get("status")
            episode.error = (
                f"{detail}; game_id={item.job.game_id}; opponent={item.job.opponent_id}; "
                f"focal_first={item.job.focal_first}; seed={item.job.seed}; "
                f"worker_exitcode={worker_exitcode}; last_decision={item.last_decision!r}; "
                f"last_action={item.last_action!r}; recent_actions={item.recent_actions!r}"
            )
        episode.diagnostics = {
            "engine_selections": int(message.get("engine_selections", 0)),
            "termination_status": message.get("status"),
            "termination_error": message.get("error"),
            "source_policy_update": item.job.source_policy_update,
            "worker_exitcode": worker_exitcode,
            "last_decision": item.last_decision,
            "last_action": item.last_action,
        }
        return episode

    def _infer_focal(self, requests: list[tuple[_Session, dict]]) -> list[object]:
        started = time.perf_counter()
        cpu_features = collate_canonical_records(
            [message["record"] for _, message in requests]
        )
        encoded = time.perf_counter()
        batch = move_batch(cpu_features, self.device)
        prepared = time.perf_counter()
        actions = (
            sample_actions(
                self.model,
                batch,
                generators=[item.policy_generator for item, _ in requests],
            )
            if self.mode == "sample"
            else greedy_actions(self.model, batch)
        )
        inferred = time.perf_counter()
        self.focal_encode_seconds += encoded - started
        self.focal_collate_move_seconds += prepared - encoded
        self.focal_model_seconds += inferred - prepared
        for index, (item, message) in enumerate(requests):
            action = actions[index]
            turn = message.get("turn")
            if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
                raise RuntimeError(f"official observation has invalid turn: {turn!r}")
            item.decisions.append(
                TrajectoryDecision(
                    cpu_batch({
                        name: value[index : index + 1]
                        for name, value in cpu_features.items()
                    }),
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

    def _infer_opponent(self, requests: list[tuple[_Session, dict]]) -> list[tuple[int, ...]]:
        started = time.perf_counter()
        cpu_features = collate_canonical_records(
            [message["record"] for _, message in requests]
        )
        encoded = time.perf_counter()
        batch = move_batch(cpu_features, self.device)
        prepared = time.perf_counter()
        with torch.inference_mode():
            decoded = self.opponent.deterministic_action_tensors(batch)
        if not bool(decoded.legal.all()):
            raise RuntimeError("Frozen Policy-0806 opponent produced an illegal action")
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
        if not jobs:
            return []
        if len({job.game_id for job in jobs}) != len(jobs):
            raise ValueError("rollout game IDs must be unique")
        sessions = {job.game_id: self._session(job) for job in jobs}
        pools = self._start_pools(jobs)
        result_owners = {pool.result_connection: pool for pool in pools}
        inference_owners = {
            connection: pool
            for pool in pools
            for connection in pool.inference_connections
        }
        active = set(result_owners) | set(inference_owners)
        messages: dict[str, tuple[dict, _PoolProcess]] = {}

        def sample_rss() -> None:
            total = 0
            for pid in (os.getpid(), *(pool.process.pid for pool in pools)):
                if pid is None:
                    continue
                try:
                    status = (Path(f"/proc/{pid}/status")).read_text()
                    line = next(row for row in status.splitlines() if row.startswith("VmRSS:"))
                    total += int(line.split()[1]) * 1024
                except (FileNotFoundError, StopIteration, ValueError, PermissionError):
                    pass
            self.peak_process_tree_rss_bytes = max(
                self.peak_process_tree_rss_bytes, total
            )

        try:
            while len(messages) < len(jobs):
                sample_rss()
                ready = list(wait(list(active), timeout=self.timeout_seconds))
                if not ready:
                    raise TimeoutError("official CPU engine workers stopped responding")
                deadline = time.perf_counter() + self.coalesce_seconds
                while len(ready) < len(active) and time.perf_counter() < deadline:
                    more = wait(
                        [connection for connection in active if connection not in ready],
                        timeout=max(0.0, deadline - time.perf_counter()),
                    )
                    if not more:
                        break
                    ready.extend(more)
                focal: list[tuple[_Session, dict, Connection]] = []
                opponent: list[tuple[_Session, dict, Connection]] = []
                for connection in ready:
                    try:
                        message = connection.recv()
                    except EOFError:
                        active.discard(connection)
                        connection.close()
                        continue
                    if message.get("kind") == "decision":
                        game_id = message.get("game_id")
                        if game_id not in sessions:
                            raise RuntimeError(f"unknown pooled rollout session: {game_id!r}")
                        item = sessions[game_id]
                        selection = message.get("select") or {}
                        options = selection.get("option") or []
                        item.last_decision = {
                            "selection_index": int(message.get("selection_index", -1)),
                            "role": message.get("role"),
                            "actor": message.get("actor"),
                            "turn": message.get("turn"),
                            "select_type": selection.get("type"),
                            "select_context": selection.get("context"),
                            "option_count": len(options),
                            "min_count": selection.get("minCount"),
                            "max_count": selection.get("maxCount"),
                            "options": [
                                {
                                    key: option.get(key)
                                    for key in ("type", "id", "card", "skill", "source", "target")
                                    if key in option
                                }
                                for option in options
                                if isinstance(option, dict)
                            ],
                        }
                        if message.get("role") == "focal":
                            self.focal_encode_seconds += float(message.get("compile_seconds", 0.0))
                        else:
                            self.opponent_encode_seconds += float(message.get("compile_seconds", 0.0))
                        request = (item, message, connection)
                        (focal if message.get("role") == "focal" else opponent).append(request)
                    elif message.get("kind") == "result":
                        game_id = message.get("game_id")
                        if game_id not in sessions or game_id in messages:
                            raise RuntimeError(f"invalid duplicate pooled result: {game_id!r}")
                        messages[game_id] = (message, result_owners[connection])
                    elif message.get("kind") == "worker_done":
                        active.discard(connection)
                        connection.close()
                    else:
                        raise RuntimeError("unknown official CPU worker message")
                started = time.perf_counter()
                routed: list[tuple[_Session, tuple[int, ...], Connection]] = []
                if focal:
                    routed.extend(
                        (item, tuple(action.indices), connection)
                        for (item, _, connection), action in zip(
                            focal,
                            self._infer_focal([(item, message) for item, message, _ in focal]),
                            strict=True,
                        )
                    )
                if opponent:
                    routed.extend(
                        (item, action, connection)
                        for (item, _, connection), action in zip(
                            opponent,
                            self._infer_opponent([(item, message) for item, message, _ in opponent]),
                            strict=True,
                        )
                    )
                if routed:
                    elapsed = time.perf_counter() - started
                    self.inference_batches += int(bool(focal)) + int(bool(opponent))
                    self.inference_requests += len(routed)
                    self.focal_requests += len(focal)
                    self.opponent_requests += len(opponent)
                    self.max_batch_size = max(self.max_batch_size, len(focal), len(opponent))
                    self.inference_seconds += elapsed
                    for item, action, connection in routed:
                        item.last_action = action
                        item.recent_actions.append(
                            {"decision": item.last_decision, "action": action}
                        )
                        del item.recent_actions[:-12]
                        connection.send({"kind": "action", "action": list(action)})
        finally:
            for connection in active:
                try:
                    connection.close()
                except OSError:
                    pass
            for pool in pools:
                pool.process.join(timeout=5)
                if pool.process.is_alive():
                    pool.process.terminate()
                    pool.process.join(timeout=2)
        results: list[EpisodeTrajectory] = []
        for job in jobs:
            payload = messages.get(job.game_id)
            if payload is None:
                message = {
                    "valid": False,
                    "status": "worker_eof",
                    "error": "pool worker exited without a game result",
                }
                owner = next(pool for pool in pools if job in pool.jobs)
            else:
                message, owner = payload
            results.append(self._finish(
                sessions[job.game_id], message, owner.process.exitcode
            ))
        return results

    def metrics(self) -> dict[str, float]:
        focal_cache = self.model.actor.prototype_cache_stats()
        opponent_cache = self.opponent.prototype_cache_stats()
        return {
            "rollout/inference_batches": float(self.inference_batches),
            "rollout/inference_requests": float(self.inference_requests),
            "rollout/focal_requests": float(self.focal_requests),
            "rollout/opponent_requests": float(self.opponent_requests),
            "rollout/max_batch_size": float(self.max_batch_size),
            "rollout/mean_batch_size": (
                float(self.inference_requests) / self.inference_batches
                if self.inference_batches else 0.0
            ),
            "rollout/inference_seconds": self.inference_seconds,
            "rollout/focal_encode_seconds": self.focal_encode_seconds,
            "rollout/focal_collate_move_seconds": self.focal_collate_move_seconds,
            "rollout/focal_model_seconds": self.focal_model_seconds,
            "rollout/opponent_encode_seconds": self.opponent_encode_seconds,
            "rollout/opponent_collate_move_seconds": self.opponent_collate_move_seconds,
            "rollout/opponent_model_seconds": self.opponent_model_seconds,
            "rollout/peak_process_tree_rss_bytes": float(
                self.peak_process_tree_rss_bytes
            ),
            "rollout/worker_processes": float(self.worker_processes),
            "rollout/engines_per_worker": float(self.engines_per_worker),
            "rollout/inference_channels_per_role": float(
                self.inference_channels_per_role
            ),
            **{
                f"rollout/focal_prototype_cache_{name}": float(
                    focal_cache[name] - self._focal_prototype_start[name]
                )
                for name in ("builds", "hits", "invalidations")
            },
            **{
                f"rollout/opponent_prototype_cache_{name}": float(
                    opponent_cache[name] - self._opponent_prototype_start[name]
                )
                for name in ("builds", "hits", "invalidations")
            },
        }


__all__ = ["FullSemanticRolloutCollector"]
