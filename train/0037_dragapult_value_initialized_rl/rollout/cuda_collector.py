"""CUDA-resident official-engine collector for 0037 full-semantic PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Literal

import torch
from torch import nn

from .protocol import EpisodeTrajectory, RolloutJob, TrajectoryDecision


ROOT = Path(__file__).resolve().parents[3]
CUDA_PYTHON = ROOT / "engine_cuda/python"
if str(CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(CUDA_PYTHON))

from ptcg_cuda_engine.semantic0031_bridge import Semantic0031DeviceAdapter
from ptcg_cuda_engine.semantic0031_resident import (
    ResidentJob,
    run_resident_greedy_jobs,
)
from ptcg_cuda_engine.semantic0031_router import Semantic0031ResidentRouter


@dataclass(frozen=True)
class _StagedDecisions:
    job_indices: torch.Tensor
    features: dict[str, torch.Tensor]
    actions: torch.Tensor
    lengths: torch.Tensor
    stopped: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    turns: torch.Tensor


def _build_resident_jobs(jobs: list[RolloutJob]) -> tuple[ResidentJob, ...]:
    return tuple(
        ResidentJob(
            schedule_index=index,
            decks=(job.focal_deck, job.opponent_deck)
            if job.focal_first
            else (job.opponent_deck, job.focal_deck),
            engine_seed=job.seed,
            focal_player=0 if job.focal_first else 1,
            opponent_id=job.opponent_id,
            policy_seed=int(job.policy_seed if job.policy_seed is not None else job.seed),
        )
        for index, job in enumerate(jobs)
    )


class CudaFullSemanticRolloutCollector:
    """Collect focal trajectories through reusable CUDA-resident engine lanes."""

    def __init__(
        self,
        model: nn.Module,
        opponent: nn.Module,
        *,
        device: torch.device,
        rules_path: Path,
        extension_dir: Path,
        lane_count: int = 256,
        mode: Literal["sample", "greedy"] = "sample",
        max_select: int = 64,
        max_decisions: int = 8192,
        check_interval: int = 8,
        ability_repeat_limit: int = 20,
    ) -> None:
        if device.type != "cuda" or lane_count < 1 or check_interval < 1:
            raise ValueError("CUDA collector requires a CUDA device and positive topology")
        extension = extension_dir.resolve() / "_ptcg_cuda.so"
        if not extension.is_file():
            raise FileNotFoundError(extension)
        if not rules_path.is_file():
            raise FileNotFoundError(rules_path)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if str(extension_dir.resolve()) not in sys.path:
            sys.path.insert(0, str(extension_dir.resolve()))
        __import__("_ptcg_cuda")
        self.model = model.eval()
        self.opponent = opponent.eval().requires_grad_(False)
        self.device = device
        self.rules = rules_path.resolve().read_bytes()
        self.lane_count = int(lane_count)
        self.mode = mode
        self.max_select = int(max_select)
        self.max_decisions = int(max_decisions)
        self.check_interval = int(check_interval)
        self.ability_repeat_limit = int(ability_repeat_limit)
        self._metrics: dict[str, float] = {}

    def collect(self, jobs: list[RolloutJob]) -> list[EpisodeTrajectory]:
        if not jobs:
            return []
        if len({job.game_id for job in jobs}) != len(jobs):
            raise ValueError("rollout game IDs must be unique")
        focal_decks = {job.focal_deck for job in jobs}
        if len(focal_decks) != 1:
            raise ValueError("resident 0037 collector requires one exact focal deck")
        updates = {job.source_policy_update for job in jobs}
        if len(updates) != 1:
            raise ValueError("resident rollout must use one source policy update")

        focal_deck = next(iter(focal_decks))
        adapter = Semantic0031DeviceAdapter(
            self.model.actor, focal_deck, max_select=self.max_select
        )
        opponent_transformer = self.opponent.option_encoder.cross_attention_transformer
        router = Semantic0031ResidentRouter(
            focal_adapter=adapter,
            opponent_last_option_layer=opponent_transformer.layers[1],
            opponent_option_norm=opponent_transformer.norm,
            opponent_decoder=self.opponent.action_decoder,
            same_policy=False,
        )
        resident_jobs = _build_resident_jobs(jobs)
        expected = tuple(sorted(self.model.actor.expected_batch_keys))
        staged: list[_StagedDecisions] = []
        staged_bytes = 0

        def value_fn(validated: Any, state: Any, options: torch.Tensor) -> torch.Tensor:
            return self.model.value_from_encoded(validated, state, options)

        def sink(**payload: Any) -> None:
            nonlocal staged_bytes
            route = payload["focal_route"].bool()
            rows = route.nonzero(as_tuple=False).flatten()
            semantic = payload["semantic"]
            routed = payload["routed"]
            values = payload["values"]
            block = _StagedDecisions(
                job_indices=payload["lane_job"].index_select(0, rows).detach().clone(),
                features={
                    name: semantic[name].index_select(0, rows).detach().clone()
                    for name in expected
                },
                actions=routed.actions.index_select(0, rows).detach().clone(),
                lengths=routed.lengths.index_select(0, rows).detach().clone(),
                stopped=routed.stopped.index_select(0, rows).detach().clone(),
                logprob=routed.focal_logprob.index_select(0, rows).detach().clone(),
                entropy=routed.focal_entropy.index_select(0, rows).detach().clone(),
                values=values.index_select(0, rows).detach().clone(),
                turns=payload["turns"].index_select(0, rows).detach().clone(),
            )
            staged.append(block)
            staged_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    *block.features.values(),
                    block.actions,
                    block.lengths,
                    block.stopped,
                    block.logprob,
                    block.entropy,
                    block.values,
                    block.turns,
                    block.job_indices,
                )
            )

        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = run_resident_greedy_jobs(
            jobs=resident_jobs,
            rules=self.rules,
            router=router,
            lane_count=min(self.lane_count, len(jobs)),
            device=self.device,
            device_index=self.device.index or 0,
            max_select=self.max_select,
            max_decisions=self.max_decisions,
            check_interval=self.check_interval,
            ability_repeat_limit=self.ability_repeat_limit,
            focal_greedy=self.mode == "greedy",
            focal_value_fn=value_fn,
            focal_decision_sink=sink,
            compact_prefixes=True,
        )
        hot_loop_seconds = time.perf_counter() - started
        hot_loop_peak_allocated = torch.cuda.max_memory_allocated(self.device)
        hot_loop_peak_reserved = torch.cuda.max_memory_reserved(self.device)

        episodes = [EpisodeTrajectory(job) for job in jobs]
        materialize_started = time.perf_counter()
        for block in staged:
            job_indices = block.job_indices.cpu()
            features = {name: value.cpu() for name, value in block.features.items()}
            actions = block.actions.cpu()
            lengths = block.lengths.cpu()
            stopped = block.stopped.cpu()
            logprob = block.logprob.cpu()
            entropy = block.entropy.cpu()
            values = block.values.cpu()
            turns = block.turns.cpu()
            for row, job_index in enumerate(job_indices.tolist()):
                length = int(lengths[row])
                episodes[job_index].decisions.append(
                    TrajectoryDecision(
                        features={
                            name: value[row : row + 1]
                            for name, value in features.items()
                        },
                        indices=tuple(int(value) for value in actions[row, :length].tolist()),
                        stopped=bool(stopped[row]),
                        log_prob=float(logprob[row]),
                        entropy=float(entropy[row]),
                        value=float(values[row]),
                        policy_update=jobs[job_index].source_policy_update,
                        turn=int(turns[row]),
                    )
                )
        materialize_seconds = time.perf_counter() - materialize_started

        forfeits = set(result.forfeit_schedule_indices)
        for index, (episode, game_result) in enumerate(
            zip(episodes, result.game_results, strict=True)
        ):
            focal_player = 0 if episode.job.focal_first else 1
            reward = 1.0 if game_result == focal_player + 1 else -1.0 if game_result else 0.0
            episode.finish(reward, result.terminal_turns[index])
            episode.diagnostics = {
                "engine_selections": result.engine_selections[index],
                "termination_status": "repeat_forfeit" if index in forfeits else "terminal",
                "termination_error": None,
                "source_policy_update": episode.job.source_policy_update,
                "cuda_resident": True,
            }
        self._metrics = {
            "rollout/inference_batches": float(result.decisions),
            "rollout/inference_requests": float(result.routed_ready_rows),
            "rollout/focal_requests": float(sum(len(ep.decisions) for ep in episodes)),
            "rollout/opponent_requests": float(
                result.routed_ready_rows - sum(len(ep.decisions) for ep in episodes)
            ),
            "rollout/max_batch_size": float(min(self.lane_count, len(jobs))),
            "rollout/mean_batch_size": result.routed_ready_rows / max(1, result.decisions),
            "rollout/inference_seconds": result.gpu_seconds,
            "rollout/cuda_hot_loop_seconds": hot_loop_seconds,
            "rollout/cuda_materialize_seconds": materialize_seconds,
            "rollout/cuda_hot_loop_games_per_second": len(jobs)
            / max(hot_loop_seconds, 1e-9),
            "rollout/cuda_games_per_second": len(jobs)
            / max(hot_loop_seconds + materialize_seconds, 1e-9),
            "rollout/cuda_lane_count": float(min(self.lane_count, len(jobs))),
            "rollout/cuda_refill_events": float(result.refill_events),
            "rollout/cuda_repeat_forfeits": float(len(forfeits)),
            "rollout/cuda_staged_trajectory_bytes": float(staged_bytes),
            "rollout/cuda_peak_allocated_bytes": float(hot_loop_peak_allocated),
            "rollout/cuda_peak_reserved_bytes": float(hot_loop_peak_reserved),
            "rollout/peak_process_tree_rss_bytes": 0.0,
            "rollout/worker_processes": 0.0,
            "rollout/engines_per_worker": 0.0,
            "rollout/inference_channels_per_role": 0.0,
        }
        return episodes

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)


__all__ = ["CudaFullSemanticRolloutCollector"]
