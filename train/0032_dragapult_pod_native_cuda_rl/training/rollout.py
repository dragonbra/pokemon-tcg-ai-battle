"""Fully CUDA-resident official-engine rollout collection for 0032."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import Tensor

from ..model.routed_policy import RoutedResidentPolicy
from .batch import FEATURE_NAMES, PreparedBatch, prepare_complete_episodes


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


@dataclass(frozen=True, slots=True)
class RolloutResult:
    batch: PreparedBatch
    metrics: dict[str, float]


class ResidentRolloutCollector:
    def __init__(
        self,
        *,
        engine: Any,
        policy: RoutedResidentPolicy,
        decks: Tensor,
        seeds: Tensor,
        opponent_head_indices: Tensor,
        opponent_deck_ids: tuple[str, ...] | None = None,
        steps: int,
        actor_execution: str = "cuda_graph",
    ) -> None:
        if steps < 1 or decks.ndim != 3 or decks.shape[1:] != (2, 60):
            raise ValueError("bad rollout dimensions")
        self.engine = engine
        self.policy = policy
        self.decks = decks
        self.seeds = seeds
        self.opponent_head_indices = opponent_head_indices
        if opponent_deck_ids is not None and len(opponent_deck_ids) != decks.shape[0]:
            raise ValueError("opponent_deck_ids must align with rollout lanes")
        self.opponent_deck_ids = opponent_deck_ids
        self.steps = steps
        if actor_execution not in {"eager", "cuda_graph"}:
            raise ValueError("actor_execution must be eager or cuda_graph")
        self.actor_execution = actor_execution
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_actions: Any | None = None
        self._graph_stochastic_changed = False
        self._storage: dict[str, Tensor] | None = None

    def collect(self, *, source_policy_update: int) -> RolloutResult:
        device = self.decks.device
        lanes = self.decks.shape[0]
        lane_mask = torch.ones(lanes, dtype=torch.bool, device=device)
        episode_id = torch.arange(lanes, dtype=torch.long, device=device)
        self.engine.reset_seeded_interactive_masked(
            self.decks, self.seeds, lane_mask
        )
        illegal = torch.zeros((), dtype=torch.long, device=device)
        decisions = torch.zeros((), dtype=torch.long, device=device)
        completed = torch.zeros((), dtype=torch.long, device=device)
        focal_wins = torch.zeros((), dtype=torch.long, device=device)
        focal_losses = torch.zeros((), dtype=torch.long, device=device)
        draws = torch.zeros((), dtype=torch.long, device=device)
        lane_completed = torch.zeros(lanes, dtype=torch.long, device=device)
        lane_wins = torch.zeros(lanes, dtype=torch.long, device=device)
        lane_losses = torch.zeros(lanes, dtype=torch.long, device=device)
        lane_draws = torch.zeros(lanes, dtype=torch.long, device=device)

        graph = self._graph
        graph_actions = self._graph_actions
        graph_stochastic_changed = self._graph_stochastic_changed
        self.engine.reset_seeded_interactive_masked(
            self.decks, self.seeds, lane_mask
        )
        self.engine.advance_to_decision()
        graph_raw = dict(self.engine.encode_policy_v1())
        template = dict(graph_raw)
        template["entity_mask"] = graph_raw["entity_mask"].bool()
        template["option_mask"] = graph_raw["option_mask"].bool()
        if self._storage is None:
            self._storage = {
                name: torch.empty(
                    (self.steps, *template[name].shape),
                    dtype=template[name].dtype,
                    device=device,
                )
                for name in FEATURE_NAMES
            }
            self._storage.update({
                "sequences": torch.empty(
                    self.steps, lanes, 64, dtype=torch.long, device=device
                ),
                "lengths": torch.empty(self.steps, lanes, dtype=torch.long, device=device),
                "logprob": torch.empty(self.steps, lanes, dtype=torch.float32, device=device),
                "value": torch.empty(self.steps, lanes, dtype=torch.float32, device=device),
                "focal_mask": torch.empty(self.steps, lanes, dtype=torch.bool, device=device),
                "episode_id": torch.empty(self.steps, lanes, dtype=torch.long, device=device),
                "terminal_episode_id": torch.empty(
                    self.steps, lanes, dtype=torch.long, device=device
                ),
                "terminal_result": torch.empty(
                    self.steps, lanes, dtype=torch.long, device=device
                ),
            })
        storage = self._storage
        if self.actor_execution == "cuda_graph" and graph is None:

            def graph_actor() -> Any:
                batch = dict(graph_raw)
                batch["entity_mask"] = graph_raw["entity_mask"].bool()
                batch["option_mask"] = graph_raw["option_mask"].bool()
                return self.policy.act(
                    batch, self.opponent_head_indices, stochastic_focal=True
                )

            with torch.inference_mode():
                graph_actor()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_actions = graph_actor()
                graph.replay()
                first_sample = graph_actions.sequences.clone()
                graph.replay()
                torch.cuda.synchronize(device)
                graph_stochastic_changed = not torch.equal(
                    first_sample, graph_actions.sequences
                )
            if not graph_stochastic_changed:
                raise RuntimeError("CUDA Graph replay did not advance stochastic actions")
            self._graph = graph
            self._graph_actions = graph_actions
            self._graph_stochastic_changed = graph_stochastic_changed

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            for step_index in range(self.steps):
                statuses = self.engine.statuses()
                terminal = statuses.eq(TERMINAL)
                results = self.engine.game_results().clone()
                storage["terminal_episode_id"][step_index].copy_(
                    torch.where(terminal, episode_id, torch.full_like(episode_id, -1))
                )
                storage["terminal_result"][step_index].copy_(
                    torch.where(terminal, results.long(), torch.zeros_like(episode_id))
                )
                completed += terminal.long().sum()
                focal_wins += (terminal & results.eq(1)).long().sum()
                focal_losses += (terminal & results.eq(2)).long().sum()
                draws += (terminal & results.eq(3)).long().sum()
                lane_completed += terminal.long()
                lane_wins += (terminal & results.eq(1)).long()
                lane_losses += (terminal & results.eq(2)).long()
                lane_draws += (terminal & results.eq(3)).long()
                self.seeds.add_(terminal.long() * lanes)
                episode_id += terminal.long() * lanes
                self.engine.reset_seeded_interactive_masked(
                    self.decks, self.seeds, terminal
                )
                self.engine.advance_to_decision()
                ready = self.engine.statuses().eq(NEEDS_ACTION)
                borrowed = dict(self.engine.encode_policy_v1())
                raw = dict(borrowed)
                raw["entity_mask"] = borrowed["entity_mask"].bool()
                raw["option_mask"] = borrowed["option_mask"].bool()
                if graph is None:
                    actions = self.policy.act(
                        raw, self.opponent_head_indices, stochastic_focal=True
                    )
                else:
                    graph.replay()
                    actions = graph_actions
                for name in FEATURE_NAMES:
                    storage[name][step_index].copy_(raw[name])
                storage["sequences"][step_index].copy_(actions.sequences)
                storage["lengths"][step_index].copy_(actions.lengths)
                storage["logprob"][step_index].copy_(actions.logprob)
                storage["value"][step_index].copy_(actions.value)
                storage["focal_mask"][step_index].copy_(actions.focal_mask)
                storage["episode_id"][step_index].copy_(episode_id)
                decisions += ready.long().sum()
                illegal += (ready & ~actions.legal).long().sum()
                self.engine.pack_actions(actions.sequences, actions.lengths)
                self.engine.apply_packed_actions()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        stacked = {name: value.cpu() for name, value in storage.items()}
        statuses = self.engine.statuses()
        error_count = int(statuses.eq(ERROR).long().sum().item())
        illegal_count = int(illegal.item())
        if error_count or illegal_count:
            raise RuntimeError(
                f"official CUDA rollout failed: errors={error_count} illegal={illegal_count}"
            )
        batch = prepare_complete_episodes(
            stacked, source_policy_update=source_policy_update
        )
        # Incomplete trajectories are intentionally excluded. Advance every lane
        # to a disjoint deterministic seed window so they are not replayed next update.
        self.seeds.add_(lanes * (self.steps + 1))
        decision_count = int(decisions.item())
        completed_count = int(completed.item())
        win_count = int(focal_wins.item())
        loss_count = int(focal_losses.item())
        draw_count = int(draws.item())
        metrics = {
            "rollout/wall_seconds": elapsed,
            "rollout/engine_decisions": float(decision_count),
            "rollout/decisions_per_second": decision_count / elapsed,
            "rollout/completed_episodes": float(completed_count),
            "rollout/focal_wins": float(win_count),
            "rollout/focal_losses": float(loss_count),
            "rollout/draws": float(draw_count),
            "rollout/focal_win_rate": win_count / max(1, completed_count),
            "rollout/episodes_per_second": completed_count / elapsed,
            "rollout/focal_training_decisions": float(batch.decisions),
            "rollout/discarded_incomplete_decisions": float(
                batch.discarded_incomplete_decisions
            ),
            "rollout/error_lanes": float(error_count),
            "rollout/illegal_rows": float(illegal_count),
            "rollout/actor_cuda_graph": float(graph is not None),
            "rollout/graph_stochastic_changed": float(graph_stochastic_changed),
        }
        if self.opponent_deck_ids is not None:
            per_deck: dict[str, list[int]] = {}
            rows = zip(
                self.opponent_deck_ids,
                lane_completed.cpu().tolist(),
                lane_wins.cpu().tolist(),
                lane_losses.cpu().tolist(),
                lane_draws.cpu().tolist(),
            )
            for deck_id, deck_episodes, deck_wins, deck_losses, deck_draws in rows:
                totals = per_deck.setdefault(deck_id, [0, 0, 0, 0])
                totals[0] += deck_episodes
                totals[1] += deck_wins
                totals[2] += deck_losses
                totals[3] += deck_draws
            for deck_id, (deck_episodes, deck_wins, deck_losses, deck_draws) in per_deck.items():
                prefix = f"rollout/opponent/{deck_id}"
                metrics[f"{prefix}/episodes"] = float(deck_episodes)
                metrics[f"{prefix}/wins"] = float(deck_wins)
                metrics[f"{prefix}/losses"] = float(deck_losses)
                metrics[f"{prefix}/draws"] = float(deck_draws)
                metrics[f"{prefix}/win_rate"] = (
                    deck_wins / deck_episodes if deck_episodes else float("nan")
                )
        return RolloutResult(batch=batch, metrics=metrics)


__all__ = ["ResidentRolloutCollector", "RolloutResult"]
