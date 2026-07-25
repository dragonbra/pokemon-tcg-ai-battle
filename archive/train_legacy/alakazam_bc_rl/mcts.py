"""Small, deterministic PUCT search utilities for official SearchState trees.

The simulator owns state transitions.  This module only manages the tree,
model priors, leaf values, and visit-count targets, so it can be tested without
loading the official C++ runtime.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


Selection = tuple[int, ...]


@dataclass
class MCTSChild:
    selection: Selection
    prior: float
    node: "MCTSNode | None" = None


@dataclass
class MCTSNode:
    state: Any
    observation: Any
    player_index: int
    value: float
    parent: "MCTSNode | None" = None
    action_from_parent: Selection | None = None
    prior: float = 1.0
    visits: int = 0
    total_value: float = 0.0
    children: list[MCTSChild] = field(default_factory=list)
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else self.value


class PUCTSearch:
    """Run a finite-budget PUCT search over an injected simulator API.

    Values are always from ``root_player``'s point of view.  The caller's
    ``evaluate`` callback must follow that contract.  ``expand`` returns legal
    selections and their policy priors for the observation at a node.
    """

    def __init__(
        self,
        *,
        root_player: int,
        simulations: int,
        cpuct: float,
        step: Callable[[Any, list[int]], Any],
        observation: Callable[[Any], Any],
        player_index: Callable[[Any], int],
        evaluate: Callable[[Any], float],
        expand: Callable[[Any], Iterable[tuple[Selection, float]]],
        rng: random.Random | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if cpuct <= 0:
            raise ValueError("cpuct must be positive")
        self.root_player = root_player
        self.simulations = simulations
        self.cpuct = cpuct
        self.step = step
        self.observation = observation
        self.player_index = player_index
        self.evaluate = evaluate
        self.expand = expand
        self.rng = rng or random.Random(0)

    def _new_node(
        self,
        state: Any,
        *,
        parent: MCTSNode | None = None,
        action_from_parent: Selection | None = None,
        prior: float = 1.0,
    ) -> MCTSNode:
        obs = self.observation(state)
        node = MCTSNode(
            state=state,
            observation=obs,
            player_index=self.player_index(obs),
            value=float(self.evaluate(obs)),
            parent=parent,
            action_from_parent=action_from_parent,
            prior=prior,
        )
        self._expand_node(node)
        return node

    def _expand_node(self, node: MCTSNode) -> None:
        if node.expanded:
            return
        raw_children = list(self.expand(node.observation))
        if raw_children:
            priors = [max(0.0, float(prior)) for _selection, prior in raw_children]
            total = sum(priors)
            if total <= 0.0:
                priors = [1.0 / len(raw_children)] * len(raw_children)
            else:
                priors = [prior / total for prior in priors]
            node.children = [
                MCTSChild(selection=selection, prior=prior)
                for (selection, _), prior in zip(raw_children, priors)
            ]
        node.expanded = True

    def _select_child(self, node: MCTSNode) -> MCTSChild:
        if not node.children:
            raise RuntimeError("cannot select from a leaf node")
        scale = self.cpuct * math.sqrt(max(1, node.visits))
        scores: list[float] = []
        for child in node.children:
            if child.node is None:
                child_value = node.value
                child_visits = 0
            else:
                child_value = child.node.mean_value
                child_visits = child.node.visits
            if node.player_index != self.root_player:
                child_value = -child_value
            scores.append(
                child_value + scale * child.prior / (1.0 + child_visits)
            )
        best = max(scores)
        candidates = [child for child, score in zip(node.children, scores) if score == best]
        return self.rng.choice(candidates)

    def _backup(self, path: list[MCTSNode], value: float) -> None:
        value = max(-1.0, min(1.0, float(value)))
        for node in path:
            node.visits += 1
            node.total_value += value

    def run(self, root_state: Any) -> MCTSNode:
        root = self._new_node(root_state)
        for _ in range(self.simulations):
            node = root
            path = [root]
            while node.children:
                child = self._select_child(node)
                if child.node is None:
                    child_state = self.step(node.state, list(child.selection))
                    child.node = self._new_node(
                        child_state,
                        parent=node,
                        action_from_parent=child.selection,
                        prior=child.prior,
                    )
                    node = child.node
                    path.append(node)
                    # A simulation expands one new leaf, then backs up its
                    # value. Continuing here would turn a finite PUCT budget
                    # into an accidental full-game rollout.
                    break
                node = child.node
                path.append(node)
                if not node.children:
                    break
            self._backup(path, node.value)
        return root

    @staticmethod
    def root_policy(root: MCTSNode, temperature: float = 1.0) -> list[float]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        weights = [
            float(child.node.visits) if child.node is not None else 0.0
            for child in root.children
        ]
        if not any(weights):
            weights = [max(0.0, child.prior) for child in root.children]
        if temperature != 1.0:
            weights = [weight ** (1.0 / temperature) for weight in weights]
        total = sum(weights)
        if total <= 0.0:
            return [1.0 / len(root.children)] * len(root.children) if root.children else []
        return [weight / total for weight in weights]

    @staticmethod
    def root_values(root: MCTSNode) -> list[float]:
        return [
            child.node.mean_value if child.node is not None else root.value
            for child in root.children
        ]
