"""Convert focal-only trajectories into episode-balanced GAE batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..rollout.protocol import EpisodeTrajectory
from .compound_gae import compound_gae
from ..integrated.prize import prize_gae
from ..policy.own_archetype import OwnArchetypeVocabulary


@dataclass(frozen=True)
class PreparedBatch:
    features: tuple[dict[str, Tensor], ...]
    sequences: Tensor
    lengths: Tensor
    stopped: Tensor
    rollout_log_prob: Tensor
    old_value: Tensor
    terminal_return: Tensor
    gae_return: Tensor
    advantage: Tensor
    episode_weight: Tensor
    episode_index: Tensor
    turns: Tensor
    terminal_turns: Tensor
    candidate_first: Tensor
    opponents: tuple[str, ...]
    source_policy_update: int
    gamma: float
    gae_lambda: float
    credit_clock: str
    loss_weighting: str
    semantic_boundaries: int
    first_decision_credit_mean: float
    macro_actions: tuple[dict | None, ...]
    prize_reward: Tensor
    own_prize_reward: Tensor
    opponent_prize_reward: Tensor
    prize_return: Tensor
    prize_advantage: Tensor
    opponent_meta_label: Tensor
    rollout_meta_logits: Tensor

    @property
    def decisions(self) -> int:
        return len(self.features)


def _episode_gae(
    values: list[float],
    reward: float,
    turns: list[int | None],
    *,
    gamma: float,
    gae_lambda: float,
    credit_clock: str,
) -> tuple[list[float], list[float], int, float]:
    if not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1]")
    if credit_clock not in {"selection", "turn"}:
        raise ValueError(f"unsupported credit clock: {credit_clock}")
    if len(turns) != len(values):
        raise ValueError("turn metadata length does not match values")
    if credit_clock == "turn" and any(
        isinstance(turn, bool) or not isinstance(turn, int) or turn < 0 for turn in turns
    ):
        raise ValueError("turn clock requires nonnegative turn metadata")
    advantages = [0.0] * len(values)
    next_value = 0.0
    next_advantage = 0.0
    boundaries = 0
    for index in range(len(values) - 1, -1, -1):
        immediate = reward if index == len(values) - 1 else 0.0
        delta = immediate + gamma * next_value - values[index]
        if index == len(values) - 1:
            transition_lambda = 1.0
        elif credit_clock == "selection":
            transition_lambda = gae_lambda
            boundaries += 1
        else:
            transition_lambda = gae_lambda if turns[index] != turns[index + 1] else 1.0
            boundaries += int(turns[index] != turns[index + 1])
        current = delta + gamma * transition_lambda * next_advantage
        advantages[index] = current
        next_advantage = current
        next_value = values[index]
    first_credit = gae_lambda**boundaries if values else 0.0
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns, boundaries, first_credit


def prepare_episodes(
    episodes: list[EpisodeTrajectory],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    credit_clock: str = "selection",
    loss_weighting: str = "episode_equal_decisions",
    prize_mode: str = "off",
    gamma_prize: float = 0.97,
    lambda_prize: float = 0.97,
    prize_scale: float = 1.0 / 24.0,
) -> PreparedBatch:
    if loss_weighting not in {"episode_equal_decisions", "episode_equal_turns"}:
        raise ValueError(f"unsupported loss_weighting: {loss_weighting}")
    valid = [
        episode for episode in episodes
        if episode.valid and episode.reward is not None
        and (episode.policy_transitions or episode.decisions)
        and (
            not episode.policy_transitions
            or all(transition.valid for transition in episode.policy_transitions)
        )
    ]
    if not valid:
        raise ValueError("no valid terminal episodes to prepare")
    use_compound = all(episode.policy_transitions for episode in valid)
    updates = (
        {
            int(transition.metadata["policy_update"])
            for episode in valid for transition in episode.policy_transitions
        }
        if use_compound else
        {decision.policy_update for episode in valid for decision in episode.decisions}
    )
    if len(updates) != 1:
        raise ValueError("PPO batch must contain exactly one source_policy_update")
    features: list[dict[str, Tensor]] = []
    actions: list[tuple[int, ...]] = []
    stopped: list[bool] = []
    log_prob: list[float] = []
    values: list[float] = []
    terminal: list[float] = []
    returns: list[float] = []
    advantages: list[float] = []
    weights: list[float] = []
    episode_indices: list[int] = []
    turns: list[int] = []
    terminal_turns: list[int] = []
    candidate_first: list[bool] = []
    opponents: list[str] = []
    macro_actions: list[dict | None] = []
    prize_rewards: list[float] = []
    own_prize_rewards: list[float] = []
    opponent_prize_rewards: list[float] = []
    prize_returns: list[float] = []
    prize_advantages: list[float] = []
    taxonomy = OwnArchetypeVocabulary.load()
    meta_labels: list[int] = []
    meta_logits: list[list[float]] = []
    semantic_boundaries = 0
    first_decision_credits: list[float] = []
    for episode_index, episode in enumerate(valid):
        if not episode.decisions and not episode.policy_transitions:
            continue
        if use_compound:
            records = episode.policy_transitions
            targets = compound_gae(
                records, gae_lambda=gae_lambda, credit_clock=credit_clock
            )
            ep_advantages = targets.advantages.tolist()
            ep_returns = targets.returns.tolist()
            record_turns = [item.metadata.get("turn") for item in records]
            ep_boundaries = sum(
                1
                for left, right in zip(record_turns, record_turns[1:])
                if credit_clock == "selection" or left != right
            )
            first_credit = gae_lambda**ep_boundaries
            prize_prediction = torch.tensor([
                float(item.metadata.get("pre_action_prize_value", 0.0)) for item in records
            ])
            prize_targets = prize_gae(
                records, prize_prediction, mode=prize_mode,
                gamma_prize=gamma_prize, lambda_prize=lambda_prize, scale=prize_scale,
            )
            ep_prize_rewards = prize_targets.rewards.tolist()
            ep_prize_returns = prize_targets.returns.tolist()
            ep_prize_advantages = prize_targets.advantages.tolist()
        else:
            records = episode.decisions
            ep_values = [decision.value for decision in records]
            record_turns = [decision.turn for decision in records]
            ep_advantages, ep_returns, ep_boundaries, first_credit = _episode_gae(
                ep_values, float(episode.reward), record_turns,
                gamma=gamma, gae_lambda=gae_lambda, credit_clock=credit_clock,
            )
            ep_prize_rewards = [0.0] * len(records)
            ep_prize_returns = [0.0] * len(records)
            ep_prize_advantages = [0.0] * len(records)
        semantic_boundaries += ep_boundaries
        first_decision_credits.append(first_credit)
        turn_counts: dict[int, int] = {}
        if loss_weighting == "episode_equal_turns":
            for record, record_turn in zip(records, record_turns, strict=True):
                if (
                    isinstance(record_turn, bool)
                    or not isinstance(record_turn, int)
                    or record_turn < 0
                ):
                    raise ValueError("turn-equal loss_weighting requires turn metadata")
                turn_counts[record_turn] = turn_counts.get(record_turn, 0) + 1
        for record, record_turn, advantage, gae_return, prize_reward, prize_return, prize_advantage in zip(
            records, record_turns, ep_advantages, ep_returns, ep_prize_rewards,
            ep_prize_returns, ep_prize_advantages, strict=True
        ):
            if loss_weighting == "episode_equal_turns":
                weight = 1.0 / (len(turn_counts) * turn_counts[record_turn])
            else:
                weight = 1.0 / len(records)
            if use_compound:
                features.append(record.pre_action_features)
                actions.append(record.canonical_macro_action.root_indices)
                stopped.append(record.canonical_macro_action.root_stopped)
                log_prob.append(record.joint_old_logprob)
                values.append(record.pre_action_value)
                macro_actions.append(
                    record.canonical_macro_action.parameters
                    if record.canonical_macro_action.family == "phantom_dive" else None
                )
            else:
                features.append(record.features)
                actions.append(record.indices)
                stopped.append(record.stopped)
                log_prob.append(record.log_prob)
                values.append(record.value)
                macro_actions.append(record.macro_action)
            terminal.append(float(episode.reward))
            returns.append(gae_return)
            advantages.append(advantage)
            weights.append(weight)
            episode_indices.append(episode_index)
            turns.append(record_turn if record_turn is not None else -1)
            if record_turn is not None and record_turn > episode.turns:
                raise ValueError("decision turn exceeds terminal engine turn")
            terminal_turns.append(episode.turns)
            candidate_first.append(episode.job.focal_first)
            opponents.append(episode.job.opponent_id)
            prize_rewards.append(prize_reward)
            if use_compound:
                own_prize_rewards.append(
                    float(record.metadata.get("focal_prizes_taken", 0)) * prize_scale
                )
                opponent_prize_rewards.append(
                    float(record.metadata.get("opponent_prizes_taken", 0)) * prize_scale
                )
            else:
                own_prize_rewards.append(0.0)
                opponent_prize_rewards.append(0.0)
            prize_returns.append(prize_return)
            prize_advantages.append(prize_advantage)
            meta_labels.append(
                taxonomy.classify_opponent_target(episode.job.opponent_deck)
            )
            raw_meta = (
                record.metadata.get("opponent_meta_logits")
                if use_compound else record.auxiliary_values.get("opponent_meta_logits")
            )
            if not isinstance(raw_meta, list) or len(raw_meta) != 15:
                raise ValueError("0042 trajectory is missing 15-class pretrained Meta logits")
            meta_logits.append([float(value) for value in raw_meta])
    if not features:
        raise ValueError("valid episodes contain no focal decisions")
    width = max(1, max(len(action) for action in actions))
    sequences = torch.full((len(actions), width), -1, dtype=torch.long)
    lengths = torch.tensor([len(action) for action in actions], dtype=torch.long)
    for index, action in enumerate(actions):
        if action:
            sequences[index, : len(action)] = torch.tensor(action, dtype=torch.long)
    raw_advantage = torch.tensor(advantages, dtype=torch.float32)
    episode_weight = torch.tensor(weights, dtype=torch.float32)
    mean = (raw_advantage * episode_weight).sum() / episode_weight.sum()
    variance = (
        (raw_advantage - mean).square() * episode_weight
    ).sum() / episode_weight.sum()
    normalized = (raw_advantage - mean) / variance.sqrt().clamp_min(1e-6)
    raw_prize_advantage = torch.tensor(prize_advantages, dtype=torch.float32)
    prize_mean = (raw_prize_advantage * episode_weight).sum() / episode_weight.sum()
    prize_variance = (
        (raw_prize_advantage - prize_mean).square() * episode_weight
    ).sum() / episode_weight.sum()
    normalized_prize = (
        (raw_prize_advantage - prize_mean) / prize_variance.sqrt().clamp_min(1e-6)
        if prize_mode != "off" else torch.zeros_like(raw_prize_advantage)
    )
    return PreparedBatch(
        features=tuple(features),
        sequences=sequences,
        lengths=lengths,
        stopped=torch.tensor(stopped, dtype=torch.bool),
        rollout_log_prob=torch.tensor(log_prob, dtype=torch.float32),
        old_value=torch.tensor(values, dtype=torch.float32),
        terminal_return=torch.tensor(terminal, dtype=torch.float32),
        gae_return=torch.tensor(returns, dtype=torch.float32),
        advantage=normalized,
        episode_weight=episode_weight,
        episode_index=torch.tensor(episode_indices, dtype=torch.long),
        turns=torch.tensor(turns, dtype=torch.long),
        terminal_turns=torch.tensor(terminal_turns, dtype=torch.long),
        candidate_first=torch.tensor(candidate_first, dtype=torch.bool),
        opponents=tuple(opponents),
        source_policy_update=next(iter(updates)),
        gamma=gamma,
        gae_lambda=gae_lambda,
        credit_clock=credit_clock,
        loss_weighting=loss_weighting,
        semantic_boundaries=semantic_boundaries,
        first_decision_credit_mean=sum(first_decision_credits) / len(first_decision_credits),
        macro_actions=tuple(macro_actions),
        prize_reward=torch.tensor(prize_rewards, dtype=torch.float32),
        own_prize_reward=torch.tensor(own_prize_rewards, dtype=torch.float32),
        opponent_prize_reward=torch.tensor(opponent_prize_rewards, dtype=torch.float32),
        prize_return=torch.tensor(prize_returns, dtype=torch.float32),
        prize_advantage=normalized_prize,
        opponent_meta_label=torch.tensor(meta_labels, dtype=torch.long),
        rollout_meta_logits=torch.tensor(meta_logits, dtype=torch.float32),
    )


def training_batch_metrics(batch: PreparedBatch, *, include_detailed: bool = True) -> dict[str, float]:
    weights = batch.episode_weight / batch.episode_weight.sum()

    def moments(values: Tensor) -> tuple[Tensor, Tensor]:
        mean = (values * weights).sum()
        return mean, ((values - mean).square() * weights).sum()

    advantage_mean, advantage_variance = moments(batch.advantage)
    return_mean, return_variance = moments(batch.gae_return)
    value_mean, value_variance = moments(batch.old_value)
    residual = batch.gae_return - batch.old_value
    residual_mean = (residual * weights).sum()
    residual_variance = ((residual - residual_mean).square() * weights).sum()
    explained = torch.where(
        return_variance > 1e-8,
        1.0 - residual_variance / return_variance,
        torch.zeros_like(return_variance),
    )
    result = {
        "ppo/advantage_mean": float(advantage_mean),
        "ppo/advantage_std": float(advantage_variance.sqrt()),
        "ppo/return_mean": float(return_mean),
        "ppo/return_std": float(return_variance.sqrt()),
        "ppo/old_value_mean": float(value_mean),
        "ppo/old_value_std": float(value_variance.sqrt()),
        "ppo/explained_variance": float(explained),
        "ppo/semantic_boundaries": float(batch.semantic_boundaries),
        "ppo/semantic_boundaries_per_episode": float(
            batch.semantic_boundaries / max(1, len(set(batch.episode_index.tolist())))
        ),
        "ppo/first_decision_terminal_credit_mean": batch.first_decision_credit_mean,
        "ppo/credit_clock_turn": float(batch.credit_clock == "turn"),
        "ppo/loss_weighting_turn_equal": float(
            batch.loss_weighting == "episode_equal_turns"
        ),
        "reward/terminal_episode_sum_mean": float(torch.stack([
            batch.terminal_return[batch.episode_index.eq(index).nonzero(as_tuple=False)[0, 0]]
            for index in batch.episode_index.unique()
        ]).mean()),
        "reward/own_prize_component_sum": float(batch.own_prize_reward.sum()),
        "reward/opponent_prize_component_sum": float(batch.opponent_prize_reward.sum()),
        "reward/net_prize_component_sum": float(batch.prize_reward.sum()),
        "ppo/prize_to_win_advantage_std_ratio": float(
            batch.prize_advantage.std(unbiased=False)
            / batch.advantage.std(unbiased=False).clamp_min(1e-8)
        ),
    }
    if include_detailed:
        result.update(value_diagnostic_metrics(batch))
        result.update(meta_diagnostic_metrics(batch))
    return result


def meta_diagnostic_metrics(batch: PreparedBatch) -> dict[str, float]:
    probabilities = batch.rollout_meta_logits.softmax(dim=-1)
    prediction = probabilities.argmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    groups = {
        "all": torch.ones_like(batch.turns, dtype=torch.bool),
        "raw_turn_0": batch.turns.eq(0),
        "early": batch.turns.le(3),
        "middle": batch.turns.ge(4) & batch.turns.le(7),
        "late": batch.turns.ge(8),
    }
    metrics: dict[str, float] = {}
    for name, mask in groups.items():
        prefix = f"meta/{name}"
        count = int(mask.sum())
        metrics[f"{prefix}/decisions"] = float(count)
        if count:
            metrics[f"{prefix}/accuracy"] = float(
                prediction[mask].eq(batch.opponent_meta_label[mask]).float().mean()
            )
            metrics[f"{prefix}/entropy"] = float(entropy[mask].mean())
    class_accuracy = []
    for class_id in range(15):
        mask = batch.opponent_meta_label.eq(class_id)
        if bool(mask.any()):
            accuracy = prediction[mask].eq(class_id).float().mean()
            metrics[f"meta/class_{class_id:02d}/accuracy"] = float(accuracy)
            class_accuracy.append(accuracy)
    metrics["meta/macro_accuracy"] = float(
        torch.stack(class_accuracy).mean() if class_accuracy else torch.zeros(())
    )
    return metrics


def value_diagnostic_metrics(batch: PreparedBatch) -> dict[str, float]:
    """Audit rollout Value predictions without an additional Encoder pass."""
    turns = batch.turns
    remaining = batch.terminal_turns - turns
    win = batch.terminal_return == 1.0
    loss = batch.terminal_return == -1.0
    draw = batch.terminal_return == 0.0
    opening = turns <= 3
    near_terminal = remaining <= 1
    groups = {
        "all": torch.ones_like(turns, dtype=torch.bool),
        "turn_00_03": opening,
        "turn_04_07": (turns >= 4) & (turns <= 7),
        "turn_08_11": (turns >= 8) & (turns <= 11),
        "turn_12_plus": turns >= 12,
        "remaining_00_01": near_terminal,
        "remaining_02_03": (remaining >= 2) & (remaining <= 3),
        "remaining_04_plus": remaining >= 4,
        "focal_win": win,
        "focal_loss": loss,
        "draw": draw,
        "focal_win_opening": win & opening,
        "focal_win_near_terminal": win & near_terminal,
        "focal_loss_opening": loss & opening,
        "focal_loss_near_terminal": loss & near_terminal,
    }

    def weighted_moments(values: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        weights = weights / weights.sum()
        mean = (values * weights).sum()
        variance = ((values - mean).square() * weights).sum()
        return mean, variance

    metrics: dict[str, float] = {}
    for name, mask in groups.items():
        prefix = f"value_diag/{name}"
        count = int(mask.sum())
        metrics[f"{prefix}/decisions"] = float(count)
        if count == 0:
            continue
        weights = batch.episode_weight[mask]
        value = batch.old_value[mask]
        gae = batch.gae_return[mask]
        terminal = batch.terminal_return[mask]
        value_mean, value_variance = weighted_moments(value, weights)
        gae_mean, gae_variance = weighted_moments(gae, weights)
        terminal_mean, terminal_variance = weighted_moments(terminal, weights)
        gae_residual = gae - value
        terminal_residual = terminal - value
        _, gae_residual_variance = weighted_moments(gae_residual, weights)
        terminal_bias, terminal_residual_variance = weighted_moments(terminal_residual, weights)
        metrics.update({
            f"{prefix}/value_mean": float(value_mean),
            f"{prefix}/value_std": float(value_variance.sqrt()),
            f"{prefix}/gae_target_mean": float(gae_mean),
            f"{prefix}/terminal_target_mean": float(terminal_mean),
            f"{prefix}/terminal_residual_mean": float(terminal_bias),
            f"{prefix}/explained_variance_gae": float(
                1.0 - gae_residual_variance / gae_variance
            ) if float(gae_variance) > 1e-8 else 0.0,
            f"{prefix}/explained_variance_terminal": float(
                1.0 - terminal_residual_variance / terminal_variance
            ) if float(terminal_variance) > 1e-8 else 0.0,
        })
    return metrics


__all__ = [
    "PreparedBatch", "meta_diagnostic_metrics", "prepare_episodes",
    "training_batch_metrics", "value_diagnostic_metrics",
]
