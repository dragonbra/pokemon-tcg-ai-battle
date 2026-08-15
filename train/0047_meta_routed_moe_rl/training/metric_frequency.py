"""Low-overhead metric frequency contract for 0038."""

UPDATE_SCALARS = (
    "rollout_games", "rollout_focal_strategic_decisions", "forced_shortcuts",
    "macro_actions", "invalid", "fallback", "error", "policy_loss", "v_win_loss",
    "v_prize_loss", "opponent_meta_loss", "root_entropy", "allocation_entropy",
    "approx_kl", "clip_fraction", "explained_variance", "learning_rate_actor",
    "learning_rate_value", "rollout_wall_time", "ppo_wall_time",
    "evaluation_wall_time", "games_per_sec", "strategic_decisions_per_sec",
    "cuda_allocated", "cuda_reserved", "cpu_rss",
)

FROZEN_OFFLINE_METRICS = (
    "win_rate", "wilson_ci", "paired_flips", "seat", "matchup", "first_prize_turn",
    "prizes_per_game", "five_prize_loss_rate", "immediate_prize_opportunity",
    "turn_two_attack_rate", "attack_streak", "meta_accuracy_f1_ece",
    "macro_fallback_error",
)

SPARSE_DIAGNOSTICS = (
    "gradient_norm_win", "gradient_norm_prize", "gradient_norm_meta",
    "gradient_cosine", "allocation_top1_top2_margin", "value_calibration_detail",
)


def is_sparse_diagnostic_update(update: int) -> bool:
    if update < 0:
        raise ValueError("update must be nonnegative")
    return update in {0, 5, 10} or (update > 10 and update % 10 == 0)


__all__ = [
    "FROZEN_OFFLINE_METRICS", "SPARSE_DIAGNOSTICS", "UPDATE_SCALARS",
    "is_sparse_diagnostic_update",
]
