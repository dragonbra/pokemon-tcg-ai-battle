#!/usr/bin/env bash
set -euo pipefail

ROOT="${POKEMON_REPO_ROOT:-/root/autodl-tmp/pokemon/repo}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/0020_dragapult_decoder_ppo_v1.json}"
OUT="${OUT:-$ROOT/rl_runs/0020_pluggable_deck_rl/versions/V11_dragapult_decoder_ppo_6x60_12w_b512_v2}"
ITERS="${ITERS:-}"
GAMES_PER_ITER="${GAMES_PER_ITER:-}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_MODE="${WANDB_MODE:-online}"

cd "$ROOT"
export WANDB_MODE

args=(
  -m train.0020_pluggable_deck_rl.train_decoder_ppo
  --config "$CONFIG"
  --out "$OUT"
)
if [[ -n "$ITERS" ]]; then
  args+=(--iters "$ITERS")
fi
if [[ -n "$GAMES_PER_ITER" ]]; then
  args+=(--games-per-iter "$GAMES_PER_ITER")
fi
if [[ -n "$ROLLOUT_WORKERS" ]]; then
  args+=(--rollout-workers "$ROLLOUT_WORKERS")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  args+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ -n "$WANDB_RUN_ID" ]]; then
  args+=(--wandb-run-id "$WANDB_RUN_ID")
fi
if [[ -n "$WANDB_RESUME" ]]; then
  args+=(--wandb-resume "$WANDB_RESUME")
fi

exec "$PYTHON_BIN" "${args[@]}"
