#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${POKEMON_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_DIR="${PPO_SMOKE_RUN_DIR:-$ROOT/engine_cuda/artifacts/ppo_smoke_6bc_decoder_only/run}"
MODE="${1:-fresh}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INITIAL_CHECKPOINT="$ROOT/bc_models/archive/agent_pure_lucario_v1_bc512_e4_probe/policy.pt"
GAMES_PER_ITER="${PPO_SMOKE_GAMES_PER_ITER:-6}"
ROLLOUT_ENVS="${PPO_SMOKE_ROLLOUT_ENVS:-6}"
ENGINE_THREADS="${PPO_SMOKE_ENGINE_THREADS:-15}"

case "$MODE" in
  fresh)
    CHECKPOINT="$INITIAL_CHECKPOINT"
    ITERATIONS=1
    RESUME_ARGS=()
    ;;
  resume)
    CHECKPOINT="$RUN_DIR/checkpoint_last.pt"
    ITERATIONS=2
    RESUME_ARGS=(--resume-run)
    ;;
  *)
    echo "usage: $0 [fresh|resume]" >&2
    exit 2
    ;;
esac

cd "$ROOT"
exec "$PYTHON_BIN" tools/train_pure_lucario_ppo.py \
  --config "$ROOT/engine_cuda/configs/pure_lucario_bc_pool_ppo_smoke.json" \
  --lib "$ROOT/tmp/seeded_cpp_shim_policy_codec_v1/libcg_seeded.so" \
  --checkpoint "$CHECKPOINT" \
  --reference-checkpoint "$INITIAL_CHECKPOINT" \
  --out "$RUN_DIR" \
  --iters "$ITERATIONS" \
  --games-per-iter "$GAMES_PER_ITER" \
  --rollout-envs "$ROLLOUT_ENVS" \
  --max-steps 700 \
  --seed 2026073131 \
  --temperature 1.0 \
  --rollout-policy-mode sample \
  --hard-multiplier 1.0 \
  --opponent-sampling stratified \
  --gamma 1.0 \
  --gae-lambda 0.95 \
  --train-scope decoder_only \
  --lr 1e-6 \
  --weight-decay 0.01 \
  --ppo-epochs 1 \
  --batch-size 64 \
  --update-pipeline cached \
  --ppo-bucket-multiplier 4 \
  --clip-eps 0.1 \
  --value-clip 0.2 \
  --value-coef 0.5 \
  --entropy-coef 0.002 \
  --reference-kl-coef 0.10 \
  --bc-coef 0.0 \
  --max-grad-norm 1.0 \
  --target-kl 0.02 \
  --checkpoint-every 1 \
  --keep-last 1 \
  --checkpoint-mode last \
  --env-backend batch \
  --engine-threads "$ENGINE_THREADS" \
  --no-rollout-amp \
  --rollout-profile \
  --native-rollout-codec \
  --opponent-policy-process \
  --opponent-policy-workers 6 \
  --no-opponent-policy-shard-same-agent \
  --no-opponent-actor-profile \
  --device cuda \
  "${RESUME_ARGS[@]}"
