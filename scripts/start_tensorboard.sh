#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${TENSORBOARD_LOGDIR:-$repo_root/rl/_runs/tensorboard}"
host="${TENSORBOARD_HOST:-127.0.0.1}"
port="${TENSORBOARD_PORT:-6006}"
python_bin="${PYTHON:-python3}"

if [[ ! -d "$log_dir" ]]; then
  echo "TensorBoard log directory does not exist: $log_dir" >&2
  exit 1
fi

if ! "$python_bin" -c "import tensorboard" >/dev/null 2>&1; then
  echo "TensorBoard is not installed for $python_bin; install the RL dependencies first." >&2
  echo "Suggested command: $python_bin -m pip install -e '.[rl]'" >&2
  exit 1
fi

exec "$python_bin" -m tensorboard.main \
  --logdir "$log_dir" \
  --host "$host" \
  --port "$port" \
  "$@"
