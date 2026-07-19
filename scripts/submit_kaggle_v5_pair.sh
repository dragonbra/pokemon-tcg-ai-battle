#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kaggle_bin="/opt/homebrew/bin/kaggle"
competition="pokemon-tcg-ai-battle"
log_dir="${HOME}/Library/Logs/pokemon-tcg-ai-battle"
state_dir="${HOME}/Library/Application Support/pokemon-tcg-ai-battle"
lock_dir="/tmp/pokemon-tcg-ai-battle-v5-submit.lock"

mkdir -p "$log_dir" "$state_dir"
exec >> "$log_dir/v5-pair-submit.log" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

echo "[$(timestamp)] scheduled V5 pair submission started"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[$(timestamp)] another submission process is active; exiting"
  exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

if [[ ! -x "$kaggle_bin" ]]; then
  echo "[$(timestamp)] Kaggle CLI not found: $kaggle_bin" >&2
  exit 1
fi

submit_one() {
  local submission_name="$1"
  local message="$2"
  local marker="$state_dir/${submission_name}.submitted"
  local archive

  if [[ -f "$marker" ]]; then
    echo "[$(timestamp)] $submission_name already submitted; skipping"
    return 0
  fi

  archive="$(bash "$root_dir/scripts/package_submission.sh" "$submission_name")"
  [[ -f "$archive" ]] || {
    echo "[$(timestamp)] package missing: $archive" >&2
    return 1
  }

  tar -tzf "$archive" | sort
  "$kaggle_bin" competitions submit "$competition" \
    -f "$archive" \
    -m "$message"

  printf 'submitted_at=%s\narchive=%s\n' "$(timestamp)" "$archive" > "$marker"
  echo "[$(timestamp)] $submission_name submitted successfully"
}

submit_one alakazam_v5 "Alakazam V5 scheduled submission"
submit_one alakazam_v5_mixed_f1 "Alakazam V5 Mixed F1 scheduled submission"

echo "[$(timestamp)] scheduled V5 pair submission finished"
