#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user_home="/Users/hejinyu"
kaggle_bin="/opt/homebrew/bin/kaggle"
competition="pokemon-tcg-ai-battle"
submission_root="$repo_dir/submission/alakazam_v7_auto_iter"
best_marker="$submission_root/BEST_STRATEGY.json"
log_dir="$user_home/Library/Logs/pokemon-tcg-ai-battle"
state_dir="$user_home/Library/Application Support/pokemon-tcg-ai-battle"
lock_dir="/tmp/pokemon-tcg-ai-battle-v7-best-submit.lock"
dry_run=false

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ $# -gt 0 ]]; then
  echo "用法: $0 [--dry-run]" >&2
  exit 2
fi

mkdir -p "$log_dir" "$state_dir"
exec >> "$log_dir/v7-best-submit.log" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

echo "[$(timestamp)] scheduled V7 best submission started (dry_run=$dry_run)"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[$(timestamp)] another V7 best submission process is active; exiting"
  exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

[[ -x "$kaggle_bin" ]] || {
  echo "[$(timestamp)] Kaggle CLI not found: $kaggle_bin" >&2
  exit 1
}
[[ -f "$best_marker" ]] || {
  echo "[$(timestamp)] best strategy marker missing: $best_marker" >&2
  exit 1
}

best_info="$(python3 "$repo_dir/scripts/v7_best_submission.py" validate \
  --marker "$best_marker" --repo-root "$repo_dir")" || {
  echo "[$(timestamp)] best strategy archive validation failed" >&2
  exit 1
}
IFS=$'\t' read -r archive best_iteration best_label artifact_sha256 <<< "$best_info"
[[ -s "$archive" ]] || {
  echo "[$(timestamp)] immutable best archive missing: $archive" >&2
  exit 1
}

echo "[$(timestamp)] best strategy: iteration=$best_iteration label=$best_label"
echo "[$(timestamp)] archive: $archive"
echo "[$(timestamp)] archive sha256: $artifact_sha256"
tar -tzf "$archive" | sort

if [[ "$dry_run" == true ]]; then
  echo "[$(timestamp)] dry-run complete; Kaggle submission skipped"
  exit 0
fi

message="Alakazam V7 AutoIter best strategy iteration=${best_iteration} label=${best_label} sha256=${artifact_sha256} scheduled evaluation"
set +e
submit_output="$("$kaggle_bin" competitions submit "$competition" -f "$archive" -m "$message" 2>&1)"
submit_status=$?
set -e
printf '%s\n' "$submit_output"
if (( submit_status != 0 )); then
  printf 'submitted_at=%s\niteration=%s\nlabel=%s\narchive=%s\nexit_code=%s\n%s\n' \
    "$(timestamp)" "$best_iteration" "$best_label" "$archive" "$submit_status" "$submit_output" \
    > "$state_dir/v7-best-${best_label}.failed"
  echo "[$(timestamp)] scheduled V7 best submission failed (exit_code=$submit_status)" >&2
  exit "$submit_status"
fi
printf 'submitted_at=%s\niteration=%s\nlabel=%s\narchive=%s\n%s\n' \
  "$(timestamp)" "$best_iteration" "$best_label" "$archive" "$submit_output" \
  > "$state_dir/v7-best-${best_label}.submitted"
echo "[$(timestamp)] scheduled V7 best submission finished"
