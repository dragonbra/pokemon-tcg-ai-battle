#!/usr/bin/env bash
set -euo pipefail

user_home="/Users/hejinyu"
kaggle_bin="/opt/homebrew/bin/kaggle"
competition="pokemon-tcg-ai-battle"
state_dir="$user_home/Library/Application Support/pokemon-tcg-ai-battle/v7-best"
marker="$state_dir/BEST_STRATEGY.json"
log_dir="$user_home/Library/Logs/pokemon-tcg-ai-battle"
lock_dir="/tmp/pokemon-tcg-ai-battle-v7-best-submit.lock"
dry_run=false

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ $# -gt 0 ]]; then
  echo "用法: $0 [--dry-run]" >&2
  exit 2
fi

mkdir -p "$log_dir"
exec >> "$log_dir/v7-best-submit.log" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

echo "[$(timestamp)] launchd V7 best submission started (dry_run=$dry_run)"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[$(timestamp)] another V7 best submission process is active; exiting"
  exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

[[ -x "$kaggle_bin" ]] || {
  echo "[$(timestamp)] Kaggle CLI not found: $kaggle_bin" >&2
  exit 1
}
[[ -f "$marker" ]] || {
  echo "[$(timestamp)] staged best marker missing: $marker" >&2
  exit 1
}

best_info="$('/opt/homebrew/bin/python3' - "$marker" <<'PY'
import json
import re
import sys
from pathlib import Path

marker = Path(sys.argv[1])
payload = json.loads(marker.read_text(encoding="utf-8"))
iteration = payload.get("best_iteration")
label = payload.get("label")
artifact = payload.get("artifact")
digest = payload.get("artifact_sha256")
if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
    raise SystemExit("invalid best_iteration")
if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", label):
    raise SystemExit("invalid label")
if not isinstance(artifact, str) or Path(artifact).name != artifact:
    raise SystemExit("staged artifact must be a filename")
if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
    raise SystemExit("invalid artifact_sha256")
print(f"{iteration}\t{label}\t{artifact}\t{digest}")
PY
)"
IFS=$'\t' read -r best_iteration best_label artifact artifact_sha256 <<< "$best_info"
archive="$state_dir/$artifact"
[[ -s "$archive" ]] || {
  echo "[$(timestamp)] staged best archive missing: $archive" >&2
  exit 1
}

actual_sha256="$(shasum -a 256 "$archive" | awk '{print $1}')"
[[ "$actual_sha256" == "$artifact_sha256" ]] || {
  echo "[$(timestamp)] staged best archive SHA-256 mismatch" >&2
  exit 1
}
tar -tzf "$archive" | sort

if [[ "$dry_run" == true ]]; then
  echo "[$(timestamp)] dry-run complete; Kaggle submission skipped"
  exit 0
fi

message="Alakazam V7 AutoIter best strategy iteration=${best_iteration} label=${best_label} sha256=${artifact_sha256} scheduled evaluation"
set +e
submit_output="$($kaggle_bin competitions submit "$competition" -f "$archive" -m "$message" 2>&1)"
submit_status=$?
set -e
printf '%s\n' "$submit_output"
if (( submit_status != 0 )); then
  printf 'submitted_at=%s\niteration=%s\nlabel=%s\narchive=%s\nexit_code=%s\n%s\n' \
    "$(timestamp)" "$best_iteration" "$best_label" "$archive" "$submit_status" "$submit_output" \
    > "$state_dir/v7-best-${best_label}.failed"
  echo "[$(timestamp)] launchd V7 best submission failed (exit_code=$submit_status)" >&2
  exit "$submit_status"
fi
printf 'submitted_at=%s\niteration=%s\nlabel=%s\narchive=%s\n%s\n' \
  "$(timestamp)" "$best_iteration" "$best_label" "$archive" "$submit_output" \
  > "$state_dir/v7-best-${best_label}.submitted"
echo "[$(timestamp)] launchd V7 best submission finished"
