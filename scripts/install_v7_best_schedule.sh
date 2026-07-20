#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user_home="/Users/hejinyu"
state_dir="$user_home/Library/Application Support/pokemon-tcg-ai-battle/v7-best"
launch_agent="$user_home/Library/LaunchAgents/com.hejinyu.pokemon-tcg-ai-battle.submit-v7-best.plist"
label="com.hejinyu.pokemon-tcg-ai-battle.submit-v7-best"

mkdir -p "$state_dir" "$user_home/Library/LaunchAgents"
cp "$repo_dir/scripts/submit_v7_best_launchd.sh" "$state_dir/submit_v7_best_launchd.sh"
chmod 755 "$state_dir/submit_v7_best_launchd.sh"
python3 "$repo_dir/scripts/v7_best_submission.py" stage-schedule \
  --marker "$repo_dir/submission/alakazam_v7_auto_iter/BEST_STRATEGY.json" \
  --repo-root "$repo_dir" \
  --state-dir "$state_dir"
cp "$repo_dir/scripts/com.hejinyu.pokemon-tcg-ai-battle.submit-v7-best.plist" "$launch_agent"

launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$launch_agent"
echo "已安装并加载 ${label}；每天 08:05 提交 $state_dir 中标记的 immutable best。"
