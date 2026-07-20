#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_submission="alakazam_v7_auto_iter"
marker="$repo_dir/submission/$source_submission/BEST_STRATEGY.json"

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <iteration> <label>" >&2
  exit 2
fi
iteration="$1"
label="$2"
[[ "$iteration" =~ ^[0-9]+$ ]] || {
  echo "iteration 必须是非负整数" >&2
  exit 2
}
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "label 只能包含字母、数字、点、下划线和连字符" >&2
  exit 2
}

base_archive="$(bash "$repo_dir/scripts/package_submission.sh" "$source_submission")"
artifact="$repo_dir/submission/dist/${source_submission}_best_${label}.tar.gz"
cp "$base_archive" "$artifact"

python3 "$repo_dir/scripts/v7_best_submission.py" write-marker \
  --marker "$marker" \
  --repo-root "$repo_dir" \
  --source-submission "$source_submission" \
  --iteration "$iteration" \
  --label "$label" \
  --artifact "$artifact" \
  --evidence "由 promote_v7_best.sh 从当前工作区生成不可变归档；仅在对应迭代被接受后调用。"

python3 "$repo_dir/scripts/v7_best_submission.py" stage-schedule \
  --marker "$marker" \
  --repo-root "$repo_dir" \
  --state-dir "/Users/hejinyu/Library/Application Support/pokemon-tcg-ai-battle/v7-best"

echo "$artifact"
