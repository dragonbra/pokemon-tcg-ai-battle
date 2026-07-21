#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submission_name="${1:-alakazam_v1}"
is_complete_submission_dir() {
  local candidate_dir="$1"
  [[
    -d "$candidate_dir"
    && -f "$candidate_dir/main.py"
    && -f "$candidate_dir/deck.csv"
    && -d "$candidate_dir/cg"
  ]]
}

if is_complete_submission_dir "$root_dir/work/$submission_name"; then
  src_dir="$root_dir/work/$submission_name"
elif is_complete_submission_dir "$root_dir/submission/$submission_name"; then
  src_dir="$root_dir/submission/$submission_name"
else
  src_dir="$root_dir/work/$submission_name"
fi
out_dir="$root_dir/submission/dist"
out_file="$out_dir/${submission_name}.tar.gz"

if [[ ! -d "$src_dir" || ! -f "$src_dir/main.py" || ! -f "$src_dir/deck.csv" || ! -d "$src_dir/cg" ]]; then
  echo "完整 submission 不存在: work/$submission_name 或 submission/$submission_name" >&2
  exit 1
fi

mkdir -p "$out_dir"
rm -f "$out_file"

cd "$src_dir"
archive_entries=(main.py deck.csv cg)
if [[ -d strategy ]]; then
  archive_entries+=(strategy)
fi
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
  -czf "$out_file" "${archive_entries[@]}"

echo "$out_file"
