#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submission_name="${1:-alakazam_v1}"
src_dir="$root_dir/submission/$submission_name"
out_dir="$root_dir/dist"
out_file="$out_dir/${submission_name}.tar.gz"

if [[ ! -d "$src_dir" || ! -f "$src_dir/main.py" || ! -f "$src_dir/deck.csv" || ! -d "$src_dir/cg" ]]; then
  echo "完整 submission 不存在: submission/$submission_name" >&2
  exit 1
fi

mkdir -p "$out_dir"
rm -f "$out_file"

cd "$src_dir"
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.pyc' -czf "$out_file" main.py deck.csv cg

echo "$out_file"
