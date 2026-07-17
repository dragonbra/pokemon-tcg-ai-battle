#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$root_dir/submission"
out_dir="$root_dir/dist"
out_file="$out_dir/submission.tar.gz"

mkdir -p "$out_dir"
rm -f "$out_file"

cd "$src_dir"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$out_file" main.py deck.csv cg

echo "$out_file"
