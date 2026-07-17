#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${PTCG_ENGINE_ARCHIVE:-$root_dir/ptcg_engine.zip}"
source_root="${PTCG_ENGINE_SOURCE_ROOT:-$root_dir/engine/source}"
source_dir="${PTCG_ENGINE_SOURCE:-$source_root/ptcgProgram 22}"

if [[ ! -f "$source_dir/Export.cpp" ]]; then
  if [[ ! -f "$archive" ]]; then
    echo "Official engine archive not found: $archive" >&2
    echo "Set PTCG_ENGINE_ARCHIVE or PTCG_ENGINE_SOURCE." >&2
    exit 2
  fi
  mkdir -p "$source_root"
  unzip -q "$archive" -d "$source_root"
fi

if [[ ! -f "$source_dir/Export.cpp" ]]; then
  echo "Invalid official engine source directory: $source_dir" >&2
  exit 2
fi

cxx="${CXX:-g++}"
system="$(uname -s)"
output="${PTCG_ENGINE_OUTPUT:-$root_dir/engine/build/libcg.so}"
mkdir -p "$(dirname "$output")"

case "$system" in
  Darwin)
    link_flag="-dynamiclib"
    output="${PTCG_ENGINE_OUTPUT:-$root_dir/engine/build/libcg.dylib}"
    ;;
  Linux)
    link_flag="-shared"
    ;;
  *)
    echo "This helper supports Linux and macOS; use game.sln on Windows." >&2
    exit 2
    ;;
esac

if ! "$cxx" -std=c++20 -x c++ -include ranges -E -o /dev/null - < /dev/null 2>/dev/null; then
  echo "C++20 compiler with <ranges> is required; current compiler: $cxx" >&2
  echo "Use a newer GCC/Clang toolchain, or use game.sln on Windows." >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"
"$cxx" -std=c++20 -DNDEBUG -O2 -fPIC "$link_flag" \
  "$source_dir/Export.cpp" \
  -o "$output"

echo "$output"
