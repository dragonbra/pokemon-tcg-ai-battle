#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The official Linux binary needs a newer libstdc++ than Ubuntu 20.04 ships.
# Point PTCG_CXX_RUNTIME at a compatible environment when one is available;
# otherwise leave the loader untouched so the native error remains visible.
if [[ -n "${PTCG_CXX_RUNTIME:-}" ]]; then
  runtime_lib="$PTCG_CXX_RUNTIME/lib"
  if [[ ! -f "$runtime_lib/libstdc++.so.6" ]]; then
    echo "PTCG_CXX_RUNTIME does not contain lib/libstdc++.so.6: $PTCG_CXX_RUNTIME" >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_PRELOAD="$runtime_lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec python3 "$root_dir/scripts/run_local_battle.py" "$@"
