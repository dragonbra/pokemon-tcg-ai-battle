"""Generate deterministic unordered deck-pair TSV shards for CUDA parity."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.shards <= 0:
        parser.error("--shards must be positive")
    decks = sorted((args.pool / "decks").glob("*/deck.csv"))
    if len(decks) < 2:
        parser.error("pool must contain at least two deck.csv files")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{ordinal}\t{left.as_posix()}\t{right.as_posix()}\n"
        for ordinal, (left, right) in enumerate(combinations(decks, 2))
    ]
    (args.output / "all_cases.tsv").write_bytes("".join(rows).encode("utf-8"))
    for shard in range(args.shards):
        selected = rows[shard::args.shards]
        (args.output / f"cases_{shard:02d}.tsv").write_bytes(
            "".join(selected).encode("utf-8")
        )
    print(f"decks={len(decks)} pairs={len(rows)} shards={args.shards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
