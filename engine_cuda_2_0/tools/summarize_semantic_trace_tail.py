"""Print compact tail records from an official semantic JSONL trace."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    records = deque(maxlen=args.count)
    for raw in args.trace.open(encoding="utf-8"):
        records.append(json.loads(raw))
    for record in records:
        observation = record["actor_observation"]
        selection = observation["select"]
        print(json.dumps({
            "seed": record["seed"],
            "decision": record["decision"],
            "select": {
                key: selection.get(key) for key in (
                    "type", "context", "minCount", "maxCount",
                    "contextCard", "effect",
                )
            },
            "options": selection.get("option"),
            "ordered_action": record["ordered_action"],
            "logs": observation.get("logs", [])[-12:],
        }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
