from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CUDA setup draw order with a frozen official trace."
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--cuda-exe", type=Path, required=True)
    parser.add_argument(
        "--deck0",
        type=Path,
        default=CUDA_ENGINE_ROOT / "fixtures" / "decks" / "marnie_prize_control.csv",
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=CUDA_ENGINE_ROOT / "fixtures" / "decks" / "alakazam_battle_cage.csv",
    )
    parser.add_argument("--details", action="store_true")
    return parser.parse_args()


def initial_hand(trace: dict[str, Any], player: int) -> list[int]:
    compact = trace.get("initial_hands")
    if isinstance(compact, list) and len(compact) == 2:
        return [int(card_id) for card_id in compact[player]]
    for frame in trace["frames"]:
        if int(frame["meta"][7]) != player:
            continue
        state_player = frame["observation"]["current"]["players"][player]
        hand = state_player.get("hand")
        if state_player.get("deckCount") == 53 and isinstance(hand, list) and len(hand) == 7:
            return [int(card["id"]) for card in hand]
    raise RuntimeError(f"oracle trace for seed {trace['seed']} has no initial hand for player {player}")


def main() -> None:
    args = parse_args()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    seeds = [int(trace["seed"]) for trace in oracle["traces"]]
    basic_ids = sorted(
        int(card["cardId"])
        for card in oracle["card_metadata"]
        if bool(card.get("basic"))
    )
    completed = subprocess.run(
        [
            str(args.cuda_exe),
            "--deck0",
            str(args.deck0),
            "--deck1",
            str(args.deck1),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--basic-ids",
            ",".join(str(card_id) for card_id in basic_ids),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cuda = json.loads(completed.stdout)
    cuda_by_seed = {int(trace["seed"]): trace for trace in cuda["traces"]}
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for trace in oracle["traces"]:
        seed = int(trace["seed"])
        cuda_trace = cuda_by_seed[seed]
        for player in range(2):
            expected = initial_hand(trace, player)
            actual = [
                int(value)
                for value in cuda_trace[f"player{player}_draw_order"][:7]
            ]
            equal = actual == expected
            mismatches += int(not equal)
            rows.append(
                {
                    "seed": seed,
                    "player": player,
                    "expected": expected,
                    "actual": actual,
                    "equal": equal,
                }
            )
    result = {
        "device": cuda["device"],
        "seeds": len(seeds),
        "hands_compared": len(rows),
        "basic_card_ids": basic_ids,
        "mismatches": mismatches,
        "mismatch_rows": [row for row in rows if not row["equal"]],
        "sample_rows": rows if args.details else rows[: min(8, len(rows))],
    }
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
