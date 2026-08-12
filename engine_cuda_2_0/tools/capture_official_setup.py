from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
TOOLS_ROOT = WORKSPACE_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from seeded_cpp_shim import SeededCppLib, read_deck_csv  # noqa: E402

from capture_official_oracle import card_metadata, canonical_hash, deterministic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture compact official setup hands over a contiguous seed range."
    )
    parser.add_argument(
        "--lib",
        type=Path,
        default=WORKSPACE_ROOT / "tmp" / "seeded_cpp_shim_pure_v1" / "libcg_seeded.so",
    )
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
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--max-setup-decisions", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def capture_setup_seed(
    shim: SeededCppLib,
    deck0: list[int],
    deck1: list[int],
    seed: int,
    max_decisions: int,
) -> dict[str, Any]:
    hands: dict[int, list[int]] = {}
    hand_digests: dict[int, int] = {}
    select_error = 0
    with shim.start_battle(deck0, deck1, seed=seed) as game:
        for _ in range(max_decisions):
            snapshot = game.snapshot()
            observation = game.observation()
            player = int(snapshot.meta.select_player)
            if player in (0, 1) and snapshot.meta.select_context == 2:
                state_player = observation["current"]["players"][player]
                hand = state_player.get("hand")
                if (
                    state_player.get("deckCount") == 53
                    and isinstance(hand, list)
                    and len(hand) == 7
                ):
                    hands[player] = [int(card["id"]) for card in hand]
                    hand_digests[player] = snapshot.digest
                    if len(hands) == 2:
                        break
            action = deterministic_action(
                snapshot.meta.select_min,
                snapshot.meta.option_count,
            )
            select_error = game.select(action)
            if select_error:
                break

    if set(hands) != {0, 1}:
        raise RuntimeError(
            f"seed {seed} did not expose both setup hands: "
            f"players={sorted(hands)} select_error={select_error}"
        )
    return {
        "seed": seed,
        "initial_hands": [hands[0], hands[1]],
        "hand_digests": [hand_digests[0], hand_digests[1]],
    }


def main() -> None:
    args = parse_args()
    if args.seed_count <= 0 or args.max_setup_decisions <= 0:
        raise SystemExit("seed count and max setup decisions must be positive")
    deck0 = read_deck_csv(args.deck0)
    deck1 = read_deck_csv(args.deck1)
    shim = SeededCppLib(args.lib)
    traces = [
        capture_setup_seed(
            shim,
            deck0,
            deck1,
            args.seed_start + offset,
            args.max_setup_decisions,
        )
        for offset in range(args.seed_count)
    ]
    payload = {
        "version": 1,
        "kind": "official_setup_hands",
        "oracle_library": str(args.lib.resolve()),
        "oracle_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
        "deck0_sha256": hashlib.sha256(args.deck0.read_bytes()).hexdigest(),
        "deck1_sha256": hashlib.sha256(args.deck1.read_bytes()).hexdigest(),
        "card_metadata": card_metadata(shim, deck0, deck1),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "traces": traces,
    }
    payload["corpus_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seeds": len(traces),
                "hands": len(traces) * 2,
                "oracle_sha256": payload["oracle_sha256"],
                "corpus_sha256": payload["corpus_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
