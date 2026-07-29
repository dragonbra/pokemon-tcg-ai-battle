from __future__ import annotations

import argparse
import ctypes
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture deterministic decision traces from the frozen CPU oracle."
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
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument(
        "--action-policy",
        choices=("first_min", "attach_attack"),
        default="first_min",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_action(select_min: int, option_count: int) -> list[int]:
    if select_min < 0 or select_min > option_count:
        raise RuntimeError(
            f"invalid oracle selection bounds: min={select_min}, options={option_count}"
        )
    return list(range(select_min))


def choose_action(snapshot: Any, policy: str) -> list[int]:
    if policy == "attach_attack" and snapshot.meta.select_context == 1:
        for preferred_type in (8, 13, 14):
            for option in snapshot.options:
                if int(option[1]) == preferred_type:
                    return [int(option[0])]
    return deterministic_action(
        snapshot.meta.select_min,
        snapshot.meta.option_count,
    )


def scrub_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result = dict(observation)
    result.pop("search_begin_input", None)
    return result


def capture_seed(
    shim: SeededCppLib,
    deck0: list[int],
    deck1: list[int],
    seed: int,
    max_decisions: int,
    action_policy: str,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    select_error = 0
    with shim.start_battle(deck0, deck1, seed=seed) as game:
        for decision in range(max_decisions):
            snapshot = game.snapshot()
            observation = scrub_observation(game.observation())
            if snapshot.meta.game_result != 0:
                break
            action = choose_action(snapshot, action_policy)
            frame = {
                "decision": decision,
                "digest_before": snapshot.digest,
                "meta": list(snapshot.meta.raw),
                "options": [list(option) for option in snapshot.options],
                "state_tensor": list(snapshot.tensor),
                "observation": observation,
                "action": action,
            }
            select_error = game.select(action)
            frame["select_error"] = select_error
            frame["digest_after"] = game.digest()
            frames.append(frame)
            if select_error:
                break

        final = game.snapshot()
        result = {
            "seed": seed,
            "decisions": len(frames),
            "select_error": select_error,
            "game_result": final.meta.game_result,
            "finish_reason": final.meta.finish_reason,
            "final_digest": final.digest,
            "frames": frames,
        }
    result["trace_sha256"] = canonical_hash(result)
    return result


def card_metadata(shim: SeededCppLib, deck0: list[int], deck1: list[int]) -> list[dict[str, Any]]:
    shim.lib.AllCard.restype = ctypes.c_char_p
    shim.lib.AllCard.argtypes = []
    raw = shim.lib.AllCard()
    if not raw:
        raise RuntimeError("official AllCard export returned null")
    wanted = set(deck0) | set(deck1)
    cards = json.loads(raw.decode("utf-8"))
    selected = [card for card in cards if int(card["cardId"]) in wanted]
    found = {int(card["cardId"]) for card in selected}
    if found != wanted:
        raise RuntimeError(f"official metadata is missing card IDs: {sorted(wanted - found)}")
    return selected


def main() -> None:
    args = parse_args()
    if args.max_decisions <= 0:
        raise SystemExit("max decisions must be positive")
    deck0 = read_deck_csv(args.deck0)
    deck1 = read_deck_csv(args.deck1)
    shim = SeededCppLib(args.lib)
    payload = {
        "version": 1,
        "oracle_library": str(args.lib.resolve()),
        "oracle_sha256": hashlib.sha256(args.lib.read_bytes()).hexdigest(),
        "deck0": {
            "path": str(args.deck0.resolve()),
            "sha256": hashlib.sha256(args.deck0.read_bytes()).hexdigest(),
        },
        "deck1": {
            "path": str(args.deck1.resolve()),
            "sha256": hashlib.sha256(args.deck1.read_bytes()).hexdigest(),
        },
        "action_policy": args.action_policy,
        "card_metadata": card_metadata(shim, deck0, deck1),
        "traces": [
            capture_seed(
                shim,
                deck0,
                deck1,
                seed,
                args.max_decisions,
                args.action_policy,
            )
            for seed in args.seeds
        ],
    }
    payload["corpus_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "oracle_sha256": payload["oracle_sha256"],
                "corpus_sha256": payload["corpus_sha256"],
                "traces": [
                    {
                        "seed": trace["seed"],
                        "decisions": trace["decisions"],
                        "select_error": trace["select_error"],
                        "game_result": trace["game_result"],
                        "trace_sha256": trace["trace_sha256"],
                    }
                    for trace in payload["traces"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
