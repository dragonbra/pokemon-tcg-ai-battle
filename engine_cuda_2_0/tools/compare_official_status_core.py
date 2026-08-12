from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the CUDA poison/confusion battle-core trace against a frozen "
            "official CPU oracle corpus."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument(
        "--deck0",
        type=Path,
        default=CUDA_ENGINE_ROOT / "fixtures" / "decks" / "status_core_poison.csv",
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=CUDA_ENGINE_ROOT / "fixtures" / "decks" / "status_core_confuse.csv",
    )
    parser.add_argument("--write-cuda-output", type=Path)
    parser.add_argument("--max-mismatches", type=int, default=25)
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_oracle(payload: dict[str, Any]) -> None:
    recorded = payload.get("corpus_sha256")
    unhashed = dict(payload)
    unhashed.pop("corpus_sha256", None)
    actual = canonical_hash(unhashed)
    if recorded != actual:
        raise RuntimeError(
            f"oracle corpus hash mismatch: recorded={recorded} recomputed={actual}"
        )
    if payload.get("action_policy") != "attach_attack":
        raise RuntimeError("status-core comparator requires action_policy=attach_attack")


def official_meta(frame: dict[str, Any]) -> list[int]:
    raw = frame["meta"]
    return [
        raw[1],   # turn
        raw[2],   # phase
        raw[3],   # game result
        raw[4],   # finish reason
        raw[5],   # select type
        raw[6],   # select context
        raw[7],   # select player
        raw[8],   # select min
        raw[9],   # select max
        raw[11],  # turn action count
        raw[16],  # energy attached this turn
        raw[19],  # first player
        raw[20],  # coin head count
    ]


def card_pair(card: dict[str, Any]) -> list[int]:
    return [int(card["id"]), int(card["serial"])]


def compare_value(
    mismatches: list[dict[str, Any]],
    path: str,
    expected: Any,
    actual: Any,
    max_mismatches: int,
) -> int:
    if expected == actual:
        return 1
    if len(mismatches) < max_mismatches:
        mismatches.append({"path": path, "expected": expected, "actual": actual})
    return 1


def compare_player(
    mismatches: list[dict[str, Any]],
    path: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    max_mismatches: int,
) -> int:
    checks = 0
    checks += compare_value(
        mismatches, f"{path}.deck_count", expected["deckCount"], actual["deck_count"], max_mismatches
    )
    checks += compare_value(
        mismatches, f"{path}.hand_count", expected["handCount"], len(actual["hand"]), max_mismatches
    )
    if expected.get("hand") is not None:
        checks += compare_value(
            mismatches,
            f"{path}.hand",
            [card_pair(card) for card in expected["hand"]],
            actual["hand"],
            max_mismatches,
        )
    checks += compare_value(
        mismatches,
        f"{path}.prize_count",
        len(expected.get("prize") or []),
        actual["prize_count"],
        max_mismatches,
    )
    checks += compare_value(
        mismatches,
        f"{path}.discard",
        [card_pair(card) for card in expected.get("discard") or []],
        actual["discard"],
        max_mismatches,
    )

    expected_active_list = expected.get("active") or []
    expected_present = bool(expected_active_list)
    checks += compare_value(
        mismatches,
        f"{path}.active_present",
        expected_present,
        actual["active"] is not None,
        max_mismatches,
    )
    expected_active = expected_active_list[0] if expected_active_list else None
    if expected_active is not None:
        checks += compare_value(
            mismatches,
            f"{path}.active.card",
            card_pair(expected_active),
            actual["active"]["card"],
            max_mismatches,
        )
        checks += compare_value(
            mismatches,
            f"{path}.active.hp",
            expected_active["hp"],
            actual["active"]["hp"],
            max_mismatches,
        )
        checks += compare_value(
            mismatches,
            f"{path}.active.max_hp",
            expected_active["maxHp"],
            actual["active"]["max_hp"],
            max_mismatches,
        )
        checks += compare_value(
            mismatches,
            f"{path}.attached",
            [card_pair(card) for card in expected_active.get("energyCards") or []],
            actual["attached"],
            max_mismatches,
        )
    checks += compare_value(
        mismatches,
        f"{path}.poisoned",
        bool(expected["poisoned"]),
        bool(actual["poisoned"]),
        max_mismatches,
    )
    checks += compare_value(
        mismatches,
        f"{path}.confused",
        bool(expected["confused"]),
        bool(actual["confused"]),
        max_mismatches,
    )
    return checks


def compare_payloads(
    oracle: dict[str, Any],
    cuda: dict[str, Any],
    max_mismatches: int,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checks = 0
    decisions = 0
    cuda_by_seed = {int(trace["seed"]): trace for trace in cuda["traces"]}
    for expected_trace in oracle["traces"]:
        seed = int(expected_trace["seed"])
        path = f"seed[{seed}]"
        actual_trace = cuda_by_seed.get(seed)
        checks += compare_value(
            mismatches, f"{path}.present", True, actual_trace is not None, max_mismatches
        )
        if actual_trace is None:
            continue
        checks += compare_value(
            mismatches,
            f"{path}.error",
            0,
            actual_trace["error"],
            max_mismatches,
        )
        for key in ("decisions", "game_result", "finish_reason"):
            checks += compare_value(
                mismatches,
                f"{path}.{key}",
                expected_trace[key],
                actual_trace[key],
                max_mismatches,
            )
        for frame_index, expected_frame in enumerate(expected_trace["frames"]):
            decisions += 1
            frame_path = f"{path}.frame[{frame_index}]"
            if frame_index >= len(actual_trace["frames"]):
                checks += compare_value(
                    mismatches, f"{frame_path}.present", True, False, max_mismatches
                )
                continue
            actual_frame = actual_trace["frames"][frame_index]
            checks += compare_value(
                mismatches,
                f"{frame_path}.decision",
                expected_frame["decision"],
                actual_frame["decision"],
                max_mismatches,
            )
            checks += compare_value(
                mismatches,
                f"{frame_path}.meta",
                official_meta(expected_frame),
                actual_frame["meta"],
                max_mismatches,
            )
            checks += compare_value(
                mismatches,
                f"{frame_path}.options",
                expected_frame["options"],
                actual_frame["options"],
                max_mismatches,
            )
            checks += compare_value(
                mismatches,
                f"{frame_path}.action",
                expected_frame["action"],
                actual_frame["action"],
                max_mismatches,
            )
            # Official JSON logs use one delivery cursor per observing player,
            # so a single physical coin flip can appear again when control
            # returns to the other player. Count the event exactly once from
            # the transition that caused it: a confused player-0 attack in
            # the immediately preceding decision. The raw meta coin count is
            # the authoritative outcome and is already compared above.
            expected_coin = -1
            expected_coin_player = -1
            if frame_index > 0:
                previous = expected_trace["frames"][frame_index - 1]
                previous_action = previous["action"]
                if previous_action:
                    selected = previous["options"][previous_action[0]]
                    previous_players = previous["observation"]["current"]["players"]
                    if (
                        selected[1] == 13
                        and previous["meta"][7] == 0
                        and previous_players[0]["confused"]
                    ):
                        expected_coin = int(expected_frame["meta"][20] > 0)
                        expected_coin_player = 0
            checks += compare_value(
                mismatches,
                f"{frame_path}.coin_event",
                expected_coin,
                actual_frame["coin_event"],
                max_mismatches,
            )
            checks += compare_value(
                mismatches,
                f"{frame_path}.coin_player",
                expected_coin_player,
                actual_frame["coin_player"],
                max_mismatches,
            )
            expected_players = expected_frame["observation"]["current"]["players"]
            for player_index in range(2):
                checks += compare_player(
                    mismatches,
                    f"{frame_path}.player[{player_index}]",
                    expected_players[player_index],
                    actual_frame["players"][player_index],
                    max_mismatches,
                )

    expected_seeds = {int(trace["seed"]) for trace in oracle["traces"]}
    unexpected = sorted(set(cuda_by_seed) - expected_seeds)
    checks += compare_value(
        mismatches, "unexpected_cuda_seeds", [], unexpected, max_mismatches
    )
    return {
        "passed": not mismatches,
        "oracle_corpus_sha256": oracle["corpus_sha256"],
        "oracle_library_sha256": oracle["oracle_sha256"],
        "device": cuda.get("device"),
        "trace_bytes": cuda.get("trace_bytes"),
        "trace_count": len(oracle["traces"]),
        "decisions_compared": decisions,
        "field_checks": checks,
        "mismatch_count_shown": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    verify_oracle(oracle)
    seeds = ",".join(str(trace["seed"]) for trace in oracle["traces"])
    completed = subprocess.run(
        [
            str(args.binary),
            "--deck0",
            str(args.deck0),
            "--deck1",
            str(args.deck1),
            "--seeds",
            seeds,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cuda = json.loads(completed.stdout)
    if args.write_cuda_output:
        args.write_cuda_output.parent.mkdir(parents=True, exist_ok=True)
        args.write_cuda_output.write_text(
            json.dumps(cuda, indent=2) + "\n", encoding="utf-8"
        )
    report = compare_payloads(oracle, cuda, args.max_mismatches)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
