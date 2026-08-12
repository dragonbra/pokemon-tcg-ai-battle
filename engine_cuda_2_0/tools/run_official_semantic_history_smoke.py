from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from run_official_seeded_reset_paired import read_deck, read_fixture  # noqa: E402


SEMANTIC_LOG_NAMES = {
    0: "Shuffle",
    1: "HasBasicPokemon",
    2: "TurnStart",
    3: "TurnEnd",
    4: "Draw",
    5: "DrawReverse",
    6: "MoveCard",
    7: "MoveCardReverse",
    8: "Switch",
    9: "Change",
    10: "Play",
    11: "Attach",
    12: "Evolve",
    13: "Devolve",
    14: "MoveAttached",
    15: "Attack",
    16: "HpChange",
    17: "Poisoned",
    18: "Burned",
    19: "Asleep",
    20: "Paralyzed",
    21: "Confused",
    22: "Coin",
    23: "Result",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test semantic history reset and raw CUDA views."
    )
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser.add_argument(
        "--manifest",
        type=Path,
        default=private / "0022_deck40_focal_s1_1_n2" / "manifest.json",
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "build" / "torch_0031_linux",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("first-min", "interactive"),
        default="first-min",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "artifacts"
        / "official_semantic_history_smoke.json",
    )
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def require_cuda_tensor(name: str, tensor: Any, shape: tuple[int, ...]) -> None:
    if tensor.device.type != "cuda" or tuple(tensor.shape) != shape:
        raise RuntimeError(
            f"invalid semantic history tensor {name}: "
            f"device={tensor.device} shape={tuple(tensor.shape)} expected={shape}"
        )


def active_event_slots(total_count: int, write_index: int, capacity: int = 64) -> list[int]:
    live_count = min(total_count, capacity)
    if live_count == 0:
        return []
    if total_count <= capacity:
        return list(range(live_count))
    oldest = write_index % capacity
    return [(oldest + offset) % capacity for offset in range(live_count)]


def semantic_event_preview(
    log_type: Any,
    param_count: Any,
    params: Any,
    total_count: int,
    write_index: int,
    limit: int = 16,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for slot in active_event_slots(total_count, write_index)[:limit]:
        event_type = int(log_type[slot])
        count = int(param_count[slot])
        events.append(
            {
                "slot": int(slot),
                "type": event_type,
                "name": SEMANTIC_LOG_NAMES.get(event_type, f"Unknown({event_type})"),
                "param_count": count,
                "params": [int(value) for value in params[slot, :count].tolist()],
            }
        )
    return events


def validate_setup_events(
    *,
    mode: str,
    total_count: list[int],
    write_index: list[int],
    log_type: Any,
    param_count: Any,
) -> None:
    # Interactive setup intentionally yields before any event is emitted.  The
    # first-min setup also does not expose the pre-deal shuffle through the
    # official API, so an empty history or a history without Shuffle is valid.
    for lane, count in enumerate(total_count):
        slots = active_event_slots(count, write_index[lane])
        lane_types = [int(log_type[lane, slot]) for slot in slots]
        lane_param_counts = [int(param_count[lane, slot]) for slot in slots]
        if mode == "first-min" and count > 0 and 4 not in lane_types:
            raise RuntimeError(f"lane {lane} first-min semantic setup events lack Draw")
        for event_type, event_param_count in zip(lane_types, lane_param_counts):
            if event_type == 0 and event_param_count != 1:
                raise RuntimeError(
                    f"lane {lane} Shuffle param_count={event_param_count}; expected 1"
                )
            if event_type == 4 and event_param_count != 3:
                raise RuntimeError(
                    f"lane {lane} Draw param_count={event_param_count}; expected 3"
                )
            if event_type == 23 and event_param_count != 2:
                raise RuntimeError(
                    f"lane {lane} Result param_count={event_param_count}; expected 2"
                )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    required_methods = [
        "reset_seeded_first_min_semantic",
        "reset_seeded_first_min_semantic_masked",
        "reset_seeded_interactive_semantic",
        "reset_seeded_interactive_semantic_masked",
        "semantic_history_raw",
    ]
    missing = [
        name for name in required_methods
        if not hasattr(_ptcg_cuda.OfficialCudaEngine, name)
    ]
    if missing:
        raise RuntimeError(f"_ptcg_cuda lacks semantic APIs: {missing}")

    manifest_path = args.manifest.resolve()
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("semantic history smoke manifest contains no cases")
    case = cases[0]
    fixture = read_fixture(workspace_path(str(case["fixture"])))
    batch = len(fixture.seeds)
    if batch <= 0:
        raise RuntimeError("fixture contains no seeds")

    deck_rows = [
        read_deck(workspace_path(str(case["deck0"]))),
        read_deck(workspace_path(str(case["deck1"]))),
    ]
    device = torch.device("cuda", args.device_index)
    decks_i32 = torch.tensor(
        [deck_rows for _ in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.tensor(fixture.seeds, dtype=torch.int64, device=device)
    rules = workspace_path(str(manifest["rules"]))
    engine = create_official_engine(
        rules.read_bytes(),
        batch_size=batch,
        device_index=args.device_index,
    )

    if args.mode == "interactive":
        engine.reset_seeded_interactive_semantic(decks_i32, seeds)
    else:
        engine.reset_seeded_first_min_semantic(decks_i32, seeds)

    history = engine.semantic_history_raw()
    expected = {
        "total_count": (batch,),
        "write_index": (batch,),
        "log_type": (batch, 64),
        "param_count": (batch, 64),
        "params": (batch, 64, 7),
    }
    for name, shape in expected.items():
        require_cuda_tensor(name, history[name], shape)
    total_count = history["total_count"].detach().cpu()
    write_index = history["write_index"].detach().cpu()
    if bool((total_count < 0).any()):
        raise RuntimeError("semantic history total_count contains negatives")
    if bool(((write_index < 0) | (write_index >= 64)).any()):
        raise RuntimeError("semantic history write_index out of ring bounds")
    log_type = history["log_type"].detach().cpu()
    param_count = history["param_count"].detach().cpu()
    params = history["params"].detach().cpu()
    total_count_list = [int(value) for value in total_count.tolist()]
    write_index_list = [int(value) for value in write_index.tolist()]
    validate_setup_events(
        mode=args.mode,
        total_count=total_count_list,
        write_index=write_index_list,
        log_type=log_type,
        param_count=param_count,
    )

    # Legacy reset should remain callable after semantic reset and should leave
    # the history buffers resident and viewable.  It deliberately does not
    # advertise semantic history through state metadata.
    engine.reset_seeded_first_min(decks_i32, seeds)
    legacy_history = engine.semantic_history_raw()
    for name, shape in expected.items():
        require_cuda_tensor(f"legacy_{name}", legacy_history[name], shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "batch": batch,
        "case": case.get("name"),
        "mode": args.mode,
        "extension": str(Path(_ptcg_cuda.__file__).resolve()),
        "history_shapes": {
            name: list(tensor.shape) for name, tensor in history.items()
        },
        "semantic_event_preview": semantic_event_preview(
            log_type[0],
            param_count[0],
            params[0],
            total_count_list[0],
            write_index_list[0],
        ),
        "semantic_total_count": total_count_list,
        "semantic_write_index": write_index_list,
        "legacy_total_count": [
            int(value) for value in legacy_history["total_count"].detach().cpu().tolist()
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
