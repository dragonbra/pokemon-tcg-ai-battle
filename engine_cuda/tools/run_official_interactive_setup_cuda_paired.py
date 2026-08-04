from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from run_official_seeded_reset_paired import (  # noqa: E402
    STATE_BYTES,
    first_mismatch,
    read_deck,
    read_fixture,
    tensor_bytes,
)


SELECT_CONTEXT_OFFSET = 2594
MAIN_CONTEXT = 1


def parse_args() -> argparse.Namespace:
    private_root = (
        CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Replay first-min setup through the interactive CUDA action path "
            "and compare its final Main boundary with real official fixtures."
        )
    )
    parser.add_argument("--rules", type=Path, default=private_root / "official_rules.bin")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=private_root / "official_seeded_reset_marnie_alakazam_s1_10.bin",
    )
    parser.add_argument(
        "--deck0",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "agent_marnie_prize_control_v4_3121746f_20260728"
            / "deck.csv"
        ),
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "agent_yushin_idonly_bc_v1_20260723"
            / "deck.csv"
        ),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-setup-actions", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT
            / "artifacts"
            / "official_interactive_setup_cuda_paired.json"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    rules = args.rules.resolve()
    fixture_path = args.fixture.resolve()
    deck_paths = [args.deck0.resolve(), args.deck1.resolve()]
    for path in [rules, fixture_path, *deck_paths]:
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    fixture = read_fixture(fixture_path)

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if str(_ptcg_cuda.OFFICIAL_INTERACTIVE_SETUP_POLICY) != "device_action_v1":
        raise RuntimeError("interactive CUDA setup policy mismatch")

    device = torch.device("cuda", args.device_index)
    deck_rows = [read_deck(path) for path in deck_paths]
    decks = torch.tensor([deck_rows], dtype=torch.int32, device=device)
    engine = create_official_engine(
        rules.read_bytes(), batch_size=1, device_index=args.device_index
    )
    option_indices = torch.arange(
        80, dtype=torch.int64, device=device
    ).reshape(1, 80)
    setup_actions_per_seed: list[int] = []
    state_mismatches = 0
    status_mismatches = 0
    first_state_mismatch: dict[str, int] | None = None
    setup_codec_decisions = 0

    for lane, seed in enumerate(fixture.seeds):
        seeds = torch.tensor([seed], dtype=torch.int64, device=device)
        engine.reset_seeded_interactive(decks, seeds)
        action_count = 0
        while True:
            context = int(engine.state_bytes()[0, SELECT_CONTEXT_OFFSET].item())
            if context == MAIN_CONTEXT:
                break
            if action_count >= args.max_setup_actions:
                raise RuntimeError(f"setup action limit exceeded for seed {seed}")
            codec = engine.encode_policy_v1()
            if int(codec["option_mask"].sum().item()) <= 0:
                raise RuntimeError(f"setup codec exposed no options for seed {seed}")
            engine.pack_actions(option_indices, codec["min_count"])
            engine.apply_packed_actions()
            action_count += 1
            setup_codec_decisions += 1
        actual = tensor_bytes(engine.state_bytes())
        expected = fixture.states[lane * STATE_BYTES : (lane + 1) * STATE_BYTES]
        mismatch = first_mismatch(expected, actual)
        if mismatch is not None:
            state_mismatches += 1
            if first_state_mismatch is None:
                first_state_mismatch = {"seed": seed, **mismatch}
        status = int(engine.statuses()[0].item())
        if status != fixture.statuses[lane]:
            status_mismatches += 1
        setup_actions_per_seed.append(action_count)

    passed = state_mismatches == 0 and status_mismatches == 0
    result: dict[str, Any] = {
        "passed": passed,
        "contract": "official_interactive_setup_cuda_first_min_paired_v1",
        "interactive_policy": str(_ptcg_cuda.OFFICIAL_INTERACTIVE_SETUP_POLICY),
        "replay_policy": "first_min_v1",
        "seed_count": len(fixture.seeds),
        "seed_min": min(fixture.seeds),
        "seed_max": max(fixture.seeds),
        "setup_codec_decisions": setup_codec_decisions,
        "min_setup_actions": min(setup_actions_per_seed),
        "max_setup_actions": max(setup_actions_per_seed),
        "state_bytes_per_seed": STATE_BYTES,
        "state_bytes_compared": len(fixture.seeds) * STATE_BYTES,
        "state_mismatches": state_mismatches,
        "status_mismatches": status_mismatches,
        "first_state_mismatch": first_state_mismatch,
        "arena_allocated_bytes": int(engine.allocated_bytes),
        "device": torch.cuda.get_device_name(args.device_index),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rule_pack_sha256": sha256_file(rules),
        "deck_sha256": [sha256_file(path) for path in deck_paths],
        "private_fixture_sha256": sha256_file(fixture_path),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "device_action_contract": (
            "setup codec tensors, selected indices, counts, action packing, "
            "and state transitions remain on CUDA; host reads are test-only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
