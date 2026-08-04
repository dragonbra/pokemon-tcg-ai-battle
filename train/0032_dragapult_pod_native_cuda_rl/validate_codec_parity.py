"""Elementwise CPU observation codec versus resident CUDA codec parity gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = ROOT / "engine_cuda"
for path in (CUDA_ROOT / "python", CUDA_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from pure_policy_codec_v1 import PolicyCodecV1  # noqa: E402
from run_official_seeded_reset_paired import read_deck  # noqa: E402
from validate_official_idonly_device_codec import RawOfficialLib  # noqa: E402


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_lane(encoded: Any, device: dict[str, torch.Tensor], lane: int) -> list[dict[str, Any]]:
    expected = {
        "global_cat": torch.tensor(encoded.global_cat, dtype=torch.long),
        "global_num": torch.tensor(encoded.global_num, dtype=torch.float32),
        "entity_cat": torch.tensor(encoded.entity_cat, dtype=torch.long).reshape(-1, 6),
        "entity_num": torch.tensor(encoded.entity_num, dtype=torch.float32).reshape(-1, 10),
        "entity_parent": torch.tensor(encoded.entity_parent, dtype=torch.long),
        "option_cat": torch.tensor(encoded.option_cat, dtype=torch.long).reshape(-1, 12),
        "option_num": torch.tensor(encoded.option_num, dtype=torch.float32).reshape(-1, 4),
        "option_equiv": torch.tensor(encoded.option_equiv, dtype=torch.long),
        "min_count": torch.tensor(encoded.min_count, dtype=torch.long),
        "max_count": torch.tensor(encoded.max_count, dtype=torch.long),
    }
    entity_count = len(encoded.entity_cat)
    option_count = len(encoded.option_cat)
    actual = {
        "global_cat": device["global_cat"][lane],
        "global_num": device["global_num"][lane],
        "entity_cat": device["entity_cat"][lane, :entity_count],
        "entity_num": device["entity_num"][lane, :entity_count],
        "entity_parent": device["entity_parent"][lane, :entity_count],
        "option_cat": device["option_cat"][lane, :option_count],
        "option_num": device["option_num"][lane, :option_count],
        "option_equiv": device["option_equiv"][lane, :option_count],
        "min_count": device["min_count"][lane],
        "max_count": device["max_count"][lane],
    }
    mismatches: list[dict[str, Any]] = []
    if int(device["entity_mask"][lane].sum()) != entity_count:
        mismatches.append({"field": "entity_mask_count", "expected": entity_count})
    if int(device["option_mask"][lane].sum()) != option_count:
        mismatches.append({"field": "option_mask_count", "expected": option_count})
    for name in expected:
        left = expected[name]
        right = actual[name]
        equal = (
            torch.allclose(left, right, rtol=0.0, atol=1.0e-6)
            if left.dtype.is_floating_point
            else torch.equal(left, right)
        )
        if not equal:
            mismatches.append(
                {
                    "field": name,
                    "expected_shape": list(left.shape),
                    "actual_shape": list(right.shape),
                    "max_abs": (
                        float((left - right).abs().max()) if left.numel() and left.shape == right.shape else None
                    ),
                }
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--deck0", type=Path, required=True)
    parser.add_argument("--deck1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2026080501)
    args = parser.parse_args()
    sys.path.insert(0, str(args.extension_dir.resolve()))
    import _ptcg_cuda

    device = torch.device("cuda")
    deck0 = read_deck(args.deck0)
    deck1 = read_deck(args.deck1)
    pairs = [(deck0, deck1) for _ in range(args.batch)]
    seeds_list = [args.seed_start + lane for lane in range(args.batch)]
    shim = RawOfficialLib(args.lib.resolve())
    battles = shim.start_battles(pairs, seeds_list)
    decks = torch.tensor([[deck0, deck1] for _ in range(args.batch)], dtype=torch.int32, device=device)
    seeds = torch.tensor(seeds_list, dtype=torch.int64, device=device)
    engine = create_official_engine(args.rules.read_bytes(), batch_size=args.batch)
    engine.reset_seeded_interactive(decks, seeds)
    codec = PolicyCodecV1()
    decisions = 0
    episodes = 0
    status_mismatches = 0
    mismatches: list[dict[str, Any]] = []
    statuses: list[int] = []
    try:
        for step in range(args.steps):
            engine.advance_to_decision()
            metas = shim.select_meta_many(battles)
            statuses = engine.statuses().cpu().tolist()
            expected_statuses = [TERMINAL if meta.game_result else NEEDS_ACTION for meta in metas]
            status_mismatches += sum(a != b for a, b in zip(statuses, expected_statuses))
            if status_mismatches or any(status == ERROR for status in statuses):
                break
            terminals = [lane for lane, meta in enumerate(metas) if meta.game_result]
            if terminals:
                terminal_mask = torch.zeros(args.batch, dtype=torch.bool, device=device)
                terminal_mask[terminals] = True
                seeds.add_(terminal_mask.long() * args.batch)
                engine.reset_seeded_interactive_masked(decks, seeds, terminal_mask)
                for lane in terminals:
                    shim.finish_many([battles[lane]])
                    seeds_list[lane] += args.batch
                    battles[lane] = shim.start_battles([pairs[lane]], [seeds_list[lane]])[0]
                episodes += len(terminals)
                continue
            observations = shim.observation_many(battles)
            actions = [list(range(min(meta.select_min, meta.option_count))) for meta in metas]
            cpu_rows = [codec.encode(obs, action) for obs, action in zip(observations, actions)]
            gpu = {name: value.detach().cpu() for name, value in dict(engine.encode_policy_v1()).items()}
            for lane, row in enumerate(cpu_rows):
                for mismatch in compare_lane(row, gpu, lane):
                    mismatches.append({"step": step, "lane": lane, **mismatch})
            decisions += args.batch
            if mismatches:
                break
            indices = torch.full((args.batch, 64), -1, dtype=torch.long, device=device)
            lengths = torch.tensor([len(action) for action in actions], dtype=torch.long, device=device)
            for lane, action in enumerate(actions):
                if action:
                    indices[lane, : len(action)] = torch.tensor(action, dtype=torch.long, device=device)
            engine.pack_actions(indices, lengths)
            engine.apply_packed_actions()
            errors = shim.select_many(battles, actions)
            if any(errors):
                raise RuntimeError(f"official CPU action failed: {errors}")
    finally:
        shim.finish_many(battles)
    report = {
        "schema": "0032_policy_codec_cpu_cuda_parity_v1",
        "passed": decisions > 0 and not mismatches and status_mismatches == 0 and ERROR not in statuses,
        "contract": "pure_policy_codec_v1_cpu_observation_vs_official_cuda_policy_codec_v1",
        "decisions_compared": decisions,
        "episodes_reset_paired": episodes,
        "status_mismatches": status_mismatches,
        "codec_mismatch_count": len(mismatches),
        "first_codec_mismatches": mismatches[:16],
        "batch": args.batch,
        "requested_steps": args.steps,
        "rules_sha256": sha256_file(args.rules),
        "cpu_library_sha256": sha256_file(args.lib),
        "extension_sha256": sha256_file(Path(_ptcg_cuda.__file__)),
        "deck_sha256": [sha256_file(args.deck0), sha256_file(args.deck1)],
        "device": torch.cuda.get_device_name(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
