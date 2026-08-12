from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
for module_root in (
    CUDA_ENGINE_ROOT / "python",
    WORKSPACE_ROOT / "tools",
):
    sys.path.insert(0, str(module_root))

from ptcg_cuda_engine.legacy_codecs import (  # noqa: E402
    policy_codec_v1_to_idonly_codec_v1,
)
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3


class RawStartData(ctypes.Structure):
    _fields_ = [
        ("battle_ptr", ctypes.c_void_p),
        ("error_player", ctypes.c_int),
        ("error_type", ctypes.c_int),
    ]


class RawSerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("select_player", ctypes.c_int),
    ]


@dataclass(frozen=True)
class RawSelectMeta:
    turn: int
    phase: int
    game_result: int
    finish_reason: int
    select_type: int
    select_context: int
    select_player: int
    select_min: int
    select_max: int
    option_count: int


class RawOfficialLib:
    META_SIZE = 27

    def __init__(self, path: Path):
        self.lib = ctypes.cdll.LoadLibrary(str(path))
        self.lib.RawGameInitialize.restype = ctypes.c_int
        self.lib.RawGameInitialize.argtypes = []
        self.lib.RawBattleStartSeeded.restype = RawStartData
        self.lib.RawBattleStartSeeded.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_ulonglong,
        ]
        self.lib.RawGetBattleData.restype = RawSerialData
        self.lib.RawGetBattleData.argtypes = [ctypes.c_void_p]
        self.lib.RawSelect.restype = ctypes.c_int
        self.lib.RawSelect.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.RawWriteSelectMeta.restype = ctypes.c_int
        self.lib.RawWriteSelectMeta.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.RawBattleFinish.restype = None
        self.lib.RawBattleFinish.argtypes = [ctypes.c_void_p]
        if int(self.lib.RawGameInitialize()) != 0:
            raise RuntimeError("raw official shim initialization failed")

    def start_battles(
        self,
        deck_pairs: list[tuple[list[int], list[int]]],
        seeds: list[int],
    ) -> list[ctypes.c_void_p]:
        battles: list[ctypes.c_void_p] = []
        try:
            for (deck0, deck1), seed in zip(deck_pairs, seeds, strict=True):
                cards = (ctypes.c_int * 120)(*(deck0 + deck1))
                result = self.lib.RawBattleStartSeeded(
                    cards, ctypes.c_ulonglong(seed)
                )
                if not result.battle_ptr or result.error_type:
                    raise RuntimeError(
                        "raw official start failed: "
                        f"player={result.error_player} type={result.error_type}"
                    )
                battles.append(ctypes.c_void_p(result.battle_ptr))
        except Exception:
            self.finish_many(battles)
            raise
        return battles

    def select_meta_many(
        self, battles: list[ctypes.c_void_p]
    ) -> list[RawSelectMeta]:
        result: list[RawSelectMeta] = []
        for battle in battles:
            values = (ctypes.c_int * self.META_SIZE)()
            written = int(
                self.lib.RawWriteSelectMeta(battle, values, self.META_SIZE)
            )
            if written != self.META_SIZE:
                raise RuntimeError(f"raw official meta failed: {written}")
            result.append(
                RawSelectMeta(
                    turn=int(values[1]),
                    phase=int(values[2]),
                    game_result=int(values[3]),
                    finish_reason=int(values[4]),
                    select_type=int(values[5]),
                    select_context=int(values[6]),
                    select_player=int(values[7]),
                    select_min=int(values[8]),
                    select_max=int(values[9]),
                    option_count=int(values[10]),
                )
            )
        return result

    def observation_many(
        self, battles: list[ctypes.c_void_p]
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for battle in battles:
            serial = self.lib.RawGetBattleData(battle)
            if serial.select_player < 0 or not serial.json:
                raise RuntimeError(
                    f"raw official observation failed: {serial.select_player}"
                )
            observations.append(json.loads(serial.json.decode("utf-8")))
        return observations

    def select_many(
        self,
        battles: list[ctypes.c_void_p],
        actions: list[list[int]],
    ) -> list[int]:
        errors: list[int] = []
        for battle, action in zip(battles, actions, strict=True):
            selected = (ctypes.c_int * len(action))(*action)
            errors.append(int(self.lib.RawSelect(battle, selected, len(action))))
        return errors

    def finish_many(self, battles: list[ctypes.c_void_p]) -> None:
        for battle in battles:
            if battle:
                self.lib.RawBattleFinish(battle)


def read_deck_csv(path: Path) -> list[int]:
    cards = [
        int(line.split(",", 1)[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lower().startswith("card")
    ]
    if len(cards) != 60:
        raise ValueError(f"expected 60 cards in {path}, got {len(cards)}")
    return cards


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen CPU ID-only codec with the resident official "
            "CUDA codec conversion while both engines replay identical actions."
        )
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--lib",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "build"
        / "official_raw_seed_shim"
        / "libofficial_raw_seed_shim.so",
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "build" / "torch_official",
    )
    parser.add_argument(
        "--deck0",
        type=Path,
        default=WORKSPACE_ROOT
        / "bc_models"
        / "agent_cynthia_core_meanpool_epochmix_v1_c7b3253f_20260727"
        / "deck.csv",
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=WORKSPACE_ROOT
        / "bc_models"
        / "agent_yushin_idonly_bc_v1_20260723"
        / "deck.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT
        / "bc_models"
        / "agent_yushin_idonly_bc_v1_20260723"
        / "policy.pt",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=2026080101)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT
        / "artifacts"
        / "official_idonly_device_codec_paired.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_idonly_codec(checkpoint: Path, torch: Any) -> tuple[Any, Any]:
    source = checkpoint.parent / "idonly_policy.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("_paired_idonly_policy", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ID-only policy source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = module.ModelConfig(**payload["model_config"])
    return module.IDOnlyCodec(config), config


def first_min_action(meta: Any, max_action_steps: int) -> list[int]:
    if meta.select_min < 0 or meta.select_max < meta.select_min:
        raise RuntimeError(
            f"invalid select bounds {meta.select_min}/{meta.select_max}"
        )
    if meta.select_min > meta.option_count:
        raise RuntimeError(
            f"select min {meta.select_min} exceeds options {meta.option_count}"
        )
    if meta.select_min > max_action_steps:
        raise RuntimeError(
            f"select min {meta.select_min} exceeds legacy action capacity "
            f"{max_action_steps}"
        )
    return list(range(meta.select_min))


def mismatch_detail(
    *,
    decision: int,
    lane: int,
    field: str,
    expected: Any,
    actual: Any,
) -> dict[str, Any]:
    import torch

    different = expected.ne(actual)
    first = different.nonzero()
    first_index = first[0].tolist() if first.numel() else []
    detail: dict[str, Any] = {
        "decision": decision,
        "lane": lane,
        "field": field,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "different_elements": int(different.long().sum().item()),
        "first_index": first_index,
    }
    if first_index:
        index = tuple(first_index)
        detail["expected"] = expected[index].item()
        detail["actual"] = actual[index].item()
    if expected.dtype.is_floating_point:
        detail["max_abs"] = float(
            (expected - actual).abs().max().item()
        )
    return detail


def compare_row(
    *,
    row: dict[str, Any],
    device_batch: dict[str, Any],
    lane: int,
    decision: int,
) -> list[dict[str, Any]]:
    import torch

    mismatches: list[dict[str, Any]] = []
    actual_global_cat = device_batch["global_cat"][lane]
    actual_global_num = device_batch["global_num"][lane]
    actual_entity_cat = device_batch["entity_cat"][lane]
    actual_entity_num = device_batch["entity_num"][lane]
    actual_entity_mask = device_batch["entity_mask"][lane]
    actual_option_cat = device_batch["option_cat"][lane]
    actual_option_mask = device_batch["option_mask"][lane]

    entity_count = len(row["entity_cat"])
    option_count = len(row["option_cat"])
    expected = {
        "global_cat": torch.tensor(
            row["global_cat"], dtype=actual_global_cat.dtype
        ),
        "global_num": torch.tensor(
            row["global_num"], dtype=actual_global_num.dtype
        ),
        "entity_cat": torch.zeros_like(actual_entity_cat),
        "entity_num": torch.zeros_like(actual_entity_num),
        "entity_mask": torch.zeros_like(actual_entity_mask),
        "option_cat": torch.zeros_like(actual_option_cat),
        "option_mask": torch.zeros_like(actual_option_mask),
        "min_count": torch.tensor(
            int(row["min_count"]), dtype=device_batch["min_count"].dtype
        ),
        "max_count": torch.tensor(
            int(row["max_count"]), dtype=device_batch["max_count"].dtype
        ),
    }
    if entity_count > actual_entity_cat.shape[0]:
        raise RuntimeError(
            f"CPU codec emitted {entity_count} entities, CUDA capacity is "
            f"{actual_entity_cat.shape[0]}"
        )
    if option_count > actual_option_cat.shape[0]:
        raise RuntimeError(
            f"CPU codec emitted {option_count} options, CUDA capacity is "
            f"{actual_option_cat.shape[0]}"
        )
    if entity_count:
        expected["entity_cat"][:entity_count] = torch.tensor(
            row["entity_cat"], dtype=actual_entity_cat.dtype
        )
        expected["entity_num"][:entity_count] = torch.tensor(
            row["entity_num"], dtype=actual_entity_num.dtype
        )
        expected["entity_mask"][:entity_count] = True
    if option_count:
        expected["option_cat"][:option_count] = torch.tensor(
            row["option_cat"], dtype=actual_option_cat.dtype
        )
        expected["option_mask"][:option_count] = True

    actual = {
        "global_cat": actual_global_cat,
        "global_num": actual_global_num,
        "entity_cat": actual_entity_cat,
        "entity_num": actual_entity_num,
        "entity_mask": actual_entity_mask,
        "option_cat": actual_option_cat,
        "option_mask": actual_option_mask,
        "min_count": device_batch["min_count"][lane],
        "max_count": device_batch["max_count"][lane],
    }
    for field in expected:
        if not torch.equal(expected[field], actual[field]):
            mismatches.append(
                mismatch_detail(
                    decision=decision,
                    lane=lane,
                    field=field,
                    expected=expected[field],
                    actual=actual[field],
                )
            )
    return mismatches


def main() -> int:
    args = parse_args()
    if min(args.batch, args.steps) <= 0:
        raise ValueError("batch and steps must be positive")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)

    paths = (
        args.rules.resolve(),
        args.lib.resolve(),
        args.deck0.resolve(),
        args.deck1.resolve(),
        args.checkpoint.resolve(),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    rules, library, deck0_path, deck1_path, checkpoint = paths

    codec, config = load_idonly_codec(checkpoint, torch)
    max_card_id = int(config.max_card_id)
    max_action_steps = int(config.max_action_steps)
    deck0 = read_deck_csv(deck0_path)
    deck1 = read_deck_csv(deck1_path)
    deck_pairs = [(deck0, deck1) for _ in range(args.batch)]
    seeds_list = [args.seed_start + lane for lane in range(args.batch)]

    shim = RawOfficialLib(library)
    battles = shim.start_battles(deck_pairs, seeds_list)
    decks = torch.tensor(
        [[deck0, deck1] for _ in range(args.batch)],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.tensor(seeds_list, dtype=torch.int64, device=device)
    engine = create_official_engine(
        rules.read_bytes(), batch_size=args.batch, device_index=args.device_index
    )
    engine.reset_seeded_interactive(decks, seeds)

    decisions = 0
    completed_steps = 0
    episodes_reset = 0
    status_mismatches = 0
    codec_mismatches: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}
    try:
        for step in range(args.steps):
            engine.advance_to_decision()
            metas = shim.select_meta_many(battles)
            statuses = engine.statuses().detach().cpu().tolist()
            expected_statuses = [
                TERMINAL if meta.game_result else NEEDS_ACTION for meta in metas
            ]
            status_mismatches += sum(
                int(actual != expected)
                for actual, expected in zip(statuses, expected_statuses)
            )
            if status_mismatches:
                break
            if any(status == ERROR for status in statuses):
                break
            terminal_lanes = [
                lane for lane, meta in enumerate(metas) if meta.game_result
            ]
            if terminal_lanes:
                terminal_mask = torch.zeros(
                    args.batch, dtype=torch.bool, device=device
                )
                terminal_mask[terminal_lanes] = True
                seeds.add_(terminal_mask.to(torch.int64) * args.batch)
                engine.reset_seeded_interactive_masked(
                    decks, seeds, terminal_mask
                )
                for lane in terminal_lanes:
                    shim.finish_many([battles[lane]])
                    seeds_list[lane] += args.batch
                    battles[lane] = shim.start_battles(
                        [deck_pairs[lane]], [seeds_list[lane]]
                    )[0]
                episodes_reset += len(terminal_lanes)
                continue

            observations = shim.observation_many(battles)
            actions = [
                first_min_action(meta, max_action_steps) for meta in metas
            ]
            rows = [
                codec.encode(observation, action)
                for observation, action in zip(observations, actions)
            ]
            if any(row is None for row in rows):
                raise RuntimeError("legacy CPU codec rejected a valid official action")

            policy_batch = dict(engine.encode_policy_v1())
            idonly_batch = policy_codec_v1_to_idonly_codec_v1(
                policy_batch,
                max_card_id=max_card_id,
                max_action_steps=max_action_steps,
            )
            inspected = {
                name: value.detach().cpu() for name, value in idonly_batch.items()
            }
            for lane, row in enumerate(rows):
                assert row is not None
                codec_mismatches.extend(
                    compare_row(
                        row=row,
                        device_batch=inspected,
                        lane=lane,
                        decision=decisions + lane,
                    )
                )
                phase = str(metas[lane].phase)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            decisions += args.batch
            completed_steps += 1
            if codec_mismatches:
                break

            packed = torch.full(
                (args.batch, max_action_steps),
                -1,
                dtype=torch.int64,
                device=device,
            )
            lengths = torch.tensor(
                [len(action) for action in actions],
                dtype=torch.int64,
                device=device,
            )
            for lane, action in enumerate(actions):
                if action:
                    packed[lane, : len(action)] = torch.tensor(
                        action, dtype=torch.int64, device=device
                    )
            engine.pack_actions(packed, lengths)
            engine.apply_packed_actions()
            errors = shim.select_many(battles, actions)
            if any(error != 0 for error in errors):
                raise RuntimeError(f"official CPU select failed: {errors}")
    finally:
        shim.finish_many(battles)

    passed = (
        decisions > 0
        and not codec_mismatches
        and status_mismatches == 0
        and all(status != ERROR for status in statuses)
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "contract": "official_cpu_idonly_v1_vs_cuda_policy_codec_conversion_v1",
        "batch": args.batch,
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
        "decisions_compared": decisions,
        "episodes_reset_paired": episodes_reset,
        "phase_counts": phase_counts,
        "status_mismatches": status_mismatches,
        "codec_mismatch_count": len(codec_mismatches),
        "first_codec_mismatches": codec_mismatches[:16],
        "final_statuses": statuses,
        "max_card_id": max_card_id,
        "max_action_steps": max_action_steps,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "rules_sha256": sha256_file(rules),
        "library_sha256": sha256_file(library),
        "checkpoint_sha256": sha256_file(checkpoint),
        "deck_sha256": [sha256_file(deck0_path), sha256_file(deck1_path)],
        "scope": (
            "Test-only host inspection compares every tensor element. The "
            "production conversion itself performs no host transfer or sync."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
