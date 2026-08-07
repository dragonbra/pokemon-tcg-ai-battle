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


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="Smoke-test independent semantic0031_codec_v2 CUDA materialization."
    )
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
    parser.add_argument("--batch-limit", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_semantic0031_v2_smoke.json",
    )
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def require_cuda_tensor(name: str, tensor: Any, shape: tuple[int, ...], dtype: Any) -> None:
    if tensor.device.type != "cuda" or tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise RuntimeError(
            f"invalid {name}: device={tensor.device} shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} expected_shape={shape} expected_dtype={dtype}"
        )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("_ptcg_cuda lacks encode_semantic0031_v2_lanes")

    manifest: dict[str, Any] = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("semantic0031 v2 smoke manifest contains no cases")
    case = cases[0]
    fixture = read_fixture(workspace_path(str(case["fixture"])))
    batch = max(1, min(len(fixture.seeds), args.batch_limit))
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)

    deck_rows = [
        read_deck(workspace_path(str(case["deck0"]))),
        read_deck(workspace_path(str(case["deck1"]))),
    ]
    decks_i32 = torch.tensor(
        [deck_rows for _ in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    seeds = torch.tensor(fixture.seeds[:batch], dtype=torch.int64, device=device)
    rules = workspace_path(str(manifest["rules"]))
    engine = create_official_engine(
        rules.read_bytes(),
        batch_size=batch,
        device_index=args.device_index,
    )
    engine.reset_seeded_first_min_semantic(decks_i32, seeds)
    lanes = torch.arange(batch, dtype=torch.int32, device=device)
    encoded = engine.encode_semantic0031_v2_lanes(lanes)

    expected_shapes = {
        "global_cat": ((batch, 12), torch.int64),
        "global_num": ((batch, 24), torch.float32),
        "global_state": ((batch, 24), torch.int64),
        "card_cat": ((batch, 128, 9), torch.int64),
        "card_num": ((batch, 128, 7), torch.float32),
        "card_state": ((batch, 128, 7), torch.int64),
        "card_parent": ((batch, 128), torch.int64),
        "card_mask": ((batch, 128), torch.uint8),
        "resource_cat": ((batch, 64, 4), torch.int64),
        "resource_num": ((batch, 64, 15), torch.float32),
        "resource_state": ((batch, 64, 15), torch.int64),
        "resource_mask": ((batch, 64), torch.uint8),
        "event_cat": ((batch, 64, 31), torch.int64),
        "event_num": ((batch, 64, 4), torch.float32),
        "event_state": ((batch, 64, 4), torch.int64),
        "event_mask": ((batch, 64), torch.uint8),
        "event_source": ((batch, 64), torch.int64),
        "event_target": ((batch, 64), torch.int64),
        "event_before": ((batch, 64), torch.int64),
        "event_after": ((batch, 64), torch.int64),
        "option_cat": ((batch, 128, 19), torch.int64),
        "option_num": ((batch, 128, 2), torch.float32),
        "option_state": ((batch, 128, 2), torch.int64),
        "option_mask": ((batch, 128), torch.uint8),
        "option_source": ((batch, 128), torch.int64),
        "option_target": ((batch, 128), torch.int64),
        "option_context": ((batch, 128), torch.int64),
        "option_effect_card": ((batch, 128), torch.int64),
        "option_skill_id": ((batch, 1), torch.int64),
        "option_skill_role": ((batch, 1), torch.int64),
        "option_skill_parent": ((batch, 1), torch.int64),
        "option_skill_mask": ((batch, 1), torch.uint8),
        "option_effect_id": ((batch, 1), torch.int64),
        "option_effect_role": ((batch, 1), torch.int64),
        "option_effect_parent": ((batch, 1), torch.int64),
        "option_effect_mask": ((batch, 1), torch.uint8),
        "min_count": ((batch,), torch.int64),
        "max_count": ((batch,), torch.int64),
        "targets": ((batch, 1), torch.int64),
    }
    for name, (shape, dtype) in expected_shapes.items():
        require_cuda_tensor(name, encoded[name], shape, dtype)

    card_mask = encoded["card_mask"].bool()
    option_mask = encoded["option_mask"].bool()
    event_mask = encoded["event_mask"].bool()
    resource_mask = encoded["resource_mask"].bool()
    if not bool(card_mask.any()):
        raise RuntimeError("semantic0031 v2 emitted no card rows")
    if not bool(option_mask.any()):
        raise RuntimeError("semantic0031 v2 emitted no option rows")
    if not bool(resource_mask.any()):
        raise RuntimeError("semantic0031 v2 emitted no resource rows")
    if not bool(event_mask.any()):
        raise RuntimeError("semantic0031 v2 emitted no causal event rows")

    card_serial = encoded["card_cat"][..., 1]
    if not bool((card_serial[card_mask] > 0).all()):
        raise RuntimeError("visible card rows must carry one-based serials")
    card_ids = encoded["card_cat"][..., 0]
    if not bool((card_ids[card_mask] > 0).any()):
        raise RuntimeError("visible card rows must include identities")

    event_types = encoded["event_cat"][..., 0]
    if not bool((event_types[event_mask] == 5).any()):
        raise RuntimeError("semantic0031 v2 event history lacks Draw")
    draw_serials = encoded["event_cat"][..., 14]
    if not bool((draw_serials[event_types == 5] > 0).any()):
        raise RuntimeError("Draw events must carry serial")

    source_relation = encoded["option_source"]
    target_relation = encoded["option_target"]
    context_relation = encoded["option_context"]
    effect_relation = encoded["option_effect_card"]
    card_capacity = encoded["card_mask"].shape[1]
    relation_tensors = {
        "option_source": source_relation,
        "option_target": target_relation,
        "option_context": context_relation,
        "option_effect_card": effect_relation,
    }
    for name, relation in relation_tensors.items():
        active = relation[option_mask]
        if bool(((active < 0) | (active > card_capacity)).any()):
            raise RuntimeError(f"{name} relation outside one-based card range")

    summary = {
        "batch": batch,
        "case": case.get("name"),
        "extension": str(Path(_ptcg_cuda.__file__).resolve()),
        "global_cat_first": encoded["global_cat"][0].detach().cpu().tolist(),
        "card_rows": int(card_mask[0].sum().item()),
        "option_rows": int(option_mask[0].sum().item()),
        "resource_rows": int(resource_mask[0].sum().item()),
        "event_rows": int(event_mask[0].sum().item()),
        "first_event_types": [
            int(value)
            for value in event_types[0, event_mask[0]].detach().cpu().tolist()[:16]
        ],
        "first_card_serials": [
            int(value)
            for value in card_serial[0, card_mask[0]].detach().cpu().tolist()[:16]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
