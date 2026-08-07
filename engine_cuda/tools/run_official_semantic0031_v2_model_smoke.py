from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
DEFAULT_MODEL_PACKAGE = Path("/bc_models/0031_pt0805_buneary_lopunny_froslass_compact_fp16")
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))

from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.semantic0031_bridge import (  # noqa: E402
    Semantic0031DeviceAdapter,
    load_semantic0031_package,
    semantic0031_v2_ready_batch,
)
from run_official_seeded_reset_paired import read_deck, read_fixture  # noqa: E402


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="Smoke-test semantic0031 v2 CUDA materialization through the foundation model."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=private / "0022_deck40_focal_s1_1_n2" / "manifest.json",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=DEFAULT_MODEL_PACKAGE,
        help="Trusted local 0031 package directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional shared FP32 checkpoint overriding the package checkpoint.",
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=CUDA_ENGINE_ROOT / "build" / "torch_0031_linux",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-limit", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_semantic0031_v2_model_smoke.json",
    )
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("_ptcg_cuda lacks encode_semantic0031_v2_lanes")
    if not args.package.is_dir():
        raise FileNotFoundError(args.package)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    package = load_semantic0031_package(
        args.package,
        device=device,
        trusted_directory=True,
        checkpoint_override=args.checkpoint,
    )
    try:
        adapter = Semantic0031DeviceAdapter(package.model, package.deck)
        manifest: dict[str, Any] = json.loads(
            args.manifest.resolve().read_text(encoding="utf-8")
        )
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise SystemExit("semantic0031 v2 model smoke manifest contains no cases")
        case = cases[0]
        fixture = read_fixture(workspace_path(str(case["fixture"])))
        batch = max(1, min(len(fixture.seeds), args.batch_limit))
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
        normalized = semantic0031_v2_ready_batch(encoded)
        actions, lengths = adapter.act_device_semantic0031_v2(encoded)
        torch.cuda.synchronize(device)

        if actions.device != device or lengths.device != device:
            raise RuntimeError("semantic0031 v2 model actions left the CUDA device")
        if actions.shape != (batch, adapter.max_select) or lengths.shape != (batch,):
            raise RuntimeError(
                f"unexpected action shapes: actions={tuple(actions.shape)} lengths={tuple(lengths.shape)}"
            )
        valid = actions.ge(0)
        option_count = normalized["option_mask"].shape[1]
        if bool((actions[valid] >= option_count).any()):
            raise RuntimeError("semantic0031 v2 model emitted an out-of-range option index")
        if bool((lengths < 0).any()) or bool((lengths > adapter.max_select).any()):
            raise RuntimeError("semantic0031 v2 model emitted invalid action lengths")

        summary = {
            "batch": batch,
            "case": case.get("name"),
            "extension": str(Path(_ptcg_cuda.__file__).resolve()),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "model_package": str(args.package.resolve()),
            "option_rows_first": int(normalized["option_mask"][0].sum().detach().cpu().item()),
            "selected_lengths": [int(value) for value in lengths.detach().cpu().tolist()],
            "first_actions": [
                int(value)
                for value in actions[0, : int(lengths[0].detach().cpu().item())]
                .detach()
                .cpu()
                .tolist()
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        package.close()


if __name__ == "__main__":
    main()
