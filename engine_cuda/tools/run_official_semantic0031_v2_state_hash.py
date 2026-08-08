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
from run_official_seeded_reset_paired import read_deck, read_fixture  # noqa: E402


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description=(
            "Replay a deterministic legal-first action path and hash every official "
            "state and semantic0031 v2 batch."
        )
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
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare",
        type=Path,
        help="Require this run to be byte-identical to an earlier hash report.",
    )
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def hash_tensor(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def hash_semantic(encoded: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(encoded):
        digest.update(name.encode("ascii"))
        digest.update(hash_tensor(encoded[name]).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("steps must be positive")
    sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch
    import _ptcg_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not hasattr(_ptcg_cuda.OfficialCudaEngine, "encode_semantic0031_v2_lanes"):
        raise RuntimeError("_ptcg_cuda lacks encode_semantic0031_v2_lanes")

    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("semantic0031 v2 manifest contains no cases")
    case = cases[0]
    fixture = read_fixture(workspace_path(str(case["fixture"])))
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    decks = torch.tensor(
        [[
            read_deck(workspace_path(str(case["deck0"]))),
            read_deck(workspace_path(str(case["deck1"]))),
        ]],
        dtype=torch.int32,
        device=device,
    )
    engine = create_official_engine(
        workspace_path(str(manifest["rules"])).read_bytes(),
        batch_size=1,
        device_index=args.device_index,
    )
    engine.reset_seeded_first_min_semantic(
        decks,
        torch.tensor([fixture.seeds[0]], dtype=torch.int64, device=device),
    )
    lanes = torch.zeros(1, dtype=torch.int32, device=device)
    records: list[dict[str, Any]] = []
    for step in range(args.steps):
        status = int(engine.statuses()[0].item())
        if status != 1:
            records.append({"step": step, "status": status, "terminal": True})
            break
        encoded = engine.encode_semantic0031_v2_lanes(lanes)
        minimum = int(encoded["min_count"][0].item())
        option_count = int(encoded["option_mask"][0].sum().item())
        if minimum < 0 or minimum > option_count:
            raise RuntimeError(
                f"invalid action cardinality at step {step}: min={minimum} options={option_count}"
            )
        records.append(
            {
                "step": step,
                "status": status,
                # OfficialStatePod embeds a per-allocation device pointer for
                # semantic history, so its raw bytes are diagnostic only and
                # cannot be compared between two engine instances.
                "raw_state_sha256": hash_tensor(engine.state_bytes()),
                "semantic_sha256": hash_semantic(encoded),
                "min_count": minimum,
                "option_count": option_count,
            }
        )
        width = max(1, minimum)
        action = torch.full((1, width), -1, dtype=torch.int64, device=device)
        if minimum:
            action[0, :minimum] = torch.arange(minimum, dtype=torch.int64, device=device)
        engine.pack_actions(action, torch.tensor([minimum], dtype=torch.int64, device=device))
        engine.apply_packed_actions()
        engine.advance_to_decision()
    else:
        records.append(
            {
                "step": args.steps,
                "status": int(engine.statuses()[0].item()),
                "raw_state_sha256": hash_tensor(engine.state_bytes()),
                "terminal": True,
            }
        )

    report = {
        "extension": str(Path(_ptcg_cuda.__file__).resolve()),
        "case": case.get("name"),
        "seed": int(fixture.seeds[0]),
        "records": records,
    }
    if args.compare is not None:
        expected = json.loads(args.compare.resolve().read_text(encoding="utf-8"))
        def comparable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    key: value
                    for key, value in record.items()
                    if key != "raw_state_sha256"
                }
                for record in records
            ]
        if comparable(report["records"]) != comparable(expected.get("records", [])):
            raise SystemExit("state/semantic hash mismatch against comparison report")
        report["matches"] = str(args.compare.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "extension": report["extension"],
                "records": len(records),
                "matches": report.get("matches"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
