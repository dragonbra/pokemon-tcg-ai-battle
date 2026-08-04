"""Benchmark the formal 0031 train path without allocating a tracked version."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .domain.prototypes import PrototypeIndex
from .features.collate import BucketPadding
from .model import ModelConfig, SemanticPolicy
from .run_bc import DATASET_ROOT, PROTOTYPES
from .training.dataset import CanonicalDecisionDataset
from .training.trainer import _train_epoch


class _BatchPolicyDataset:
    """Select one batching policy without changing canonical dataset contents."""

    def __init__(self, dataset: CanonicalDecisionDataset, *, length_bucketed: bool):
        self.dataset = dataset
        self.length_bucketed = length_bucketed
        self.manifest = dataset.manifest
        self.manifest_sha256 = dataset.manifest_sha256
        self.split_counts = dataset.split_counts

    def batch_count(self, split: str, batch_size: int) -> int:
        return self.dataset.batch_count(split, batch_size)

    def iter_batches(
        self, split: str, batch_size: int, *, seed: int
    ) -> Any:
        return self.dataset.iter_batches(
            split,
            batch_size,
            seed=seed,
            length_bucketed=self.length_bucketed,
        )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def benchmark(
    *,
    dataset_root: Path,
    batches: int,
    batch_size: int,
    seed: int,
    prefetch_depth: int,
    length_bucketed: bool,
    shared_prototypes: bool,
    model_config: ModelConfig | None = None,
    fixed_bucket_padding: bool = False,
    fused_optimizer: bool = False,
) -> dict[str, Any]:
    if batches < 1 or batch_size < 1 or prefetch_depth < 0:
        raise ValueError("batches/batch size must be positive and prefetch nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("0031 training benchmark requires CUDA")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = CanonicalDecisionDataset(
        dataset_root,
        bucket_padding=BucketPadding() if fixed_bucket_padding else None,
    )
    selected = _BatchPolicyDataset(dataset, length_bucketed=length_bucketed)
    prototypes = PrototypeIndex.load(PROTOTYPES)
    config = model_config or ModelConfig()
    model = SemanticPolicy(
        config,
        prototypes,
        share_prototype_embeddings=shared_prototypes,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.02,
        fused=fused_optimizer,
    )
    started = time.time()
    metrics, updates = _train_epoch(
        "throughput_gate",
        model,
        optimizer,
        selected,
        epoch=1,
        batch_size=batch_size,
        seed=seed,
        device=torch.device("cuda"),
        amp=True,
        grad_clip=1.0,
        maximum_batches=batches,
        prefetch_depth=prefetch_depth,
    )
    consumer_active_fraction = 1.0 - float(metrics["data_wait_fraction"])
    return {
        "schema_version": "0031_training_throughput_benchmark_v1",
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "seed": seed,
        "batch_size": batch_size,
        "requested_batches": batches,
        "completed_batches": updates,
        "prefetch_depth": prefetch_depth,
        "length_bucketed": length_bucketed,
        "shared_prototypes": shared_prototypes,
        "fixed_bucket_padding": fixed_bucket_padding,
        "fused_optimizer": fused_optimizer,
        "model_config": config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "amp_dtype": "bfloat16",
        "cuda_device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "started_at": started,
        "completed_at": time.time(),
        "metrics": metrics,
        "consumer_active_fraction": consumer_active_fraction,
        "acceptance": {
            "minimum_consumer_active_fraction": 0.85,
            "maximum_data_wait_fraction": 0.15,
            "cpu_pipeline_gate_passed": consumer_active_fraction >= 0.85,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--no-length-bucketing", action="store_true")
    parser.add_argument("--no-shared-prototypes", action="store_true")
    parser.add_argument("--fixed-bucket-padding", action="store_true")
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--state-layers", type=int, default=4)
    parser.add_argument("--option-layers", type=int, default=2)
    parser.add_argument("--ffn-multiplier", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--state-architecture", choices=("hierarchical", "joint"), default="hierarchical"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        dataset_root=args.dataset_root,
        batches=args.batches,
        batch_size=args.batch_size,
        seed=args.seed,
        prefetch_depth=args.prefetch_depth,
        length_bucketed=not args.no_length_bucketing,
        shared_prototypes=not args.no_shared_prototypes,
        model_config=ModelConfig(
            d_model=args.d_model,
            heads=args.heads,
            state_layers=args.state_layers,
            option_layers=args.option_layers,
            ffn_multiplier=args.ffn_multiplier,
            dropout=args.dropout,
            state_architecture=args.state_architecture,
        ),
        fixed_bucket_padding=args.fixed_bucket_padding,
        fused_optimizer=args.fused_optimizer,
    )
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["benchmark"]
