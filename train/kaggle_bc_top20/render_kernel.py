"""Render one uploadable private Kaggle Kernel directory from the Notebook template."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any


_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def render_kernel(
    *,
    template_root: Path,
    output: Path,
    owner: str,
    dataset_source: str | None,
    replay_dataset_sources: list[str] | None = None,
    kernel_sources: list[str] | None = None,
    job_order: int,
    input_archive: Path | None = None,
    kernel_slug: str | None = None,
    accelerator: str | None = None,
    retention: str = "dataset",
    mode: str = "full",
    corpus_mode: str = "exact_submission",
) -> dict[str, Any]:
    if job_order < 1:
        raise ValueError("job_order must be positive")
    if retention not in {"dataset", "package"}:
        raise ValueError(f"unsupported retention: {retention}")
    if mode not in {"full", "prepare", "train"}:
        raise ValueError(f"unsupported worker mode: {mode}")
    if corpus_mode not in {"exact_submission", "daily_team_winners"}:
        raise ValueError(f"unsupported corpus mode: {corpus_mode}")
    if (dataset_source is None) == (input_archive is None):
        raise ValueError("provide exactly one of dataset_source or input_archive")
    default_slug = {
        "full": f"pokemon-tcg-bc-job-{job_order:02d}",
        "prepare": f"pokemon-tcg-bc-prepare-{job_order:02d}",
        "train": f"pokemon-tcg-bc-train-{job_order:02d}",
    }[mode]
    slug = kernel_slug or default_slug
    if not _SLUG.fullmatch(owner) or not _SLUG.fullmatch(slug):
        raise ValueError(
            "owner and kernel slug must contain only lowercase letters, digits, hyphens"
        )
    all_dataset_sources = [
        *([dataset_source] if dataset_source is not None else []),
        *(replay_dataset_sources or []),
    ]
    if any("/" not in source for source in all_dataset_sources):
        raise ValueError("dataset sources must use owner/dataset-slug format")
    if len(set(all_dataset_sources)) != len(all_dataset_sources):
        raise ValueError("dataset sources must be unique")
    kernel_sources = list(kernel_sources or [])
    if any("/" not in source for source in kernel_sources):
        raise ValueError("kernel sources must use owner/kernel-slug format")
    if mode == "train" and len(kernel_sources) != 1:
        raise ValueError("train mode requires exactly one CPU prepare kernel source")
    if mode != "train" and kernel_sources:
        raise ValueError("kernel sources are only valid in train mode")
    if mode == "prepare" and not replay_dataset_sources:
        raise ValueError("prepare mode requires at least one daily Episode Dataset")
    if mode == "train" and replay_dataset_sources:
        raise ValueError("train mode must not attach daily Episode Datasets")
    effective_accelerator = accelerator or (
        "cpu" if mode == "prepare" else "NvidiaTeslaT4"
    )
    template_root = template_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Kernel directory: {output}")
    notebook_path = template_root / "ptcg_single_expert_bc.ipynb"
    metadata_path = template_root / "kernel-metadata.json"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    archive_base64 = ""
    archive_sha256 = ""
    if input_archive is not None:
        input_archive = input_archive.resolve()
        if not input_archive.is_file():
            raise FileNotFoundError(input_archive)
        archive_bytes = input_archive.read_bytes()
        archive_base64 = base64.b64encode(archive_bytes).decode("ascii")
        archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    replaced_order = False
    replaced_mode = False
    replaced_corpus_mode = False
    replaced_prebuilt_source = False
    replaced_retention = False
    replaced_archive = False
    replaced_archive_hash = False
    for cell in notebook.get("cells") or []:
        source = cell.get("source") or []
        if corpus_mode == "daily_team_winners" and cell.get("id") == "dependencies":
            cell["source"] = [
                "print(\"Daily winner mode uses preinstalled Kaggle packages; "
                "no network install is required.\")"
            ]
            source = cell["source"]
        for index, line in enumerate(source):
            if line.startswith("JOB_ORDER = "):
                source[index] = f"JOB_ORDER = {job_order}\n"
                replaced_order = True
            elif line.startswith("WORKER_MODE = "):
                source[index] = f'WORKER_MODE = "{mode}"\n'
                replaced_mode = True
            elif line.startswith("CORPUS_MODE = "):
                source[index] = f'CORPUS_MODE = "{corpus_mode}"\n'
                replaced_corpus_mode = True
            elif line.startswith("PREBUILT_KERNEL_SOURCE = "):
                value = kernel_sources[0] if kernel_sources else ""
                source[index] = f'PREBUILT_KERNEL_SOURCE = "{value}"\n'
                replaced_prebuilt_source = True
            elif line.startswith("RETENTION = "):
                suffix = "\n" if line.endswith("\n") else ""
                source[index] = f'RETENTION = "{retention}"{suffix}'
                replaced_retention = True
            elif line.startswith("EMBEDDED_INPUT_ARCHIVE_B64 = "):
                source[index] = f'EMBEDDED_INPUT_ARCHIVE_B64 = "{archive_base64}"\n'
                replaced_archive = True
            elif line.startswith("EMBEDDED_INPUT_ARCHIVE_SHA256 = "):
                source[index] = (
                    f'EMBEDDED_INPUT_ARCHIVE_SHA256 = "{archive_sha256}"\n'
                )
                replaced_archive_hash = True
    if not all(
        (
            replaced_order,
            replaced_mode,
            replaced_corpus_mode,
            replaced_prebuilt_source,
            replaced_retention,
            replaced_archive,
            replaced_archive_hash,
        )
    ):
        raise ValueError("Notebook template no longer contains the worker config cell")

    metadata.update(
        {
            "id": f"{owner}/{slug}",
            "title": f"Pokemon TCG BC {mode.title()} {job_order:02d}",
            "enable_gpu": mode != "prepare",
            "dataset_sources": all_dataset_sources,
            "kernel_sources": kernel_sources,
        }
    )
    if corpus_mode == "daily_team_winners":
        metadata["enable_internet"] = False
    if mode == "prepare":
        metadata.pop("machine_shape", None)
    else:
        metadata["machine_shape"] = effective_accelerator
    output.mkdir(parents=True)
    (output / notebook_path.name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    (output / metadata_path.name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "kernel": metadata["id"],
        "job_order": job_order,
        "accelerator": effective_accelerator,
        "mode": mode,
        "corpus_mode": corpus_mode,
        "retention": retention,
        "input_mode": "embedded_archive" if input_archive is not None else "private_dataset",
        "input_archive_sha256": archive_sha256 or None,
        "replay_dataset_sources": list(replay_dataset_sources or []),
        "kernel_sources": kernel_sources,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset-source")
    input_group.add_argument("--input-archive", type=Path)
    parser.add_argument(
        "--replay-dataset-source",
        action="append",
        default=[],
        help="public daily Episode Dataset source; may be repeated",
    )
    parser.add_argument(
        "--kernel-source",
        action="append",
        default=[],
        help="completed CPU prepare Kernel source; required once for train mode",
    )
    parser.add_argument("--job-order", type=int, required=True)
    parser.add_argument("--kernel-slug")
    parser.add_argument("--accelerator")
    parser.add_argument("--retention", choices=("dataset", "package"), default="dataset")
    parser.add_argument("--mode", choices=("full", "prepare", "train"), default="full")
    parser.add_argument(
        "--corpus-mode",
        choices=("exact_submission", "daily_team_winners"),
        default="exact_submission",
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    result = render_kernel(
        template_root=repository_root / "notebooks/kaggle_bc_worker",
        output=args.output,
        owner=args.owner,
        dataset_source=args.dataset_source,
        replay_dataset_sources=args.replay_dataset_source,
        kernel_sources=args.kernel_source,
        job_order=args.job_order,
        input_archive=args.input_archive,
        kernel_slug=args.kernel_slug,
        accelerator=args.accelerator,
        retention=args.retention,
        mode=args.mode,
        corpus_mode=args.corpus_mode,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
