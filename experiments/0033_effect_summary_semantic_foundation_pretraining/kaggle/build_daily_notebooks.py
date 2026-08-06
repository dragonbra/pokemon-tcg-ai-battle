"""Build the four independent Kaggle data notebooks with nbformat."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
CONFIGS = (
    ("01_day_0728", "01_day_0728.ipynb", ["2026-07-28"]),
    ("02_day_0729", "02_day_0729.ipynb", ["2026-07-29"]),
    ("03_day_0730", "03_day_0730.ipynb", ["2026-07-30"]),
    ("04_days_0731_0801", "04_days_0731_0801.ipynb", ["2026-07-31", "2026-08-01"]),
)


SETUP = dedent(
    """
    from __future__ import annotations

    from collections import Counter
    import gzip
    import hashlib
    import importlib
    import json
    import os
    from pathlib import Path
    import shutil
    import sys

    import torch

    INPUT = Path(os.environ.get("PTCG_KAGGLE_INPUT", "/kaggle/input"))
    WORKING = Path(os.environ.get("PTCG_KAGGLE_WORKING", "/kaggle/working"))
    OUTPUT_ROOT = WORKING / f"ptcg_0032_{PART_NAME}"
    WORKERS = int(
        os.environ.get("PTCG_WORKERS", str(max(1, min(4, os.cpu_count() or 1))))
    )
    if WORKERS < 1:
        raise ValueError("PTCG_WORKERS must be positive")
    RECORDS_PER_SHARD = 4096
    VALIDATION_BATCH_SIZE = 256
    EXPECTED_PARAMETER_COUNT = 20_567_042
    GIB = 1024 ** 3
    KAGGLE_DATASET_LIMIT_BYTES = 18 * GIB
    KAGGLE_FREE_FLOOR_BYTES = 1 * GIB

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)


    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )


    def find_source_root() -> Path:
        package_name = "0033_effect_summary_semantic_foundation_pretraining"
        package_candidates = []
        flat_candidates = []
        for public_asset in INPUT.rglob("official_public_prototypes_v1.json"):
            root = public_asset.parent.parent
            if (
                (root / "assets" / "official_full_engine_prototypes_v2.json").is_file()
                and (root / "features" / "audit.py").is_file()
            ):
                if root.name == package_name:
                    package_candidates.append(root)
                else:
                    flat_candidates.append(root)
        package_unique = sorted(set(package_candidates))
        if len(package_unique) == 1:
            return package_unique[0]
        flat_unique = sorted(set(flat_candidates))
        if len(flat_unique) == 1:
            staged = WORKING / package_name
            if staged.exists():
                shutil.rmtree(staged)
            shutil.copytree(flat_unique[0], staged)
            return staged
        raise FileNotFoundError(
            "expected one complete 0032 source root or one flat source dataset, "
            f"found package={package_unique}, flat={flat_unique}"
        )


    def episode_files(day: str) -> list[Path]:
        files = sorted(
            path
            for path in INPUT.rglob("*.json")
            if path.stem.isdigit() and day in str(path)
        )
        if not files:
            raise FileNotFoundError(f"official Episode dataset is empty for {day}")
        return files


    SOURCE_ROOT = find_source_root()
    sys.path.insert(0, str(SOURCE_ROOT.parent))
    PACKAGE = SOURCE_ROOT.name

    winner_catalog = importlib.import_module(f"{PACKAGE}.data.winner_catalog")
    raw_decisions = importlib.import_module(f"{PACKAGE}.data.raw_decisions")
    materialize = importlib.import_module(f"{PACKAGE}.data.materialize")
    fields = importlib.import_module(f"{PACKAGE}.contracts.fields")
    batch_contract = importlib.import_module(f"{PACKAGE}.contracts.batch")
    collate_module = importlib.import_module(f"{PACKAGE}.features.collate")
    prototypes_module = importlib.import_module(f"{PACKAGE}.domain.prototypes")
    model_module = importlib.import_module(f"{PACKAGE}.model")

    SCHEMA_VERSION = fields.SCHEMA_VERSION
    EXPECTED_WIDTHS = {
        "global_cat": 7,
        "global_num": 11,
        "card_cat": 6,
        "card_num": 14,
        "resource_cat": 3,
        "resource_num": 10,
        "event_cat": 17,
        "event_num": 4,
        "option_cat": 11,
        "option_num": 2,
    }
    actual_widths = {
        name: getattr(fields.WIDTHS, name) for name in EXPECTED_WIDTHS
    }
    if SCHEMA_VERSION != "0033_effect_summary_semantic_decision_v1":
        raise ValueError(f"unexpected actor schema: {SCHEMA_VERSION}")
    if actual_widths != EXPECTED_WIDTHS:
        raise ValueError(f"0032 field width drift: {actual_widths}")

    PROTOTYPE_PATH = SOURCE_ROOT / "assets" / "official_public_prototypes_v1.json"
    prototypes = prototypes_module.PrototypeIndex.load(PROTOTYPE_PATH)
    prototype_counts = {
        "cards": len(prototypes.engine_cards),
        "attacks": len(prototypes.engine_attacks),
        "skills": len(prototypes.skills),
    }
    if prototype_counts != {"cards": 1267, "attacks": 1556, "skills": 433}:
        raise ValueError(f"incomplete full-engine prototype asset: {prototype_counts}")

    print(
        json.dumps(
            {
                "part": PART_NAME,
                "dates": TARGET_DATES,
                "schema": SCHEMA_VERSION,
                "widths": actual_widths,
                "prototype_counts": prototype_counts,
                "workers": WORKERS,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    """
)


VALIDATION = dedent(
    """
    def validate_canonical_partition(
        day: str,
        canonical_root: Path,
        manifest: dict,
    ) -> dict:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("canonical manifest schema mismatch")
        if manifest.get("status") != "complete":
            raise ValueError("canonical materialization is not complete")
        if manifest.get("decisions", 0) <= 0:
            raise ValueError("canonical partition has no decisions")
        if any(manifest["split_counts"].get(split, 0) <= 0 for split in ("train", "validation")):
            raise ValueError(f"canonical partition has an empty split: {manifest['split_counts']}")

        feature_audit = manifest.get("feature_input_audit", {})
        if feature_audit.get("status") != "passed":
            raise ValueError("pre-write feature input audit did not pass")
        if feature_audit.get("audited_decisions") != manifest["decisions"]:
            raise ValueError("not every canonical decision passed the source-observation audit")
        for fact in ("in_play_pokemon", "attached_children", "attach_options"):
            if feature_audit.get(fact, 0) <= 0:
                raise ValueError(f"required feature audit fact is empty: {fact}")

        observed = Counter()
        relation_counts = Counter()
        validation_batch = []
        forward_records = []

        def validate_batch() -> None:
            if not validation_batch:
                return
            tensors = collate_module.collate_canonical_records(validation_batch)
            batch_contract.DecisionBatch.from_mapping(tensors)
            validation_batch.clear()

        for split in ("train", "validation"):
            for shard in manifest["shards"][split]:
                path = canonical_root / shard["path"]
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"missing or empty canonical shard: {path}")
                if path.stat().st_size != shard["bytes"]:
                    raise ValueError(f"canonical shard byte count mismatch: {path}")
                if sha256_file(path) != shard["sha256"]:
                    raise ValueError(f"canonical shard SHA256 mismatch: {path}")
                shard_records = 0
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        if record.get("schema_version") != SCHEMA_VERSION:
                            raise ValueError("canonical record schema mismatch")
                        if record["audit"]["identity"].get("date") != day:
                            raise ValueError("cross-date record found in daily partition")
                        if record["audit"].get("split") != split:
                            raise ValueError("canonical record split disagrees with its shard")
                        actor = record["actor"]
                        if set(actor) != fields.ACTOR_KEYS:
                            raise ValueError("canonical actor key set drift")
                        for name in (
                            "card_parent",
                            "event_source",
                            "event_target",
                            "option_source",
                            "option_target",
                        ):
                            relation_counts[name] += sum(value > 0 for value in actor[name])
                        validation_batch.append(record)
                        if len(forward_records) < 2:
                            forward_records.append(record)
                        if len(validation_batch) >= VALIDATION_BATCH_SIZE:
                            validate_batch()
                        observed[split] += 1
                        shard_records += 1
                if shard_records != shard["count"]:
                    raise ValueError(f"canonical shard record count mismatch: {path}")
        validate_batch()

        if dict(observed) != manifest["split_counts"]:
            raise ValueError(
                f"observed split counts disagree with manifest: {dict(observed)}"
            )
        if observed.total() != manifest["decisions"]:
            raise ValueError("observed decision count disagrees with manifest")

        forward_batch = batch_contract.DecisionBatch.from_mapping(
            collate_module.collate_canonical_records(forward_records)
        )
        model = model_module.SemanticPolicy(model_module.ModelConfig(), prototypes).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ValueError(f"0032 model parameter drift: {parameter_count}")
        with torch.inference_mode():
            logits = model.teacher_logits(forward_batch)
        if not torch.isfinite(logits).all():
            raise ValueError("0032 model produced non-finite logits on canonical records")

        return {
            "status": "passed",
            "validated_records": observed.total(),
            "split_counts": dict(observed),
            "relation_counts": dict(sorted(relation_counts.items())),
            "model_forward": {
                "rows": len(forward_records),
                "logits_shape": list(logits.shape),
                "finite": True,
                "parameter_count": parameter_count,
            },
        }
    """
)


PROCESS = dedent(
    """
    def process_day(day: str) -> dict:
        day_root = OUTPUT_ROOT / f"date={day}"
        day_root.mkdir(parents=True)
        files = episode_files(day)

        catalog = winner_catalog.scan_archives([], files, workers=WORKERS)
        if catalog["contracts"]["selection"] != "unique_positive_terminal_winner_perspective":
            raise ValueError("winner-only catalog contract drift")
        if not catalog["episodes"]:
            raise ValueError(f"winner-only catalog is empty for {day}")
        wrong_dates = sorted(
            {row["episode_date"] for row in catalog["episodes"] if row["episode_date"] != day}
        )
        if wrong_dates:
            raise ValueError(f"catalog contains cross-date Episodes: {wrong_dates}")

        catalog_path = day_root / "winner_catalog.json"
        winner_catalog.write_catalog(catalog, catalog_path)
        raw_root = day_root / "raw"
        raw_reference = raw_decisions.build_raw_dataset(
            catalog_path,
            raw_root,
            shard_size=100_000,
            workers=WORKERS,
            start_date=day,
            end_date=day,
            dataset_limit_bytes=KAGGLE_DATASET_LIMIT_BYTES,
            linux_free_floor_bytes=KAGGLE_FREE_FLOOR_BYTES,
            c_free_floor_bytes=0,
        )
        canonical_root = day_root / "canonical"
        canonical_manifest = materialize.materialize_canonical(
            raw_root,
            canonical_root,
            PROTOTYPE_PATH,
            catalog_path,
            records_per_shard=RECORDS_PER_SHARD,
            workers=WORKERS,
            require_complete_catalog=True,
            dataset_limit_bytes=KAGGLE_DATASET_LIMIT_BYTES,
            linux_free_floor_bytes=KAGGLE_FREE_FLOOR_BYTES,
            c_free_floor_bytes=0,
        )
        acceptance = validate_canonical_partition(day, canonical_root, canonical_manifest)

        raw_decision_count = sum(
            row["decisions"] for row in raw_reference["decision_counts"]
        )
        if raw_decision_count != canonical_manifest["decisions"]:
            raise ValueError("raw and canonical decision counts disagree")

        daily_manifest = {
            "schema_version": "0032_daily_semantic_partition_v1",
            "actor_schema": SCHEMA_VERSION,
            "status": "complete",
            "experiment": "0033_effect_summary_semantic_foundation_pretraining",
            "part": PART_NAME,
            "date": day,
            "winner_only": True,
            "source_dataset": f"kaggle/pokemon-tcg-ai-battle-episodes-{day}",
            "source_files": len(files),
            "winning_episodes": catalog["totals"]["unique_winning_episodes"],
            "records": canonical_manifest["decisions"],
            "split_counts": canonical_manifest["split_counts"],
            "episode_split_counts": canonical_manifest["episode_split_counts"],
            "shards": canonical_manifest["shards"],
            "canonical_path": "canonical",
            "canonical_manifest_sha256": sha256_file(canonical_root / "manifest.json"),
            "winner_catalog_sha256": catalog["catalog_sha256"],
            "feature_input_audit": canonical_manifest["feature_input_audit"],
            "acceptance": acceptance,
        }
        write_json(day_root / "manifest.json", daily_manifest)
        shutil.rmtree(raw_root)
        print(
            json.dumps(
                {
                    "date": day,
                    "records": daily_manifest["records"],
                    "split_counts": daily_manifest["split_counts"],
                    "feature_input_audit": daily_manifest["feature_input_audit"],
                    "model_forward": acceptance["model_forward"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return daily_manifest


    daily_manifests = [process_day(day) for day in TARGET_DATES]
    partition_manifest = {
        "schema_version": "0032_daily_partition_group_v1",
        "status": "complete",
        "experiment": "0033_effect_summary_semantic_foundation_pretraining",
        "part": PART_NAME,
        "dates": TARGET_DATES,
        "winner_only": True,
        "actor_schema": SCHEMA_VERSION,
        "total_records": sum(row["records"] for row in daily_manifests),
        "day_manifests": [
            {
                "date": row["date"],
                "records": row["records"],
                "split_counts": row["split_counts"],
                "manifest_path": f"date={row['date']}/manifest.json",
                "manifest_sha256": sha256_file(
                    OUTPUT_ROOT / f"date={row['date']}" / "manifest.json"
                ),
            }
            for row in daily_manifests
        ],
    }
    write_json(OUTPUT_ROOT / "partition_manifest.json", partition_manifest)
    print(json.dumps(partition_manifest, indent=2, sort_keys=True), flush=True)
    """
)


def build_notebook(part_name: str, dates: list[str]) -> nbf.NotebookNode:
    header = (
        f"# PTCG 0032 daily canonical data: {part_name}\n\n"
        "Produces independent winner-perspective canonical shards. Every decision is audited "
        "against its source observation before writing, then reloaded through DecisionBatch and "
        "the exact 0032 model input path."
    )
    parameters = f"PART_NAME = {part_name!r}\nTARGET_DATES = {dates!r}"
    notebook = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_markdown_cell(header),
            nbf.v4.new_code_cell(parameters),
            nbf.v4.new_markdown_cell("## Environment and immutable contracts"),
            nbf.v4.new_code_cell(SETUP),
            nbf.v4.new_markdown_cell("## Full-record acceptance checks"),
            nbf.v4.new_code_cell(VALIDATION),
            nbf.v4.new_markdown_cell("## Winner-only materialization"),
            nbf.v4.new_code_cell(PROCESS),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.validate(notebook)
    return notebook


def main() -> None:
    for directory, filename, dates in CONFIGS:
        path = ROOT / directory / filename
        notebook = build_notebook(directory, dates)
        nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
