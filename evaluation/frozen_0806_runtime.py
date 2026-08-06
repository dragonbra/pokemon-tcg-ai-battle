"""Executable two-policy runtime catalog for the Frozen-0806 deck pool."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from evaluation.cards import card_image_url, load_card_catalog
from evaluation.frozen_0806 import (
    POLICY_0019_SHA256,
    POLICY_0806_SHA256,
    Frozen0806Pool,
    load_frozen_0806_pool,
)
from evaluation.packages.loader import (
    PackageValidationError,
    SubmissionPackage,
    load_submission_package,
)
from evaluation.runtime.loader import assert_cg_compatible


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
DEFAULT_CONFIG = EVALUATION_ROOT / "configs" / "frozen_0806.json"
DEFAULT_POOL_ROOT = (
    EVALUATION_ROOT / "arena" / "frozen_pools" / "0806_kaggle_top100_plus_v1"
)
POLICY_0019_SOURCE = (
    ROOT / "archive" / "evaluation" / "pre_0806_0019"
    / "evaluation_arena" / "frozen" / "_policy"
)
POLICY_0806_CHECKPOINT = (
    ROOT / "archive" / "pretrained" / "0031_friend_0806_epoch11_best_validation_loss"
    / "model.pt"
)


@dataclass(frozen=True)
class Frozen0806RuntimeCatalog:
    pool: Frozen0806Pool
    candidate_policy: SubmissionPackage
    opponent_policy: SubmissionPackage
    candidates: tuple[SubmissionPackage, ...]
    opponents: tuple[SubmissionPackage, ...]

    def candidate(self, deck_id: str) -> SubmissionPackage:
        return _one(self.candidates, deck_id, "candidate")

    def opponent(self, deck_id: str) -> SubmissionPackage:
        return _one(self.opponents, deck_id, "opponent")


def _one(
    packages: tuple[SubmissionPackage, ...], deck_id: str, role: str
) -> SubmissionPackage:
    matches = [package for package in packages if package.name == deck_id]
    if len(matches) != 1:
        raise PackageValidationError(f"unknown Frozen-0806 {role} deck: {deck_id}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def materialize_policy_runtimes(
    *, pool_root: Path = DEFAULT_POOL_ROOT, policy_0019_source: Path = POLICY_0019_SOURCE
) -> None:
    policies = pool_root / "policies"
    policy_0019 = policies / "policy_0019"
    policy_0806 = policies / "policy_0806"
    if policies.exists():
        raise FileExistsError(f"Frozen-0806 policy runtimes already exist: {policies}")
    if not policy_0019_source.is_dir():
        raise FileNotFoundError(policy_0019_source)
    first_deck = next(iter(sorted((pool_root / "decks").glob("*/deck.csv"))), None)
    if first_deck is None:
        raise FileNotFoundError("Frozen-0806 has no exact deck identities")
    try:
        shutil.copytree(
            policy_0019_source,
            policy_0019,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        exporter = importlib.import_module(
            "train.0031_rule_faithful_semantic_foundation_pretraining.export_candidate"
        )
        exporter.export_candidate(
            checkpoint=POLICY_0806_CHECKPOINT,
            deck_path=first_deck,
            cg_source=policy_0019 / "cg",
            output=policy_0806,
            deck_id="frozen_0806_shared_runtime",
        )
        manifest_path = pool_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["policy_runtimes"] = {
            "opponent": {
                "path": "policies/policy_0019",
                "tree_sha256": _tree_sha256(policy_0019),
            },
            "main": {
                "path": "policies/policy_0806",
                "tree_sha256": _tree_sha256(policy_0806),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(policies, ignore_errors=True)
        raise


def _policy_identity(
    package: SubmissionPackage, *, role: str, expected_sha256: str
) -> None:
    manifest = package.package_manifest or {}
    actual = (
        manifest.get("foundation_sha256")
        if role == "opponent"
        else manifest.get("checkpoint_sha256")
    )
    if actual != expected_sha256:
        raise PackageValidationError(f"Frozen-0806 {role} runtime policy SHA mismatch")
    if role == "opponent" and manifest.get("update") != 0:
        raise PackageValidationError("Frozen-0806 opponent runtime is not Policy-0019 update 0")


def _routed_package(
    policy: SubmissionPackage,
    pool: Frozen0806Pool,
    deck_id: str,
    role: str,
    card_catalog: dict[int, dict],
) -> SubmissionPackage:
    deck = next(item for item in pool.decks if item.deck_id == deck_id)
    schedule = next(item for item in pool.schedule if item.deck_id == deck_id)
    representative_cards = []
    for card_id in deck.manifest["representative_card_ids"]:
        metadata = card_catalog[card_id]
        representative_cards.append(
            {
                "card_id": card_id,
                "name": metadata["name"],
                "image_url": card_image_url(
                    metadata["expansion"], metadata["collection_number"]
                ),
            }
        )
    identity_hash = hashlib.sha256(
        f"{policy.package_hash}:{role}:{deck.exact_deck_sha256}".encode("ascii")
    ).hexdigest()
    package_manifest = dict(policy.package_manifest or {})
    package_manifest.update(
        {
            "frozen_pool_id": pool.pool_id,
            "frozen_role": role,
            "deck_id": deck_id,
            "exact_deck_sha256": deck.exact_deck_sha256,
            "schedule_games": schedule.games,
            "schedule_segment": schedule.segment,
        }
    )
    return replace(
        policy,
        name=deck_id,
        deck=list(deck.cards),
        package_hash=identity_hash,
        deck_hash=_sha256(deck.root / "deck.csv"),
        display_name=schedule.archetype,
        representative_cards=tuple(representative_cards),
        package_manifest=package_manifest,
    )


def load_frozen_0806_runtime_catalog(
    config_path: Path = DEFAULT_CONFIG, evaluation_root: Path = EVALUATION_ROOT
) -> Frozen0806RuntimeCatalog:
    pool = load_frozen_0806_pool(config_path, evaluation_root)
    official_cards = load_card_catalog(
        evaluation_root.parent / "data" / "official" / "EN_Card_Data.csv"
    )
    official_ids = set(official_cards)
    policies_root = pool.root / "policies"
    runtime_manifest = pool.manifest.get("policy_runtimes")
    if not isinstance(runtime_manifest, dict) or set(runtime_manifest) != {"main", "opponent"}:
        raise PackageValidationError("Frozen-0806 policy runtime manifest is missing")
    for role, directory in (("main", "policy_0806"), ("opponent", "policy_0019")):
        record = runtime_manifest.get(role)
        if (
            not isinstance(record, dict)
            or record.get("path") != f"policies/{directory}"
            or record.get("tree_sha256") != _tree_sha256(policies_root / directory)
        ):
            raise PackageValidationError(f"Frozen-0806 {role} runtime tree mismatch")
    opponent_policy = load_submission_package(
        policies_root / "policy_0019", official_ids, name="policy_0019"
    )
    candidate_policy = load_submission_package(
        policies_root / "policy_0806", official_ids, name="policy_0806"
    )
    _policy_identity(
        opponent_policy, role="opponent", expected_sha256=POLICY_0019_SHA256
    )
    _policy_identity(candidate_policy, role="candidate", expected_sha256=POLICY_0806_SHA256)
    assert_cg_compatible(candidate_policy, opponent_policy)
    ordered_ids = tuple(entry.deck_id for entry in pool.schedule)
    candidates = tuple(
        _routed_package(candidate_policy, pool, deck_id, "candidate", official_cards)
        for deck_id in ordered_ids
    )
    opponents = tuple(
        _routed_package(opponent_policy, pool, deck_id, "opponent", official_cards)
        for deck_id in ordered_ids
    )
    return Frozen0806RuntimeCatalog(
        pool=pool,
        candidate_policy=candidate_policy,
        opponent_policy=opponent_policy,
        candidates=candidates,
        opponents=opponents,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    if args.materialize:
        materialize_policy_runtimes()
    catalog = load_frozen_0806_runtime_catalog()
    print(
        json.dumps(
            {
                "pool_id": catalog.pool.pool_id,
                "decks": len(catalog.candidates),
                "games": catalog.pool.total_games,
                "candidate_policy": POLICY_0806_SHA256,
                "opponent_policy": POLICY_0019_SHA256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Frozen0806RuntimeCatalog",
    "load_frozen_0806_runtime_catalog",
    "materialize_policy_runtimes",
]
