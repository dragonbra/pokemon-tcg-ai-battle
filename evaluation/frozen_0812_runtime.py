"""Runtime routing catalog for the immutable Frozen-0812 pool."""

from __future__ import annotations

from pathlib import Path

from evaluation.cards import load_card_catalog
from evaluation.frozen_0806 import POLICY_0806_SHA256
from evaluation.frozen_0806_runtime import (
    Frozen0806RuntimeCatalog,
    _policy_identity,
    _routed_package,
)
from evaluation.frozen_0812 import load_frozen_0812_pool
from evaluation.packages.loader import PackageValidationError, SubmissionPackage


ROOT = Path(__file__).resolve().parents[1]


def _portable_tree_sha256(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    paths = (item for item in root.rglob("*") if item.is_file())
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest.hexdigest()


def load_frozen_0812_runtime_catalog() -> Frozen0806RuntimeCatalog:
    pool = load_frozen_0812_pool()
    policies_root = pool.root / "policies"
    runtime_manifest = pool.manifest.get("policy_runtimes")
    main = (runtime_manifest or {}).get("main")
    if (
        not isinstance(main, dict)
        or main.get("path") != "policies/policy_0806"
        or main.get("tree_sha256") != _portable_tree_sha256(policies_root / "policy_0806")
    ):
        raise PackageValidationError("Frozen-0812 shared CUDA runtime mismatch")
    official_cards = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    policy_root = policies_root / "policy_0806"
    package_manifest = __import__("json").loads(
        (policy_root / "manifest.json").read_text(encoding="utf-8")
    )
    policy = SubmissionPackage(
        name="policy_0806_metadata_only",
        root=policy_root,
        deck=list(pool.decks[0].cards),
        entrypoint=policy_root / "main.py",
        package_hash=main["tree_sha256"],
        deck_hash=pool.decks[0].exact_deck_sha256,
        cg_manifest={},
        package_manifest=package_manifest,
    )
    number_by_id = {
        deck.deck_id: int(deck.root.name.split("_", 1)[0]) for deck in pool.decks
    }
    candidates = tuple(
        _routed_package(
            policy, pool, entry.deck_id, number_by_id[entry.deck_id],
            "candidate", "Policy-0809-CPU-checkpoint-2.0", official_cards,
        )
        for entry in pool.schedule
    )
    return Frozen0806RuntimeCatalog(
        pool=pool,
        candidate_policy=policy,
        opponent_policy=policy,
        candidates=candidates,
        opponents=candidates,
    )


if __name__ == "__main__":
    catalog = load_frozen_0812_runtime_catalog()
    print(catalog.pool.pool_id, len(catalog.candidates), catalog.pool.total_games)
