"""Immutable League version initialization and audit."""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any, Mapping

import torch

from rl_environment.runs import (
    VersionPaths,
    initialize_version as allocate_version,
    project_version_paths,
    write_version_status,
)

from . import PROJECT_ID
from .decoder import DECODER_COMPONENTS, create_value_head, load_decoder_checkpoint, save_decoder_checkpoint
from .decks import DeckPlugin, DeckRole, load_deck_plugins, write_catalog_snapshot
from .foundation import REPOSITORY_ROOT, load_foundation, verify_foundation


DEFAULT_DECK_ROOT = Path(__file__).with_name("deck")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deck_initialization_seed(version_name: str, plugin: DeckPlugin) -> int:
    identity = f"{version_name}:{plugin.deck_id}:{plugin.deck_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**31)


def _validate_catalog_for_initialization(plugins: tuple[DeckPlugin, ...]) -> None:
    if not plugins:
        raise ValueError("League initialization requires at least one deck plugin")
    if not any(plugin.role is DeckRole.LIVE and plugin.focal for plugin in plugins):
        raise ValueError("League initialization requires at least one focal Live deck")


def _verify_external_decoder(plugin: DeckPlugin, foundation_sha: str) -> None:
    if plugin.decoder_ref == "foundation":
        return
    path = REPOSITORY_ROOT / plugin.decoder_ref
    if not path.is_file():
        raise FileNotFoundError(f"referenced decoder checkpoint does not exist: {path}")
    if _sha256(path) != plugin.decoder_sha256:
        raise ValueError(f"referenced decoder SHA mismatch for {plugin.deck_id}")
    load_decoder_checkpoint(
        path,
        expected_foundation_sha256=foundation_sha,
        expected_deck_id=plugin.deck_id,
        expected_deck_sha256=plugin.deck_sha256,
    )


def initialize_league_version(
    version_name: str,
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    initial_checkpoints: Mapping[str, Path | None] | None = None,
) -> VersionPaths:
    plugins = load_deck_plugins(deck_root)
    _validate_catalog_for_initialization(plugins)
    identity = verify_foundation()
    for plugin in plugins:
        _verify_external_decoder(plugin, identity.weights_sha256)
    if initial_checkpoints is not None:
        expected = {plugin.deck_id for plugin in plugins if plugin.role is DeckRole.LIVE}
        if set(initial_checkpoints) != expected:
            raise ValueError(
                "initial checkpoint map must cover every Live deck exactly; "
                f"missing={sorted(expected - set(initial_checkpoints))}, "
                f"unknown={sorted(set(initial_checkpoints) - expected)}"
            )

    paths = allocate_version(PROJECT_ID, version_name)
    try:
        catalog_path = paths.artifact / "league_catalog.json"
        catalog_sha = write_catalog_snapshot(catalog_path, plugins)
        frozen_anchors = {
            plugin.deck_id: {
                "decoder_ref": "foundation",
                "deck_sha256": plugin.deck_sha256,
                "materialized": False,
            }
            for plugin in plugins
            if plugin.frozen_anchor
        }
        checkpoints: dict[str, dict[str, object]] = {}
        deck_checkpoint_root = paths.checkpoints / "decks"
        # Construct the immutable Foundation exactly once. Initializing deck-local
        # heads must not repeatedly instantiate the full Transformer.
        foundation_model, _ = load_foundation("cpu")
        foundation_decoder_state = {
            component: copy.deepcopy(getattr(foundation_model, component).state_dict())
            for component in DECODER_COMPONENTS
        }
        for plugin in plugins:
            if plugin.role is DeckRole.FROZEN:
                checkpoints[plugin.deck_id] = {
                    "decoder_ref": plugin.decoder_ref,
                    "decoder_sha256": plugin.decoder_sha256,
                    "materialized": False,
                    "role": plugin.role.value,
                }
                continue
            model = foundation_model
            for component, state in foundation_decoder_state.items():
                getattr(model, component).load_state_dict(state)
            initialization_seed = _deck_initialization_seed(version_name, plugin)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed)
                value_head = create_value_head(model.config.d_model)
            source_checkpoint = (
                initial_checkpoints[plugin.deck_id]
                if initial_checkpoints is not None
                else (
                    REPOSITORY_ROOT / plugin.decoder_ref
                    if plugin.decoder_ref != "foundation"
                    else None
                )
            )
            if source_checkpoint is not None:
                load_decoder_checkpoint(
                    source_checkpoint,
                    expected_foundation_sha256=identity.weights_sha256,
                    expected_deck_id=plugin.deck_id,
                    expected_deck_sha256=plugin.deck_sha256,
                    model=model,
                    value_head=value_head,
                )
            checkpoint_path = deck_checkpoint_root / f"{plugin.deck_id}.pt"
            digest = save_decoder_checkpoint(
                checkpoint_path,
                model,
                value_head,
                foundation_sha256=identity.weights_sha256,
                deck_id=plugin.deck_id,
                deck_sha256=plugin.deck_sha256,
                policy_role="live",
                policy_version=version_name,
                update=0,
            )
            checkpoints[plugin.deck_id] = {
                "decoder_ref": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
                "decoder_sha256": digest,
                "initialization_seed": initialization_seed,
                "initialization_source": (
                    str(source_checkpoint.relative_to(REPOSITORY_ROOT))
                    if source_checkpoint is not None
                    else "foundation"
                ),
                "materialized": True,
                "role": plugin.role.value,
            }
        config = {
            "schema_version": "0023_league_initialization_v1",
            "project_id": PROJECT_ID,
            "version": version_name,
            "foundation": identity.as_dict(),
            "catalog": {
                "path": str(catalog_path.relative_to(REPOSITORY_ROOT)),
                "sha256": catalog_sha,
                "deck_count": len(plugins),
                "frozen_count": sum(p.frozen_anchor for p in plugins),
                "live_count": sum(p.role is DeckRole.LIVE for p in plugins),
                "focal_deck_ids": [p.deck_id for p in plugins if p.focal],
            },
            "frozen_anchors": frozen_anchors,
            "checkpoints": checkpoints,
            "training": {
                "encoder_frozen": True,
                "deployment_source_id": 0,
                "reward": {"win": 1.0, "loss": -1.0, "draw": 0.0, "gamma": 1.0},
                "wandb": {
                    "mode": "disabled_until_training_starts",
                    "reason": "version initialization performs no optimization",
                },
            },
        }
        _atomic_json(paths.config, config)
        write_version_status(
            paths,
            {
                "state": "initialized",
                "foundation_verified": True,
                "catalog_sha256": catalog_sha,
                "official_engine_strength_evidence": False,
                "training_started": False,
                "wandb_sync": "not_applicable_initialization_only",
            },
        )
        return paths
    except Exception:
        write_version_status(paths, {"state": "initialization_failed"})
        raise


def audit_league_version(version_name: str) -> dict[str, object]:
    paths = project_version_paths(PROJECT_ID, version_name)
    if not paths.config.is_file():
        raise FileNotFoundError(f"League version config does not exist: {paths.config}")
    config = json.loads(paths.config.read_text(encoding="utf-8"))
    identity = verify_foundation()
    if config.get("foundation") != identity.as_dict():
        raise ValueError("League version Foundation identity mismatch")
    catalog = config.get("catalog") or {}
    catalog_path = REPOSITORY_ROOT / catalog.get("path", "")
    if _sha256(catalog_path) != catalog.get("sha256"):
        raise ValueError("League catalog snapshot SHA mismatch")
    raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    deck_by_id = {deck["deck_id"]: deck for deck in raw_catalog.get("decks", [])}
    frozen_audits: dict[str, object] = {}
    for deck_id, anchor in (config.get("frozen_anchors") or {}).items():
        deck = deck_by_id.get(deck_id)
        if deck is None:
            raise ValueError(f"Frozen anchor deck is absent from catalog: {deck_id}")
        if anchor != {
            "decoder_ref": "foundation",
            "deck_sha256": deck["deck_sha256"],
            "materialized": False,
        }:
            raise ValueError(f"invalid Foundation Frozen anchor for {deck_id}")
        frozen_audits[deck_id] = {"sentinel": "foundation"}
    audits: dict[str, object] = {}
    for deck_id, checkpoint in (config.get("checkpoints") or {}).items():
        deck = deck_by_id.get(deck_id)
        if deck is None:
            raise ValueError(f"checkpoint deck is absent from catalog: {deck_id}")
        if not checkpoint.get("materialized"):
            if checkpoint.get("decoder_ref") == "foundation":
                audits[deck_id] = {"role": checkpoint.get("role"), "sentinel": "foundation"}
                continue
            _verify_external_decoder(
                DeckPlugin(
                    deck_id=deck_id,
                    display_name=deck["display_name"],
                    role=DeckRole(deck["role"]),
                    frozen_anchor=bool(deck["frozen_anchor"]),
                    focal=deck["focal"],
                    decoder_ref=checkpoint["decoder_ref"],
                    decoder_sha256=checkpoint["decoder_sha256"],
                    deck=tuple(deck["deck"]),
                    deck_sha256=deck["deck_sha256"],
                    plugin_sha256=deck["plugin_sha256"],
                    provenance=deck["provenance"],
                    root=Path("."),
                ),
                identity.weights_sha256,
            )
            audits[deck_id] = {"role": checkpoint.get("role"), "external": True}
            continue
        checkpoint_path = REPOSITORY_ROOT / checkpoint["decoder_ref"]
        audit = load_decoder_checkpoint(
            checkpoint_path,
            expected_foundation_sha256=identity.weights_sha256,
            expected_deck_id=deck_id,
            expected_deck_sha256=deck["deck_sha256"],
        )
        if audit.checkpoint_sha256 != checkpoint.get("decoder_sha256"):
            raise ValueError(f"version checkpoint SHA mismatch for {deck_id}")
        audits[deck_id] = {
            "role": checkpoint.get("role"),
            "checkpoint_sha256": audit.checkpoint_sha256,
            "tensor_count": audit.tensor_count,
            "update": audit.update,
        }
    return {
        "project_id": PROJECT_ID,
        "version": version_name,
        "foundation": identity.as_dict(),
        "catalog_sha256": catalog["sha256"],
        "frozen_anchors": frozen_audits,
        "decks": audits,
        "valid": True,
    }


__all__ = ["audit_league_version", "initialize_league_version"]
