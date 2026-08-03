"""Verified batched GPU service for the legacy 0019 Frozen Arena policy."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_ROOT = REPOSITORY_ROOT / "evaluation" / "arena" / "frozen" / "_policy"
PROVENANCE_ROOT = (
    REPOSITORY_ROOT / "archive" / "pretrained" / "0019_universal_winner_bc_0730_epoch13"
)
EXPECTED_ASSET_ID = "0019-0730-epoch13"
EXPECTED_POOL_ID = "0019_foundation_51_exact_decks_v4"
EXPECTED_DECK_COUNT = 51
EXPECTED_PROVENANCE_WEIGHTS_SHA256 = (
    "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
)
EXPECTED_EXPORTED_MODEL_SHA256 = (
    "2a3e3224b9fbc0bda95de45583faa1f3923d37bc55e26c6279315cc720d7f918"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d"
)

_VARIABLE_SECOND_DIM = frozenset({
    "entity_cat",
    "entity_num",
    "entity_mask",
    "option_cat",
    "option_mask",
    "registered_card_ids",
    "registered_multiplicity",
    "registered_mask",
    "ledger_cat",
    "ledger_num",
    "ledger_mask",
    "event_cat",
    "event_num",
    "event_mask",
    "known_opponent_hand_card_ids",
    "known_opponent_hand_mask",
})


@dataclass(frozen=True)
class LegacyFoundationIdentity:
    asset_id: str
    pool_id: str
    deck_count: int
    deployment_source_id: int
    provenance_weights_sha256: str
    exported_model_sha256: str
    ontology_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "deck_count": self.deck_count,
            "deployment_source_id": self.deployment_source_id,
            "exported_model_sha256": self.exported_model_sha256,
            "ontology_sha256": self.ontology_sha256,
            "pool_id": self.pool_id,
            "provenance_weights_sha256": self.provenance_weights_sha256,
        }


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_legacy_foundation(
    policy_root: Path = DEFAULT_POLICY_ROOT,
) -> LegacyFoundationIdentity:
    policy_root = Path(policy_root)
    model_path = policy_root / "strategy" / "model.bin"
    ontology_path = policy_root / "strategy" / "card_ontology.json"
    pool_manifest_path = policy_root.parent / "manifest.json"
    provenance_path = PROVENANCE_ROOT / "model.pt"
    try:
        pool = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Frozen Arena manifest: {pool_manifest_path}") from error
    actual = {
        "asset_id": (pool.get("foundation") or {}).get("asset_id"),
        "deck_count": pool.get("deck_count"),
        "deployment_source_id": (pool.get("foundation") or {}).get(
            "deployment_source_id"
        ),
        "exported_model_sha256": _sha256(model_path),
        "ontology_sha256": _sha256(ontology_path),
        "pool_id": pool.get("pool_id"),
        "provenance_weights_sha256": _sha256(provenance_path),
    }
    expected = {
        "asset_id": EXPECTED_ASSET_ID,
        "deck_count": EXPECTED_DECK_COUNT,
        "deployment_source_id": 0,
        "exported_model_sha256": EXPECTED_EXPORTED_MODEL_SHA256,
        "ontology_sha256": EXPECTED_ONTOLOGY_SHA256,
        "pool_id": EXPECTED_POOL_ID,
        "provenance_weights_sha256": EXPECTED_PROVENANCE_WEIGHTS_SHA256,
    }
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            raise ValueError(
                f"legacy Foundation {name} mismatch: {actual[name]!r} != {expected_value!r}"
            )
    return LegacyFoundationIdentity(**actual)


def _pad_second_dim(value: Tensor, width: int) -> Tensor:
    if value.size(1) == width:
        return value
    shape = list(value.shape)
    shape[1] = width - value.size(1)
    return torch.cat((value, torch.zeros(shape, dtype=value.dtype)), dim=1)


def collate_legacy_batches(items: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not items:
        raise ValueError("cannot collate an empty legacy feature batch")
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("legacy feature batch keys disagree")
    output: dict[str, Tensor] = {}
    for key in sorted(keys):
        values = [item[key] for item in items]
        if key in _VARIABLE_SECOND_DIM:
            width = max(value.size(1) for value in values)
            values = [_pad_second_dim(value, width) for value in values]
        output[key] = torch.cat(values, dim=0)
    return output


class LegacyFoundationService:
    def __init__(self, device: torch.device) -> None:
        self.identity = verify_legacy_foundation()
        module = importlib.import_module(
            "evaluation.arena.frozen._policy.strategy.portable_inference"
        )
        policy = module.PortablePolicy.from_checkpoint(
            DEFAULT_POLICY_ROOT / "strategy" / "model.bin",
            DEFAULT_POLICY_ROOT / "strategy" / "card_ontology.json",
            [0] * 60,
        )
        self.device = device
        self.actor = policy.actor.to(device).eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.encoder_type = module.OnlineCausalEncoder

    def new_encoder(self, actor: int, deck: Sequence[int]):
        return self.encoder_type(actor, deck, self.actor.config)

    def greedy(self, rows: Sequence[dict[str, Tensor]]) -> list[tuple[int, ...]]:
        batch = {
            key: value.to(self.device, non_blocking=True)
            for key, value in collate_legacy_batches(rows).items()
        }
        with torch.inference_mode():
            decoded = self.actor.deterministic_action_tensors(batch)
        sequences = decoded.sequences.detach().cpu()
        lengths = decoded.lengths.detach().cpu()
        return [
            tuple(
                int(value)
                for value in sequences[index, : lengths[index]].tolist()
            )
            for index in range(sequences.size(0))
        ]


__all__ = [
    "LegacyFoundationIdentity",
    "LegacyFoundationService",
    "collate_legacy_batches",
    "verify_legacy_foundation",
]
