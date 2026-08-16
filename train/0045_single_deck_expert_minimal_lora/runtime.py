"""Project-local strict policy loading and executable forward parity gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .assets import AssetIntegrityError, AssetRegistry, sha256_file
from .semantic_runtime.contracts.batch import DecisionBatch
from .semantic_runtime.contracts.fields import WIDTHS
from .semantic_runtime.deployment.compound_inference import PortableCompoundSemanticPolicy
from .semantic_runtime.deployment.inference import PortableSemanticPolicy
from .own_archetype import OwnArchetypeVocabulary


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "semantic_runtime"


def audit_runtime_tree() -> str:
    manifest = json.loads((RUNTIME_ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "0044_semantic_runtime_manifest_v1":
        raise AssetIntegrityError("unsupported semantic runtime manifest")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise AssetIntegrityError("semantic runtime manifest is empty")
    for row in rows:
        path = (RUNTIME_ROOT / row["path"]).resolve()
        if RUNTIME_ROOT.resolve() not in path.parents or sha256_file(path) != row["sha256"]:
            raise AssetIntegrityError(f"semantic runtime integrity mismatch: {row['path']}")
    digest = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    if digest != manifest.get("runtime_tree_sha256"):
        raise AssetIntegrityError("semantic runtime tree identity mismatch")
    return digest


def _deck(deck_id: str) -> tuple[int, ...]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    deck = next((item for item in registry.decks if item.deck_id == deck_id), None)
    if deck is None:
        raise AssetIntegrityError(f"unregistered 0044 deck: {deck_id}")
    path = PROJECT_ROOT / deck.deck_path
    if sha256_file(path) != deck.file_sha256:
        raise AssetIntegrityError(f"deck artifact mismatch: {deck_id}")
    return tuple(int(line) for line in path.read_text(encoding="utf-8").splitlines())


def load_policy(policy_id: str, *, deck_id: str):
    audit_runtime_tree()
    registry = AssetRegistry.load(PROJECT_ROOT)
    policy = next((item for item in registry.policies if item.policy_id == policy_id), None)
    if policy is None:
        raise AssetIntegrityError(f"unregistered 0044 policy: {policy_id}")
    if policy_id not in {"Policy-0809", "Policy-0814"} and not (
        policy_id.startswith("Champion-G") and policy.generation is not None
    ):
        raise AssetIntegrityError("0044 runtime admits only registered complete policies")
    purpose = (
        "complete_base_checkpoint"
        if policy_id in {"Policy-0809", "Policy-0814"}
        else "portable_fp16_artifact"
    )
    artifact = next((item for item in policy.artifacts if item.purpose == purpose), None)
    if artifact is None:
        raise AssetIntegrityError(f"{policy_id} has no {purpose}")
    path = PROJECT_ROOT / artifact.path
    if sha256_file(path) != artifact.sha256:
        raise AssetIntegrityError(f"policy artifact mismatch: {policy_id}")
    cards = _deck(deck_id)
    if policy_id in {"Policy-0809", "Policy-0814"}:
        return PortableSemanticPolicy.from_checkpoint(path, cards)
    if policy_id.startswith("Champion-G") and policy.generation is not None:
        loaded = PortableCompoundSemanticPolicy.from_checkpoint(path, cards)
        vocabulary = OwnArchetypeVocabulary.load_version(
            "own_archetypes_v2", project_root=PROJECT_ROOT
        )
        mapping = next(row for row in vocabulary.mappings if row.deck_id == deck_id)
        embedding_count = loaded.value_adapter.own_embedding.num_embeddings
        loaded.metadata["own_archetype_id"] = (
            mapping.archetype_id if embedding_count == vocabulary.class_count
            else vocabulary.classes[mapping.archetype_id].embedding_init_from
        )
        return loaded
    raise AssetIntegrityError(f"0044 has no executable loader for {policy_id}")


def load_focal_seed(*, deck_id: str):
    """Load the immutable G2 U407 initialization for focal deck 007."""
    if deck_id != "007":
        raise AssetIntegrityError("0044 V1 focal seed is defined only for exact deck 007")
    path = PROJECT_ROOT / "assets/policies/definitions/champion_g002/model.bin"
    cards = _deck(deck_id)
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    loaded = PortableCompoundSemanticPolicy.from_checkpoint(path, cards)
    loaded.metadata["own_archetype_id"] = vocabulary.resolve_exact_deck(deck_id, cards).value
    if loaded.value_adapter.own_embedding.num_embeddings != vocabulary.class_count:
        raise AssetIntegrityError("V1 focal own-taxonomy runtime shape mismatch")
    return loaded


def synthetic_batch(*, batch_size: int = 2) -> DecisionBatch:
    """Deterministic legal-shaped batch used only for executable parity tests."""
    b, c, r, e, o, s, f, steps = batch_size, 3, 4, 2, 5, 3, 4, 2
    values: dict[str, torch.Tensor] = {
        "global_cat": torch.zeros(b, WIDTHS.global_cat, dtype=torch.long),
        "global_num": torch.zeros(b, WIDTHS.global_num),
        "global_state": torch.ones(b, WIDTHS.global_state, dtype=torch.long),
        "card_cat": torch.zeros(b, c, WIDTHS.card_cat, dtype=torch.long),
        "card_num": torch.zeros(b, c, WIDTHS.card_num),
        "card_state": torch.ones(b, c, WIDTHS.card_state, dtype=torch.long),
        "card_parent": torch.zeros(b, c, dtype=torch.long),
        "card_mask": torch.ones(b, c, dtype=torch.bool),
        "resource_cat": torch.zeros(b, r, WIDTHS.resource_cat, dtype=torch.long),
        "resource_num": torch.zeros(b, r, WIDTHS.resource_num),
        "resource_state": torch.ones(b, r, WIDTHS.resource_state, dtype=torch.long),
        "resource_mask": torch.ones(b, r, dtype=torch.bool),
        "event_cat": torch.zeros(b, e, WIDTHS.event_cat, dtype=torch.long),
        "event_num": torch.zeros(b, e, WIDTHS.event_num),
        "event_state": torch.ones(b, e, WIDTHS.event_state, dtype=torch.long),
        "event_source": torch.zeros(b, e, dtype=torch.long),
        "event_target": torch.zeros(b, e, dtype=torch.long),
        "event_before": torch.zeros(b, e, dtype=torch.long),
        "event_after": torch.zeros(b, e, dtype=torch.long),
        "event_mask": torch.ones(b, e, dtype=torch.bool),
        "option_cat": torch.zeros(b, o, WIDTHS.option_cat, dtype=torch.long),
        "option_num": torch.zeros(b, o, WIDTHS.option_num),
        "option_state": torch.zeros(b, o, WIDTHS.option_state, dtype=torch.long),
        "option_source": torch.zeros(b, o, dtype=torch.long),
        "option_target": torch.zeros(b, o, dtype=torch.long),
        "option_context": torch.zeros(b, o, dtype=torch.long),
        "option_effect_card": torch.zeros(b, o, dtype=torch.long),
        "option_mask": torch.ones(b, o, dtype=torch.bool),
        "option_skill_id": torch.zeros(b, s, dtype=torch.long),
        "option_skill_role": torch.zeros(b, s, dtype=torch.long),
        "option_skill_parent": torch.zeros(b, s, dtype=torch.long),
        "option_skill_mask": torch.ones(b, s, dtype=torch.bool),
        "option_effect_id": torch.zeros(b, f, dtype=torch.long),
        "option_effect_role": torch.zeros(b, f, dtype=torch.long),
        "option_effect_parent": torch.zeros(b, f, dtype=torch.long),
        "option_effect_mask": torch.ones(b, f, dtype=torch.bool),
        "min_count": torch.ones(b, dtype=torch.long),
        "max_count": torch.full((b,), steps, dtype=torch.long),
        "targets": torch.zeros(b, steps, dtype=torch.long),
    }
    values["global_cat"][:, 2] = 1  # public relative-first-player identity
    return DecisionBatch.from_mapping(values)


def _modules(policy: Any) -> tuple[torch.nn.Module, ...]:
    if isinstance(policy, PortableSemanticPolicy):
        return (policy.model,)
    return (
        policy.actor, policy.value_head, policy.allocation_head,
        policy.value_adapter, policy.policy_strategy_adapter,
    )


def _forward(policy: Any, batch: DecisionBatch) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        if isinstance(policy, PortableSemanticPolicy):
            logits = policy.model(batch)
            action = policy.model.deterministic_action_tensors(batch).sequences
            return logits.float().cpu(), action.cpu()
        validated, state, options, value, _, context = policy.encode_with_strategy(batch)
        sequences, _, legal = policy._greedy_strategy(validated, state, options, context)
        if not bool(legal.all()):
            raise RuntimeError("synthetic forward produced an illegal action")
        return value.float().cpu(), sequences.cpu()


@dataclass(frozen=True, slots=True)
class ForwardParity:
    policy_id: str
    deck_id: str
    cpu_shape: tuple[int, ...]
    actions_equal: bool
    max_abs_error: float | None
    gpu_status: str


def forward_parity(policy_id: str, *, deck_id: str, gpu: bool) -> ForwardParity:
    policy = load_policy(policy_id, deck_id=deck_id)
    batch = synthetic_batch()
    cpu_output, cpu_actions = _forward(policy, batch)
    if not gpu:
        return ForwardParity(policy_id, deck_id, tuple(cpu_output.shape), True, None, "NOT_RUN")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA forward parity requested but CUDA is unavailable")
    for module in _modules(policy):
        module.to("cuda:0")
    gpu_output, gpu_actions = _forward(policy, batch.to("cuda:0"))
    maximum = float((cpu_output - gpu_output).abs().max().item())
    actions_equal = bool(torch.equal(cpu_actions, gpu_actions))
    if not actions_equal or not torch.allclose(cpu_output, gpu_output, rtol=2e-3, atol=2e-4):
        raise RuntimeError(
            f"CPU/CUDA forward parity failed: actions_equal={actions_equal}, max_abs_error={maximum}"
        )
    return ForwardParity(policy_id, deck_id, tuple(cpu_output.shape), actions_equal, maximum, "PASS")


__all__ = [
    "ForwardParity", "audit_runtime_tree", "forward_parity", "load_focal_seed",
    "load_policy", "synthetic_batch",
]
