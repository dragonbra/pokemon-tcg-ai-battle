"""Self-contained 0038 compound-action inference runtime copied into packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..action_boundary.decision_gate import DecisionClass, DecisionGate
from ..action_boundary.dragapult import (
    PHANTOM_DIVE_ATTACK_ID,
    StableTargetIdentity,
)
from ..action_boundary.macro_planner import MacroPlanner
from ..action_boundary.macro_protocol import MacroProtocolError, PendingMacroTransaction
from ..action_boundary.public_card_features import card_prize_counts, with_public_prize
from ..contracts.batch import DecisionBatch
from ..domain.prototypes import PrototypeIndex
from ..model import ModelConfig, SemanticPolicy
from .inference import _expanded_portable_state_dict
from .online_runtime import OnlineCausalEncoder, _prototype_paths
from .value_network import LatentQueryValueHead


SCHEMA_VERSION = "0038_compound_kaggle_candidate_v3"
PROTOTYPES = (
    Path(__file__).resolve().parents[1]
    / "assets/official_full_engine_prototypes_v2.json"
)


class DragapultAllocationHead(nn.Module):
    VISIBLE_FEATURES = 12

    def __init__(self, width: int) -> None:
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(width + self.VISIBLE_FEATURES, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.structure = nn.Sequential(nn.Linear(6, width), nn.GELU())
        self.score = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, state_summary, root_option, target_embeddings,
                visible_features, target_mask, allocation_mask):
        mask = target_mask.bool()
        rows = self.target(torch.cat((target_embeddings, visible_features), dim=-1))
        pooled = (rows * mask.unsqueeze(-1)).sum(dim=-2)
        counters = visible_features[..., 0]
        positive = counters.gt(0) & mask
        total = counters.sum(dim=-1).clamp_min(1)
        concentration = (counters.square().sum(dim=-1) / total.square()).unsqueeze(-1)
        maximum = counters.max(dim=-1).values.unsqueeze(-1)
        minimum = torch.where(positive, counters, torch.inf).min(dim=-1).values
        minimum = torch.where(torch.isfinite(minimum), minimum, 0).unsqueeze(-1)
        structure = torch.cat((
            mask.sum(dim=-1, keepdim=True).to(counters.dtype),
            positive.sum(dim=-1, keepdim=True).to(counters.dtype),
            total.unsqueeze(-1), maximum, minimum, concentration,
        ), dim=-1)
        structural = self.structure(structure)
        batch, allocations = allocation_mask.shape
        summary = state_summary[:, None].expand(batch, allocations, -1)
        root = root_option[:, None].expand(batch, allocations, -1)
        logits = self.score(torch.cat((summary, root, pooled, structural), dim=-1)).squeeze(-1)
        return logits.masked_fill(~allocation_mask.bool(), -torch.inf)


class OpponentMetaHead(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, classes),
        )

    def forward(self, state_summary):
        return self.classifier(state_summary)


class OpponentMetaConditioner(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.archetype_embeddings = nn.Parameter(torch.empty(classes, width))
        self.output = nn.Linear(width, width)

    def forward(self, state_summary, logits):
        distribution = logits.softmax(dim=-1)
        return state_summary + self.output(distribution @ self.archetype_embeddings)


class PortableCompoundSemanticPolicy:
    """Greedy 0038 actor with forced shortcuts and cached macro expansion."""

    def __init__(self, actor: SemanticPolicy, value_head: nn.Module,
                 allocation_head: nn.Module,
                 meta_head: nn.Module, meta_conditioner: nn.Module,
                 deck: Sequence[int], metadata: Mapping[str, Any]) -> None:
        torch.set_num_threads(1)
        self.actor = actor.eval()
        # The critic is carried and strict-loaded for audit/parity.  It is not
        # called by select(), so deployment action latency and semantics remain
        # actor-only.
        self.value_head = value_head.eval()
        self.allocation_head = allocation_head.eval()
        self.meta_head = meta_head.eval()
        self.meta_conditioner = meta_conditioner.eval()
        self.deck = tuple(int(card) for card in deck)
        if len(self.deck) != 60 or any(card <= 0 for card in self.deck):
            raise ValueError("deck must contain exactly 60 positive card IDs")
        self.metadata = dict(metadata)
        self.encoder: OnlineCausalEncoder | None = None
        self.gate = DecisionGate()
        self.planner = MacroPlanner(self.allocation_head)
        self.prizes = card_prize_counts(PROTOTYPES)
        self.pending: PendingMacroTransaction | None = None
        self.session_serial = 0
        self.battle_id = "kaggle-session-0"

    def prototype_cache_stats(self) -> dict[str, int]:
        """Expose low-cardinality deployment diagnostics without changing policy."""
        return self.actor.prototype_cache_stats()

    @classmethod
    def from_checkpoint(cls, path: Path, deck: Sequence[int]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "schema_version", "actor_state_dict", "value_head_state_dict",
            "allocation_head_state_dict",
            "opponent_meta_head_state_dict", "opponent_meta_conditioner_state_dict",
            "metadata",
        }
        if set(payload) != expected or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint is not the 0038 compound Kaggle contract")
        metadata = payload["metadata"]
        actor_metadata = metadata.get("actor_metadata")
        contract = actor_metadata.get("model_config") if isinstance(actor_metadata, Mapping) else None
        config_payload = contract.get("config") if isinstance(contract, Mapping) else None
        if not isinstance(config_payload, Mapping) or contract.get("model") != "SemanticPolicy":
            raise ValueError("compound checkpoint has no audited actor config")
        public_path, engine_path = _prototype_paths()
        actor = SemanticPolicy(
            ModelConfig(**dict(config_payload)),
            PrototypeIndex.load(public_path, engine_path),
        )
        actor.load_state_dict(
            _expanded_portable_state_dict(payload["actor_state_dict"]), strict=True
        )
        width = int(actor.config.d_model)
        value_head = LatentQueryValueHead(width, int(actor.config.heads))
        value_head.load_state_dict(payload["value_head_state_dict"], strict=True)
        classes = int(metadata["opponent_meta_class_count"])
        allocation_head = DragapultAllocationHead(width)
        allocation_head.load_state_dict(payload["allocation_head_state_dict"], strict=True)
        meta_head = OpponentMetaHead(width, classes)
        meta_head.load_state_dict(payload["opponent_meta_head_state_dict"], strict=True)
        conditioner = OpponentMetaConditioner(width, classes)
        conditioner.load_state_dict(
            payload["opponent_meta_conditioner_state_dict"], strict=True
        )
        return cls(
            actor, value_head, allocation_head, meta_head, conditioner, deck, metadata
        )

    def value_from_encoded(self, validated, state, options):
        """Diagnostic-only Value parity path; never called by select()."""
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, memory_mask)
        meta_logits = self.meta_head(state.summary)
        conditioned = self.meta_conditioner(state.summary, meta_logits)
        value_query = queries[:, 0] + (conditioned - state.summary)
        logit = self.value_head.heads.value(value_query).squeeze(-1)
        return 2.0 * logit.sigmoid() - 1.0

    def reset(self) -> None:
        self.encoder = None
        self.pending = None
        self.session_serial += 1
        self.battle_id = f"kaggle-session-{self.session_serial}"

    def _ensure_encoder(self, actor: int) -> OnlineCausalEncoder:
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.actor.config)
            self.pending = None
        return self.encoder

    def _observe_only(self, observation: Mapping[str, Any], actor: int) -> None:
        self._ensure_encoder(actor).knowledge.consume(observation)

    @staticmethod
    def _confused(current: Mapping[str, Any], actor: int) -> bool:
        players = current.get("players") or []
        return bool(
            0 <= actor < len(players) and isinstance(players[actor], Mapping)
            and players[actor].get("confused") is True
        )

    def select(self, observation: dict[str, Any]) -> list[int]:
        current = observation.get("current") or {}
        actor = current.get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        encoder = self._ensure_encoder(int(actor))

        if self.pending is not None:
            self._observe_only(observation, int(actor))
            try:
                action = self.pending.next_primitive(
                    observation, battle_id=self.battle_id
                )
            except MacroProtocolError:
                self.pending = None
                raise
            else:
                if self.pending.complete:
                    self.pending = None
                return action

        gate = self.gate.classify(observation)
        if gate.classification is DecisionClass.MASK_ERROR:
            raise RuntimeError(f"DecisionGate MASK_ERROR: {gate.reason}")
        if gate.classification in {DecisionClass.FORCED, DecisionClass.LEGAL_EMPTY_PASS}:
            self._observe_only(observation, int(actor))
            return list(gate.forced_action or ())
        if gate.classification is not DecisionClass.STRATEGIC:
            raise RuntimeError(f"unsupported decision class: {gate.classification}")

        batch = DecisionBatch.from_mapping(encoder.encode(observation))
        with torch.inference_mode():
            validated, state, options = self.actor.encode(batch)
            meta_logits = self.meta_head(state.summary)
            summary = self.meta_conditioner(state.summary, meta_logits)
            decoded = self.actor.action_decoder.greedy(validated, options, summary)
        if not bool(decoded.legal[0]):
            raise RuntimeError("greedy decode violated selection bounds")
        action = decoded.sequences[0, : int(decoded.lengths[0])].tolist()

        root_options = (observation.get("select") or {}).get("option") or []
        phantom_root = next((
            index for index in action
            if 0 <= index < len(root_options)
            and isinstance(root_options[index], Mapping)
            and root_options[index].get("attackId") == PHANTOM_DIVE_ATTACK_ID
        ), None)
        players = current.get("players") or []
        raw_bench = (
            list(players[1 - int(actor)].get("bench") or [])
            if len(players) == 2 and isinstance(players[1 - int(actor)], Mapping)
            else []
        )
        if (
            phantom_root is not None and 1 <= len(raw_bench) <= 8
            and not self._confused(current, int(actor))
        ):
            identities = sorted(
                StableTargetIdentity(
                    1 - int(actor), int(raw["serial"]), int(raw["id"]), slot
                )
                for slot, raw in enumerate(raw_bench)
            )
            raw_by_serial = {
                int(raw["serial"]): {**raw, "benchSlot": slot}
                for slot, raw in enumerate(raw_bench)
            }
            target_rows = []
            for target in identities:
                card_mask = (
                    validated.card_mask[0]
                    & validated.card_cat[0, :, 2].eq(2)
                    & validated.card_cat[0, :, 3].eq(6)
                    & validated.card_cat[0, :, 4].eq(target.initial_bench_slot + 1)
                )
                positions = card_mask.nonzero(as_tuple=False).flatten()
                if positions.numel() != 1:
                    raise RuntimeError("Phantom target does not map to one state token")
                target_rows.append(state.cards[0, int(positions[0])])
            planned = self.planner.plan(
                state_summary=state.summary[0],
                root_option=options[0, phantom_root],
                target_embeddings=torch.stack(target_rows),
                target_identities=identities,
                visible_targets=[
                    with_public_prize(raw_by_serial[target.serial], self.prizes)
                    for target in identities
                ],
                greedy=True,
            )
            self.pending = PendingMacroTransaction(
                self.battle_id, int(actor), planned.allocation
            )
        return action


__all__ = ["PortableCompoundSemanticPolicy", "SCHEMA_VERSION"]
