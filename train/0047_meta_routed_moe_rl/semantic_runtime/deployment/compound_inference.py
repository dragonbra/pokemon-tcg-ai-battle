"""Self-contained 0038 compound-action inference runtime copied into packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

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


SCHEMA_VERSION = "0042_strategy_conditioned_kaggle_candidate_v1"
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


class ValueResidualAdapter(nn.Module):
    def __init__(self, width: int = 320, embedding_dim: int = 16,
                 own_archetype_classes: int | None = None) -> None:
        super().__init__()
        if own_archetype_classes is None or own_archetype_classes < 1:
            raise ValueError("own_archetype_classes must come from checkpoint taxonomy metadata")
        self.norm = nn.LayerNorm(width)
        self.own_embedding = nn.Embedding(own_archetype_classes, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(width + embedding_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, own_id):
        delta = self.mlp(torch.cat((self.norm(hidden), self.own_embedding(own_id)), dim=-1))
        return hidden + self.gate.tanh() * delta, delta


@dataclass(frozen=True, slots=True)
class StrategyContext:
    side_onehot: torch.Tensor
    z_meta: torch.Tensor
    meta_probabilities: torch.Tensor
    value: torch.Tensor
    own_archetype_id: torch.Tensor

    @classmethod
    def build(cls, relative_first_player, z_meta, meta_logits, value, own_archetype_id):
        if not bool(relative_first_player.eq(1).logical_or(relative_first_player.eq(2)).all()):
            raise ValueError("public first/second identity is unavailable")
        side = torch.nn.functional.one_hot(
            relative_first_player - 1, num_classes=2
        ).to(z_meta.dtype)
        return cls(
            side,
            z_meta.detach(),
            meta_logits.detach().softmax(dim=-1),
            value.detach(),
            own_archetype_id,
        )


class PolicyStrategyAdapter(nn.Module):
    def __init__(self, width: int = 320, embedding_dim: int = 16,
                 own_archetype_classes: int | None = None) -> None:
        super().__init__()
        if own_archetype_classes is None or own_archetype_classes < 1:
            raise ValueError("own_archetype_classes must come from checkpoint taxonomy metadata")
        self.hidden_norm = nn.LayerNorm(width)
        self.meta_context_norm = nn.LayerNorm(width)
        self.own_embedding = nn.Embedding(own_archetype_classes, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(width + 2 + width + 15 + 1 + embedding_dim, width),
            nn.GELU(), nn.Linear(width, width),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def context_tensor(self, context):
        return torch.cat((
            context.side_onehot,
            self.meta_context_norm(context.z_meta),
            context.meta_probabilities,
            context.value.unsqueeze(-1),
            self.own_embedding(context.own_archetype_id),
        ), dim=-1)

    def forward(self, hidden, context):
        delta = self.mlp(torch.cat((self.hidden_norm(hidden), self.context_tensor(context)), dim=-1))
        return hidden + self.gate.tanh() * delta, delta


class MetaActorResidual(nn.Module):
    def __init__(self, width: int = 320, rank: int = 4,
                 archetype_classes: int = 29, alpha: float = 4.0) -> None:
        super().__init__()
        self.width = width
        self.rank = rank
        self.archetype_classes = archetype_classes
        self.scale = float(alpha) / rank
        self.down = nn.Parameter(torch.empty(archetype_classes, rank, width))
        self.up = nn.Parameter(torch.zeros(archetype_classes, width, rank))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def forward(self, hidden, own_archetype_id):
        ids = own_archetype_id.to(device=hidden.device, dtype=torch.long)
        if ids.shape != (hidden.shape[0],) or not bool(
            ids.ge(0).logical_and(ids.lt(self.archetype_classes)).all()
        ):
            raise ValueError("Meta Actor Residual own-Meta identity mismatch")
        low_rank = torch.bmm(self.down[ids], hidden.unsqueeze(-1))
        delta = torch.bmm(self.up[ids], low_rank).squeeze(-1) * self.scale
        return hidden + delta, delta


class LoRAQVDelta(nn.Module):
    """Portable rank-4 Q/V delta for one merged Option QKV projection."""

    def __init__(self, width: int, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.q_a = nn.Parameter(torch.empty(rank, width))
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.empty(rank, width))
        self.v_b = nn.Parameter(torch.zeros(width, rank))
        self.scale = float(alpha) / int(rank)

    def merged_weight(self, base: torch.Tensor) -> torch.Tensor:
        width = base.shape[0] // 3
        if base.shape != (3 * width, width):
            raise ValueError("Option Q/V LoRA requires a square merged QKV projection")
        q = (self.q_b @ self.q_a).to(base.dtype) * self.scale
        v = (self.v_b @ self.v_a).to(base.dtype) * self.scale
        return base + torch.cat((q, torch.zeros_like(q), v), dim=0)


class PolicyOnlyOptionLoRA(nn.Module):
    """Portable policy-only final Option block; Value keeps the immutable block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attention = LoRAQVDelta(width)
        self.cross_attention = LoRAQVDelta(width)

    def forward(self, option_encoder, batch, state, prefix):
        transformer = option_encoder.cross_attention_transformer
        if len(transformer.layers) != 2:
            raise RuntimeError("0044 requires exactly two Option Transformer blocks")
        final = transformer.layers[1]
        encoded = functional_call(
            final,
            {
                "self_attn.in_proj_weight": self.self_attention.merged_weight(
                    final.self_attn.in_proj_weight
                ),
                "multihead_attn.in_proj_weight": self.cross_attention.merged_weight(
                    final.multihead_attn.in_proj_weight
                ),
            },
            (prefix, state.tokens),
            option_encoder._masks(batch, state),
            strict=False,
        )
        if transformer.norm is not None:
            encoded = transformer.norm(encoded)
        return encoded * batch.option_mask.unsqueeze(-1)


class PortableCompoundSemanticPolicy:
    """Greedy 0042 strategy-conditioned actor with cached macro expansion."""

    def __init__(self, actor: SemanticPolicy, value_head: nn.Module,
                 allocation_head: nn.Module, value_adapter: nn.Module,
                 policy_strategy_adapter: nn.Module | None,
                 deck: Sequence[int], metadata: Mapping[str, Any],
                 policy_option_lora: nn.Module | None = None,
                 meta_actor_residual: nn.Module | None = None) -> None:
        torch.set_num_threads(1)
        self.actor = actor.eval()
        # Legacy 0042-0044 policies consume Critic context in Actor inference.
        # 0045 keeps these modules only for deployment-identity compatibility;
        # its Actor path never evaluates or consumes them.
        self.value_head = value_head.eval()
        self.allocation_head = allocation_head.eval()
        self.value_adapter = value_adapter.eval()
        self.policy_strategy_adapter = (
            policy_strategy_adapter.eval()
            if policy_strategy_adapter is not None else None
        )
        self.policy_option_lora = (
            policy_option_lora.eval() if policy_option_lora is not None else None
        )
        self.meta_actor_residual = (
            meta_actor_residual.eval() if meta_actor_residual is not None else None
        )
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
        legacy_expected = {
            "schema_version", "actor_state_dict", "value_head_state_dict",
            "allocation_head_state_dict",
            "value_adapter_state_dict", "policy_strategy_adapter_state_dict",
            "metadata",
        }
        option_lora_expected = legacy_expected | {"policy_option_lora_state_dict"}
        meta_residual_expected = option_lora_expected | {
            "meta_actor_residual_state_dict"
        }
        minimal_0045_expected = {
            "schema_version", "actor_state_dict", "value_head_state_dict",
            "allocation_head_state_dict", "value_adapter_state_dict",
            "policy_option_lora_state_dict", "metadata",
        }
        supported_schemas = {
            SCHEMA_VERSION,
            "0043_focal_v1_kaggle_candidate_v1",
            "0044_g2_policy_option_lora_kaggle_candidate_v1",
            "0044_policy_value_split_option_lora_candidate_v1",
            "0044_policy_value_split_option_lora_meta_residual_candidate_v2",
            "0045_minimal_lora_candidate_v1",
        }
        expected = (
            minimal_0045_expected
            if payload.get("schema_version") == "0045_minimal_lora_candidate_v1"
            else
            meta_residual_expected
            if payload.get("schema_version") == "0044_policy_value_split_option_lora_meta_residual_candidate_v2"
            else option_lora_expected
            if payload.get("schema_version") == "0044_policy_value_split_option_lora_candidate_v1"
            else legacy_expected
        )
        if set(payload) != expected or payload.get("schema_version") not in supported_schemas:
            raise ValueError("checkpoint is not the 0042 compound Kaggle contract")
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
        actor_state = _expanded_portable_state_dict(payload["actor_state_dict"])
        actor.load_state_dict(actor_state, strict=True)
        width = int(actor.config.d_model)
        value_head = LatentQueryValueHead(width, int(actor.config.heads))
        value_head.load_state_dict(payload["value_head_state_dict"], strict=True)
        allocation_head = DragapultAllocationHead(width)
        allocation_head.load_state_dict(payload["allocation_head_state_dict"], strict=True)
        has_policy_option_lora = "policy_option_lora_state_dict" in payload
        if not has_policy_option_lora and metadata.get("no_option_lora") is not True:
            raise ValueError("0042 portable checkpoint must declare no_option_lora=true")
        own_classes = int(payload["value_adapter_state_dict"]["own_embedding.weight"].shape[0])
        declared_classes = metadata.get("own_archetype_class_count")
        if declared_classes is not None and int(declared_classes) != own_classes:
            raise ValueError("declared own-taxonomy size disagrees with checkpoint tensor")
        own_version = metadata.get("own_archetype_vocabulary_version")
        if own_classes == 15 and own_version != "0042_own_archetypes_v1":
            raise ValueError("G1 own-taxonomy identity mismatch")
        if own_classes != 15 and (
            payload.get("schema_version") not in {
                "0043_focal_v1_kaggle_candidate_v1",
                "0044_g2_policy_option_lora_kaggle_candidate_v1",
                "0044_policy_value_split_option_lora_candidate_v1",
                "0044_policy_value_split_option_lora_meta_residual_candidate_v2",
                "0045_minimal_lora_candidate_v1",
            }
            or own_version != "own_archetypes_v2"
        ):
            raise ValueError("dynamic own-taxonomy checkpoint identity mismatch")
        value_adapter = ValueResidualAdapter(width, own_archetype_classes=own_classes)
        value_adapter.load_state_dict(payload["value_adapter_state_dict"], strict=True)
        minimal_0045 = payload.get("schema_version") == "0045_minimal_lora_candidate_v1"
        if minimal_0045:
            if metadata.get("policy_strategy_adapter") != "removed":
                raise ValueError("0045 candidate did not retire Policy Strategy Adapter")
            if metadata.get("meta_actor_residual") != "removed":
                raise ValueError("0045 candidate did not retire Meta Actor Residual")
            if metadata.get("critic_outputs_consumed_by_actor") is not False:
                raise ValueError("0045 candidate does not prove Critic-free Actor inference")
            policy_adapter = None
        else:
            policy_adapter = PolicyStrategyAdapter(
                width, own_archetype_classes=own_classes
            )
            policy_adapter.load_state_dict(
                payload["policy_strategy_adapter_state_dict"], strict=True
            )
        policy_option_lora = None
        if has_policy_option_lora:
            policy_option_lora = PolicyOnlyOptionLoRA(width)
            policy_option_lora.load_state_dict(
                payload["policy_option_lora_state_dict"], strict=True
            )
        meta_actor_residual = None
        if "meta_actor_residual_state_dict" in payload:
            meta_actor_residual = MetaActorResidual(
                width, rank=4, archetype_classes=own_classes
            )
            meta_actor_residual.load_state_dict(
                payload["meta_actor_residual_state_dict"], strict=True
            )
        return cls(
            actor, value_head, allocation_head, value_adapter, policy_adapter,
            deck, metadata, policy_option_lora, meta_actor_residual,
        )

    def _encode_options(self, features):
        """Return immutable Value options and independently adapted Policy options."""
        if self.policy_option_lora is None:
            validated, state, options = self.actor.encode(features)
            return validated, state, options, options
        validated = self.actor.validate_batch(features)
        prototype_memory = self.actor.prototype_memory()
        state = self.actor.state_encoder(validated, prototype_memory)
        encoder = self.actor.option_encoder
        inputs = encoder.encode_inputs(validated, state, prototype_memory)
        prefix = encoder.encode_prefix(validated, state, inputs)
        value_options = encoder.encode_final(validated, state, prefix)
        policy_options = self.policy_option_lora(encoder, validated, state, prefix)
        return validated, state, value_options, policy_options

    def value_from_encoded(self, validated, state, options, own_archetype_id=None):
        """Expose the same adapted q0 Value used to build strategy context."""
        value, _ = self.value_and_aux_from_encoded(
            validated, state, options, own_archetype_id=own_archetype_id
        )
        return value

    def value_and_aux_from_encoded(
        self, validated, state, options, own_archetype_id=None
    ):
        """Expose q0/q1 tensors for deployment-parity diagnostics."""
        memory = torch.cat((state.tokens, options), dim=1)
        memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
        queries = self.value_head.decode(memory, memory_mask)
        z_meta = queries[:, 1]
        meta_logits = self.value_head.heads.archetype(z_meta)
        own_id = (
            torch.full(
                (queries.shape[0],), int(self.metadata["own_archetype_id"]),
                dtype=torch.long, device=queries.device,
            )
            if own_archetype_id is None
            else own_archetype_id.to(device=queries.device, dtype=torch.long)
        )
        if own_id.shape != (queries.shape[0],):
            raise ValueError("own_archetype_id must have shape [B]")
        value_query, _ = self.value_adapter(queries[:, 0], own_id)
        logit = self.value_head.heads.value(value_query).squeeze(-1)
        value = 2.0 * logit.sigmoid() - 1.0
        return value, {
            "z_meta": z_meta,
            "meta_logits": meta_logits,
            "own_archetype_id": own_id,
        }

    def encode_with_strategy(self, features, own_archetype_id=None):
        """Mirror the training actor-critic boundary for parity tooling."""
        validated, state, value_options, policy_options = self._encode_options(features)
        value, auxiliary = self.value_and_aux_from_encoded(
            validated, state, value_options, own_archetype_id=own_archetype_id
        )
        context = StrategyContext.build(
            validated.global_cat[:, 2],
            auxiliary["z_meta"],
            auxiliary["meta_logits"],
            value,
            auxiliary["own_archetype_id"],
        )
        return validated, state, policy_options, value, auxiliary, context

    def _greedy_strategy(self, validated, state, options, context=None):
        decoder = self.actor.action_decoder
        decoder_state = decoder.initialize(validated, state.summary)
        maximum_steps = min(
            decoder.config.max_action_steps,
            validated.option_count,
            int(validated.max_count.max()),
        )
        sequences = torch.full(
            (validated.batch_size, maximum_steps), -1,
            dtype=torch.long, device=options.device,
        )
        active = torch.ones(validated.batch_size, dtype=torch.bool, device=options.device)
        legal = torch.ones_like(active)
        for step in range(maximum_steps):
            readout = decoder_state.hidden
            if self.policy_strategy_adapter is not None:
                if context is None:
                    raise RuntimeError("legacy strategy-conditioned policy has no context")
                readout, _ = self.policy_strategy_adapter(readout, context)
            if self.meta_actor_residual is not None:
                if context is None:
                    raise RuntimeError("Meta Actor Residual has no context")
                readout, _ = self.meta_actor_residual(
                    readout, context.own_archetype_id
                )
            scores = decoder.logits(
                validated, options, decoder_state, readout_hidden=readout
            )
            choice = scores.argmax(dim=1)
            chose_stop = choice.eq(validated.option_count)
            selecting = active & ~chose_stop
            legal &= ~(active & chose_stop & decoder_state.selected_count.lt(validated.min_count))
            active &= ~chose_stop
            if not bool(selecting.any()):
                break
            chosen = choice.clamp_max(validated.option_count - 1)
            sequences[selecting, step] = chosen[selecting]
            decoder_state = decoder.consume(
                options, decoder_state, torch.where(selecting, chosen, -1)
            )
            active &= ~decoder_state.selected_count.ge(validated.max_count)
        legal &= decoder_state.selected_count.ge(validated.min_count)
        return sequences, decoder_state.selected_count, legal

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
            validated, state, value_options, options = self._encode_options(batch)
            strategy_context = None
            if self.policy_strategy_adapter is not None:
                memory = torch.cat((state.tokens, value_options), dim=1)
                memory_mask = torch.cat((state.mask, validated.option_mask), dim=1)
                queries = self.value_head.decode(memory, memory_mask)
                z_meta = queries[:, 1]
                meta_logits = self.value_head.heads.archetype(z_meta)
                own_id = torch.full(
                    (validated.batch_size,), int(self.metadata["own_archetype_id"]),
                    dtype=torch.long, device=options.device,
                )
                adapted_value, _ = self.value_adapter(queries[:, 0], own_id)
                value = 2.0 * self.value_head.heads.value(
                    adapted_value
                ).squeeze(-1).sigmoid() - 1.0
                strategy_context = StrategyContext.build(
                    validated.global_cat[:, 2], z_meta, meta_logits, value, own_id
                )
            sequences, lengths, legal = self._greedy_strategy(
                validated, state, options, strategy_context
            )
        if not bool(legal[0]):
            raise RuntimeError("greedy decode violated selection bounds")
        action = sequences[0, : int(lengths[0])].tolist()

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
