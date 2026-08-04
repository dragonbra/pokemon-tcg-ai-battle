from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def foundation_r15_static_fields(registered_deck: list[int], device: Any) -> dict[str, Any]:
    """Build resident, deck-specific 0020 foundation side-channel tensors.

    This is the neutral zero-history contract used by the CUDA resident smoke:
    registered deck facts are real and device-resident, while causal ledger,
    event, and known-hand streams start empty.  A later device ledger can
    replace these tensors without changing the policy interface.
    """

    import torch

    deck = [int(card) for card in registered_deck]
    if len(deck) != 60 or any(card <= 0 for card in deck):
        raise ValueError("registered deck must contain 60 positive card IDs")
    counts = Counter(deck)
    card_ids = sorted(counts)
    width = len(card_ids)
    return {
        "registered_card_ids": torch.tensor(card_ids, dtype=torch.long, device=device).view(1, width),
        "registered_multiplicity": torch.tensor(
            [counts[card] for card in card_ids],
            dtype=torch.float32,
            device=device,
        ).view(1, width),
        "registered_mask": torch.ones((1, width), dtype=torch.bool, device=device),
        "ledger_cat": torch.zeros((1, width, 4), dtype=torch.long, device=device),
        "ledger_num": torch.zeros((1, width, 15), dtype=torch.float32, device=device),
        "ledger_mask": torch.ones((1, width), dtype=torch.bool, device=device),
        "event_cat": torch.zeros((1, 1, 8), dtype=torch.long, device=device),
        "event_num": torch.zeros((1, 1, 4), dtype=torch.float32, device=device),
        "event_mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
        "known_opponent_hand_card_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
        "known_opponent_hand_mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
        "unknown_opponent_hand_count": torch.zeros((1,), dtype=torch.float32, device=device),
        "source_id": torch.zeros((1,), dtype=torch.long, device=device),
    }


def foundation_r15_static_fields_for_decks(
    registered_decks: list[list[int]], device: Any
) -> dict[str, Any]:
    """Build one padded, device-resident foundation side-channel batch.

    The foundation weights are shared across every row; only the registered
    deck facts vary by lane.  All rows use the same fixed neutral-history
    contract as :func:`foundation_r15_static_fields` until a resident causal
    ledger is available.  Construction is intentionally outside the rollout
    hot path, so the Python deck lists are never consulted during inference or
    PPO updates.
    """

    import torch

    decks = [[int(card) for card in deck] for deck in registered_decks]
    if not decks:
        raise ValueError("registered_decks must not be empty")
    counters = [Counter(deck) for deck in decks]
    if any(len(deck) != 60 or any(card <= 0 for card in deck) for deck in decks):
        raise ValueError("every registered deck must contain 60 positive card IDs")
    card_rows = [sorted(counter) for counter in counters]
    width = max(len(row) for row in card_rows)
    batch_size = len(decks)
    card_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    multiplicity = torch.zeros(
        (batch_size, width), dtype=torch.float32, device=device
    )
    registered_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    for row, (ids, counter) in enumerate(zip(card_rows, counters)):
        count = len(ids)
        card_ids[row, :count] = torch.tensor(ids, dtype=torch.long, device=device)
        multiplicity[row, :count] = torch.tensor(
            [counter[card] for card in ids], dtype=torch.float32, device=device
        )
        registered_mask[row, :count] = True
    return {
        "registered_card_ids": card_ids,
        "registered_multiplicity": multiplicity,
        "registered_mask": registered_mask,
        "zone_inventory_num": torch.zeros(
            (batch_size, 2, 16), dtype=torch.float32, device=device
        ),
        "ledger_cat": torch.zeros(
            (batch_size, width, 4), dtype=torch.long, device=device
        ),
        "ledger_num": torch.zeros(
            (batch_size, width, 15), dtype=torch.float32, device=device
        ),
        "ledger_mask": registered_mask.clone(),
        "event_cat": torch.zeros(
            (batch_size, 1, 8), dtype=torch.long, device=device
        ),
        "event_num": torch.zeros(
            (batch_size, 1, 4), dtype=torch.float32, device=device
        ),
        "event_mask": torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
        "known_opponent_hand_card_ids": torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        ),
        "known_opponent_hand_mask": torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=device
        ),
        "unknown_opponent_hand_count": torch.zeros(
            (batch_size,), dtype=torch.float32, device=device
        ),
        "source_id": torch.zeros((batch_size,), dtype=torch.long, device=device),
    }


class StaticBatchFieldsDeviceAdapter:
    """Inject policy-specific resident tensors into a device adapter.

    Deck-conditioned legacy checkpoints expect their registered 60-card deck
    on every row.  The one-row tensors are uploaded once at construction and
    expanded as views for each fixed route cohort; no rollout-time H2D copy or
    host materialization is required.
    """

    def __init__(self, adapter: Any, fields: Mapping[str, Any]) -> None:
        if not fields:
            raise ValueError("static device fields must not be empty")
        self.adapter = adapter
        self.fields = dict(fields)

    @property
    def outputs_normalized(self) -> bool:
        return bool(getattr(self.adapter, "outputs_normalized", False))

    @property
    def shared_batch_key(self) -> Any:
        return getattr(self.adapter, "shared_batch_key", None)

    @property
    def shared_static_fields(self) -> Mapping[str, Any]:
        return self.fields

    def act_device_shared_static(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        return self.adapter.act_device(batch)

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        option_mask = batch["option_mask"]
        batch_size = option_mask.shape[0]
        augmented = dict(batch)
        for name, value in self.fields.items():
            if value.device != option_mask.device:
                raise ValueError(f"static field {name} is on the wrong device")
            if value.ndim == 0:
                augmented[name] = value.expand(batch_size)
            elif value.shape[0] == 1:
                augmented[name] = value.expand(batch_size, *value.shape[1:])
            else:
                raise ValueError(
                    f"static field {name} must have a singleton batch axis"
                )
        return self.adapter.act_device(augmented)


class AutocastDeviceAdapter:
    """Run a resident adapter under one explicitly frozen CUDA math dtype."""

    def __init__(self, adapter: Any, dtype: Any) -> None:
        self.adapter = adapter
        self.dtype = dtype

    @property
    def outputs_normalized(self) -> bool:
        return bool(getattr(self.adapter, "outputs_normalized", False))

    @property
    def shared_batch_key(self) -> Any:
        return getattr(self.adapter, "shared_batch_key", None)

    @property
    def shared_static_fields(self) -> Mapping[str, Any] | None:
        return getattr(self.adapter, "shared_static_fields", None)

    def act_device_shared_static(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import torch

        if batch["option_mask"].device.type != "cuda":
            raise ValueError("CUDA autocast adapter requires a CUDA batch")
        method = getattr(self.adapter, "act_device_shared_static", None)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            if method is not None:
                return method(batch)
            return self.adapter.act_device(batch)

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import torch

        if batch["option_mask"].device.type != "cuda":
            raise ValueError("CUDA autocast adapter requires a CUDA batch")
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.adapter.act_device(batch)


class CompiledDeviceAdapter:
    """Compile an adapter's resident action function as an opt-in experiment."""

    def __init__(
        self,
        adapter: Any,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> None:
        self.adapter = adapter
        self.mode = mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self._compiled: Any | None = None
        self._compiled_shared: Any | None = None

    @property
    def outputs_normalized(self) -> bool:
        return bool(getattr(self.adapter, "outputs_normalized", False))

    @property
    def shared_batch_key(self) -> Any:
        return getattr(self.adapter, "shared_batch_key", None)

    @property
    def shared_static_fields(self) -> Mapping[str, Any] | None:
        return getattr(self.adapter, "shared_static_fields", None)

    def _act_device_shared_static(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        method = getattr(self.adapter, "act_device_shared_static", None)
        if method is not None:
            return method(batch)
        return self.adapter.act_device(batch)

    def act_device_shared_static(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import torch

        if self._compiled_shared is None:
            self._compiled_shared = torch.compile(
                self._act_device_shared_static,
                mode=self.mode,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
            )
        return self._compiled_shared(batch)

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import torch

        if self._compiled is None:
            self._compiled = torch.compile(
                self.adapter.act_device,
                mode=self.mode,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
            )
        return self._compiled(batch)


class EntityPointerPolicyV1DeviceAdapter:
    """Greedy, fixed-shape device decoder for EntityPointerPolicy checkpoints."""

    outputs_normalized = True

    def __init__(self, model: Any, *, max_select: int = 80) -> None:
        if max_select <= 0:
            raise ValueError("max_select must be positive")
        self.model = model
        self.max_select = max_select

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import torch

        option_mask = batch["option_mask"].bool()
        if option_mask.ndim != 2:
            raise ValueError("option_mask must be rank two")
        batch_size, option_count = option_mask.shape
        if option_count <= 0:
            raise ValueError("option capacity must be positive")

        route_mask = batch.get("_route_mask")
        if route_mask is None:
            route_mask = torch.ones(batch_size, dtype=torch.bool, device=option_mask.device)
        else:
            route_mask = route_mask.bool().view(batch_size)

        outputs = self.model.encode(dict(batch))
        option_repr = outputs["option_repr"]
        hidden = self.model.initial_decoder_state(outputs["state_repr"])
        selected_mask = torch.zeros_like(option_mask)
        actions = torch.full(
            (batch_size, self.max_select),
            -1,
            dtype=torch.long,
            device=option_mask.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=option_mask.device)
        min_count = batch["min_count"].long().view(batch_size)
        max_count = batch["max_count"].long().view(batch_size).clamp(
            min=0,
            max=self.max_select,
        )
        active = route_mask & max_count.gt(0)

        for step in range(self.max_select):
            valid_options = option_mask & ~selected_mask & active.unsqueeze(1)
            stop_valid = active & min_count.le(step)
            logits = self.model.pointer_logits(
                option_repr,
                hidden,
                valid_options,
                stop_valid,
            )
            choice = logits.argmax(dim=1)
            chosen_valid = active & choice.lt(option_count)
            safe_choice = choice.clamp(min=0, max=option_count - 1)
            actions[:, step] = torch.where(chosen_valid, safe_choice, -1)
            previously_selected = selected_mask.gather(1, safe_choice.unsqueeze(1))
            selected_mask.scatter_(
                1,
                safe_choice.unsqueeze(1),
                previously_selected | chosen_valid.unsqueeze(1),
            )
            lengths = lengths + chosen_valid.long()
            active = chosen_valid & lengths.lt(max_count) & route_mask
            selected_repr = option_repr.gather(
                1,
                safe_choice.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
            ).squeeze(1)
            hidden = self.model.advance_decoder(selected_repr, hidden, chosen_valid)

        return actions, lengths


class IDOnlyPointerPolicyDeviceAdapter:
    """Fixed-shape greedy decoder shared by the ID-only BC model family."""

    outputs_normalized = True

    def __init__(self, model: Any, *, max_select: int = 16) -> None:
        if max_select <= 0:
            raise ValueError("max_select must be positive")
        self.model = model
        self.max_select = max_select

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import math

        import torch

        option_mask = batch["option_mask"].bool()
        if option_mask.ndim != 2:
            raise ValueError("option_mask must be rank two")
        batch_size, option_count = option_mask.shape
        if option_count <= 0:
            raise ValueError("option capacity must be positive")
        route_mask = batch.get("_route_mask")
        if route_mask is None:
            route_mask = torch.ones(batch_size, dtype=torch.bool, device=option_mask.device)
        else:
            route_mask = route_mask.bool().view(batch_size)

        encoded = self.model.encode(dict(batch))
        state_repr, option_repr = encoded[0], encoded[1]
        option_keys = self.model.pointer_key(option_repr)
        hidden = torch.tanh(self.model.decoder_init(state_repr))
        decoder_layers = int(getattr(self.model.config, "decoder_layers", 1))
        if decoder_layers > 1:
            hidden = hidden.view(batch_size, decoder_layers, -1)

        selected_mask = torch.zeros_like(option_mask)
        actions = torch.full(
            (batch_size, self.max_select),
            -1,
            dtype=torch.long,
            device=option_mask.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=option_mask.device)
        min_count = batch["min_count"].long().view(batch_size)
        max_count = batch["max_count"].long().view(batch_size).clamp(
            min=0,
            max=self.max_select,
        )
        active = route_mask & max_count.gt(0)

        for step in range(self.max_select):
            query_hidden = hidden[:, -1] if hidden.ndim == 3 else hidden
            pointer = (
                self.model.pointer_query(query_hidden).unsqueeze(1) * option_keys
            ).sum(-1) / math.sqrt(self.model.config.d_model)
            pointer = pointer + self.model.option_bias(option_repr).squeeze(-1)
            valid_options = option_mask & ~selected_mask & active.unsqueeze(1)
            pointer = pointer.masked_fill(
                ~valid_options,
                torch.finfo(pointer.dtype).min,
            )
            stop = self.model.stop(query_hidden).squeeze(-1)
            stop_valid = active & min_count.le(step)
            stop = stop.masked_fill(~stop_valid, torch.finfo(stop.dtype).min)
            choice = torch.cat([pointer, stop.unsqueeze(1)], dim=1).argmax(dim=1)
            chosen_valid = active & choice.lt(option_count)
            safe_choice = choice.clamp(min=0, max=option_count - 1)
            actions[:, step] = torch.where(chosen_valid, safe_choice, -1)
            previously_selected = selected_mask.gather(1, safe_choice.unsqueeze(1))
            selected_mask.scatter_(
                1,
                safe_choice.unsqueeze(1),
                previously_selected | chosen_valid.unsqueeze(1),
            )
            lengths = lengths + chosen_valid.long()
            active = chosen_valid & lengths.lt(max_count) & route_mask
            selected_repr = option_repr.gather(
                1,
                safe_choice.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
            ).squeeze(1)
            if hasattr(self.model, "decode_step"):
                candidate_hidden = self.model.decode_step(selected_repr, hidden)
            else:
                candidate_hidden = self.model.decoder(selected_repr, hidden)
            update_shape = (-1, 1, 1) if hidden.ndim == 3 else (-1, 1)
            hidden = torch.where(
                chosen_valid.view(*update_shape),
                candidate_hidden,
                hidden,
            )

        return actions, lengths


class FoundationR15DeviceAdapter:
    """Device greedy adapter for the 0020 source-conditioned R15 foundation."""

    outputs_normalized = True

    def __init__(self, model: Any, *, max_select: int = 64) -> None:
        if max_select <= 0:
            raise ValueError("max_select must be positive")
        self.model = model
        self.max_select = max_select

    @property
    def shared_batch_key(self) -> tuple[str, int, int]:
        return ("foundation_r15", id(self.model), self.max_select)

    @staticmethod
    def _with_neutral_side_channels(batch: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        option_mask = batch["option_mask"]
        batch_size = int(option_mask.shape[0])
        device = option_mask.device
        out = dict(batch)
        if "zone_inventory_num" not in out:
            out["zone_inventory_num"] = torch.zeros(
                (batch_size, 2, 16), dtype=torch.float32, device=device
            )
        for required in (
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
            "unknown_opponent_hand_count",
            "source_id",
        ):
            if required not in out:
                raise KeyError(f"foundation R15 batch is missing {required}")
        return out

    def act_device(self, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        import math

        import torch

        batch = self._with_neutral_side_channels(batch)
        option_mask = batch["option_mask"].bool()
        if option_mask.ndim != 2:
            raise ValueError("option_mask must be rank two")
        batch_size, option_count = option_mask.shape
        if option_count <= 0:
            raise ValueError("option capacity must be positive")
        route_mask = batch.get("_route_mask")
        if route_mask is None:
            route_mask = torch.ones(batch_size, dtype=torch.bool, device=option_mask.device)
        else:
            route_mask = route_mask.bool().view(batch_size)

        state_repr, option_repr = self.model.encode(dict(batch))
        option_keys = self.model.pointer_key(option_repr)
        hidden = torch.tanh(self.model.decoder_init(state_repr))
        selected_mask = torch.zeros_like(option_mask)
        actions = torch.full(
            (batch_size, self.max_select),
            -1,
            dtype=torch.long,
            device=option_mask.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=option_mask.device)
        min_count = batch["min_count"].long().view(batch_size)
        max_count = batch["max_count"].long().view(batch_size).clamp(
            min=0,
            max=self.max_select,
        )
        active = route_mask & max_count.gt(0)

        for step in range(self.max_select):
            pointer = (
                self.model.pointer_query(hidden).unsqueeze(1) * option_keys
            ).sum(-1) / math.sqrt(self.model.config.d_model)
            pointer = pointer + self.model.option_bias(option_repr).squeeze(-1)
            valid_options = option_mask & ~selected_mask & active.unsqueeze(1)
            pointer = pointer.masked_fill(
                ~valid_options,
                torch.finfo(pointer.dtype).min,
            )
            stop = self.model.stop(hidden).squeeze(-1)
            stop_valid = active & min_count.le(step)
            stop = stop.masked_fill(~stop_valid, torch.finfo(stop.dtype).min)
            choice = torch.cat([pointer, stop.unsqueeze(1)], dim=1).argmax(dim=1)
            chosen_valid = active & choice.lt(option_count)
            safe_choice = choice.clamp(min=0, max=option_count - 1)
            actions[:, step] = torch.where(chosen_valid, safe_choice, -1)
            previously_selected = selected_mask.gather(1, safe_choice.unsqueeze(1))
            selected_mask.scatter_(
                1,
                safe_choice.unsqueeze(1),
                previously_selected | chosen_valid.unsqueeze(1),
            )
            lengths = lengths + chosen_valid.long()
            active = chosen_valid & lengths.lt(max_count) & route_mask
            selected_repr = option_repr.gather(
                1,
                safe_choice.view(-1, 1, 1).expand(-1, 1, option_repr.size(-1)),
            ).squeeze(1)
            candidate_hidden = self.model.decoder(selected_repr, hidden)
            hidden = torch.where(chosen_valid.unsqueeze(-1), candidate_hidden, hidden)

        return actions, lengths
