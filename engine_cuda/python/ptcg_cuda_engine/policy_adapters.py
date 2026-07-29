from __future__ import annotations

from typing import Any, Mapping


class EntityPointerPolicyV1DeviceAdapter:
    """Greedy, fixed-shape device decoder for EntityPointerPolicy checkpoints."""

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
