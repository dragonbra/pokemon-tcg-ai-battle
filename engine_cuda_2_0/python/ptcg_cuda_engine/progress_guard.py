"""GPU-resident progress guards for policy-created action cycles."""

from __future__ import annotations

from typing import Any


class DeviceRepeatForfeitGuard:
    """Forfeit after one actor starts 20 choices with the same option."""

    def __init__(
        self,
        *,
        batch_size: int,
        limit: int = 20,
        device: Any,
        slot_capacity: int = 64,
    ) -> None:
        import torch

        if batch_size < 1 or limit < 1 or slot_capacity < 1:
            raise ValueError("repeat guard dimensions and limit must be positive")
        self.batch_size = int(batch_size)
        self.limit = int(limit)
        self.device = torch.device(device)
        self.keys = torch.zeros(
            (batch_size, 2, slot_capacity, 6),
            dtype=torch.int64,
            device=self.device,
        )
        self.counts = torch.zeros(
            (batch_size, 2, slot_capacity), dtype=torch.int16, device=self.device
        )
        self.turn = torch.full(
            (batch_size,), -1, dtype=torch.int64, device=self.device
        )

    def reset(self, lane_mask: Any) -> None:
        """Clear repeat history for lanes assigned to a new Episode."""

        import torch

        mask = lane_mask.bool().view(-1)
        if mask.shape != (self.batch_size,):
            raise ValueError("repeat guard reset mask has incompatible shape")
        self.keys.masked_fill_(mask[:, None, None, None], 0)
        self.counts.masked_fill_(mask[:, None, None], 0)
        self.turn.masked_fill_(mask, -1)

    def observe(
        self,
        *,
        option_cat: Any,
        selection_type: Any,
        turn: Any,
        actor: Any,
        actions: Any,
        lengths: Any,
        ready: Any,
    ) -> Any:
        import torch

        if (
            option_cat.ndim != 3
            or option_cat.shape[0] != self.batch_size
            or option_cat.shape[2] < 19
            or actions.ndim != 2
            or actions.shape[0] != self.batch_size
        ):
            raise ValueError("repeat guard received incompatible option/action tensors")
        turn = turn.long().view(self.batch_size)
        actor = actor.long().view(self.batch_size)
        # The guard detects a policy-created loop inside one engine turn.  A
        # later turn is real game progress: once-per-turn Abilities may be used
        # again and must never accumulate toward a whole-game forfeit.
        changed_turn = self.turn.ge(0) & self.turn.ne(turn)
        self.keys.masked_fill_(changed_turn[:, None, None, None], 0)
        self.counts.masked_fill_(changed_turn[:, None, None], 0)
        self.turn.copy_(turn)
        valid_actor = actor.ge(0) & actor.lt(2)
        safe_actor = actor.clamp(min=0, max=1)

        first = actions[:, 0].long()
        option_count = option_cat.shape[1]
        valid_index = first.ge(0) & first.lt(option_count)
        safe = first.clamp(min=0, max=max(0, option_count - 1))
        selected = option_cat.gather(
            1, safe[:, None, None].expand(-1, 1, option_cat.shape[2])
        ).squeeze(1).long()
        eligible = (
            ready.bool().view(self.batch_size)
            & lengths.long().view(self.batch_size).ge(1)
            & valid_index
            & valid_actor
        )
        select_kind = selection_type.long().view(self.batch_size)

        # Canonicalize the selected Ability by stable semantic identity.  The
        # raw option row also contains presentation-order and target fields;
        # including those would split one repeatedly selected Ability whenever
        # the legal-option ordering changes between callbacks.
        identity = selected[:, (0, 5, 11, 12, 18)]
        key = torch.cat((identity, select_kind[:, None]), dim=1)
        rows = torch.arange(self.batch_size, device=self.device)
        actor_keys = self.keys[rows, safe_actor]
        actor_counts = self.counts[rows, safe_actor]
        matches = actor_keys.eq(key[:, None, :]).all(dim=2) & eligible[:, None]
        empty = actor_counts.eq(0)
        has_match = matches.any(dim=1)
        has_empty = empty.any(dim=1)
        match_slot = matches.long().argmax(dim=1)
        empty_slot = empty.long().argmax(dim=1)
        slot = torch.where(has_match, match_slot, empty_slot)
        tracked = eligible & (has_match | has_empty)
        previous = actor_counts[rows, slot].long()
        updated = previous + tracked.long()
        self.keys[
            rows[tracked], safe_actor[tracked], slot[tracked], :
        ] = key[tracked]
        self.counts[
            rows[tracked], safe_actor[tracked], slot[tracked]
        ] = updated[tracked].to(torch.int16)
        return tracked & updated.ge(self.limit)


def apply_loop_forfeits(engine: Any, forfeit_mask: Any, losing_players: Any) -> None:
    """Mark guarded lanes terminal with the other physical player as winner."""

    import torch

    mask = forfeit_mask.bool().view(-1)
    losing = losing_players.long().view(-1)
    if mask.shape != losing.shape:
        raise ValueError("forfeit mask and losing-player tensors disagree")
    winners = torch.where(losing.eq(0), 2, 1).to(torch.uint8)
    engine.game_results().copy_(
        torch.where(mask, winners, engine.game_results())
    )
    engine.statuses().copy_(
        torch.where(mask, torch.full_like(engine.statuses(), 2), engine.statuses())
    )


__all__ = ["DeviceRepeatForfeitGuard", "apply_loop_forfeits"]
