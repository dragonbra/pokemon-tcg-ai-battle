"""Fail-closed per-lane exact-deck audits for resident Semantic0031 batches."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Sequence

import torch


def exact_deck_sha256(deck: Sequence[int]) -> str:
    cards = tuple(int(card) for card in deck)
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError("exact deck must contain 60 positive card IDs")
    return hashlib.sha256(
        ",".join(str(card) for card in sorted(cards)).encode("ascii")
    ).hexdigest()


def audit_exact_deck_rows(
    *,
    semantic: dict[str, torch.Tensor],
    ready: torch.Tensor,
    focal_route: torch.Tensor,
    lane_job: torch.Tensor,
    jobs: Sequence[Any],
    already_audited: set[tuple[int, str]],
) -> set[tuple[int, str]]:
    """Verify each newly observed job/role resource ledger against its own deck."""
    required = {"resource_cat", "resource_num", "resource_mask"}
    if required - set(semantic):
        raise RuntimeError("exact-deck routing audit lacks Semantic0031 resources")
    rows_tensor = ready.nonzero(as_tuple=False).flatten()
    rows = rows_tensor.tolist()
    job_indices = lane_job.index_select(0, rows_tensor).long().tolist()
    focal_flags = focal_route.index_select(0, rows_tensor).bool().tolist()
    additions: set[tuple[int, str]] = set()
    audit_rows: list[int] = []
    expected_rows: list[dict[int, int]] = []
    audit_keys: list[tuple[int, str]] = []
    for row, job_index, is_focal in zip(
        rows, job_indices, focal_flags, strict=True
    ):
        job_index = int(job_index)
        if not 0 <= job_index < len(jobs):
            raise RuntimeError("exact-deck routing audit received an invalid job index")
        is_focal = bool(is_focal)
        role = "focal" if is_focal else "opponent"
        key = (job_index, role)
        if key in already_audited or key in audit_keys:
            continue
        job = jobs[job_index]
        player = int(job.focal_player) if is_focal else 1 - int(job.focal_player)
        expected_counts = Counter(int(card) for card in job.decks[player])
        audit_rows.append(row)
        expected_rows.append(dict(sorted(expected_counts.items())))
        audit_keys.append(key)
    if not audit_rows:
        return additions
    capacity = semantic["resource_mask"].shape[1]
    expected_ids = torch.zeros(
        (len(audit_rows), capacity), dtype=torch.long, device=ready.device
    )
    expected_counts = torch.zeros(
        (len(audit_rows), capacity), dtype=torch.float32, device=ready.device
    )
    expected_mask = torch.zeros(
        (len(audit_rows), capacity), dtype=torch.bool, device=ready.device
    )
    for offset, counts in enumerate(expected_rows):
        if len(counts) > capacity:
            raise RuntimeError("exact deck exceeds Semantic0031 resource capacity")
        width = len(counts)
        expected_ids[offset, :width] = torch.tensor(
            list(counts), dtype=torch.long, device=ready.device
        )
        expected_counts[offset, :width] = torch.tensor(
            list(counts.values()), dtype=torch.float32, device=ready.device
        )
        expected_mask[offset, :width] = True
    selected = torch.tensor(audit_rows, dtype=torch.long, device=ready.device)
    actual_ids = semantic["resource_cat"].index_select(0, selected)[..., 0].long()
    actual_counts = semantic["resource_num"].index_select(0, selected)[..., 0].float()
    actual_mask = semantic["resource_mask"].index_select(0, selected).bool()
    mismatched = (
        actual_mask.ne(expected_mask).any(dim=1)
        | (actual_ids.ne(expected_ids) & expected_mask).any(dim=1)
        | (actual_counts.ne(expected_counts) & expected_mask).any(dim=1)
    )
    if bool(mismatched.any()):
        offset = int(mismatched.nonzero(as_tuple=False)[0])
        key = audit_keys[offset]
        mask = actual_mask[offset]
        actual = dict(zip(
            actual_ids[offset, mask].cpu().tolist(),
            actual_counts[offset, mask].round().long().cpu().tolist(),
            strict=True,
        ))
        job_index, role = key
        job = jobs[job_index]
        player = int(job.focal_player) if role == "focal" else 1 - int(job.focal_player)
        expected = expected_rows[offset]
        if actual != expected:
            raise RuntimeError(
                "FATAL: exact-deck routing mismatch: "
                f"job={job_index} role={role} "
                f"expected_sha256={exact_deck_sha256(job.decks[player])} "
                f"expected={expected} actual={actual}"
            )
    additions.update(audit_keys)
    return additions


__all__ = ["audit_exact_deck_rows", "exact_deck_sha256"]
