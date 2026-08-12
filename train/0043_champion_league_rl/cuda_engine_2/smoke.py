"""Run the bounded 0043 CUDA Engine 2.0 end-to-end policy smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .arena import create_official_cuda_arena
from .build import DEFAULT_BUILD_DIR
from .inference import CudaPolicyCohort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def _deck(deck_id: str) -> list[int]:
    path = PROJECT_ROOT / f"assets/decks/definitions/{deck_id}/deck.csv"
    return [int(row) for row in path.read_text().splitlines()]


def run(rule_pack: Path, build_dir: Path = DEFAULT_BUILD_DIR) -> dict[str, object]:
    arena = create_official_cuda_arena(
        REPOSITORY_ROOT, rule_pack=rule_pack, batch_size=2, build_dir=build_dir,
    )
    decks = torch.tensor(
        [[_deck("001"), _deck("048")], [_deck("048"), _deck("001")]],
        dtype=torch.int32, device="cuda:0",
    )
    seeds = torch.tensor([430043001, 430043002], dtype=torch.int64, device="cuda:0")
    arena.engine.reset_seeded_first_min_semantic(decks, seeds)
    cohorts = (
        CudaPolicyCohort.load(PROJECT_ROOT, policy_id="Policy-0809", deck_id="001"),
        CudaPolicyCohort.load(PROJECT_ROOT, policy_id="Champion-G1", deck_id="048"),
    )
    decisions = 0
    semantic_shapes: dict[str, list[int]] | None = None
    for decisions in range(1, 1001):
        statuses = arena.engine.statuses()
        if bool(statuses.eq(2).all()):
            decisions -= 1
            break
        if bool(statuses.eq(3).any()):
            raise RuntimeError(f"official CUDA engine error: {statuses.cpu().tolist()}")
        ready_lanes = statuses.eq(1).nonzero(as_tuple=False).flatten().to(torch.int32)
        encoded = arena.engine.encode_semantic0031_v2_lanes(ready_lanes)
        if semantic_shapes is None:
            semantic_shapes = {
                "global_cat": list(encoded["global_cat"].shape),
                "option_cat": list(encoded["option_cat"].shape),
            }
        actors = arena.engine.decision_actors().index_select(0, ready_lanes.long()).long()
        # In both reverse-seat lanes, actor == lane identifies the deck-001
        # Policy-0809 seat; the other actor is the deck-048 Champion-G1 seat.
        route_anchor = ready_lanes.long().eq(actors)
        routed: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        action_width = 1
        for cohort, route in zip(cohorts, (route_anchor, ~route_anchor), strict=True):
            rows = route.nonzero(as_tuple=False).flatten()
            if rows.numel():
                result = cohort.greedy(
                    {name: value.index_select(0, rows) for name, value in encoded.items()}
                )
                routed.append((rows, result))
                action_width = max(action_width, int(result[0].shape[1]))
        sequences = torch.full(
            (2, action_width), -1, dtype=torch.long, device="cuda:0"
        )
        lengths = torch.zeros(2, dtype=torch.long, device="cuda:0")
        for rows, (cohort_sequences, cohort_lengths) in routed:
            lanes = ready_lanes.long().index_select(0, rows)
            sequences[lanes, : cohort_sequences.shape[1]] = cohort_sequences
            lengths[lanes] = cohort_lengths
        arena.engine.pack_actions(sequences, lengths)
        arena.engine.apply_packed_actions()
        arena.engine.advance_to_decision()
    else:
        raise RuntimeError("0043 CUDA games did not finish within 1,000 decisions")
    torch.cuda.synchronize()
    statuses = [int(value) for value in arena.engine.statuses().cpu().tolist()]
    if statuses != [2, 2]:
        raise RuntimeError(f"0043 CUDA transition smoke failed: statuses={statuses}")
    return {
        "schema_version": "0043_cuda_engine_2_policy_transition_smoke_v1",
        "status": "PASS",
        "engine_identity": arena.identity.to_manifest(),
        "lanes": [
            {
                "focal_deck_id": "001", "opponent_deck_id": "048",
                "policy_id": cohorts[0].policy_id,
                "effective_policy_sha256": cohorts[0].effective_policy_sha256,
            },
            {
                "focal_deck_id": "048", "opponent_deck_id": "001",
                "policy_id": cohorts[1].policy_id,
                "effective_policy_sha256": cohorts[1].effective_policy_sha256,
            },
        ],
        "semantic_shapes": semantic_shapes,
        "decisions": decisions,
        "terminal_statuses": statuses,
        "game_results": [int(value) for value in arena.engine.game_results().cpu().tolist()],
        "terminal_turns": [int(value) for value in arena.engine.turns().cpu().tolist()],
        "engine_errors": 0,
        "unfinished": 0,
        "cpu_fallbacks": 0,
        "scope": "two complete seeded official CUDA games; not CPU/CUDA parity or strength evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-pack", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args.rule_pack.resolve(), args.build_dir.resolve())
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
