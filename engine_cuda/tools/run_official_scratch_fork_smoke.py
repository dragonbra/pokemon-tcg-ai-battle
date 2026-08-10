"""Prove scratch-lane fork parity and source isolation on a CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ptcg_cuda_engine.native import create_official_engine, load_extension  # noqa: E402


def sha256_tensor(value) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clone_tensor_dict(values):
    return {name: value.clone() for name, value in values.items()}


def tensor_dict_equal(left, right) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    deck = [int(row) for row in args.deck.read_text(encoding="utf-8").splitlines() if row]
    if len(deck) != 60:
        raise ValueError("scratch smoke requires an exact 60-card deck")
    rule_pack = args.rules.read_bytes()
    source = create_official_engine(rule_pack, batch_size=4)
    scratch = create_official_engine(rule_pack, batch_size=2)
    decks = torch.tensor([[deck, deck]] * 4, dtype=torch.int64, device="cuda")
    seeds = torch.tensor([36001, 36002, 36003, 36004], dtype=torch.int64, device="cuda")
    source.reset_seeded_first_min_semantic(decks, seeds)
    torch.cuda.synchronize()
    source_before = source.state_bytes().clone()
    source_rng_before = source.rng_bytes().clone()
    source_status_before = source.statuses().clone()
    source_actions_before = source.action_bytes().clone()
    source_history_before = clone_tensor_dict(source.semantic_history_raw())
    # Duplicate one root so two hidden-information particles can be compared.
    indices = torch.tensor([3, 3], dtype=torch.int64, device="cuda")
    source.fork_lanes_to(scratch, indices)
    torch.cuda.synchronize()

    extension = load_extension()
    pointer_offset = int(extension.OFFICIAL_SEMANTIC_HISTORY_POINTER_OFFSET)
    pointer_bytes = int(extension.OFFICIAL_POINTER_BYTES)
    expected_state = source_before.index_select(0, indices).clone()
    scratch_before = scratch.state_bytes().clone()
    actual_state = scratch_before.clone()
    expected_state[:, pointer_offset : pointer_offset + pointer_bytes] = 0
    actual_state[:, pointer_offset : pointer_offset + pointer_bytes] = 0
    state_equal_ignoring_rebound_pointer = torch.equal(expected_state, actual_state)
    status_equal = torch.equal(source_status_before.index_select(0, indices), scratch.statuses())
    history_equal = all(
        torch.equal(value.index_select(0, indices), scratch.semantic_history_raw()[name])
        for name, value in source_history_before.items()
    )
    source_unchanged_after_fork = torch.equal(source_before, source.state_bytes())
    source_rng_unchanged_after_fork = torch.equal(source_rng_before, source.rng_bytes())

    # The learner-visible encodings must be invariant to hidden-membership and
    # future-RNG particles. Raw state_bytes remain privileged search internals.
    visible_policy_before = clone_tensor_dict(scratch.encode_policy_v1())
    lane_indices = torch.arange(2, dtype=torch.int64, device="cuda")
    visible_semantic_before = clone_tensor_dict(
        scratch.encode_semantic0031_v2_lanes(lane_indices)
    )
    exact_decks = torch.tensor([[deck, deck]] * 2, dtype=torch.int32, device="cuda")
    observer_seats = scratch.decision_actors().to(dtype=torch.int64).contiguous()
    membership_seeds = torch.tensor([91001, 91002], dtype=torch.int64, device="cuda")
    future_rng_seeds = torch.tensor([92001, 92002], dtype=torch.int64, device="cuda")
    belief_codes = scratch.redeterminize_public_belief_clean(
        exact_decks,
        observer_seats,
        membership_seeds,
        future_rng_seeds,
    )
    torch.cuda.synchronize()
    particle_states_without_rng = scratch.state_bytes().clone()
    rng_offset = int(extension.OFFICIAL_RNG_OFFSET)
    rng_bytes = int(extension.OFFICIAL_RNG_BYTES)
    particle_states_without_rng[:, rng_offset : rng_offset + rng_bytes] = 0
    hidden_particles_actually_differ = not torch.equal(
        particle_states_without_rng[0], particle_states_without_rng[1]
    )
    visible_policy_after = clone_tensor_dict(scratch.encode_policy_v1())
    visible_semantic_after = clone_tensor_dict(
        scratch.encode_semantic0031_v2_lanes(lane_indices)
    )
    belief_particles_accepted = torch.equal(
        belief_codes, torch.ones_like(belief_codes)
    )
    policy_visible_invariant = tensor_dict_equal(
        visible_policy_before, visible_policy_after
    ) and all(value[0].equal(value[1]) for value in visible_policy_after.values())
    semantic_visible_invariant = tensor_dict_equal(
        visible_semantic_before, visible_semantic_after
    ) and all(value[0].equal(value[1]) for value in visible_semantic_after.values())

    codec = scratch.encode_policy_v1()
    option_indices = torch.arange(128, dtype=torch.int64, device="cuda").expand(2, -1).contiguous()
    scratch.pack_actions(option_indices, codec["min_count"].contiguous())
    scratch.apply_packed_actions()
    torch.cuda.synchronize()
    source_unchanged_after_scratch_step = torch.equal(source_before, source.state_bytes())
    source_rng_unchanged_after_scratch_step = torch.equal(
        source_rng_before, source.rng_bytes()
    )
    source_status_unchanged = torch.equal(source_status_before, source.statuses())
    source_actions_unchanged = torch.equal(source_actions_before, source.action_bytes())
    source_history_unchanged = tensor_dict_equal(
        source_history_before, source.semantic_history_raw()
    )
    scratch_changed = not torch.equal(scratch_before, scratch.state_bytes())
    checks = {
        "state_equal_ignoring_rebound_pointer": state_equal_ignoring_rebound_pointer,
        "status_equal": status_equal,
        "semantic_history_equal": history_equal,
        "source_unchanged_after_fork": source_unchanged_after_fork,
        "source_rng_unchanged_after_fork": source_rng_unchanged_after_fork,
        "source_unchanged_after_scratch_step": source_unchanged_after_scratch_step,
        "source_rng_unchanged_after_scratch_step": source_rng_unchanged_after_scratch_step,
        "source_status_unchanged": source_status_unchanged,
        "source_actions_unchanged": source_actions_unchanged,
        "source_semantic_history_unchanged": source_history_unchanged,
        "belief_particles_accepted": belief_particles_accepted,
        "hidden_particles_actually_differ_excluding_rng": hidden_particles_actually_differ,
        "policy_visible_invariant_under_belief_particles": policy_visible_invariant,
        "semantic0031_visible_invariant_under_belief_particles": semantic_visible_invariant,
        "scratch_changed_after_step": scratch_changed,
    }
    if not all(checks.values()):
        raise RuntimeError(f"scratch-fork isolation gate failed: {checks}")
    result = {
        "schema_version": "cuda_scratch_fork_smoke_v1",
        "source_batch": 4,
        "scratch_batch": 2,
        "source_indices": [3, 3],
        "source_state_sha256": sha256_tensor(source_before),
        "source_rng_sha256": sha256_tensor(source_rng_before),
        "scratch_state_sha256_ignoring_pointer": sha256_tensor(actual_state),
        "belief_result_codes": belief_codes.detach().cpu().tolist(),
        "learner_interface": ["encode_policy_v1", "encode_semantic0031_v2_lanes"],
        "checks": checks,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
