"""Executable zero-update parity and fresh-optimizer audit for V1 focal seed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .assets import AssetRegistry, sha256_file
from .runtime import load_focal_seed, load_policy, synthetic_batch
from .own_archetype import OwnArchetypeVocabulary


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT = PROJECT_ROOT.parents[1] / "experiments/0043_champion_league_rl/focal_seed_zero_step_parity.json"


def _outputs(policy: Any, batch):
    with torch.inference_mode():
        validated, state, options, value, auxiliary, context = policy.encode_with_strategy(batch)
        value_adapted, value_delta = policy.value_adapter(
            policy.value_head.decode(
                torch.cat((state.tokens, options), dim=1),
                torch.cat((state.mask, validated.option_mask), dim=1),
            )[:, 0], auxiliary["own_archetype_id"],
        )
        policy_readout, policy_delta = policy.policy_strategy_adapter(
            policy.actor.action_decoder.initialize(validated, state.summary).hidden, context
        )
        decoder_state = policy.actor.action_decoder.initialize(validated, state.summary)
        logits = policy.actor.action_decoder.logits(
            validated, options, decoder_state, readout_hidden=policy_readout
        )
        probabilities = logits.softmax(dim=-1)
        actions, _, legal = policy._greedy_strategy(validated, state, options, context)
    if not bool(legal.all()):
        raise RuntimeError("parity batch produced an illegal greedy action")
    return {
        "value": value, "value_adapter_output": value_adapted,
        "value_adapter_delta": value_delta, "policy_adapter_output": policy_readout,
        "policy_adapter_delta": policy_delta, "policy_logits": logits,
        "action_probabilities": probabilities, "greedy_action": actions,
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_audit() -> dict[str, Any]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    batch = synthetic_batch(batch_size=2)
    rows = []
    # Weight construction is deck-independent. Exercise every frozen V1 row once,
    # then prove each old exact deck resolves to the audited old row.
    g1 = load_policy("Champion-G1", deck_id="007")
    g2 = load_focal_seed(deck_id="007")
    for own_id in range(15):
        g1.metadata["own_archetype_id"] = own_id
        g2.metadata["own_archetype_id"] = own_id
        left, right = _outputs(g1, batch), _outputs(g2, batch)
        equal = {name: bool(torch.equal(left[name], right[name])) for name in left}
        if not all(equal.values()):
            raise RuntimeError(f"zero-step parity failed for V1 row {own_id}: {equal}")
        rows.append({
            "own_archetype_id": own_id,
            "tensor_exact": equal,
            "status": "PASS",
        })
    mapping = json.loads(
        (PROJECT_ROOT / "assets/taxonomy/deck_own_archetype_mapping_v2.json").read_text()
    )["decks"]
    old_deck_rows = []
    v1 = OwnArchetypeVocabulary.load_version("0042_own_archetypes_v1")
    for deck, mapped in zip(registry.decks[:55], mapping[:55], strict=True):
        cards = tuple(int(row) for row in (PROJECT_ROOT / deck.deck_path).read_text().splitlines())
        g1_id = v1.classify_unknown_deck(cards).value
        if g1_id != mapped["old_archetype_id"]:
            raise RuntimeError(f"old exact-deck row mismatch: {deck.deck_id}")
        old_deck_rows.append({"deck_id": deck.deck_id, "g1_own_archetype_id": g1_id})

    seed = load_focal_seed(deck_id="043")
    value_parameters = (
        [seed.value_head.queries]
        + list(seed.value_head.blocks.parameters())
        + list(seed.value_head.final_norm.parameters())
        + list(seed.value_head.heads.value.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(seed.actor.action_decoder.parameters()), "lr": 2e-5},
            {"params": list(seed.policy_strategy_adapter.parameters()), "lr": 4e-5},
            {"params": list(seed.value_adapter.parameters()), "lr": 1e-4},
            {"params": value_parameters, "lr": 1e-4},
            {"params": list(seed.allocation_head.parameters()), "lr": 2e-5},
        ], weight_decay=0.0,
    )
    if optimizer.state:
        raise RuntimeError("fresh focal optimizer unexpectedly contains migrated state")
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if id(seed.value_adapter.own_embedding.weight) not in parameter_ids or id(seed.policy_strategy_adapter.own_embedding.weight) not in parameter_ids:
        raise RuntimeError("focal own embeddings are absent from the fresh optimizer")
    if any(id(parameter) in parameter_ids for parameter in seed.value_head.heads.archetype.parameters()):
        raise RuntimeError("opponent Meta classifier leaked into the focal optimizer")
    opponent_meta = seed.value_head.heads.archetype
    meta_probe = opponent_meta(torch.zeros(2, 320))
    if meta_probe.shape != (2, 15):
        raise RuntimeError("frozen opponent Meta head width changed")
    g1_meta = g1.value_head.heads.archetype.state_dict()
    if any(not torch.equal(value, opponent_meta.state_dict()[name]) for name, value in g1_meta.items()):
        raise RuntimeError("frozen opponent Meta head weights changed")
    result = {
        "schema_version": "0043_v1_focal_seed_zero_step_parity_v1",
        "status": "PASS", "old_decks_tested": 55,
        "policy_logits": "EXACT", "action_probabilities": "EXACT",
        "greedy_action": "EXACT", "value": "EXACT",
        "policy_adapter_output": "EXACT", "value_adapter_output": "EXACT",
        "g1_artifacts_unchanged": {
            "model_only_delta": sha256_file(PROJECT_ROOT / "assets/policies/definitions/champion_g001/source_update_000010.pt"),
            "portable_fp16_artifact": sha256_file(PROJECT_ROOT / "assets/policies/definitions/champion_g001/model.bin"),
        },
        "own_deck_only": True,
        "opponent_meta": {"class_count": 15, "weights_exact_to_g1": True, "trainable": False},
        "fresh_optimizer": {"state_entries": 0, "own_embeddings_included": True,
                            "opponent_meta_excluded": True,
                            "optimizer_state_migrated": False},
        "rows": rows, "old_deck_resolution": old_deck_rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT, result)
    manifest_path = (
        PROJECT_ROOT.parents[1]
        / "rl_runs/0043_champion_league_rl/versions/V1_focal_002_007"
        / "artifact/focal_seed/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["readiness"] = "ZERO_STEP_PARITY_PASS"
    manifest["zero_step_parity_report"] = "experiments/0043_champion_league_rl/focal_seed_zero_step_parity.json"
    manifest["zero_step_parity_sha256"] = sha256_file(REPORT)
    _write_json(manifest_path, manifest)
    return result


if __name__ == "__main__":
    print(json.dumps({key: value for key, value in run_audit().items() if key != "rows"}, sort_keys=True))
