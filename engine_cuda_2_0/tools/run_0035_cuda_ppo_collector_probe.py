"""CUDA-resident 0035 semantic PPO collector contract probe.

This is deliberately a small diagnostic gate, rather than a policy-strength
experiment or a formal RL version.  It verifies that the frozen 50-deck
snapshot can be materialized as CUDA lanes, that the 0031 model consumes the
engine's semantic-v2 tensors directly, and that sampled learner trajectories
round-trip through a device-side PPO update without moving rollout tensors to
the host.

The current compact-checkpoint loader lives with the historical 0034 runtime,
so this probe imports it only to recover its immutable 0031 model definition.
Any formal CUDA-native continuation must first copy that implementation into a
new numbered project and record a new formal version.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.cuda_ppo import CudaRolloutBuffer, ppo_update_device  # noqa: E402
from ptcg_cuda_engine.native import create_official_engine  # noqa: E402
from ptcg_cuda_engine.semantic0031_bridge import semantic0031_v2_ready_batch  # noqa: E402


NEEDS_ACTION = 1
TERMINAL = 2
ERROR = 3
EXPECTED_CHECKPOINT_SHA256 = (
    "83b4ad133e414ca14c7602c0aec3920bf86439253e22d77d658c68ceb8892f7a"
)
DEFAULT_PROJECT = "0035_lucario_semantic0031_cpu_ppo"
DEFAULT_VERSION = "V4_lucario_shared0031_cpu_512u_first50_20260807"


def parse_args() -> argparse.Namespace:
    private = CUDA_ENGINE_ROOT / "generated" / "private" / "official_3aaeaa92"
    parser = argparse.ArgumentParser(
        description="Run the 50-opponent CUDA semantic PPO collector contract probe."
    )
    parser.add_argument("--rules", type=Path, default=private / "official_rules.bin")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "rl_runs"
            / DEFAULT_PROJECT
            / "versions"
            / DEFAULT_VERSION
            / "artifact/opponent_snapshot.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "bc_models/semantic0031_0806_shared_prototype_fp32.pt",
    )
    parser.add_argument(
        "--focal-deck",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "train/0034_dragapult_third_large_model_rl/league/decks"
            / "mega_lucario_ex_solrock_77a53ffc32f8/deck.csv"
        ),
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--deck-limit",
        type=int,
        default=50,
        help="Use the first N audited snapshot decks; default covers all 50 once.",
    )
    parser.add_argument(
        "--require-terminal",
        action="store_true",
        help="Fail unless at least one selected lane reaches an official terminal result.",
    )
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--seed", type=int, default=350031001)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1.0e-5)
    parser.add_argument("--value-lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / ".tmp/enginecuda_ppo/0035_collector_probe.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_deck(path: Path) -> tuple[int, ...]:
    cards = tuple(int(value) for value in path.read_text(encoding="utf-8").splitlines() if value)
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError(f"expected an exact 60-card deck: {path}")
    return cards


def _load_first50_snapshot(path: Path) -> tuple[list[dict[str, Any]], Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("opponents") if isinstance(payload, dict) else None
    if (
        payload.get("selected_count") != 50
        or payload.get("source_pool_count") != 55
        or payload.get("source_pool_id") != "0806_kaggle_top100_plus_v1"
        or not isinstance(rows, list)
        or len(rows) != 50
    ):
        raise ValueError("snapshot is not the audited Frozen-0806 first-50 pool")

    league = importlib.import_module(
        "train.0034_dragapult_third_large_model_rl.league"
    )
    catalog = list(league.load_frozen_catalog())
    expected = catalog[:50]
    if [str(row.get("deck_id")) for row in rows] != [item.deck_id for item in expected]:
        raise ValueError("first-50 snapshot deck order disagrees with frozen catalog")
    if [str(row.get("deck_sha256")) for row in rows] != [item.deck_sha256 for item in expected]:
        raise ValueError("first-50 snapshot deck hashes disagree with frozen catalog")
    return rows, expected


def _semantic_mapping(batch: Mapping[str, Any], expected: frozenset[str]) -> dict[str, Any]:
    return {name: batch[name] for name in expected}


_SEMANTIC_PREFIX_FAMILIES = {
    "card": (
        "card_cat", "card_num", "card_state", "card_mask", "card_parent",
    ),
    "resource": (
        "resource_cat", "resource_num", "resource_state", "resource_mask",
    ),
    "event": (
        "event_cat", "event_num", "event_state", "event_mask", "event_source",
        "event_target", "event_before", "event_after",
    ),
    "option": (
        "option_cat", "option_num", "option_state", "option_mask", "option_source",
        "option_target", "option_context", "option_effect_card",
    ),
    "option_skill": (
        "option_skill_id", "option_skill_role", "option_skill_parent", "option_skill_mask",
    ),
    "option_effect": (
        "option_effect_id", "option_effect_role", "option_effect_parent", "option_effect_mask",
    ),
}
_SEMANTIC_PREFIX_MASKS = {
    "card": "card_mask",
    "resource": "resource_mask",
    "event": "event_mask",
    "option": "option_mask",
    "option_skill": "option_skill_mask",
    "option_effect": "option_effect_mask",
}


def _compact_semantic_prefixes(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Trim only right-padding that is excluded by the semantic masks."""

    import torch

    mask_names = [
        _SEMANTIC_PREFIX_MASKS[family] for family in _SEMANTIC_PREFIX_FAMILIES
    ]
    maxima = torch.stack(
        [batch[name].long().sum(dim=1).amax() for name in mask_names]
    ).tolist()
    widths = {
        family: max(1, int(maximum))
        for family, maximum in zip(_SEMANTIC_PREFIX_FAMILIES, maxima, strict=True)
    }
    output = dict(batch)
    for family, fields in _SEMANTIC_PREFIX_FAMILIES.items():
        width = widths[family]
        for name in fields:
            output[name] = batch[name][:, :width]
    return output, widths


class CudaSemantic0031ActorCritic:  # instantiated as a torch module below
    """Minimal device PPO protocol around the frozen 0031 representation."""

    @staticmethod
    def build(source_model: Any, *, max_select: int) -> Any:
        import torch
        from torch import nn

        class _ActorCritic(nn.Module):
            def __init__(self, policy: Any, limit: int) -> None:
                super().__init__()
                self.policy = policy
                self.max_select = int(limit)
                self.expected = frozenset(policy.expected_batch_keys)
                width = int(policy.config.d_model)
                self.value_head = nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width),
                    nn.GELU(),
                    nn.Linear(width, 1),
                    nn.Tanh(),
                )
                final = self.value_head[-2]
                assert isinstance(final, nn.Linear)
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
                self.policy.requires_grad_(False)
                self.policy.action_decoder.requires_grad_(True)
                self._prototype_memory = None
                self.prototype_cache_builds = 0
                self.prototype_cache_bytes = 0

            def clear_prototype_cache(self) -> None:
                """Discard the transient frozen-representation cache."""

                self._prototype_memory = None
                self.prototype_cache_bytes = 0

            def _frozen_prototype_memory(self) -> Any:
                encoder = self.policy.prototype_encoder
                if any(parameter.requires_grad for parameter in encoder.parameters()):
                    raise RuntimeError("prototype cache requires a frozen prototype encoder")
                parameter = next(encoder.parameters())
                cached = self._prototype_memory
                if (
                    cached is None
                    or cached.cards.device != parameter.device
                    or cached.cards.dtype != parameter.dtype
                ):
                    with torch.no_grad():
                        memory = encoder.encode_all()
                    self._prototype_memory = memory
                    self.prototype_cache_builds += 1
                    self.prototype_cache_bytes = sum(
                        int(value.numel() * value.element_size())
                        for value in (
                            memory.cards,
                            memory.attacks,
                            memory.skills,
                            memory.effects,
                        )
                    )
                return self._prototype_memory

            def _encode(
                self,
                batch: Mapping[str, Any],
                *,
                use_prototype_cache: bool = True,
            ) -> tuple[Any, Any, Any]:
                semantic = semantic0031_v2_ready_batch(
                    _semantic_mapping(batch, self.expected),
                    max_action_steps=self.max_select,
                )
                with torch.no_grad():
                    validated = self.policy.validate_batch(semantic)
                    prototype_memory = (
                        self._frozen_prototype_memory()
                        if use_prototype_cache
                        else self.policy.prototype_encoder.encode_all()
                    )
                    state = self.policy.state_encoder(validated, prototype_memory)
                    options = self.policy.option_encoder(
                        validated, state, prototype_memory
                    )
                return validated, state.summary.detach(), options.detach()

            def encode(self, batch: Mapping[str, Any]) -> tuple[Any, Any, Any]:
                return self._encode(batch)

            def encode_uncached(self, batch: Mapping[str, Any]) -> tuple[Any, Any, Any]:
                return self._encode(batch, use_prototype_cache=False)

            @staticmethod
            def _evaluate_decoder(
                decoder: Any,
                value_head: Any,
                batch: Any,
                summary: Any,
                options: Any,
                targets: Any,
                lengths: Any,
                stopped: Any,
                max_select: int,
            ) -> tuple[Any, Any, Any]:
                state = decoder.initialize(batch, summary)
                total_logprob = torch.zeros(batch.batch_size, device=options.device)
                total_entropy = torch.zeros_like(total_logprob)
                option_count = int(options.shape[1])
                option_keys = decoder.key(options)
                option_bias = decoder.option_bias(options).squeeze(-1)
                required_steps = int(
                    (lengths + stopped.to(lengths.dtype)).amax().item()
                )
                for step in range(min(max_select, required_steps)):
                    selecting = lengths.gt(step)
                    choosing_stop = stopped & lengths.eq(step)
                    active = selecting | choosing_stop
                    logits = decoder.logits(
                        batch,
                        options,
                        state,
                        option_keys=option_keys,
                        option_bias=option_bias,
                    )
                    raw = targets[:, step]
                    if bool((selecting & (raw.lt(0) | raw.ge(option_count))).any()):
                        raise ValueError("rollout target is outside the legal option range")
                    token = torch.where(selecting, raw, torch.full_like(raw, option_count))
                    distribution = torch.distributions.Categorical(logits=logits.float())
                    total_logprob += torch.where(active, distribution.log_prob(token), 0.0)
                    total_entropy += torch.where(active, distribution.entropy(), 0.0)
                    state = decoder.consume(options, state, torch.where(selecting, raw, -1))
                return total_logprob, total_entropy, value_head(summary).squeeze(-1)

            def evaluate_targets_from_encoding(
                self, batch: Mapping[str, Any], encoded: tuple[Any, Any, Any]
            ) -> tuple[Any, Any]:
                validated, summary, options = encoded
                logprob, _entropy, value = self._evaluate_decoder(
                    self.policy.action_decoder,
                    self.value_head,
                    validated,
                    summary,
                    options,
                    batch["targets"],
                    batch["sequence_lengths"],
                    batch["sequence_stopped"].bool(),
                    self.max_select,
                )
                return logprob, value

            def target_entropy_from_encoding(
                self, batch: Mapping[str, Any], encoded: tuple[Any, Any, Any]
            ) -> Any:
                validated, summary, options = encoded
                _logprob, entropy, _value = self._evaluate_decoder(
                    self.policy.action_decoder,
                    self.value_head,
                    validated,
                    summary,
                    options,
                    batch["targets"],
                    batch["sequence_lengths"],
                    batch["sequence_stopped"].bool(),
                    self.max_select,
                )
                return entropy.mean()

            def evaluate_targets_with_entropy_from_encoding(
                self, batch: Mapping[str, Any], encoded: tuple[Any, Any, Any]
            ) -> tuple[Any, Any, Any]:
                validated, summary, options = encoded
                logprob, entropy, value = self._evaluate_decoder(
                    self.policy.action_decoder,
                    self.value_head,
                    validated,
                    summary,
                    options,
                    batch["targets"],
                    batch["sequence_lengths"],
                    batch["sequence_stopped"].bool(),
                    self.max_select,
                )
                return logprob, entropy.mean(), value

        return _ActorCritic(source_model, max_select)


def _decode_device(
    decoder: Any,
    batch: Any,
    summary: Any,
    options: Any,
    *,
    max_select: int,
    greedy: bool,
    route_mask: Any | None = None,
    trim_to_routed_max_count: bool = True,
    compute_entropy: bool = False,
) -> dict[str, Any]:
    """Sample or greedily decode legal ordered actions without host action data."""

    import torch

    state = decoder.initialize(batch, summary)
    batch_size, option_count, _ = options.shape
    actions = torch.full(
        (batch_size, max_select), -1, dtype=torch.long, device=options.device
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
    logprob = torch.zeros(batch_size, device=options.device)
    entropy = torch.zeros_like(logprob)
    route = (
        torch.ones(batch_size, dtype=torch.bool, device=options.device)
        if route_mask is None
        else route_mask.bool().view(batch_size)
    )
    active = route & batch.max_count.gt(0)
    # The option projections are invariant across autoregressive steps.  The
    # previous probe recomputed both large projections up to 64 times for each
    # of two decoders, which dominated the resident rollout hot path.
    option_keys = decoder.key(options)
    option_bias = decoder.option_bias(options).squeeze(-1)
    decode_steps = max_select
    if trim_to_routed_max_count:
        routed_max_count = torch.where(route, batch.max_count, 0).amax()
        decode_steps = min(max_select, int(routed_max_count.item()))
    for step in range(decode_steps):
        logits = decoder.logits(
            batch,
            options,
            state,
            option_keys=option_keys,
            option_bias=option_bias,
        )
        # Non-routed terminal lanes may have no legal option nor legal STOP.
        # Give them a finite inert categorical row while keeping their action,
        # log-probability and recurrent state excluded by ``active``.
        logits = torch.where(route[:, None], logits, torch.zeros_like(logits))
        if greedy:
            choice = logits.argmax(dim=1)
        else:
            distribution = torch.distributions.Categorical(logits=logits.float())
            choice = distribution.sample()
            logprob += torch.where(active, distribution.log_prob(choice), 0.0)
            if compute_entropy:
                entropy += torch.where(active, distribution.entropy(), 0.0)
        selecting = active & choice.lt(option_count)
        choosing_stop = active & choice.eq(option_count)
        raw = torch.where(selecting, choice, torch.full_like(choice, -1))
        actions[:, step] = raw
        stopped |= choosing_stop
        state = decoder.consume(options, state, raw)
        lengths = state.selected_count
        active = selecting & lengths.lt(batch.max_count)
    if bool((route & (lengths < batch.min_count)).any()):
        raise RuntimeError("sampled action violates min_count")
    return {
        "actions": actions,
        "lengths": lengths,
        "stopped": stopped,
        "logprob": logprob,
        "entropy": entropy,
    }


def _focal_reward(game_result: Any, focal_player: Any) -> Any:
    import torch

    focal_win = (focal_player.eq(0) & game_result.eq(1)) | (
        focal_player.eq(1) & game_result.eq(2)
    )
    focal_loss = (focal_player.eq(0) & game_result.eq(2)) | (
        focal_player.eq(1) & game_result.eq(1)
    )
    return focal_win.to(torch.float32) - focal_loss.to(torch.float32)


def main() -> int:
    raise RuntimeError(
        "FATAL: legacy 0035 shared-encoder/frozen-decoder routing has no "
        "Policy Identity Protocol V1 manifest or effective-weight audit; "
        "historical results are retained, but this entrypoint is disabled"
    )
    args = parse_args()
    if min(
        args.steps, args.deck_limit, args.max_select, args.ppo_epochs, args.minibatch_size
    ) <= 0:
        raise ValueError(
            "steps, deck-limit, max-select, PPO epochs, and minibatch size must be positive"
        )
    if not 1 <= args.max_select <= 64:
        raise ValueError("max-select must be in [1, 64]")
    for path in (args.rules, args.snapshot, args.checkpoint, args.focal_deck):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows, catalog = _load_first50_snapshot(args.snapshot.resolve())
    if args.deck_limit > len(catalog):
        raise ValueError(f"deck-limit exceeds snapshot size: {args.deck_limit} > {len(catalog)}")
    selected_rows = rows[: args.deck_limit]
    selected_catalog = catalog[: args.deck_limit]
    focal_deck = _read_deck(args.focal_deck.resolve())
    loader = importlib.import_module(
        "train.0034_dragapult_third_large_model_rl.policy.compact_actor_critic"
    )
    loaded, source, storage = loader.load_compact_actor_critic(
        args.checkpoint.resolve(),
        focal_deck,
        device=device,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    model = CudaSemantic0031ActorCritic.build(
        loaded.actor, max_select=args.max_select
    ).to(device)
    model.policy.eval()
    model.value_head.eval()
    opponent_decoder = copy.deepcopy(model.policy.action_decoder).to(device).eval()
    opponent_decoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.policy.action_decoder.parameters(), "lr": args.actor_lr},
            {"params": model.value_head.parameters(), "lr": args.value_lr},
        ]
    )

    batch_size = len(selected_catalog)
    focal_player = torch.arange(batch_size, device=device, dtype=torch.long).remainder(2)
    deck_rows = []
    for player, opponent in zip(focal_player.tolist(), selected_catalog, strict=True):
        opponent_deck = tuple(int(card) for card in opponent.deck)
        deck_rows.append(
            [focal_deck, opponent_deck]
            if player == 0
            else [opponent_deck, focal_deck]
        )
    decks = torch.tensor(deck_rows, dtype=torch.int32, device=device)
    seeds = torch.arange(args.seed, args.seed + batch_size, dtype=torch.int64, device=device)
    lanes = torch.arange(batch_size, dtype=torch.int32, device=device)
    engine = create_official_engine(
        args.rules.resolve().read_bytes(),
        batch_size=batch_size,
        device_index=args.device_index,
    )
    engine.reset_seeded_interactive_semantic(decks, seeds)
    engine.advance_to_decision()
    buffer = CudaRolloutBuffer(max_steps=args.steps, batch_size=batch_size)
    decision_counts = []
    behavior_logprob_mae = 0.0
    terminal_before_horizon = 0
    error_before_horizon = 0
    started = time.perf_counter()

    for _ in range(args.steps):
        statuses = engine.statuses()
        if bool(statuses.eq(ERROR).any()):
            error_before_horizon = int(statuses.eq(ERROR).sum().item())
            break
        if bool(statuses.eq(TERMINAL).any()):
            terminal_before_horizon = int(statuses.eq(TERMINAL).sum().item())
            break
        if not bool(statuses.eq(NEEDS_ACTION).all()):
            raise RuntimeError("collector probe expected every active lane to need an action")

        semantic = semantic0031_v2_ready_batch(
            engine.encode_semantic0031_v2_lanes(lanes),
            max_action_steps=args.max_select,
        )
        with torch.no_grad():
            validated, summary, options = model.encode(semantic)
            learner = _decode_device(
                model.policy.action_decoder,
                validated,
                summary,
                options,
                max_select=args.max_select,
                greedy=False,
            )
            opponent = _decode_device(
                opponent_decoder,
                validated,
                summary,
                options,
                max_select=args.max_select,
                greedy=True,
            )
            actor = (semantic["global_cat"][:, 3] - 1).clamp(min=0, max=1)
            learner_turn = actor.eq(focal_player)
            actions = torch.where(
                learner_turn[:, None], learner["actions"], opponent["actions"]
            ).contiguous()
            lengths = torch.where(learner_turn, learner["lengths"], opponent["lengths"])
            stopped = torch.where(learner_turn, learner["stopped"], opponent["stopped"])
            record = dict(semantic)
            record["targets"] = actions
            record["sequence_lengths"] = lengths
            record["sequence_stopped"] = stopped
            record["train_mask"] = learner_turn
            checked_logprob, checked_value = model.evaluate_targets_from_encoding(
                record, (validated, summary, options)
            )
            if bool(learner_turn.any()):
                discrepancy = (checked_logprob[learner_turn] - learner["logprob"][learner_turn]).abs().max()
                behavior_logprob_mae = max(behavior_logprob_mae, float(discrepancy.item()))

        engine.pack_actions(actions, lengths)
        engine.apply_packed_actions()
        engine.advance_to_decision()
        statuses_after = engine.statuses()
        done = statuses_after.eq(TERMINAL)
        reward = torch.where(
            done,
            _focal_reward(engine.game_results().long(), focal_player),
            torch.zeros(batch_size, dtype=torch.float32, device=device),
        )
        buffer.append(
            record,
            old_logprob=torch.where(learner_turn, learner["logprob"], torch.zeros_like(learner["logprob"])),
            old_value=torch.where(learner_turn, checked_value, torch.zeros_like(checked_value)),
            reward=reward,
            done=done,
        )
        decision_counts.append(int(learner_turn.long().sum().item()))
        if bool(done.any()):
            terminal_before_horizon = int(done.sum().item())
            break

    torch.cuda.synchronize(device)
    rollout_seconds = time.perf_counter() - started
    final_statuses = engine.statuses()
    final_results = engine.game_results().long()
    terminal_mask = final_statuses.eq(TERMINAL)
    terminal_rewards = _focal_reward(final_results, focal_player)
    terminal_count = int(terminal_mask.sum().item())
    error_count = int(final_statuses.eq(ERROR).sum().item())
    terminal_outcomes = {
        "focal_wins": int((terminal_mask & terminal_rewards.gt(0)).sum().item()),
        "focal_losses": int((terminal_mask & terminal_rewards.lt(0)).sum().item()),
        "draws": int((terminal_mask & terminal_rewards.eq(0)).sum().item()),
    }
    ppo: dict[str, float] | None = None
    if len(buffer) > 0 and error_count == 0:
        stats = ppo_update_device(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            last_value=torch.zeros(batch_size, dtype=torch.float32, device=device),
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_eps=0.10,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            gamma=1.0,
            gae_lambda=0.97,
            target_kl=0.02,
        )
        torch.cuda.synchronize(device)
        ppo = stats.as_host()

    passed = (
        bool(decision_counts)
        and error_count == 0
        and behavior_logprob_mae <= 1.0e-5
        and ppo is not None
        and (not args.require_terminal or terminal_count > 0)
    )
    output = {
        "schema_version": 1,
        "passed": passed,
        "scope": "nonformal_cuda_native_0035_collector_and_ppo_contract_probe",
        "not_policy_strength_evidence": True,
        "snapshot": {
            "path": str(args.snapshot.resolve()),
            "sha256": _sha256(args.snapshot.resolve()),
            "source_selected_count": len(catalog),
            "probe_selected_count": batch_size,
            "one_lane_per_selected_exact_deck": True,
            "weighted_512_schedule_not_collected": True,
            "first_ids": [str(row["deck_id"]) for row in selected_rows[:3]],
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": source.checkpoint_sha256,
            "storage_schema": storage.schema_version,
        },
        "collector": {
            "batch_size": batch_size,
            "max_steps": args.steps,
            "steps_collected": len(decision_counts),
            "require_terminal": bool(args.require_terminal),
            "learner_decisions_by_step": decision_counts,
            "terminal_before_horizon": terminal_before_horizon,
            "error_before_horizon": error_before_horizon,
            "terminal_count": terminal_count,
            "terminal_outcomes": terminal_outcomes,
            "error_count": error_count,
            "behavior_logprob_max_abs": behavior_logprob_mae,
            "rollout_seconds": rollout_seconds,
            "actions": "learner_stochastic_opponent_frozen_greedy",
            "reward": "terminal_only_minus_one_zero_plus_one",
        },
        "cuda_resident": {
            "official_engine": True,
            "semantic0031_v2_observation": True,
            "learner_decoder_sampling": True,
            "frozen_opponent_decoder": True,
            "rollout_buffer": True,
            "gae": True,
            "ppo_update": True,
            "per_step_host_status_sync": True,
            "note": "The status sync is diagnostic-only and must be removed from a throughput collector.",
        },
        "ppo": ppo,
        "extension": importlib.import_module("_ptcg_cuda").__file__,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
