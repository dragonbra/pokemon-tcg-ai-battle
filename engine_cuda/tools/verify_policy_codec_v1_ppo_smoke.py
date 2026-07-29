from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_numbers_finite(item) for item in value)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = args.run_dir / "training_metrics.json"
    checkpoint_path = args.run_dir / "checkpoint_last.pt"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_opponents = {str(item["name"]) for item in config["opponents"]}
    initial = load_checkpoint(args.initial_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    history = list(metrics.get("history") or [])
    extra = dict(checkpoint.get("extra") or {})
    training_state = dict(extra.get("training_state") or {})
    optimizer_state = dict(training_state.get("optimizer_state_dict") or {})
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "history_length",
        len(history) == args.expected_iterations,
        {"actual": len(history), "expected": args.expected_iterations},
    )
    iterations = [int(row.get("iteration", -1)) for row in history]
    check(
        "iteration_sequence",
        iterations == list(range(1, args.expected_iterations + 1)),
        iterations,
    )
    check(
        "checkpoint_iteration",
        int(extra.get("iteration", -1)) == args.expected_iterations,
        extra.get("iteration"),
    )
    check(
        "checkpoint_history_length",
        len(extra.get("history") or []) == args.expected_iterations,
        len(extra.get("history") or []),
    )

    iteration_summaries: list[dict[str, Any]] = []
    for row in history:
        rollout = dict(row.get("rollout") or {})
        update = dict(row.get("update") or {})
        matchup = dict(row.get("matchup") or {})
        games_by_opponent = {
            name: int(dict(stats).get("games", 0)) for name, stats in matchup.items()
        }
        iteration = int(row.get("iteration", -1))
        check(f"iteration_{iteration}_games", rollout.get("games") == 12, rollout.get("games"))
        check(
            f"iteration_{iteration}_usable_games",
            rollout.get("usable_games") == 12,
            rollout.get("usable_games"),
        )
        check(f"iteration_{iteration}_errors", rollout.get("errors") == 0, rollout.get("errors"))
        check(
            f"iteration_{iteration}_opponent_coverage",
            set(matchup) == expected_opponents
            and all(games_by_opponent.get(name) == 2 for name in expected_opponents),
            games_by_opponent,
        )
        check(
            f"iteration_{iteration}_optimizer_steps",
            float(update.get("optimizer_step_end", 0.0))
            > float(update.get("optimizer_step_start", 0.0)),
            {
                "start": update.get("optimizer_step_start"),
                "end": update.get("optimizer_step_end"),
            },
        )
        check(f"iteration_{iteration}_finite", all_numbers_finite(row), None)
        iteration_summaries.append(
            {
                "iteration": iteration,
                "games": rollout.get("games"),
                "decisions": rollout.get("decisions"),
                "errors": rollout.get("errors"),
                "wins": rollout.get("wins"),
                "optimizer_step_start": update.get("optimizer_step_start"),
                "optimizer_step_end": update.get("optimizer_step_end"),
                "matchup_games": games_by_opponent,
                "peak_cuda_memory_gib": row.get("peak_cuda_memory_gib"),
            }
        )

    required_training_state = {
        "optimizer_state_dict",
        "python_random_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    check(
        "resumable_training_state",
        required_training_state.issubset(training_state),
        sorted(training_state),
    )
    check(
        "optimizer_state_nonempty",
        bool(optimizer_state.get("state")) and bool(optimizer_state.get("param_groups")),
        {
            "state_entries": len(optimizer_state.get("state") or {}),
            "param_groups": len(optimizer_state.get("param_groups") or []),
        },
    )
    resume = dict((extra.get("run_provenance") or {}).get("resume") or {})
    if args.expected_iterations > 1:
        check("resume_requested", resume.get("requested") is True, resume)
        check(
            "optimizer_rng_state_restored",
            resume.get("optimizer_rng_state_restored") is True,
            resume,
        )

    initial_state = dict(initial.get("model_state_dict") or {})
    final_state = dict(checkpoint.get("model_state_dict") or {})
    missing_initial_keys = sorted(set(initial_state) - set(final_state))
    added_keys = sorted(set(final_state) - set(initial_state))
    initial_keys_preserved = not missing_initial_keys
    changed_tensors = 0
    changed_elements = 0
    squared_delta = 0.0
    max_abs_delta = 0.0
    if initial_keys_preserved:
        for name, before in initial_state.items():
            after = final_state[name]
            if not (torch.is_tensor(before) and torch.is_tensor(after)):
                continue
            if before.shape != after.shape:
                initial_keys_preserved = False
                missing_initial_keys.append(name + " (shape mismatch)")
                break
            if not (torch.is_floating_point(before) or torch.is_complex(before)):
                continue
            delta = after.to(torch.float64) - before.to(torch.float64)
            nonzero = int(torch.count_nonzero(delta).item())
            if nonzero:
                changed_tensors += 1
                changed_elements += nonzero
                squared_delta += float(torch.sum(delta * delta).item())
                max_abs_delta = max(max_abs_delta, float(torch.max(torch.abs(delta)).item()))
    check(
        "initial_model_state_preserved",
        initial_keys_preserved,
        {
            "initial_keys": len(initial_state),
            "final_keys": len(final_state),
            "missing_or_mismatched": missing_initial_keys,
            "added_keys": added_keys,
        },
    )
    check("learner_parameters_changed", changed_elements > 0, changed_elements)

    report = {
        "schema_version": 1,
        "passed": all(item["passed"] for item in checks),
        "run_dir": str(args.run_dir.resolve()),
        "expected_iterations": args.expected_iterations,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "iteration": extra.get("iteration"),
            "changed_tensors": changed_tensors,
            "changed_elements": changed_elements,
            "parameter_delta_l2": math.sqrt(squared_delta),
            "parameter_delta_max_abs": max_abs_delta,
        },
        "resume": resume,
        "iterations": iteration_summaries,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
