# 0040 Policy Identity Audit — 2026-08-11

**Classification:** Historical Hybrid opponent; invalid as Frozen-0806.

The V2 CUDA path constructed `Semantic0031DeviceAdapter` from the focal 0809 actor,
computed the focal prototype encoder, state encoder, option input encoder and option
transformer layer 0, then attached only the loaded 0806 option layer 1, final norm and
action decoder. The CPU collector called the complete 0806 model, so training/evaluation
policy resolution was also inconsistent across execution paths.

Affected evidence includes all outputs produced by the old CUDA resident path under:

```text
rl_runs/0040_dragapult_0809_action_boundary_rl/versions/
  V2_snapshot_loader_fix_long_run/artifact/frozen_results/
  V2_snapshot_loader_fix_long_run/artifact/champion.json
  V2_snapshot_loader_fix_long_run/artifact/status.json
  V2_snapshot_loader_fix_long_run/artifact/training_summary.json
```

This includes update 250, the automatically labeled update 270 leader, and update 280.
The files are retained unmodified for provenance. They must not be cited as Full-0806,
used as a Frozen-0806 baseline, or used as Promote Champion evidence.

The repaired path resolves `Policy-0806` before routing and verifies every effective
component against `train/0040_dragapult_0809_action_boundary_rl/policy_registry.json`.
The complete 0806 model then receives its own CUDA device adapter. Missing or mismatched
identity audits abort before inference.

The required lane-1 greedy CPU/CUDA Full-0806 lockstep gate passed all four historical
divergence seeds on 2026-08-11. The audit resolved `Policy-0806` to effective identity
`94a9aff63de8e2e728413e84d0647c1409326b2e5e41876a5df9062107dd2d5b`; all four
trajectories matched exactly through 181, 210, 158, and 213 decisions respectively,
with no first divergence. Result:

```text
.tmp/evaluation/policy_identity_full0806_four_seed/full0806-u270-four-seed.json
```

This gate permits broader parity work. It is not a CUDA-2048 strength result and does
not promote any candidate.

Legacy executable routes which do not resolve registered full policies are also
fail-closed. The 0035 hand-written shared-encoder/frozen-decoder probes and the early
0040 POD routed-head runtime retain their source and historical artifacts, but abort at
entry until they are migrated to V1 manifests and effective-weight audits. The 0037 and
0038 partial Semantic0031 callers abort in the shared router constructor.
