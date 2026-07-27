# Decision 002: target-only evaluation contract

**Date:** 2026-07-27
**Decision:** T0–T3 are judged only on the target Dragapult + Dusknoir build. Pure Dragapult and
Starmie + Dusknoir are auxiliary training sources and diagnostic strata, not success endpoints.
No runnable teacher policy is available, so official-engine strength is necessarily self-relative:
our frozen arms/checkpoints are compared with our own T0, never with a teacher candidate.

## Rationale

Behavior-cloning agreement measures reproduction of a demonstrator on logged states. It does not
measure the state distribution induced by the learned policy and is not synonymous with playing
strength. Auxiliary data can therefore lower target exact-action agreement while improving legal,
recoverable decisions encountered in actual games. Conversely, higher agreement can reproduce the
teacher more faithfully without improving wins.

The experiment consequently has two distinct evidence layers:

1. **Target-only offline diagnostics:** loss, token accuracy, greedy full-action exact, legality and
   key action slices on held-out Dragapult + Dusknoir episodes. These detect collapse and explain
   behavior; they are not the primary strength endpoint.
2. **Target-only official-engine strength:** W/L/error for frozen candidates that all use the same
   target `deck.csv`, opponent catalog snapshot, seat schedule, seeds, games and metric profile.
   This is the primary endpoint.

## Checkpoint and test separation

- Use one predeclared checkpoint-selection rule based only on target validation data for every arm;
  do not select an epoch using official-engine test wins.
- Freeze one candidate per arm before the formal comparison.
- Use a development engine batch only for package/runtime checks and preliminary uncertainty.
- After choosing the claimed winner, run one fresh confirmation batch with previously unused seeds
  and the same frozen package. Do not tune or choose another checkpoint from that result.
- Report paired win-rate difference against T0, both seats, uncertainty interval, completion/error
  rate and per-opponent results. A point-estimate increase alone is exploratory.

## Interpretation matrix

| Target imitation | Target engine strength | Interpretation |
|---|---|---|
| Up | Up | Clean positive transfer. |
| Down | Up | Potential policy improvement beyond literal teacher reproduction; accept only after confirmation. |
| Up | Flat/down | Better imitation, no evidence of stronger play. |
| Down | Down | Negative transfer / learned off-target. |

Auxiliary-build metrics may be retained to diagnose whether the source modules were learned, but
they cannot rescue a model that fails on the target, and they do not enter the headline score.
