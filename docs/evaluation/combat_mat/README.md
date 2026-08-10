# Combat Mat

This directory is the canonical home for durable Combat Mat evaluation reports.
Runtime packages remain under `evaluation/`; generated formal evidence belongs here
rather than under `evaluation/arena/` or disposable `.tmp/` output.

## Policy-0806 Reset

The former `policy_0806/` standard and its generated reports were retired on
2026-08-11. They predated Policy Identity Protocol V1 and did not contain the mandatory
component-level `policy_identity_audit: PASS`. They must not be cited, restored, or
renamed as Full-0806 or Promote Champion evidence.

New Policy-0806 reports must be generated from the canonical Full-0806 resolver. The
requested and materialized policy must both be `Policy-0806`, with effective policy
SHA-256 `94a9aff63de8e2e728413e84d0647c1409326b2e5e41876a5df9062107dd2d5b`
and checkpoint SHA-256
`0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`.

The normative rules are in
[`RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`](../../rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md).
Known Hybrid results remain invalid as Frozen-0806; deleting this old publication
standard does not retroactively relabel any underlying experiment.

## Publication Contract

A formal result must contain the requested candidate reports, the contract's complete
per-candidate game count, zero errors, zero unfinished games, exact schedule and seed
provenance, and a passing policy-identity audit. Promotion remains a separate human
decision.
