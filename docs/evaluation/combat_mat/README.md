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

Every new Combat Mat publication must pass
`evaluation.combat_mat_contract` before any HTML is written. The visual and data
contract has two required levels:

1. The main `index.html` contains the ordered 001-055 exact-deck catalog, at least one
   representative card image for every row, explicit `tested` or `pending` semantics,
   report links only for completed candidates, an ordered 14-row meta-archetype
   aggregate, and a visibly equivalent class-15 `Other` row. Untested decks use
   missing results, never synthetic 0% values.
2. Every completed `reports/*.html` contains the candidate's exact 60-card construction
   with card art and quantities, one evidence-backed matchup row for each opponent
   001-055, and an ordered 14-row opponent meta-archetype aggregation followed by a
   class-15 `Other` row using the same win-rate bar and W-L-D presentation. The opponent
   rows and the 14 classes plus `Other` must each conserve the report's total W-L-D and
   game count.

The canonical detail-page visual language is implemented by
`evaluation/reporting/html.py`. New CPU or CUDA runners must feed their official-engine
evidence into that shared renderer; a runner-specific simplified detail page is not a
valid Combat Mat publication.
