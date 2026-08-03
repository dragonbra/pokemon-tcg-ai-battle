# 0031 Decisions

## 2026-08-04: rule-faithful semantic foundation

- Freeze 0025 and 0028 as historical artifacts. 0031 has no executable dependency on another numbered project.
- Use the full-engine v2 exporter as the static semantic authority. Public compressed EnergyTypeIndex values never overwrite engine bitmasks.
- Represent every physical card instance separately and connect attachment/evolution/source/target relations explicitly.
- Preserve exact facts and field missingness. Do not expose energy gaps, surplus/removal preferences, attach counterfactuals, newly-enabled attacks, final damage, or KO answers.
- Keep effect order through stable effect identities and ordinal fields. Preserve Target Area and Target Condition values in ordered slots.
- Formal dataset materialization and training are intentionally deferred. Current status is training-ready after unit and two-batch smoke verification.

Verification completed on 2026-08-04: 38 unit tests passed; eight chronological real 0025 raw decisions compiled, four collated into a finite forward; and a noncanonical CUDA smoke completed two optimizer updates with validation and model-only checkpoint retention at `.tmp/0031_rule_faithful_bc_smoke/run-088a7d1cc1`.
