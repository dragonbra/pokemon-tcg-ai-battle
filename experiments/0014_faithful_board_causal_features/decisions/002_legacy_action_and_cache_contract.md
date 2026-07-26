# 002 - Legacy action compatibility and immutable cache publication

Date: 2026-07-27  
Status: accepted

The full raw corpus contains 36 decisions whose ordered action length is 17–21, above the original
0010 `max_action_steps=16`. The immutable full-feature cache retains all 162,128 decisions and
stores an explicit `a0_eligible` bit. Strict A0 uses 145,928 train and 16,164 validation decisions;
the primary AC comparison uses the same rows so feature value is not confounded by label coverage.

0010 always appended a STOP target, including forced-max actions. The corrected 0013 action
contract omits STOP after forced-max; 161,549 of 162,128 decisions are forced-max. Therefore the
primary A0 and AC comparison deliberately uses the same legacy always-STOP objective. Raw ordered
actions and termination kind remain in the shared cache so a corrected-objective control can be
run later under a new version without rebuilding data.

The published cache is `V1_full_feature_superset`, content SHA-256
`e341a7cbc761797f4fa97a4a01fddf2d25d69b8c3c60e4e41bcb3b4e2708686a`.
Legacy numerics are FP32; auxiliary numerics are FP16. A0 constructs dynamic batch widths and STOP
indices after selecting only legacy fields, before pinning or host-to-device transfer. A0 does not
permute legal options.
