# 0044 Protocol Pointer

This file does not duplicate protocol rules. The authoritative current evaluation
contract is `docs/rl/0044_Benchmark_V2_Protocol.md`; the repository-wide
policy-identity and deployment contract remains
`docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`.

For V6, PPO rollout opponents use immutable Champion-G2 with uniform deck sampling
over `001`–`067` and no PFSP state. Benchmark V2 is a separate CUDA-2048 greedy
evaluation against immutable Policy-0809, balanced across Meta IDs `00`–`13`, `17`
and `27`, with common random numbers. The repository-wide protocol remains
non-negotiable for complete effective policy identity, focal/opponent independence,
no-hybrid behavior, FP16-storage/FP32-runtime candidate evidence, and manual
Promotion.
