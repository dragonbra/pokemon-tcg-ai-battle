# 0045 — Single-Deck Expert Minimal-LoRA

Status: architecture/migration hard gates pass. V1 standard-LR control completed through U5; V2 runs the project-default expert cold-start LR without an automatic update limit, under human observation.

## Goal and identity

0045 trains one specialist only: exact deck `007` (Dragapult ex), 60-card content SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`. It is intentionally not a 29-deck generalist. The copied 0044 assets are project-local immutable inputs; 0045 runtime code must not import another numbered training project.

The source checkpoint is selected only after the live 0044 V24 process stops at a complete model-only checkpoint boundary. Its path and SHA-256 are therefore a pending run identity, not an architectural unknown. The 0045 specification explicitly selects the G2 generalist: immutable `Champion-G2`, effective SHA-256 `5f314275c576c1957fe011ab554ca4807cc73680fc89f443394894a7441cb082`. Active V24’s `Champion-G3` (`5d15f503...`) is recorded as a non-selected identity and is never silently substituted.

## Audited 0044 pre-change graph

```text
official observation + exact deck/resource state + legal options
  -> frozen prototype/card/state/event/resource encoders
  -> frozen Option prefix
     -> base final Option block ---------------------------> Critic queries
     -> policy final Option Q/V LoRA (r=4, alpha=8) ------> Actor options

Critic queries -> z_value -> ValueResidualAdapter(Own Meta) -> V_win
               -> z_meta  -> opponent-Meta logits/probs
               -> PrizeAuxHead -> V_prize

state.hidden + side + z_meta + meta_probs + V_win + Own Meta
  -> PolicyStrategyAdapter
  -> 29-way MetaActorResidual
  -> ActionDecoder -> option/STOP logits

state/options/public features -> AllocationHead
```

Thus 0044 used detached Critic outputs in Actor inference. Detachment blocked gradients but did not remove the semantic dependency.

## 0045 V1 post-change graph

```text
official observation + exact deck/resource state + legal options
  -> frozen semantic backbone
  -> frozen Option prefix
     +-> policy final Option Q/V LoRA (r=4, alpha=8)
     |    -> ActionDecoder -> option/STOP logits
     |    -> AllocationHead when applicable
     |
     +-> base final Option block -> Critic latent queries
          -> ValueResidualAdapter (Critic-only deck-007 ID) -> V_win
          -> opponent-Meta head
          -> PrizeAuxHead -> V_prize
```

The Actor API is `encode_policy(batch) -> (validated, state, policy_options)`. `DecoderPolicyHead.logits` accepts only batch, policy options, decoder state, and decoder-owned keyword arguments. It has no field or argument capable of carrying `V_win`, `z_meta`, Meta probabilities, or Own-Archetype IDs.

The Critic API is `encode_critic(batch) -> (validated, state, value_options, value, auxiliary)`. The two paths share only frozen semantic computation. Trainable parameter objects are disjoint.

## Tensor and parameter contract

- Semantic width: 320.
- State Transformer: 4 layers, 8 heads.
- Event encoder: 1 layer.
- Option Transformer: 2 blocks, 8 heads; only final-block self/cross attention Q/V LoRA is trainable in the policy fork.
- Maximum legal options: 128; maximum decoder action steps: 64.
- Actor trainables measured at runtime: Action Decoder 1,027,202; Allocation Head 621,761; Option LoRA 10,240; total 1,659,203.
- Critic trainables measured at runtime: Value head 3,397,121; Value adapter 211,665; Prize head 103,681; total 3,712,467.
- Total trainables: 5,371,670. Frozen plus trainable total parameters: 60,707,058.

Removed Actor modules are the 320,465-parameter Policy Strategy Adapter and 74,240-parameter Meta Actor Residual. Their source tensors remain in 0044 checkpoints and are classified as dropped by migration; they do not exist in the 0045 model or optimizer.

Exact-deck/resource information remains in the frozen semantic observation. The Critic’s Own-Archetype embedding remains a training-only value feature; it is never policy-visible.

## Migration and identity

`migration.from_0044.migrate_0044_checkpoint` starts from the copied immutable complete semantic base, rebinds the Critic-only Own-Archetype ID to exact deck 007, and copies every shape-compatible checkpoint tensor except the two explicitly retired Actor prefixes. Unclassified tensors, missing inherited tensors, shape mismatches, or failed post-load tensor equality are fatal. The audit records copied/dropped/rebound/base-materialized/new tensors and hashes.

The migrated U0 model is serialized as `0045_minimal_lora_model_only_v1`. An immutable same-architecture `Frozen-0045-Init` snapshot is the primary reference policy. `ppo/reference_kl` means distance to this U0 snapshot; any historical 0044/BC KL must use a separate metric name. PPO/master weights remain FP32. Evaluation first materializes a full effective candidate, stores FP16, strict-loads FP32 runtime weights, and records the source FP32 checkpoint, portable artifact, and deployment-effective hash.

## PPO objective and expert learning-rate stages

The architecture is the controlled variable. V1 preserves 0044 V24 settings: 512 rollout games, 3 PPO epochs, logical minibatch 4096, physical/probe batch 256, clip 0.10, entropy coefficient 0.003, reference-KL coefficient 0.02, behavior-KL target/hard guard 0.015/0.025, max grad norm 0.5, gamma 1.0, GAE lambda 0.95, win-value coefficient 0.5, Meta anchor 0.1, Prize-value weight 0.5, and Prize actor-advantage contribution 0.1.

A fresh AdamW optimizer has only the intended Actor/Critic groups. V1 measured the inherited conservative/limit profile:

- Action Decoder: LR `5e-6`.
- Allocation Head: LR `5e-6`.
- final Option Q/V LoRA: LR `1e-5`.
- Value head/trunk: LR `2e-5`.
- Value adapter: LR `2e-5`.
- Prize head: LR `2e-5`.

V1 stopped at its complete U5 boundary. Its fixed common-seed Tiny V2 moved from U0 `324-188-0` (63.28125%) to U5 `326-186-0` (63.671875%), while reference KL reached only the low `e-6` range. This control confirmed that the safe steps were too slow for rapid expert viability screening.

For every newly initialized 0045 expert, the project default is now named `0045_expert_cold_start_lr_v1`:

- Action Decoder: LR `1e-5`.
- Allocation Head: LR `1e-5`.
- final Option Q/V LoRA: LR `2e-5`.
- Value head/trunk, Value adapter, and Prize head remain LR `2e-5`.

Only Actor LR is doubled. Entropy, rollout size, PPO epochs, clipping, KL coefficients/guards, reward shaping, and Critic LR remain unchanged. The purpose is rapid traversal of the near-U0 region and early measurement of whether reference movement converts into fixed greedy evaluation gains. Reference KL is a distance diagnostic, never a quality target by itself.

The old V1 rates are retained explicitly as `0045_limit_finetune_lr_v1`. After a high-LR cold-start run identifies a useful checkpoint/region, a new version may clone that model-only checkpoint with a fresh optimizer and use the limit profile for fine adjustment. LR stage changes never append to an existing repository version.

No optimizer/scheduler/RNG state is migrated. Checkpoints are model-only and retain every update.

## Rollout and evaluation

The focal schedule is fixed deck 007. The first controlled opponent deck schedule stays aligned with current 0044; opponent policy identity is immutable and explicit. Focal and opponent weights, storage, routing, and caches are independent.

Benchmark Tiny V2 is a deterministic CUDA-512 common-random-number contract over Benchmark V2’s frozen 16 Meta classes, exactly 32 games per class. It runs every five updates: U5, U10, U15, and so on. Multiples of ten are also the requested periodic evaluate points and reuse the same 512-game report rather than launching a duplicate evaluation. Seeded toss winners invoke each Agent’s real context-41 first-player decision.

Every scored evaluation requires `kaggle_fp16_storage_fp32_runtime_v1` deployment identity PASS, 512 terminal games, zero errors, and zero unfinished games. It logs total/first/second win rate, per-Meta and per-deck results, and `eval/checkpoint_update`.

Training logs include rollout win rate, behavior KL, specialist reference KL, optional historical KL, Value loss, explained variance, policy entropy, policy/critic grad norms, decoder/LoRA/allocation update norms, and guarded eval-win-rate improvement per reference-KL movement. Sampled rollout win rate is never reported as greedy checkpoint strength.

## Hard gates and stage

CPU hard gates cover Critic-mutation→Actor invariance, Policy-LoRA-mutation→Critic invariance, gradient ownership, migration equality, opponent immutability, optimizer ownership, and Critic-free export parity. They pass. 0044 stopped at V24 U21 (`814765…c83e`), migrated U0 is `3c13e0…91bc7`, and CUDA smoke passes. V1 is an immutable standard/limit-LR U0→U5 control. V2 restarts from the exact same Frozen-0045-Init with the cold-start profile and a fresh optimizer. It has no automatic update limit; U0/U5/U10 common-seed Tiny V2 reports are early viability gates, and later five-update reports continue until the user stops the run.

## Disabled expansion ladder

V1 adds no capacity. If later controlled evidence shows a ceiling, the documented order is final Option Q/V LoRA rank 4→8→16 with function-preserving zero delta; then Q/V LoRA on the previous Option block; then a small final State-block LoRA/zero residual. Full State Encoder unfreezing is not the first response to a plateau.
