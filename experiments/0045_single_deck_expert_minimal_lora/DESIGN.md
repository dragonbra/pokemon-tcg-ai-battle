# 0045 — Single-Deck Expert Minimal-LoRA

Status: architecture/migration hard gates pass. V1 standard-LR control completed through U5. V2 reached a durable U45 model-only checkpoint, then hard-failed before its U45 evaluation because a packaging-time deployment-runtime edit changed the audited semantic-runtime tree. V3 continues from V2 U45 with a fresh optimizer, the same cold-start LR profile, and no automatic update limit.

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

The migrated U0 model is serialized as `0045_minimal_lora_model_only_v1`. An immutable same-architecture `Frozen-0045-Init` snapshot is the primary reference policy. `ppo/reference_kl` means distance to this U0 snapshot; any historical 0044/BC KL must use a separate metric name. A later version may initialize its focal model from a selected model-only parent checkpoint while loading the reference model independently from Frozen-0045-Init. This separation prevents a version boundary from silently resetting reference KL. PPO/master weights remain FP32. Evaluation first materializes a full effective candidate, stores FP16, strict-loads FP32 runtime weights, and records the source FP32 checkpoint, portable artifact, and deployment-effective hash.

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

As of 2026-08-16, the append-only live training/opponent deck registry is contiguous `001`–`070`. Deck `070` is the user-provided exact 60-card `Dragapult ex / Dusknoir / Munkidori` list with content SHA-256 `2693a7b610ccdf2ed10a76a8872003dcb53456e54b006d3a3feff580520374fb`. It is mapped to the existing 29-way Own Archetype class `15` (`dragapult_dusknoir`); no new Actor/Critic class or embedding dimension is introduced. Current `meta_balanced_training_pool` and aggressive training-pool schedules resolve the registry dynamically and therefore include `070`. Historical Benchmark V2/Core-16 schedules and already-recorded run manifests remain frozen and are not rewritten to include it.

V4 is a fresh-optimizer continuation from the exact V3 U200 model-only checkpoint. Its sampled training rollout still uses immutable `Champion-G2`; changing the evaluation opponent does not silently change the behavior-policy environment. The 512-game rollout schedule is Meta-first and then exact-deck-balanced within each Meta. Meta `03` and `05` have weight `6`; Meta `00`, `01`, `02`, and `04` have weight `3`; Meta `27` explicitly returns to weight `1`; every other active Meta also has implicit weight `1`. The first U200-source schedule therefore assigns 67 lanes each to `03/05`, 34 each to `00/01/02/04`, and 11 each to all remaining active classes including `27`.

Starting with V4, the every-five-update longitudinal evaluation is a new three-pool contract against complete immutable `Policy-0809`. Each pool independently contains 512 greedy official-engine games and is balanced first over its selected active Meta classes, then over their exact decks:

- `eval/low_score/*`: Meta `02/03/05`;
- `eval/priority/*`: Meta `00/01/04/27`;
- `eval/remaining/*`: every other non-empty Meta class, sampled uniformly by Meta.

The three sets are disjoint and cover all 28 non-empty classes in `own_archetypes_v2`; empty class `14` produces no fabricated games. One deployment-effective candidate is materialized per checkpoint and reused for all 1,536 games. Each pool is logged as its own CUDA-512 curve. `eval/three_pool_aggregate/*` is explicitly named as a 1,536-game diagnostic and does not reuse or impersonate the historical `eval/core/*` series. Common-random schedules exclude focal deployment identity, while opponent identity is fixed to Policy-0809. Multiples of ten reuse the same five-update evaluation rather than running a duplicate.

Benchmark Tiny V2 is a deterministic CUDA-512 common-random-number contract over Benchmark V2’s frozen 16 Meta classes, exactly 32 games per class. It runs every five updates: U5, U10, U15, and so on. Multiples of ten are also the requested periodic evaluate points and reuse the same 512-game report rather than launching a duplicate evaluation. Seeded toss winners invoke each Agent’s real context-41 first-player decision.

Every scored evaluation requires `kaggle_fp16_storage_fp32_runtime_v1` deployment identity PASS, 512 terminal games, zero errors, and zero unfinished games. It logs total/first/second win rate, per-Meta and per-deck results, and `eval/checkpoint_update`.

The formal Kaggle package uses schema `0045_single_deck_expert_kaggle_package_v1`. It materializes the same `0045_minimal_lora_candidate_v1` FP16 artifact and deployment-effective hash used by periodic evaluation. The portable runtime retains Critic tensors only because the U40 qualifying identity includes the complete training candidate payload; for 0045 schemas it does not instantiate a Policy Strategy Adapter and never evaluates or consumes Value/Meta outputs in Actor selection. Runtime regression tests strongly mutate the retained Critic and require identical greedy Actor sequences.

Training logs include rollout win rate, behavior KL, specialist reference KL, optional historical KL, Value loss, explained variance, policy entropy, policy/critic grad norms, decoder/LoRA/allocation update norms, and guarded eval-win-rate improvement per reference-KL movement. Sampled rollout win rate is never reported as greedy checkpoint strength. W&B defines `eval/checkpoint_update` as the explicit step metric for every `eval/*` curve, so the separately emitted evaluation row remains horizontally aligned with the checkpoint/update it evaluates rather than W&B's internal log-row counter or `env/episodes`.

## Hard gates and stage

CPU hard gates cover Critic-mutation→Actor invariance, Policy-LoRA-mutation→Critic invariance, gradient ownership, migration equality, opponent immutability, optimizer ownership, Critic-free export parity, and Critic-free Kaggle runtime selection. They pass. 0044 stopped at V24 U21 (`814765…c83e`), migrated U0 is `3c13e0…91bc7`, and CUDA smoke passes. V1 is an immutable standard/limit-LR U0→U5 control. V2 restarted from the exact same Frozen-0045-Init with the cold-start profile and a fresh optimizer; U40 is the packaged candidate after a valid common-seed Tiny V2 result of `329-183-0` (64.2578125%), versus U0 `324-188-0` (63.28125%). V2's U45 PPO row and checkpoint are durable, but its synchronous U45 evaluation never ran because the runtime-integrity hard gate detected the deployment-file edit made during U40 packaging. V2 is therefore recorded as interrupted, not resumed in place. V3 started at U45 from that model-only checkpoint with a fresh optimizer and freshly collected on-policy data, kept Frozen-0045-Init as reference, and stopped cleanly at U200. V4 binds parent SHA-256 `d8a2cd…a7306` at U200 and independent Frozen-U0 SHA-256 `3c13e0…91bc7`, uses the unchanged cold-start LR/PPO objective, fresh optimizer/on-policy data, targeted rollout Meta weights, and the Policy-0809 three-pool evaluation contract. It has no automatic update limit.

V4 passed its U200 baseline before its first update: low-score `241-271-0` (47.0703125%), priority `329-183-0` (64.2578125%), remaining `400-112-0` (78.125%), all 1,536 games terminal with zero errors and identity/storage-isolation PASS. Its first fresh targeted Champion-G2 sampled rollout was `275-237-0`; U201 was durably written with behavior KL `3.43e-5` (far below the `0.025` hard guard), reference KL `0.006496`, Value loss `0.4023`, and explained variance `0.4933`.

V4 was explicitly stopped after its complete U275 checkpoint and three-pool evaluation. U275 passed all 1,536 games with low-score `241-271-0` (47.0703125%), priority `336-176-0` (65.625%), remaining `409-103-0` (79.8828125%), and named aggregate `986-550-0` (64.1927083%). No U276 checkpoint exists. The V4 status is `stopped`, and its U275 checkpoint SHA-256 is `ed0634…ecdb`.

V6 is an intentionally aggressive sampling ablation from exact V4 U275. V5 is already occupied by the formal deck-023 U105 evaluation identity and is not reused. V6 creates a fresh optimizer, collects fresh on-policy data, keeps the cold-start LR/entropy/PPO objective/model architecture unchanged, and retains Frozen-0045-Init as reference. For this experiment only, the sampled rollout opponent is also locked to complete immutable Policy-0809. Every 512-game rollout has exact Meta quotas: `03=200`, `05=200`, `02=50`; the remaining 62 lanes are balanced uniformly over all other 25 active Meta classes (2–3 each), then balanced over exact decks within each class. Every completed V6 update synchronously runs the unchanged three disjoint Policy-0809 CUDA-512 evaluation pools before the next rollout. The existing V4 U275 PASS report is hash-bound baseline provenance and is not re-materialized or duplicated in V6.

V6 exposed a telemetry-only integration defect after its first PPO update: the exact-quota Policy-0809 rollout completed `249-263-0` and U276 model-only checkpoint was durable, but the telemetry consumer rejected the new sampling-mode string before writing canonical metrics or starting eval. V6 is retained as failed with zero metric rows. A minimal regression test reproduced the independent allowlist omission; telemetry now records both `sampling/mode_meta_balanced=1` and `sampling/mode_aggressive_meta_quota=1`. V7 binds exact V6 U276 SHA-256 `a75e18…05df`, creates another fresh optimizer, reruns the missing U276 three-pool baseline, and then continues the identical aggressive experiment with every-update evaluation.

V7 was explicitly stopped after durable U282. U281 is its last complete every-update three-pool evaluation; U282 then passed the formal common-seed Policy-0809 Benchmark V2 CUDA-2048 at `1343-705-0` (65.576171875%), with all 2,048 games terminal and zero errors. V8 started from exact V7 U282 SHA-256 `25f312…2d5c`, with a fresh optimizer and fresh on-policy rollout. It first ran the missing U282 three-pool baseline, then preserved the exact 512-game Policy-0809 quota schedule (`03=200`, `05=200`, `02=50`, other active Meta total 62) while changing only the synchronous three-pool evaluation cadence from every update to every two checkpoint updates. V8 was explicitly stopped after durable U299 SHA-256 `042e91…b56e`; U298 is its final complete three-pool evaluation at `990-546-0` (64.453125%), and no U300 or partial checkpoint exists. LR, entropy, PPO objective, model architecture, opponent identity, and Frozen-0045-Init reference remained unchanged. The authoritative longitudinal report covers all 31 same-contract V4/V7/V8 points from U200 through U298 and, in separately labeled tables, all 25 valid existing Deck-007 Policy-0809 Benchmark V2 CUDA-2048 points (U40, U90/U95, U100–U200 every five updates, and U282). In total it exposes 56 measured points / 98,816 terminal games under both `own_archetypes_v2` and the actual 15-way Value-loss opponent taxonomy; the contracts remain separated because their sampling distributions differ.

V9 is a diagnostic, explicitly cheating `Experimental-Oracle-MetaRouter-V1`, not a trainable/deployable policy. Before each game it reads the opponent exact deck's true `own_archetypes_v2` 29-way class and uses literal routing: `06/08/10/27 -> U200`, `01/02/17 -> U40`, `03 -> U90`, and every other class (including `00/04/05`) -> U282. All four checkpoints are independently materialized through `kaggle_fp16_storage_fp32_runtime_v1`; 279 frozen Actor tensors compare exactly equal, so runtime retains one U282 semantic backbone and switches only Policy Q/V LoRA, Action Decoder, and Allocation Head. The U282 Critic remains training/evaluation scaffolding and its outputs do not enter Actor inference. Under the unchanged Policy-0809 Benchmark V2 CUDA-2048 common-seed schedule, the oracle scored `1381-667-0` (67.431640625%), with 2,048 terminal games and zero errors. This result is labeled `diagnostic_oracle_not_promote_not_kaggle` and must never be used as submission-strength or Promote evidence.

V10–V12 replace the oracle input with a small monotonic memory inside the focal Agent policy layer. It consumes only opponent Pokémon identities that the official observation marks public/certain; it never reads exact opponent deck ID, hidden cards, the 29-way ground-truth label, or Critic Meta output. Frozen semantic tensors remain shared, and only Policy Q/V LoRA, Action Decoder, and Allocation Head switch at a strategic decision boundary; the decoder has no hidden state carried between official decisions. V12 `Experimental-Public-MetaRouter-V3-GrassFoldU200` routes `01/02/17 -> U40`, `03 -> U90`, and the public `06/08/27/28` grass/Festival family -> U200, with U282 as default. The U200 route fires immediately on public Grookey/Thwackey, Festival Applin/Dipplin, Teal Mask Ogerpon, or any registered 08/27/28 representative Pokémon line; ambiguous Applin/Dipplin may route before a fine Meta label can be locked. The same routing core is used by the CPU Agent wrapper and project-local CUDA evaluator; official engine source and CUDA engine routing are unchanged.

On the unchanged common-seed Policy-0809 Benchmark V2 CUDA-2048 schedule, V10 scored `1373-675-0` (67.041015625%), V11's temporary `08/27/28 -> U282` fold scored `1370-678-0` (66.89453125%), and V12's early `06/08/27/28 -> U200` fold scored `1382-666-0` (67.48046875%). V12 completed 2,048/2,048 terminal games with zero errors and candidate/opponent/CUDA identity PASS. Its Core-16 results for Meta 06/08/27 were respectively `116/128`, `109/128`, and `95/128`, equal to the V9 oracle on those rows; Meta 28 is absent from the frozen Core-16 schedule and therefore has no fabricated score. Public-router evidence remains experimental and is not automatically Promote or Kaggle evidence.

## Compact public-router Kaggle deployment

The user explicitly selected V12 for a Kaggle-ready Deck-007 package. The deployment keeps U282 as the full default policy for every observation that does not match a public routing rule. It stores the complete U282 portable candidate once and stores only the Policy Q/V LoRA, Action Decoder, and Allocation Head tensors for U40, U90, and U200. It never stores three duplicate semantic backbones or three duplicate Critics. All floating checkpoint tensors are FP16 on disk and are strict-loaded into FP32 modules before inference under `kaggle_fp16_storage_fp32_runtime_v1`.

The compact artifact hard-verifies exact equality of all shared Actor tensors, the four per-head effective hashes, the public-rule manifest, and the V12 composite identity `1c8125…dc1`. Unknown/unclassified public Pokémon remain on U282. The package entry point owns the monotonic opponent-public-card memory and routing; neither official CPU engine code nor CUDA engine code is modified. The package schema is `0045_public_meta_router_kaggle_package_v1`.

The resulting archive is `0045_dragapult_ex_007_public_meta_router_v3_compact_fp16_storage_fp32_runtime.tar.gz`, 123,652,167 bytes, SHA-256 `d3c1e50bd1b30c0f279e075ded7ed7dea530cef201d6a9086d9d08dd4be2a633`. Its default portable U282 file is 125,122,571 bytes and its three compact alternate heads total 9,988,377 bytes. Independent archive extraction verified 54 declared files, the exact 60-card deck, the V12 composite hash, and FP32 runtime parameters. The full 0045 regression suite passes 44 tests. A 30-game official CPU-engine package smoke covering U40, U90, and the U282 default completed 30/30 games with zero errors/unfinished games; U200 exact tensor identity and FP32 loading are covered by the hard hash regression. The qualifying strength evidence remains the existing V12 Policy-0809 CUDA-2048 result `1382-666-0` (67.48046875%), rather than the small smoke score.

## Disabled expansion ladder

V1 adds no capacity. If later controlled evidence shows a ceiling, the documented order is final Option Q/V LoRA rank 4→8→16 with function-preserving zero delta; then Q/V LoRA on the previous Option block; then a small final State-block LoRA/zero residual. Full State Encoder unfreezing is not the first response to a plateau.
