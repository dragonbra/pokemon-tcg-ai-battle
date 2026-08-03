# 0030 Pure Dragapult Shared-Encoder Decoder RL

## Objective and evidence boundary

This project starts from the 0028 V3 epoch-4 `latest.pt` semantic policy (SHA-256 `5e0a6eea42bf228a9bd977cf56fdd14ad360fbceffcf5c19e2d9fe1b39713d98`) and optimizes only the action decoder and a value head for the exact 60-card `dragapult_ex_001` deck. Rollout opponents are 102 `(foundation, exact deck)` identities: all 51 exact decks from `0019_foundation_51_exact_decks_v4` served once by the frozen legacy 0019 Epoch-13 actor and once by the frozen 0028 semantic actor. There is no live league and no opponent update.

Official rules establish legality, attack-as-turn-commitment, evolution timing, action budgets, zones, Prize and victory. Current card behavior and state transitions come from the unmodified official engine. PPO hyperparameters and the hypothesis that decoder-only adaptation improves pure Dragapult are project assumptions.

## Actor input contract

The actor consumes the unchanged 0028 canonical typed schema. It includes global categorical/numeric fields; variable card, resource and recent-event memories; option categorical/numeric/state fields; card-parent and option source/target relations; skill/effect IDs and roles; option masks; and `min_count`/`max_count`. Team/source identity is provenance only and never actor-visible.

The model width is 320 with 8 attention heads, 4 state layers, 3 option layers, FFN multiplier 3, maximum 128 options and maximum 64 ordered action steps. Prototype, state and option representation parameters total 20,809,880 and remain frozen. The ordered action decoder has 1,027,202 trainable parameters. The zero-initialized value head has 103,681 trainable parameters and emits actor-relative `V(s)` in `[-1,+1]`.

## Shared rollout data flow

Each game remains in an isolated spawned official-engine worker. The worker import path is deliberately Torch-light: it loads the rollout protocol, evaluation loader and official `libcg`, but not Torch, CUDA, the collector, model or feature compiler. Actor-local causal encoders remain in the parent and maintain chronological knowledge. The parent keeps both frozen foundations resident on the same GPU. Focal rows always use the 0028 semantic representation and live decoder/value. Opponent rows route by job to either the frozen 0019 encoder/decoder or the frozen 0028 representation/opponent decoder. Each foundation shares one service across all 51 exact decks; no per-deck model copy exists. Opponent rows are always greedy. Focal rows sample during rollout and are greedy during the separate frozen evaluation.

For each focal decision the trajectory stores the frozen state summary `[1,320]`, contextual option memory `[1,O,320]`, option mask `[1,O]`, scalar selection bounds, action sequence, STOP state, behavior log probability, entropy and value. `O` is cropped to that row's real legal-option count before CPU storage; summary/options are stored as fp16 and restored to fp32 for decoder math. PPO pads and collates these cached decoder inputs. It never recomputes prototype/state/option encoding.

## PPO contract

- terminal official-engine reward: loss `-1`, draw `0`, win `+1`;
- gamma `1.0`, GAE lambda `0.95`, episode-balanced decision weights;
- 4 epochs, minibatch 1024, clip ratio `0.10`;
- decoder LR `1e-5`, value LR `1e-4`, max gradient norm `0.5`;
- value coefficient `0.5`, entropy coefficient `0.01`;
- frozen initial-decoder anchor coefficient `0.02`;
- target behavior KL `0.02` with early stop.

Only focal decisions enter PPO. `rollout/source_policy_update=k-1` identifies the sampled behavior policy; PPO produces `checkpoint/update=k`. Each 512-game update allocates 256 games to each foundation, with 128 focal-first and 128 focal-second games per foundation over the same 51 exact decks. Sampled rollout metrics are diagnostics and are also split under `rollout/foundation_0019/*` and `rollout/foundation_0028/*`.

Every five updates, strength evidence comes from two fixed-seed greedy evaluations of 102 games each, covering all 51 decks in both seats. Metrics remain separate under `eval/foundation_0019/*` and `eval/foundation_0028/*`; they are never averaged. The legacy 0019 win rate is the current checkpoint-selection anchor for comparison with 0022, while the 0028 rate independently measures performance against the current foundation.

## Storage and current stage

`V1_shared_encoder_cached_decoder` was interrupted before update 1 after its padded trajectory cache caused falling RAM and swap growth. Its update-0 checkpoint and baseline metrics are preserved as failure evidence and it will not be resumed. `V2_trimmed_fp16_prototype_cache` fixed the trajectory cache and completed five PPO updates. Its update-5 model-only checkpoint is the accepted V3 branch point: SHA-256 `7065bdd8fa49472b64c871a1c4e470c58a390ed5710968550386e297d1b7a7bf`. V2 update 5 sampled 208-304 (40.62%); the separately evaluated checkpoint was 60-42 (58.82%) over the frozen 102-game contract.

V2 was intentionally stopped after that evaluation because its spawned workers imported the rollout package initializer, which eagerly imported the collector and Torch. Live PSS evidence measured roughly 280 MiB per worker and about 17.3 GiB across 64 workers. V3 changes the collector export to a lazy import and adds a subprocess contract that fails if importing the worker loads Torch. The production CLI now measures roughly 13-17 MiB PSS per worker with zero Torch/CUDA mappings.

The fixed-checkpoint 128-game official-engine calibration produced 2.080, 2.190 and 2.384 games/s at 64, 96 and 128 workers respectively, all with zero errors, zero worker Torch/CUDA mappings and no swap growth. At 128 workers, worker PSS peaked at 1.57 GiB total and host available RAM stayed above 17.8 GiB. V3 therefore uses 128 workers without weakening the 60-second response timeout.

`V3_lightweight_engine_workers` loads only the V2 update-5 decoder/value weights, verifies the checkpoint and representation hashes, and then constructs a fresh PPO optimizer. It starts a new metrics/W&B timeline and recollects all on-policy episodes. Checkpoints remain atomic model-only payloads containing only `action_decoder.*` and `value_head.*`; optimizer, scheduler, scaler, RNG, dataloader, rollout and replay state remain forbidden. The feature schema, tensor shapes, trainable parameter set, action contract, reward, value target and PPO objective are unchanged from V2.

Before V3 launch, the accepted update-5 checkpoint completed a 204-game dual-foundation baseline with zero official-engine errors: 55-47 (53.92%) against 0019 and 61-41 (59.80%) against 0028. Both models remained on one GPU; the run took 83.20 seconds and peaked at about 0.53 GiB CUDA allocation. Adding 0019 to training is an explicit opponent-curriculum change from V2, so V3 begins a fresh optimizer and newly collected on-policy data.

The final 512-game pre-launch gate completed 512/512 official-engine games in 195.14 seconds at 2.624 games/s with 128 workers. It retained zero worker Torch/CUDA mappings, zero PPO representation calls, no swap growth, and about 1.06 GiB CUDA peak allocation. This matches or slightly exceeds the 0022 2.565 games/s reference despite routing both frozen foundations.

For context, 0022 V2's first sampled update was 140-372 (27.34%) and its initial frozen snapshot was 43-53 (44.79%), but its 48-deck/older-policy contract differs from 0030's 51-deck shared-0028 contract. 0022 update 1 reached 2.565 games/s with 128 workers; the V3 128-game gate reached 2.384 games/s. A benchmark is not a complete 512-game update, so V3's first full update is the production throughput confirmation. Only future same-contract frozen greedy evaluations can establish policy improvement.
