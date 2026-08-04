# CUDA RL Throughput Acceptance

## Result

The first 308.727 decisions/s number was not the CUDA engine limit. Profiling showed that 87.7% of the step was spent in 64 rounds of eager autoregressive decoder launches. A graph-capturable fixed-shape decoder plus CUDA Graph replay raises the accepted 38-lane result to 1,923.473 decisions/s. The recommended 152-lane rollout batch reaches 3,579.865 decisions/s and 35.328 episodes/s on the RTX 5080.

Machine-readable reports:

- [fixed-seed eager, 38 lanes](0032_pod_native_throughput_eager_seed32032.json)
- [fixed-seed CUDA Graph, 38 lanes](0032_pod_native_throughput_cuda_graph_seed32032.json)
- [fixed-seed CUDA Graph, 152 lanes](0032_pod_native_throughput_cuda_graph_b152_seed32032.json)
- [original resident baseline](0032_pod_native_throughput.json)

## Audited results

| Execution | Lanes | Decisions | Episodes | Wall | Decisions/s | Episodes/s | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eager, model seed 32032 | 38 | 19,000 | 187 | 37.908 s | 501.212 | 4.933 | 0 |
| CUDA Graph, model seed 32032 | 38 | 19,000 | 187 | 9.878 s | 1,923.473 | 18.931 | 0 |
| CUDA Graph, 4 copies/opponent, model seed 32032 | 152 | 76,000 | 750 | 21.230 s | 3,579.865 | 35.328 | 0 |
| Existing CPU reference | - | - | 100 | 82.714 s | 168.449 | 1.209 | - |

The fixed-seed 38-lane eager and graph arms have the same model, decks, seeds, 500-step horizon, 19,000 decisions, and 187 completed episodes. CUDA Graph is therefore a strict execution A/B: 3.84x faster for both decision and episode throughput, with identical capture-time actions and no legality or engine errors.

The 152-lane result is the recommended rollout configuration. Against the existing same-machine CPU report it is contextually 21.25x higher in decisions/s and 29.22x higher in episodes/s. That CPU comparison is not a strict same-model A/B because the historical CPU candidate and completion window differ.

## Root cause and rejected shortcuts

At 38 lanes, component CUDA-event timing measured about 0.35 ms for status/reset, 0.004 ms for advance, 1.10 ms for codec, 8.33 ms for actor encoding, 108.70 ms for the decoder, and 5.49 ms for pack/apply. The engine was not the dominant cost; many small dependent decoder launches were.

- A 32-step decoder reached 545.5 decisions/s but produced two real overflows and engine errors because some states require more than 32 selections. It is rejected.
- BF16 reduced encoder time but increased small-GRU decoder time and lowered total throughput. It is rejected for this model.
- CUDA Graph initially could not capture dynamic boolean indexing. Replacing that write with fixed-shape `torch.where` is output-equivalent and covered by a CUDA capture regression test.
- Graph capture must occur under the same `torch.inference_mode()` context as replay. A pre-fix failing diagnostic is retained in the V1 artifact directory.

## Residency and capacity

All accepted graph runs use the full 64-step decoder and 128-option codec. The measured hot path has zero CPU engine calls, CPU policy calls, H2D/D2H copies, and host-value synchronizations, with one final synchronization for metrics. The 152-lane run reserved about 650 MB of Torch memory and 23.4 MB for the engine arena, leaving room for a learner on the 16 GB card.

The 304-lane exploratory run only modestly exceeded 152-lane decision throughput while raising Torch reservation to about 1.20 GB. For shared rollout plus learner training, 152 lanes is the current practical operating point.

These are throughput and legality results, not policy-strength evidence and not strict 0031 continuation. The 0032 actor uses a new POD contract; compatible 0031 parameters are initialization only.

## Formal PPO topology

All 38 CUDA-supported exact decks participate in every update with four lanes
each. They use one shared 0019 Epoch 13 Foundation decoder. The 48 materialized
0022 V11 update-0 decoders were verified tensor-identical to this immutable
foundation, so no grouped deck-head routing is required.

The selected 256-step persistent CUDA Graph canary measured 2,282.910 rollout
decisions/s and 988.121 end-to-end engine decisions/s including PPO. It
completed 176 Episodes and admitted 10,704 focal PPO decisions in one update,
with all 38 decks represented, zero illegal/error rows, behavior log-prob MAE
2.33e-7, and 3.84 GiB reserved memory. This window has much higher effective
PPO-sample throughput than 64 or 128 steps despite lower raw end-to-end engine
decision throughput.

Formal V4 retains all 200 updates as action-decoder plus value-head checkpoints.
That payload has 1,130,243 FP32 parameters (about 4.31 MiB raw) and does not
duplicate the frozen encoder or store optimizer/RNG/rollout state.
