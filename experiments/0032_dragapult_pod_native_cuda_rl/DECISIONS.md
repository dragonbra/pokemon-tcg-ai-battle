# 0032 Decisions

## 2026-08-05: Follow the user-selected Mega Lopunny/Froslass 001 pathway

The formal focal deck is `mega_lopunny_ex_mega_froslass_ex_001`, using the
exact frozen deck package. Its 0031 epoch-2 validation exact-action score is
0.657664974619 over 4,925 decisions. Although Marnie's deck ranks higher on
that offline metric, the user explicitly selected Mega Lopunny/Froslass for
business-value potential. Offline ranking does not override that decision.

V2 is selection/performance preflight because its artifact path was already
used. Formal PPO is V3. The epoch-2 checkpoint remains initialization only.

## 2026-08-05: Rotate one real 0022 frozen opponent per update

Of the 38 CUDA-supported frozen decks, 35 have real 0022 V11 decoder heads.
V3 snapshots those 35. Each update repeats one head over 152 lanes and rotates
round-robin across updates. This measured faster than simultaneous five-head
and 35-head routing, while still covering every admitted real frozen head once
per 35 updates. The pool reduction is explicit rather than filling missing
heads with random or mislabeled policies.

The formal one-update canary passed at 2,346.092 rollout decisions/s and
1,390.477 end-to-end decisions/s including PPO, with log-prob replay MAE
1.84e-7 and zero engine/legality errors. This topology is accepted for V3.

## 2026-08-05: Use a POD-native actor contract

0032 consumes the CUDA engine's resident `PolicyCodecV1` tensors directly. It does not reconstruct the 0031 chronological event history, opponent-hand memory, known deck order, causal prize ledger, or semantic effect trees. Consequently, 0031 is an initialization source rather than a behavior-compatible parent policy.

The first version is `V1_pod_native_adapter`. It is an adapter and throughput validation version, not a PPO training run. A new formal PPO version may be allocated only after the tensor contract, explicit weight transfer, CPU/GPU codec parity, resident action legality, and throughput gates pass.

The focal deck is `dragapult_ex_001`. The opponent snapshot is the 38-deck admitted subset from the frozen51 CUDA support audit. Source/team identity remains dataset provenance and is not an actor-visible input.

## 2026-08-05: Expand the official resident codec to 128 options

A 500-step resident rollout reached a real official-engine state with 81 legal options against `crustle_cornerstone_mask_ogerpon_ex_001`. The original 80-option `PolicyCodecV1` buffer failed closed with `OfficialPodError::kOptionOverflow`; excluding that deck would only conceal a capacity defect because the official state supports 128 options.

The official CUDA arena and 0032 actor contract now use 128 option slots. The generic prototype engine retains its existing 80-option ABI. The old 80-slot diagnostic corpus and reports remain immutable; the replacement corpus is `artifact/adaptation_corpus_option128/`.

After the change, CPU/CUDA codec parity passed over 400 decisions with zero field or status mismatch. A 38-lane, 500-step resident run completed 19,000 decisions and 199 episodes with zero engine errors, illegal actor rows, or action-step overflow. This closes the V1 runtime adapter gate, not the policy-strength or formal PPO gate.

## 2026-08-05: Treat 0031 weights as initialization only

The explicit transfer copied 11,052,162 of 14,579,843 parameters (75.8044%). On the bounded cross-source diagnostic corpus, the transferred arm did not consistently outperform random initialization. The scanned 0031 raw dataset contains no exact `dragapult_ex_001` deck decisions, so V1 provides no Dragapult-specific imitation evidence. Formal training requires a new version with a fresh optimizer, explicit reward/value and rollout contracts, a frozen opponent snapshot, W&B online logging, model-only checkpoints, and frozen official-engine evaluation.

## 2026-08-05: Use CUDA Graph replay for the resident actor

Component timing showed that the official engine, codec, and action apply were not the primary bottleneck. At 38 lanes, the 64-step autoregressive decoder consumed about 108.7 ms per step, or 87.7% of the eager resident loop. Finished rows still execute the fixed loop so the path remains device-resident and supports rare large selections.

Reducing the loop to 32 steps reached 545.5 decisions/s but failed with two real `max_count > 32` states. BF16 reduced encoder time but made the small GRU decoder slower. Neither variant is accepted.

The decoder's dynamic boolean write was replaced by an exactly equivalent fixed-shape `torch.where` update, making greedy inference CUDA Graph capturable. A regression test requires captured and eager sequences, lengths, and legality to be identical. Capture and replay run under the same inference-mode context.

With model seed 32032 fixed, eager and CUDA Graph 38-lane runs followed the same 19,000-decision, 187-episode trajectory with zero errors. CUDA Graph improved throughput from 501.212 to 1,923.473 decisions/s and from 4.933 to 18.931 episodes/s, a strict 3.84x execution speedup. Four copies of each of the 38 opponents (152 lanes) reached 3,579.865 decisions/s and 35.328 episodes/s with zero errors. This is the recommended single-GPU rollout batch; 304 exploratory lanes showed only a small additional throughput gain while reserving substantially more memory.
