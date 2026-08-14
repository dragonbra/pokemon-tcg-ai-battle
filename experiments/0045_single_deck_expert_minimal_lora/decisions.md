# 0045 decisions

- 2026-08-15: Established `0045_single_deck_expert_minimal_lora` as a self-contained deck-007 specialist project derived from 0044 without modifying 0044.
- 2026-08-15: Interpreted the requested cadence as one CUDA-512 Benchmark Tiny V2 every five updates; U10/U20/... reports also satisfy the ten-update evaluate cadence and are not duplicated.
- 2026-08-15: Kept Prize advantage and all other V24 PPO settings unchanged for architecture isolation.
- 2026-08-15: Resolved the opponent to immutable `Champion-G2` because the 0045 specification explicitly names the G2 generalist. Live V24's Champion-G3 is recorded as a non-selected identity and is not silently substituted.
- 2026-08-15: Froze V1 at its complete U5 checkpoint/evaluation boundary. U0→U5 Tiny V2 changed 63.28125%→63.671875% while reference KL remained in the low `e-6` range; an already-started U6 rollout was interrupted before any U6 checkpoint or metric row existed.
- 2026-08-15: Made `0045_expert_cold_start_lr_v1` the default for the first run of every new expert: Decoder/Allocation `1e-5`, Option Q/V LoRA `2e-5`, Critic/Prize `2e-5`. Retained the former Actor rates as explicit `0045_limit_finetune_lr_v1` for a later fresh-version fine-tuning stage. Reference KL remains a distance diagnostic; fixed greedy evaluation determines quality.
- 2026-08-15: V2 has no automatic update limit. The initial U10 cap was revoked before U1 existed; the same V2/W&B identity retained its sole U0 baseline row and restarted with a strict U0-eval-only restart gate. Training continues until the user explicitly stops it.
