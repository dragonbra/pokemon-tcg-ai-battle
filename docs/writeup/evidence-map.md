# Write-up claim-to-evidence map

| Claim | Evidence | Class | Safe wording |
|---|---|---|---|
| Final representation combines IDs with engine-derived mechanics | `src/pokemon_tcg_ai/semantic_runtime/model/prototype_encoder.py`, field inventory, feature tests | Code + contract tests | The relationship is represented; no claim that one anecdotal error was eliminated |
| Actor and critic inference paths are separated | `src/pokemon_tcg_ai/policy/actor_critic.py`, `test_actor_critic_separation.py` | Code + mutation/gradient tests | Critic outputs do not enter actor logits in the final architecture |
| Complete-delta checkpoints retain every trainable tensor | `training/checkpointing.py`, `test_complete_delta_checkpoint.py` | Code + strict round-trip tests | Inference state reconstructs exactly; optimizer trajectory does not |
| Kaggle-facing policy uses FP16 storage and FP32 runtime | canonical RL protocol, candidate/export tests | Identity contract | Applies only when deployment audit is PASS |
| Public router scored 1,382–666–0 | `docs/model/evaluation/final-public-router-cuda2048.html` | 2,048 official-engine CUDA games | Selection-set experimental result against complete Policy-0809 |
| Historical Frozen-0806 CUDA result was invalid | canonical policy protocol | Forensic policy-identity audit | A hybrid policy identity is invalid as Frozen-0806 |
| Encoder adaptation caused the final gain | No clean single-variable official-engine ablation | Unsupported causal claim | Say only that adapter tensors changed and results were suggestive |
| A particular late archive was submitted externally | No retained Kaggle receipt | Unknown | Do not claim submission or quota exhaustion |
