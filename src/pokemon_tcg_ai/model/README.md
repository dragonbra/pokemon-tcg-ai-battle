# Model

This is the public entry point for the final 0045 model definition.

- `SemanticPolicy` defines the BC-pretrained semantic backbone.
- `SemanticActorCritic` owns the PPO Actor, training-only Critic, allocation head, and LoRA paths.
- `AdaptationConfig` declares the exact trainable boundary.

The implementation remains split between `semantic_runtime/model/` (token encoders and Transformer)
and `policy/` (Actor/Critic ownership and RL heads). Import public types from
`pokemon_tcg_ai.model`.
