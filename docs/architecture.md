# Architecture

## End-to-end data flow

```text
official observation + legal actions
              |
              v
  semantic feature compiler
  - card/effect prototypes
  - dynamic entities/events/resources
  - structured legal options
              |
              v
 frozen BC-pretrained semantic backbone
              |
       +------+------+
       |             |
       v             v
 policy Option     critic Option
 LoRA branch       base branch
       |             |
 decoder +        value / prize /
 allocation       opponent-meta heads
       |
 public-only checkpoint router
       |
 autoregressive legal action
```

The actor and critic share only the explicitly declared representation path. Critic predictions do not enter actor logits. Trainable ownership and storage independence are tested rather than inferred from module names.

## Behavior cloning foundation

The base policy imitates complete official-engine action sequences. The model does not choose from a fixed global action vocabulary: it scores the legal options supplied at each decision and autoregressively emits option selections and STOP. Conditional actions such as damage-counter allocation use the allocation head.

The semantic representation combines identity embeddings with mechanics. Static prototypes include card type, HP, stage/evolution, attack damage and ordered energy requirements, retreat, weakness/resistance, skills, effects, targets, and conditions. Dynamic tokens keep zones, current damage, attachments, turn restrictions, public knowledge, and event/resource context separate from static rules.

## Reinforcement learning

PPO starts from the BC policy and operates on official-engine trajectories. The final family uses:

- an action decoder and conditional allocation head;
- policy-side final-Option attention LoRA;
- selected shared State attention/fusion adaptation and later bounded FFN/LayerNorm experiments;
- a multi-task critic with win value, prize auxiliary value, and opponent-meta diagnostics;
- model-only checkpoints with exact tensor inventories and strict reconstruction hashes.

Rollout win rate is attributed to the behavior policy that generated the batch. Greedy checkpoint strength is reported only by a separate fixed-schedule evaluation.

## Public-information routing

The final router owns monotonic per-game memory of opponent Pokemon identities that the observation marks public and certain. It may switch among complete policy heads whose shared semantic tensors have been proven content-equal. It cannot consume evaluator deck IDs, hidden cards, final outcomes, or critic meta predictions. Unknown and conflicting evidence fail closed to the default policy.

## Policy identity

A policy is the complete effective inference function, not a decoder filename. Resolution and evaluation follow [RL Promote Champion & Frozen Policy Protocol V1](rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md): resolve the requested identity, verify every component and content hash, bind lanes, then batch. Kaggle-facing candidates are materialized as FP16 storage and strict-loaded into FP32 runtime tensors.
