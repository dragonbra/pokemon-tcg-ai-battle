# GitHub write-up outline

## 1. The real challenge: structured actions, not just card choice

Explain the official turn/action contract, legal-option enumeration, autoregressive STOP, and conditional allocation. Use one compact action trace rather than a history of experiment IDs.

## 2. Behavior cloning as the policy foundation

Describe full-action imitation, exact-deck/source provenance, held-out Episode splits, and why offline exact-action accuracy is necessary but not a strength metric.

## 3. Learning card mechanics from the engine

Contrast the early identity-heavy representation with the final identity-plus-semantics prototype encoder. Clearly state that representability tests do not prove a behavior failure was eliminated.

## 4. Building a critic that can support PPO

Present the progression from a weakly validated scalar critic initialization to replay-pretrained multi-task value features. Do not claim that the earliest PPO lacked a critic.

## 5. Adaptation placement and gradient ownership

Show the actor/critic fork, policy-only Option LoRA, shared State adaptation, checkpoint inventory, and the compute cost of moving the backward frontier earlier in the network.

## 6. Official-engine PPO

Cover on-policy collection, terminal win/loss, prize shaping chronology, GAE, KL guards, model-only checkpoints, and the distinction between source-policy rollout and post-update greedy evaluation.

## 7. Frozen opponents and the identity bug

Use the Hybrid-0806 incident as the engineering lesson: names and shapes are insufficient; full effective content hashes must be resolved before batching.

## 8. From one checkpoint to public-information routing

Explain monotonic public-card memory, fail-closed routing, shared-backbone equality, and why evaluator labels/hidden deck IDs never enter the actor.

## 9. Results with honest boundaries

Lead with the 2,048-game official-engine public-router result, label it selection-set evidence, separate package smoke from strength, and avoid unsupported external-submission claims.

## 10. What failed and what we learned

Include incomplete checkpoint inventories, hybrid policy materialization, representation/backward cost, evidence drift, and exact-deck identity confusion. These failures are the most reusable part of the write-up.

## 11. Reproducibility

Link the curated source tree, model design, canonical identity protocol, tests, LFS artifacts, checksums, and publication/licensing caveats.
