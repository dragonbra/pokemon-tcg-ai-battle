# Decision 003: First rollout eval-mode contract

Date: 2026-07-28

## Root cause

V10 update 1 reported `ppo/rollout_log_prob_mae=0.181`. V9 had the same large error only on its
first update and approximately `4e-7` thereafter. The model configuration contains dropout 0.10.
`run_ppo` loaded the model in the default train mode, collected the first rollout, and only then
called `model.eval()` inside `PPOTrainer.update`. The frozen behavior snapshot was always eval
mode, so its denominator probability did not represent the first rollout's dropout-perturbed
behavior distribution.

This violated the stated contract that legal categorical action sampling is the only policy
stochasticity. It is independent of the V10 `gae_lambda=1.0` hypothesis.

## Decision

- Stop V10 after update 1 and before update 2 backward.
- Do not use the V10 checkpoint as a branch point or strength artifact.
- Load the live PPO model directly into eval mode before constructing its collector.
- Make `RolloutCollector` reject a train-mode model so later callers cannot silently regress.
- Preserve dropout modules and checkpoint weights; eval mode disables dropout without changing
  the model graph.
- Start `V11_ppo_lambda1_eval_mode` again from the clean V9 update-50 checkpoint, with the exact
  V10 hyperparameters and a fresh optimizer/rollout.
- Require first-update rollout-log-prob MAE near floating-point tolerance before treating V11 as
  an on-policy lambda comparison.

The regression test first demonstrated that a train-mode model was accepted, then passed after
the collector guard was added. The full 0017 policy/training contract suite also passes.
