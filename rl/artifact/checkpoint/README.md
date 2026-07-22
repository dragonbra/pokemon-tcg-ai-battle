# Local checkpoints

Store model checkpoints under the exact same `0001-experiment_name` directory
name used by `rl/_runs/` and `rl/_runs/tensorboard/`. Checkpoint payloads are
intentionally ignored by Git; configs, metrics, summaries, and evaluations are
tracked in the run record.
