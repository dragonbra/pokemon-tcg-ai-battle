# V2 Host Reboot And V3 Restart Decision

Date: 2026-08-02

`V2_james_cox_raging_bolt_ablation` was interrupted by a user-requested host reboot after two complete epochs. No 0025 training or watchdog process survived the reboot.

The canonical V2 JSONL contains valid complete records for epochs 1 and 2. The reboot also left 2,159 NUL bytes after those records in the 8,192-byte metrics file. The original file, twelve model-only checkpoint slots, TensorBoard event, and W&B run identity are retained unchanged as interruption evidence; V2 is not eligible to be described as completed.

Per the user decision, V2 will not be resumed. `V3_james_cox_raging_bolt_restart` starts from newly constructed models and newly initialized AdamW optimizers using the configured seed. It reuses only the frozen paired dataset, split, schema, and experiment configuration; it does not load any V2 checkpoint, optimizer, RNG, DataLoader, or W&B run state.

V3 completed normally after 21 epoch records and 12,972 optimizer updates in 57.78 minutes. All three arms stopped under the frozen patience-6 validation-loss rule, the foreground watchdog emitted `TRAINING_MONITOR_COMPLETE`, and W&B finished in `synced` state. The semantic arm improved substantially but did not beat either legacy control on offline validation: best greedy exact was 73.42%, versus 79.62% for legacy default and 78.77% for the parameter-matched legacy arm. This is an imitation result, not an official-engine gameplay-strength conclusion.
