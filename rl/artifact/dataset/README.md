# Local BC datasets

Only behavior-cloning runs create directories here. Use the exact same
`0001-experiment_name` directory name as the corresponding tracked run,
checkpoint, and TensorBoard log. Dataset payloads are intentionally ignored by
Git; the tracked data manifest and summary belong in `rl/_runs/<run-name>/`.
