 tail -f rl_runs/0014_faithful_board_causal_features/versions/V8_0010_faithful_100epoch_strict_epoch_axis/artifact/training_metrics.jsonl \
    | jq --unbuffered -c 'select(has("bc/validation/exact_action")) | {
        epoch: ."trainer/epoch",
        exact_action: ."bc/validation/exact_action",
        teacher_exact: ."bc/validation/teacher_exact_action",
        validation_loss: ."bc/validation/loss",
        decisions_per_second: ."system/train_decisions_per_second",
        epoch_seconds: ."system/epoch_seconds"
      }'