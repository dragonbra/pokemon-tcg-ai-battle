# Experiment 0027: Semantic Foundation Fine-tuning

This experiment starts from the 0025 canonical semantic policy and its best-greedy model-only checkpoint. The vendored source is under `vendor/0025_source/0025_semantic_foundation_pretraining` and is copied from branch `dev/cyd_main` at the fetched remote revision.

The five notebooks are ordered as follows:

1. `kaggle/01_source_inventory/01_source_inventory.ipynb`: inventory date-bearing Top replay inputs for 2026-07-15 through 2026-08-01.
2. `kaggle/02_winner_slice/02_winner_slice.ipynb`: keep strict positive-reward winner trajectories only.
3. `kaggle/03_canonical_features/03_canonical_features.ipynb`: compile the 0025 canonical actor contract.
4. `kaggle/04_package_dataset/04_package_dataset.ipynb`: validate and write immutable gzip shards.
5. `kaggle/05_train_dual_t4/05_train_dual_t4.ipynb`: fine-tune on two T4 GPUs from `best_greedy_exact.pt`.

Run the first notebook to inspect the dated source inventory. In notebook 02, leave `SELECTED_DATES = None` for the full inclusive window or set an explicit date set after reviewing the inventory; notebooks 03 and 04 consume that slice and notebook 05 consumes the packaged dataset.

The checkpoint is not present in the Git tree. The training notebook requires an attached Kaggle artifact named `best_greedy_exact.pt` and verifies SHA-256 `adc4eaeca1e62a28bbd762e8e513212c94941044bbc85673f02bbeadc1865aa3` before loading it. No source/team identity enters actor tensors; all source provenance stays in audit metadata.

For notebook 05, attach the notebook 04 dataset output, the vendored 0025 source directory (including both prototype JSON files), and the V4 checkpoint artifact. The notebook fails closed unless it finds exactly one complete dataset manifest, exactly one source package, two CUDA devices, and the committed checkpoint hash.
