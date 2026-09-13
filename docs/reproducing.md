# Reproducing the release

## Local environment

```bash
git lfs install
git lfs pull
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rl]'
```

The repository targets Python 3.11. CUDA evaluation additionally requires a compatible PyTorch/CUDA toolchain, CMake, Ninja, and the hardware architecture declared by `engine_cuda_2_0`.

## Integrity gates

```bash
git lfs fsck
sha256sum -c archive/submission/dist/2026-09-13-final-packages.sha256
PYTHONPATH=src:. pytest -q src/pokemon_tcg_ai/tests
```

Model-only checkpoints deliberately exclude optimizer, scheduler, GradScaler, RNG, data-loader position, and rollout buffers. They reproduce inference and evaluation identity; they do not promise bit-exact continuation of an optimizer trajectory.

## Engine boundary

`engine/source/` is read-only. Build products belong under `engine/build/`. The CUDA engine is a separate implementation and must pass policy-identity and CPU/CUDA semantic parity gates before its aggregate results are trusted.

## Training outputs

New local runs write beneath ignored `runs/`. Canonical scalar ordering is local JSONL flush, TensorBoard, then W&B. A W&B failure cannot erase local metrics or turn a partial run into a successful one.

Inspect the complete public configuration without launching training:

```bash
ptcg-train
```

The output must report Policy-0814, 125 trainable tensors, and 5,597,590 trainable parameters. To
launch PPO, first build CUDA Engine 2.0 and generate the competition-derived `official_rules.bin`
according to `engine_cuda_2_0/docs/usage_guide_zh.md`. That rule pack is intentionally untracked due
to its upstream terms. Either place it at
`engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin` or set `PTCG_RULE_PACK`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
ptcg-train --launch-formal --updates 10 --wandb-mode online
```

The trainer constructs its own U0 from committed BC assets; no historical `runs/V*` checkpoint is
required.
