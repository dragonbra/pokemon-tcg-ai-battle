# 0015 execution preflight — 2026-07-27

- Repository state: dirty with pre-existing 0014 and other user work; unrelated changes preserved.
- Python: 3.11; PyTorch `2.11.0+cu128`; CUDA available; W&B `0.28.1`.
- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB total; approximately 12,217 MiB free at preflight.
- Storage: 834 GiB free on the workspace filesystem at preflight.
- W&B: `wandb.Api()` loaded credentials from the user netrc; formal target is private
  `dragon_bra/pokemon-tcg-policy-learning`. No credential value was logged.
- Official opponent catalog: 20 enabled packages were listed successfully.
- Official 0726 ZIP: 4,554 Episode JSON; CRC passed; SHA-256
  `9394d9c4c18dc476cc1fffd5a2dae29c4f2a7afbfa9b2d435876568e2c22fe95`.
- Targeted overlay: 310 unique Episode IDs/content hashes; 81 official-base references and 229
  newly downloaded regular files; no duplicate count.
- Shared core data: 547 trajectories, 55,764 decisions; target-only fixed validation is 12 THIRD
  PTCG Club episodes / 1,000 decisions.
- R15 feature cache: 54,764 train decisions and 1,000 validation decisions; all pass the inherited
  R15 full-action cache eligibility contract.
- Smoke gates passed on CUDA for T0, T1 and safe T3-no-deck: finite loss/gradients, full target
  validation, legal greedy rate 1.0 and reloadable checkpoints.
- Open source gate: no genuine Starmie + Dusknoir registration deck exists in the 0726 catalog;
  formal T2/T3 allocation is forbidden until real source evidence is provided.

Commands used the project module entrypoints under `train/0015_dragapult_conditioned_bc/`; no
wrapper was added to `scripts/`. No Kaggle submission/upload, commit, push or opponent admission
was performed.
