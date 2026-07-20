# Alakazam V7 AutoIter Submit V15

这是一个从不可变归档 `iter-15-patch-priority` 落盘的独立 submission 快照，供
复盘、打包和实际提交使用。它不是正在继续实验的工作目录副本。

## 与 `alakazam_v7_auto_iter` 的主要区别

- `deck.csv` 完全相同，卡组没有修改。
- 本目录的 `main.py` 冻结在 AutoIter 第 15 轮；当前
  `submission/alakazam_v7_auto_iter/` 是持续迭代中的工作区，已经包含后续候选改动。
- 因此，本目录不包含第 15 轮之后的实验规则，例如 swap 回合计数、更多 Bench 接力
  gate、Wondrous Patch/Lana's Aid/Night Stretcher 的窄恢复路线、第二回合攻击与 Boss
  优先级调整，以及 iter-28/29 的 ready-attacker 判定修正。
- 这使它成为可重复提交的 frozen best，而 `alakazam_v7_auto_iter` 继续作为下一轮
  AutoIter 的开发基线；两者可以直接对照，不会因为后续未晋升的实验改动而漂移。

## 快照信息

- iteration：15
- label：`iter-15-patch-priority`
- 原始归档：`dist/alakazam_v7_auto_iter_best_iter-15-patch-priority.tar.gz`
- 原始归档 SHA-256：
  `6ee93ac27471c59a2d19d24f3cc0b04d6325f09c928f5d0a7adc6d8a5dc3d7c7`
- 固定卡组 SHA-256：`deck.csv` 应与 `alakazam_v7_auto_iter/deck.csv` 一致。

提交包仍只包含 `main.py`、`deck.csv` 和 `cg/` 运行时：

```bash
bash scripts/package_submission.sh alakazam_v7_auto_iter_submit_v15
```
