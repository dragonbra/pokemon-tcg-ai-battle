# agent_v2_dragapult_meta_spread_lock

这是基于公开论坛、公开 meta API、以及最新 public notebooks 选出的第二版提交。

## 为什么换成 Dragapult ex

- 公开论坛和社区 meta API 显示，2026-06-22 的公开 surface 里高占比轴包括 Alakazam、Dudunsparce、Hop's Trevenant、Mega Lucario ex。
- `agent_v1` 的 Mega Lucario 路线稳定，但在公开 matchup matrix 中对 Dudunsparce、Hop's Trevenant、Crustle 这类长局/控制轴偏弱。
- Dragapult ex 的优势是 spread damage、Budew item lock、Crushing Hammer energy denial，能压制需要铺场和资源循环的 deck。
- 本地 candidate round-robin 中，Dragapult 在 6 个候选里小样本第一：`61/100 = 0.610`。

## 来源

- Public notebook: `skarin/phantom-dive-or-go-home-a-dragapult-ex-deck`
- 本地增强：加入 crash-safe wrapper，避免边缘 observation 抛异常时直接输掉。
- 结构：顶层 `main.py`、`deck.csv`、`cg/`。

## 本地验证

- `python -m py_compile agent_v2/main.py` 通过。
- deck 检查通过：60 张；非基本能量不超过 4；ACE SPEC 不超过 1。
- vs random：11 胜 / 1 负 / 0 error。
- vs `agent_v1`：15 胜 / 9 负 / 0 error。
- 提交包：`agent_v2/submission.tar.gz`，已排除 `__pycache__` 和 `.pyc`。

## Kaggle 提交

- Competition: `pokemon-tcg-ai-battle`
- Submission ref: `54076019`
- Message: `agent_v2_dragapult_meta_spread_lock: forum/meta-researched Dragapult ex rule agent`
- Initial status: `SubmissionStatus.PENDING`
