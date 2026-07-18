# 第一版官方 simulator baseline 运行记录

日期：2026-07-18

## 结论

资源已经按职责拆开：`data/official/` 保存官方参考数据，`submission/` 保存实际提交源。使用 `submission/` 内的规则型 agent 和官方 `cg` 二进制，已经完成一局本地 AI vs AI battle。

## 当时的提交（历史记录）

- 入口：[`../../submission/official_water/main.py`](../../submission/official_water/main.py)
- 卡组：[`../../submission/official_water/deck.csv`](../../submission/official_water/deck.csv)
- 模拟器 API：[`../../submission/official_water/cg/`](../../submission/official_water/cg/)
- 卡牌参考：[`../../data/official/`](../../data/official/)

当前卡组沿用下载的官方 starter deck，包含 60 张合法卡牌；`scripts/check_assets.py` 会检查数量、整数 ID 和英文 Card Data 中的存在性。

agent 的第一版策略是确定性的：

1. 初始 `select is None` 时返回 `deck.csv` 的 60 个 ID；
2. 其余时候只在当前 observation 的 `select.option` 中返回索引；
3. 主选择优先攻击，其次贴能量、进化、出牌、Ability、撤退；
4. 没有卡牌专用策略时选择第一个合法选项；可选效果选择跳过。

## 实际命令与结果

资产与 Python 语法检查：

```bash
python3 scripts/check_assets.py
python3 -m py_compile submission/official_water/main.py submission/official_water/cg/*.py scripts/*.py
```

提交包检查：

```bash
bash scripts/package_submission.sh official_water
tar -tzf dist/official_water.tar.gz
```

archive 顶层包含 `main.py`、`deck.csv` 和 `cg/`，不包含卡牌 CSV/PDF。

本机 Ubuntu 20.04 的系统 `libstdc++.so.6` 最高只有 `GLIBCXX_3.4.28`，官方 `libcg.so` 需要 `GLIBCXX_3.4.29`。这份本地 replay/debug 输出是历史临时产物，当前不再保存；正式复盘数据统一放在 `replays/kaggle_<submission-id>/` 下的 Kaggle Episode JSON。

## 当前边界

- 本地 simulator smoke test 不是卡组竞争力结论，当前不作为主要评测来源。
- `main.py` 目前不识别具体卡牌效果，主要用于验证合法动作链路。
- `cg` 是官方预编译二进制；本地运行必须使用兼容的 Python、glibc 和 C++ runtime。
- 现阶段不执行 Kaggle 上传。

## 下一步

先把 observation、OptionType、日志和结果做成稳定的回归数据，再把当前 starter deck 替换为明确设计的最小卡组，并为贴能量、进化、撤退和攻击分别增加可测试策略。
