# SP 系列 Zero-shot 构筑实验库

这里长期归档用户提出的 SP 系列 exact 60-card 构筑，以及它们在固定 Arena 合同下的
official-engine zero-shot 评测。SP 构筑是研究候选，不自动加入 Frozen Pool 或正式 opponent
catalog。

## 目录合同

- `index.html`：面向查询的总览入口，按 SP 编号展示构筑、policy、W-L-D、总胜率和先后手结果。
- `manifest.json`：机器可读的系列清单和不可变 run 引用。
- `decks/<SP编号>/deck.csv`：排序后的 exact 60-card ID，每行一张。
- `decks/<SP编号>/manifest.json`：构筑名称、exact deck SHA-256 和卡牌计数。
- `reports/<policy>/<SP编号>/<run_id>/report.html`：该次评测的完整权威 HTML，不得覆盖。

## 当前评测合同

- Candidate policy：Policy-0806 checkpoint
  `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`
- Opponent policy：Policy-0806
- Opponent pool：`0806_kaggle_top100_plus_v1`
- 每套构筑：固定分布 256 局，先攻 128、后攻 128
- official engine 达到 50 个完整回合仍未结束时记为平局
- GPU batching：FP32 inference，batch size 32，batch wait 2 ms

后续新增或复测必须保留 exact deck SHA、policy SHA、pool/schedule 身份、真实 run ID、完整
W-L-D、error/unfinished 和报告相对路径。同一 SP 编号若修改任何一张卡，应分配新的 SP 编号，
不得覆盖原构筑身份。
