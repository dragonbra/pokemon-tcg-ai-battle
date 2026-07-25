# 0012 V1--V7 SOTA comparison for user selection

## 冻结比较合同

- source corpus SHA-256：
  `36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf`。
- train / validation：221,289 / 17,067 decisions；同一 Yushin Ito winner-only policy。
- V1--V7 训练都使用 `max_action_steps=16`。原始 3,104 个 winner-player replay group
  有 238,439 个 decision frames；0009/0010 只保留 238,356 条，恰好缺失全部 83 条
  expert action length >16 的记录。该缺陷不改变本表内部的相对公平性，但限制所有
  版本的最终能力上限。
- exact 是 validation 完整 action-list 完全一致率；loss 是 token CE。
- perm exact/consistency 用于判断候选顺序依赖；V1 故意保留 position，其他有效版本
  目标是 permutation equivariant。

## 离线关键指标

| 版本 | 状态 / 核心变化 | 参数量 | best exact (epoch) | 该点 loss | best loss (该点 exact) | perm exact / consistency | 完整 epoch runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | 有效；0010 control，保留 option position | 7,154,562 | 83.8929% (9) | 0.223857 | 0.223857 (83.8929%) | 75.1040% / 85.14% | 4,450.65s |
| V2 | 有效；去 position + permutation | 7,154,562 | 83.7698% (10) | 0.230420 | 0.229691 (83.0023%) | 83.7581% / 100% | 5,035.39s |
| V3 | 有效；V2 + source/target role projection | 7,359,362 | 84.3792% (10) | 0.220765 | 0.220765 (84.3792%) | 84.3675% / 100% | 4,294.44s |
| V4 | **无效**；primitive 与 option permutation 错配 | — | 不可比较 | — | — | epoch 1 consistency 99.19% | 298.09s 后中止 |
| V5 | 有效；修复后的 primitive/context | 7,481,666 | **84.5023% (18)** | 0.237962 | **0.216032 (84.3851%)** | **84.4788% / 100%** | 7,048.93s |
| V6 | **失败**；首 batch AMP stop-mask overflow | — | 无指标 | — | — | — | 首 batch 前失败 |
| V7 | 有效但用户中止；V5 + chosen-transition aux | 7,502,574 | 84.3851% (19) | 0.250039 | 0.218391 (84.0159%) | 84.3616% / 100% | 7,942.14s / 21 epochs |

V7 在 epoch 22、约 batch 1000 按用户要求停止；表中只使用 21 个完整 epoch。V4/V6
保留审计身份，但不能作为 SOTA 候选。

## Official-engine 当前证据

首轮评测使用旧 16-step live decoder，因此 V1/V2/V3/V5 都出现 1--3 个
`game_error`，不能作为最终晋级报告：

| checkpoint | 200 局结果 | candidate error | opponent error | 结论 |
|---|---:|---:|---:|---|
| V1 best exact | 162-36，2 error | 2 | 0 | 被长 action contract 污染 |
| V2 best exact | 146-52，2 error | 1 | 1 | 被长 action contract 污染 |
| V3 best exact | 153-46，1 error | 1 | 0 | 被长 action contract 污染 |
| V5 best exact | 155-42，3 error | 1 | 2 | 被长 action contract 污染 |

V9 live-count probe/复评证明 decoder 修复方向成立，但它没有补训练监督：

- V1 probe：157-43、0 error、200/200 finished。
- V1 正式复评：162-36、2 error；两个 error 都发生在 opponent 回合，candidate error=0。
- V2 正式复评：167-33、0 error、200/200 finished。
- V3/V5 修复后复评按用户要求暂停，尚未启动。

这些 run 未共享随机种子，胜局数波动很大；不能用 V1/V2 的单轮胜率反转替代离线
消融结论，也不能据此确定 SOTA。

## 选择含义

- 选 **V3**：结构最简洁的有效提升；exact/loss 同时优于 V1/V2，没有 primitive
  边际特征，因果解释最干净。
- 选 **V5**：当前离线 exact 与 loss 都是全局最佳；best-exact 与 best-loss 分属
  epoch 18/11，重训时仍应同时保留两类 checkpoint。
- 选 **V7**：保留 transition auxiliary，但它没有超过 V5，并在 best-exact 点显著
  恶化 loss；只有明确希望继续研究 predictive representation 时才有理由选。
- V1/V2 可作 control，不是当前指标意义上的 SOTA；V4/V6 不可选。

用户指定结构后，下一版只做一个主要变化：把训练与 live action contract 的长度上限
从 16 提高到 32，恢复 83 条此前未进入 0009 的长 action 标签（最大长度 25），从随机初始化完整重训并进行
零 candidate-error 的官方评测。
