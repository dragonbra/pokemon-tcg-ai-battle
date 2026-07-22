# Kiyota Notebook 参考笔记

来源：https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code

## 方法结构

这份 notebook 是一个 sample code，不是标准 PPO。它使用：

```text
官方 battle API
+ 官方 Search API
+ Transformer-like state/action model
+ MCTS
+ self-play
+ Huber regression targets
```

模型 `MyModel` 有两个输出：

- encoder value：当前状态价值，使用 24 个稀疏状态槽位、EmbeddingBag 和 Transformer
  Encoder。
- decoder policy：对当前最多 64 个 action combination 逐个输出分数，动作特征包含
  OptionType、Attack ID、Card ID、SelectContext、目标和区域。

模型大致使用 `d_model=128`、2 个 attention heads、encoder/decoder 各 1 层、
`SEARCH_COUNT=10`。MCTS 的搜索系数和动作数量限制是固定超参数，不是 Torch 参数。

## 训练流程

每轮先用 MCTS 对随机 agent 评估，然后进行两方 self-play。self-play 结束后，根据
最终胜负（胜 1、负 -1、和 0）向后生成 value target；根据 root/child 搜索价值差生成
policy target。之后用 masked Huber loss 训练 value 和 policy。

它没有使用每一步的 Prize/damage shaping，也没有把 policy gradient 直接反向传播穿过
MCTS。Search API 和 MCTS 负责产生不可微的训练目标。

## 对本项目的启发和限制

可复用：

- 动态合法 action candidate 编码；
- policy + value 双头；
- 官方 Search API 作为转移真相；
- self-play/search target 训练循环。

不能直接照搬：

- 对手隐藏区域使用固定占位卡；
- 自己 deck/prize 独立随机采样导致资源不一致；
- 最多 64 个组合的简单截断；
- 没有 V6 的回合时点、资源账本和牌库保护状态；
- 只对随机 agent 的小规模评估。
