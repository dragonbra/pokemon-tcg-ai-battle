# Alakazam BC v1 Tried：2,021 局 exact 多专家实验

本轮冻结了已经下载的 2,021 个 unique Episode。前 100 名筛选中有 22 个
submission 的 60 张牌 multiset 与 Yushin Ito exact submission `54773249`
一致；逐 replay 再验 hash 后得到 2,085 条专家轨迹，双方都是 exact expert
的对局保留双方动作但只进入一个 split。

训练前审计通过：173,251 条唯一决策，无重复 key、split 泄漏、重复 target、
越界 target、mask target 或 min/max cardinality 违规。数据包含 9,829 条正向
多选、250 条合法空选，最长选择 25 个 target。`context=8` 的 2,057 条记录中
有 1,998 条多选。有效 `DONE` 下一行动作被保留，只有 `action:null` 终局没有
监督标签。

模型沿用 V1 结构和训练预算，从随机初始化训练 20 epochs，best epoch 为 8。
Validation exact-action 为 `76.25%`；test exact-action `75.84%`、multi-action
exact `71.49%`、selection-count `99.78%`、legal-action `100%`。

固定 17×10 评测为 118 胜、47 负、5 个 opponent-isolated engine error，胜率
口径 `69.41%`。相比原 V1 的 122/170，本轮少 4 胜，因此命名为
`Alakazam BC v1 Tried`，不晋级也不覆盖原模型。过程指标有明显取舍：Post-KO
接力从 `210/391` 提升到 `292/424`，但二回合 Powerful Hand 从 62 降到 44，
未拿奖赏攻击率从 `161/641` 上升到 `216/684`。后续奖励函数和模型设计继续
以 experiment 0001、`work/alakazam_bc_v1` 为基线。
