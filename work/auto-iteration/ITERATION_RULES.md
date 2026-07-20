# 迭代规则

## Gate 与 guardrail

先通过 correctness gate：资产、语法、合法动作、确定性和打包结构均通过。失败 case 必须分类并完成 case resolution，不能用胜率抵消规则错误。outcome guardrail 要求候选不引入崩溃、非法动作或明确的核心指标回退。

## 流程

先运行 control，再运行 candidate；先做 focused evaluation 定位主要假设，最后按同一分母比较。只有候选通过 gate、满足 guardrail 且核心指标达到预设 promotion 阈值，才晋级；否则保留 control 并记录 decision。
