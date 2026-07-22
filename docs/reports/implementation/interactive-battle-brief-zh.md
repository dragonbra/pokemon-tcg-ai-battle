# Alakazam 对战实验任务书

状态：待在实现机器上执行。

## 目标

实现一个本地可交互的 cabt 调试模式，用于观察 Alakazam 卡组和其他 agent 的行为。

## 模式

### 1. AI vs AI

支持：

- Alakazam 规则 agent vs random agent；
- Alakazam 规则 agent vs Mega Lucario/Archaludon/Crustle Wall 规则 agent；
- 同一卡组的不同策略 agent 对战。

### 2. 人类 vs AI

支持命令行输入：

- 打印当前状态；
- 打印自己手牌和合法选项；
- 输入 option index；
- 调用 `battle_select` 推进；
- 让另一侧 agent 自动行动。

### 3. 暂停/覆盖模式

规则 agent 每步先打印：

- 当前状态摘要；
- 合法选项；
- 推荐动作及原因；
- 等待用户选择“接受推荐”或手动输入另一个合法动作。

## 验收

- 使用合法 60 张卡组启动 battle；
- 至少完成一局 AI vs AI；
- 至少完成一局人工输入若干步的 Human vs AI smoke test；
- 非法输入不能推进状态；
- 输出 `result.html` 或等价 JSON 日志；
- 全部操作和说明使用中文；
- 记录 `cabt`、`kaggle-environments` 和 Python 版本。

## 注意

官方文档提供的是 AI agent 回调和 `battle_select` API，没有现成的官方图形化人类客户端。因此 CLI 人类模式需要我们自行实现，不要把它描述成 simulator 自带功能。
