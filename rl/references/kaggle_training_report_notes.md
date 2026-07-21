# Kaggle Training Report 参考笔记

来源：https://storage.googleapis.com/kaggle-forum-message-attachments/3494938/46495/report.html

该 report 从 metrics logs 生成静态 HTML，结构分为：

1. Training Health：PPO loss、value loss、entropy、KL、clip fraction。
2. Win Rates：整体胜率、每个 opponent 胜率和训练阶段热力图。
3. Game Statistics：结束原因、回合数、非法动作率。
4. Strategy：动作频率、reward component breakdown、特定策略使用率。
5. Advanced：能量附着热力图、起手牌条件胜率和 Prize tempo。

本项目第一版保留“训练健康度、冻结胜率、对局审计、策略行为、reward breakdown”五
个方向。PPO 专属的 KL/clip 指标只有使用 PPO 时才加入；特定卡组的 Proton 指标不直接
复用。最终胜率、完成率、非法动作率、step-limit 和 per-opponent 结果必须与训练 loss
分开看。
