# 0038 决策记录

- 2026-08-08：0038 作为新的 Action Boundary 基石项目建立；不继承 V5 action/trajectory contract。
- 只读复用 Policy-0806 actor 与已训练 value head 的权重。0038 不 import 任何其他编号项目的可执行代码。
- 官方 engine、ABI、observation schema 与 `select -> list[int]` 返回合同保持不变。
- 第一版 compound action 只覆盖 Dragapult ex 的 Phantom Dive 六指示物分配。检索、随机、setup、对手控制和信息揭示链不压缩。
- allocation head 随机初始化，因此在完成经批准 trace 的 BC/distillation warm start 以前禁止正式 PPO。
- 组合数修正为 `C(n+5,6)=C(n+5,n-1)`；这与 `n=1..5` 的 `1,7,28,84,210` 一致。需求原式 `C(n+5,5)` 与这些计数不一致。
