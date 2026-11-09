# 4. 架构相关的算子优化

入口：agents/kernel.py、architectures、integrations/kda；方法：playbooks。
后续细化 S5000、SM90、SM103 的 ISA、编译工具、矩阵指令、异步搬运/同步及资源分配。

可复用部分：数学与数值契约、数据流、shape/路由矩阵、候选谱系和验证流程。
目标架构需重新核实：指令合法性、layout、寄存器/shared memory、占用率、同步和性能代价。
MoE 不只包含 GEMM；router、padding、dispatch、activation、combine 和 EP 通信都要保留语义。
通信融合与 Stream overlap 作为独立假设，检查跨 rank 顺序和 buffer 生命周期。

待接：KDA/coding agent bridge、架构优化器、编译/资源采集、隔离 patch executor。
最终产物必须通过完整 forward 和所需服务场景复验。
