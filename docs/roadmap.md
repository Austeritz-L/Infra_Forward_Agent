# 路线图与验收里程碑

| 阶段 | 产出 | 完成门槛 |
| --- | --- | --- |
| M0：工作流基线（当前） | CPU replay、分析、证据门禁、日志、任务模板、CI | 合成多轮演示；错误/重叠/rank 隔离测试 |
| M1：首个真实 forward | 一个镜像 + runner + 精度器 + trace adapter | 全 rank、C1/C16、固定长度矩阵，前后精度与重复 A/B |
| M2：自动定位 | CPU/GPU correlation、通信依赖、跨 rank DAG、关键路径候选 | 人工归因对照集；未知依赖显式保留 |
| M3：算子与通信优化 | KDA/coding-agent bridge、隔离 patch、回归集 | operator 正确、完整 forward 改善、失败候选留档 |
| M4：分布式与 Stream | 拓扑允许轴、KV 容量模型、融合/overlap 实验 | 固定全局负载与资源；all-rank 正确与峰值内存实测 |
| M5：连续运行 | LLM provider、预算、恢复、停止信号、知识检索 | 可中断恢复、无收益终止、完整可回放事件谱系 |

## 推荐的第一个真实案例

从已有的 S5000/MUSA 经验中选一个能重现实验边界的 MiniMax 或 DeepSeek forward，
先做 read-only 的镜像、模型元信息、服务、设备归属与版本盘点。旧实验结论不等于本仓库基线；
具体型号、模型版本、权重量化、可用节点均需重新确认。不要从历史文档自动恢复已暂停任务。

首个 case 只接一个后端、一个模型、一组固定输入长度与并发。把历史精度、profile、实验失败和
benchmark 原始产物导入统一格式后，再增加跨后端支持。暂不同时承诺 NV/S5000 全链路可用。

## 未来终止条件

除了 max_iterations，还应加入 wall time、GPU minutes、token/cost budget、连续无收益次数、
设备归属变化、用户暂停、验证失败和环境漂移。v0.1 仅实现迭代预算/候选耗尽/异常停止；
使用前端接进程时可以正常终止进程，未完成 run 保留原目录，不会自动续跑。
