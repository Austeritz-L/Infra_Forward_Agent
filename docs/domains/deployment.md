# 3. 分布式部署选择：Cache 容量与性能

入口：campaign.deployment 与 agents/deployment.py。
目标允许 capacity / performance / balanced，后续再定义约束、权重、Pareto 选择与 tie-break。

输入包括物理卡数/链路、TP/EP/PP/DP 放置、全局有效负载、每 rank 权重/KV/state/graph/workspace。
输出包括可行方案、容量上限、实测延迟/吞吐、OOM/失败原因以及未测选项。
通信融合与 overlap 的额外 buffer 必须计入容量，计算模型与实测峰值需要互相核对。
容量优先仍要满足明确性能下限；性能优先仍要满足长度/并发与精度约束。

待接：模型相关 KV/state 公式、运行时内存采集、方案搜索、跨拓扑公平比较与选型器。
本模块当前不会修改服务，原 replay 的严格拓扑可比性规则保持生效。
