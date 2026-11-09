# 分布式与 KV 容量

输入：固定物理卡数/节点、global workload、TP/EP/PP/DP/DCP 放置、权重与量化布局、
每卡显存、通信 workspace、graph/临时 buffer、KV dtype/page/layout、prefix/cache 状态。

1. 按 attention 结构建立 KV 容量公式，区分实际 KV、recurrent state、draft 与 cache 元数据。
2. 同时记录每 rank 分配/保留/峰值显存；通信融合或 overlap 可能增加 buffer 和常驻内存。
3. 声明允许改变的拓扑轴，固定有效业务负载与物理资源；保留调度/路由语义与精度。
4. 用长度×并发矩阵验证容量上限、碎片、OOM、吞吐与 forward 尾延迟。
5. 如需声称服务容量提升，另测服务接纳、TTFT/TPOT 与输出一致性。

v0.5新增独立离线选型入口，见[长上下文规范](../docs/domains/deployment.md)：
`ifa deployment <bundle.json> --output <new-directory>`。原replay仍不接受拓扑变化。
2Mi是规划目标，先核实模型原生/扩展质量和runtime限制；token池不等于单请求长度。
按独立完整请求、唯一物理Cache池、最紧张rank预算；DP不拼显存，命中不打折活动KV，
CP只计已证实的持久分片，不自动分摊recurrent/index。host/storage前缀恢复不算活动HBM扩容。
容量、性能和平衡目标共享全rank峰值/接纳、精度和逐场景SLO门槛；缺证明不推荐。
出口：解析模型、实测分配和业务矩阵三者吻合，不能只依据“空闲显存多了”。
