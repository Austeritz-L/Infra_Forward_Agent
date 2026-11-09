# 分布式与 KV 容量

输入：固定物理卡数/节点、global workload、TP/EP/PP/DP/DCP 放置、权重与量化布局、
每卡显存、通信 workspace、graph/临时 buffer、KV dtype/page/layout、prefix/cache 状态。

1. 按 attention 结构建立 KV 容量公式，区分实际 KV、recurrent state、draft 与 cache 元数据。
2. 同时记录每 rank 分配/保留/峰值显存；通信融合或 overlap 可能增加 buffer 和常驻内存。
3. 声明允许改变的拓扑轴，固定有效业务负载与物理资源；保留调度/路由语义与精度。
4. 用长度×并发矩阵验证容量上限、碎片、OOM、吞吐与 forward 尾延迟。
5. 如需声称服务容量提升，另测服务接纳、TTFT/TPOT 与输出一致性。

v0.1 只生成缺证据任务，不能从时间 trace 推算 KV 容量；严格环境匹配暂不接受拓扑变化。
出口：解析模型、实测分配和业务矩阵三者吻合，不能只依据“空闲显存多了”。
