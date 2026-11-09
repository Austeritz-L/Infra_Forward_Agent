# 通信与融合

输入：collective 类型、payload dtype/bytes、group、rank 放置、链路、dispatch/combine 语义、
路由分布、同步与通信/计算依赖。自定义通信 kernel 必须显式标注分类。

1. 区分带宽、消息规模、启动成本、慢 rank 和前驱依赖，观察各 rank 的完成边界。
2. 固定 global batch、有效 token、模型与物理资源，比较 dispatch + expert + combine 全路径。
3. 研究小消息合并、通信与计算融合、reduce-scatter/all-gather 分解；先写出语义与字节数。
4. 验证 collective 顺序、rank 对称性、buffer 生命周期、量化 scale、router 权重和归约精度。
5. 全 rank 前后精度、完整 forward 延迟/吞吐与链路计数一起验收。

出口：可解释的等待/流量变化与完整 forward 证据。通信 microbenchmark 只说明链路/消息条件。
