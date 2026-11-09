# Trace对比调查

先读[第五模块规范](../docs/domains/traces.md)，冻结当前service矩阵、revision/patch与物理部署。
全rank选择同逻辑prefill窗口，保留实际query/KV/chunk/Cache/head边界与采集开销证明。

1. 用原始timeline和来源映射确定operator/type/shape/layer，核实稳定logical invocation与collective ID。
2. 新旧Stream ID不直接配对；明确角色。Fusion/rename给有精度/语义证明的共同单元，禁止事件重复归属。
3. 比较work/占比/union/exclusive与通信/Stream交集；不跨rank累加、不把通信duration当纯wire time。
4. 对照绝对work与暴露，区分主导迁移和分母缩小。缺rank、截断、unknown或未匹配行先补证据。
5. 可选链接第四模块的原始native bundle重新验收；profile时长本身不证明precision/服务收益。
6. 新热点只产生needs_deep_analysis任务，继续第二/第四模块的counter、资源和反证流程。

入口：ifa trace-compare <bundle.json> --output <new-directory>。没有足够证据时保留unknown，
不猜shape、不补零、不动态拉伸时间线、不恢复历史GPU实验或替换服务。
