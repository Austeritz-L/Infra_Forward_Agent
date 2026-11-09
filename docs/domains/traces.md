# 5. 可选 trace 对比与分析

入口：campaign.trace_comparison.pairs（按架构 id）；规划：agents/traces.py。
没有 pair 时显式 skipped。提供 pair 只表示输入文件存在，不表示已进行语义对齐或差异计算。

后续细化：同场景/同阶段/全 rank 对齐、forward 边界、事件名称映射、算子 shape、
collective、CPU launch、Stream/event/correlation、瓶颈迁移和开销归因。
输出应同时展示完整 forward、算子 work、通信暴露、重叠与缺失依赖的变化。
Profiler 采样/采集开销与不同工具的口径必须注明；收益仍由无 profiler 计时判定。

待接：TraceComparator、跨格式 adapter、匹配算法、差异报告和依赖支持的关键路径分析。
