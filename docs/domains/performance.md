# 2. 性能统计与分析

入口：agents/performance.py；现有原始 trace 时间分析：analysis.py。
后续细化 prefill/decode forward 的定义、分位数、吞吐计数、按层/算子/shape 分类和计数器。

报告应保留三层：服务 TTFT/TPOT/请求吞吐；完整 forward 延迟/有效 token 吞吐；算子性能。
算子占比必须声明分母：累计 GPU work、forward 时间暴露或其他经过定义的测量。
累计 work 可以因 Stream overlap 大于墙钟，不直接等于优化优先级或可获得收益。
真实算子性能需绑定架构、ISA、资源、shape/dtype/layout 和原始计数器。

待接：分阶段统计聚合器、CPU/GPU correlation、归因器、真实报告渲染。
