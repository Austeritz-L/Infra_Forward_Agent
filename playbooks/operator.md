# 算子优化

输入：热点名称、调用次数、shape/dtype/layout、路由分布、local/global tokens、所属 rank/device、
对应 forward 窗口与 launch correlation。仅有累计耗时排序时，先检查该算子是否影响完成时间。

1. 建立数学与数值契约：输出、scale/量化、padding、边界 shape、误差容限。
2. 建立可执行 reference 与代表性 shape/路由矩阵；包含偏斜、短 K、空 expert 和多次调用。
3. 检查内存流量、launch 数、同步、占用率与矩阵执行效率；用当前架构的计数器核实假设。
4. 给 KDA/coding agent 一个局部任务，限制一次改动的轴，保留 parent 与 patch。
5. 算子精度通过后才计时，并在 forward 内复验；记录多 Stream/graph/cache 行为。

出口：局部正确 + 整段 forward 验收通过；或者拒绝原因与原始证据。
编译成功、launch 变少、算子 microbenchmark 更快均不单独构成 forward 收益。
