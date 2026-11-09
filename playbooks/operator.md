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


v0.6执行规范见[第四模块](../docs/domains/kernel.md)。先用ifa kernel-task导出分析任务，再由外部
执行器形成候选与原始报告，ifa kernel-evaluate做确定性验收；命令均不自动执行GPU或调用KDA。
所有优化必须插桩/源码关联、all-rank timeline、counter或已验证proxy、匹配上限和定量反证计划。
未知metric不填0；profile/replay、细粒度插桩和native timing分离；原假设不支持则回到分析。
精度覆盖默认/非默认Stream、graph（启用时）、重复state、边界shape与路由偏斜，再计时、再精度。
算子快但forward或服务退化即拒绝；额外workspace不得吃掉原已验证的长上下文KV slots。
