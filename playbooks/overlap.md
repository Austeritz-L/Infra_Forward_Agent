# 多 Stream overlap

输入：Stream/event DAG、tensor 读写、通信 group、allocator 生命周期、graph capture、
CPU launch、同步点和 forward 完成边界。

1. 先找独立工作和等待原因；时间上没有重叠不代表工作可以并行。
2. 把隐式默认 Stream 依赖改为可核对的显式 event，检查跨 rank collective 顺序。
3. 测试非默认 Stream、重复调用、不同路由、graph capture 和状态复用。
4. 使用覆盖所有相关 Stream 的 forward 边界计时，防止漏计异步尾部。
5. 比较精度、墙钟、吞吐、峰值 workspace 与计算/通信资源竞争。

出口：全路径加速和依赖正确的证据。重叠面积增大、GPU busy 增大并不自动意味着吞吐提升。
