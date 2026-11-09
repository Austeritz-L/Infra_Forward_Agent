# 架构来源与实现关系

查阅日期：2026-11-09。以下为工作流/分析思路的参考，未 vendoring 上游代码或子模块。

| 来源 | 借鉴内容 | 在本项目中的边界 |
| --- | --- | --- |
| [NVlabs/kda](https://github.com/NVlabs/kda/tree/acdd79343c5e9d0c7cb96289f26fd3f1051ce785) / [agent-flow](https://github.com/NVlabs/kda/blob/acdd79343c5e9d0c7cb96289f26fd3f1051ce785/docs/agent-flow.md) | 契约、独立工作区、计划、候选、验证、证据决策 | 本项目独立实现编排；尚无调用 KDA 的代码 |
| [HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis) | 按 rank 的计算/通信/内存/idle 拆解与 overlap 分析 | v0.1 自行实现小型时间并集；未集成 HTA 的 correlation 或关键路径算法 |
| [Perfetto Trace Processor](https://perfetto.dev/docs/analysis/trace-processor-python) | 将采集格式与可编程分析分离 | 作为未来大 trace SQL adapter 的参考；尚未依赖或下载二进制 |
| [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) | Chrome trace 作为可交换的原始证据 | 当前消费已导出的 complete events；不启动 profiler |

KDA 的 reusable workflow 与 task implementation 分离原则尤其适合跨架构工作：
控制面保留证据契约，具体 kernel、profiler、量化布局和通信实现留在任务工作区与硬件后端。
引入上游代码或自动拉取子模块时，需按当时 revision 重新检查许可和运行环境。
