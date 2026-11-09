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

v0.5长上下文选型来源与支持边界记录在[部署规范](domains/deployment.md)及
[模型长度观察](../configs/deployment/context-evidence.json)。参考vLLM版本化CP部署、
特定后端PCP/DCP布局与SGLang HiCache的分层语义；不复制实现，不将某后端支持泛化到S5000/NV。
模型长度资料不是本仓库的runtime、质量或显存验收。


v0.6算子来源与适用范围见[第四模块](domains/kernel.md)及
[架构调查配置](../configs/kernel/architectures.json)。KDA按acdd793固定流程版本借鉴任务/证据设计；
MUTLASS、NVIDIA PTX/CUDA及各厂商profiler文档只作为能力调查来源，逐工具链核实实际产物。
SM103不继承未经核实的SM100 integer tcgen05或资源表，CPU合成数据不能认证任何后端。
