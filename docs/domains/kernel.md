# 4. 热点算子优化：先深度分析，再以完整 Prefill 与服务验收

v0.6增加硬件中立的 kernel/、可选插桩接口、KDA风格任务导出和三层离线验收。
**每个优化都必须有插桩关联、性能工具证据、资源模型与可证伪假设，修改后再次分析。**
只有Top-N、源码猜测、编译成功或局部benchmark，都不足以进入正式推荐。
控制面不编译、不启动profiler、不调用KDA/LLM、不操作GPU或替换服务。

配置入口：[prefill.json](../../configs/kernel/prefill.json)；
架构调查入口：[architectures.json](../../configs/kernel/architectures.json)。
它们记录设计契约和来源，不认证某个驱动、编译器或设备已经支持这些路径。

## 与前三个模块及KDA的连接

1. **Testing**：冻结当前campaign全部service workload ID、实际输入manifest、Cache准备和精度标准。
   沿service→实际prefill→operator捕获shape；不能用客户端并发×prompt长度猜M或batch。
2. **Performance**：直接调用第二模块的build_report，重新分析原始bundle，保留all-rank时间线、
   work/union/exclusive、实际工作/流量和匹配上限。暴露时间是调查线索，不是因果节省时间。
3. **Deployment**：重跑第三模块的原始deployment bundle，核对已选结果、rank、驻留slots与KV预算。
   单纯复制一份eligible=true的JSON不构成一致证据；仍依赖外部采集器与验证器的真实性。
4. **Kernel**：task_digest绑定上述全部固定输入/SHA256、父版本、数学/精度契约及采集计划。
   生成docs/draft.md、docs/plan.md和kernel.task.json，缺分析时标记analysis_required。
5. 外部coding agent在独立工作区实现一个假设，产出patch/binary/验证报告；三层通过后重新分析。

借鉴固定版本[KDA Agent Flow](https://github.com/NVlabs/kda/blob/acdd79343c5e9d0c7cb96289f26fd3f1051ce785/docs/agent-flow.md)
的task contract、draft→plan、候选谱系和证据决策。深度分析、KV预算与服务复验是本项目增加的约束，
不是声称上游已实现这些功能。未vendoring源码/skill，也没有自动执行KDA。
任一候选被拒绝都保留原因与材料；下一轮修改已选基线后必须重新形成任务与fingerprint。

## 深度分析的必需证据

| 层次 | 必须回答 | 采集与检查 |
| --- | --- | --- |
| 服务与CPU供给 | GPU是否在等请求、调度或launch？ | request/queue/step/launch关联；成对观察TTFT和forward，不相减两个P50 |
| 时间线 | 哪个rank/Stream/调用影响结束？ | 全rank完整forward窗口、Stream/event/collective关联；完整性不足先补测 |
| 算子输入与数据流 | 实际做了多少有效、padding、重算和复制？ | 源码/IR/shape/layout、路由直方图、量化scale、真实query/chunk/KV长度 |
| 资源与上限 | 接近哪条资源界，还是实现效率低？ | native timing、FLOP/op/inst定义、HBM/L2/shared字节与匹配物理/持续/同shape参考 |
| 硬件执行 | 算力、搬运、发射、等待或资源压力？ | 矩阵/向量管线、occupancy、stall、寄存器/共享/TMEM、spill/local memory和grid waves |
| 假设检验 | 改什么，哪些观测应变化，怎样推翻？ | 单因素、量化counter/proxy预言、替代解释、修改后同口径采集与反证 |

diagnosis逐case必须给executed_work、actual_memory_bytes、occupancy、stall、spill_bytes的值、
单位、计数定义和原始来源；stall可以是明确的一类停顿比例/周期，不能把所有stall混成未知总数。
字段用于跨后端语义归一化，不要求不同厂商metric同名，更不能直接比较它们的原始百分比。
缺metric填null并保留unknown；没有该counter时，可用单独验证过的插桩/微实验proxy，
注明proxy_validation_source、误差与适用范围，否则不能进入候选实现阶段。

预言在baseline diagnosis的test中冻结metric、increase/decrease和最小相对变化。
candidate diagnosis必须保持单位/计数定义与test一致，数值独立检验预言；不支持则拒绝并继续分析。
这是一项实验机制检查，不能仅由counter变化宣布因果关键路径或服务提升已被解释。
真实收益仍由下面的native A/B决定；反证后需要更新分析与任务，不能静默改成功标准。

### 插桩规则

嵌套记录service/request → prefill step/chunk/layer → wrapper → kernel/collective，
至少带rank/device/stream、ifa_request_id、ifa_step、ifa_operator、ifa_compute_type、
ifa_signature、ifa_invocation、ifa_layer、global/local new tokens、query/KV长度和Cache tier。
保留launch correlation、设备事件和buffer生产/发布/消费/回收关系；只采ID与结构元数据即可。

kernel.instrumentation.operator_range提供默认关闭的CPU range，sink与厂商push/pop由调用方注入。
异常退出也平衡pop并标记completed=false。它不依赖torch/CUDA/MUSA，不同步设备；
输出明确是**CPU launch range，不是GPU耗时**。NVTX或MUSA range适配器负责与GPU事件对应，
本版没有内置厂商adapter，也不把CPU duration拿来验收算子性能。

阶段耗时可用同Stream设备event，全部相关Stream完成后再读；不要在每层/每kernel后插全设备同步，
那会改变原始overlap。线程内临时timestamp、原子计数或全局buffer插桩应限量抽样、检查溢出，
记录额外寄存器/访存和开销。native正式轮必须完全关闭profiler与debug插桩，并确认无新JIT。

### 性能工具与分离采集

- NVIDIA：用[Nsight Systems/NVTX](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)定位CPU/GPU、
  launch、Stream和通信；用[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  的版本可用sections/metrics与source/SASS关联深入分析。先查询工具版本/metric列表，再选最小集合。
- S5000：用Moore Perf System作系统时间线，按[Moore Perf Compute指南](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/user_guide/)
  使用mcu采集kernel级资源/内存/计算指标。保存安装版本、实际sections、原始mcu-rep及可搜索导出。
  不假设旧报告采过全指标；缺字段保留未知。工具/监控资源冲突要与资源持有人协调，不能停止他人任务。
- Compile pass保留target、operand/accumulator、编译日志、寄存器/shared/local/TMEM用量、
  binary SHA256和反汇编/IR证据；source命名或API调用不是实际指令执行证明。
- Sanitizer/竞态/越界/未初始化检查与formal timing分开；按当前工具支持能力选择检查项目。
  全rank、graph、多Stream与state复用还需要专门的运行测试，sanitizer不能替代它们。

Timeline、counter/replay、细粒度插桩、reference校准和native timing是不同实验，分别保存pass ID、
Cache控制、时钟、过滤条件、完整窗口、输入/状态重置与开销对照。不把counter replay的时间当native。
跨rank persistent kernel/collective可能依赖其他rank前进；单进程kernel replay可能破坏这种关系。
先验证采集模式能保持协作，再协调all-rank/application pass，或提取有边界的本地子阶段校准。
若出现timeout、不完整payload、NaN或profiler错误，本轮无效；不能将自旋时间直接解释为链路传输。

## S5000、SM90、SM103 的实现差异

| 方面 | Moore S5000 / MP31调查路径 | NV SM90调查路径 | NV SM103调查路径 |
| --- | --- | --- | --- |
| 矩阵与累加 | SQMMA的实际operand pair、tile/K、FP32累加及scale | mma.sync或合法WGMMA，warpgroup协作、寄存器累加 | 合法tcgen05浮点变体、TMEM累加、CTA-group与epilogue |
| 异步搬运 | TME descriptor、swizzle、barrier/完成与重用 | TMA/mbarrier、producer/consumer与phase | TMA、TMEM alloc/load/释放、group与异步完成 |
| 主要资源调查 | 每MP寄存器/shared、local spill、launch限制与实际squad布局 | accumulator/producer寄存器、shared stages、cluster/CTA occupancy | register/shared/TMEM联合限制、分配粒度与活跃CTA/group |
| 语言/库选择 | MUSA/MUTLASS；独立验收的TileLang/MUSA | CUDA/CUTLASS/CuTe；独立验收的DSL lowering | 明确SM103支持的CUDA/CUTLASS/CuTe/DSL lowering |

[官方MUTLASS SQMMA源码](https://github.com/MooreThreads/mutlass/blob/main/include/mute/arch/mma_mp31_sqmma.hpp)
可见TF32及F16S4/BF16S4等路径；[TME源码](https://github.com/MooreThreads/mutlass/blob/main/include/mute/arch/copy_mp31_tme.hpp)
提供描述符与搬运调查入口。这不推导出任意FP8×INT4组合可用，也不证明当前编译产物用了该指令。
S5000的资源数字以实际SDK/device/compiler查询为准；不能把历史MP数、显存或带宽写成新机器上限。

[Hopper指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)给出H100的64K个32-bit寄存器/SM、
228KB shared/SM及227KB shared/block等参考，但实际launch仍需opt-in、allocation granularity及occupancy检查。
[Blackwell指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)中CC10.0资源表不是SM103实测值。
架构档案继续resources={}，本模块也不预填SM103/S5000数字。

SM103不是“SM100全部功能的直接替代”。[cuBLASDx 0.8.0说明](https://docs.nvidia.com/cuda/cublasdx/0.8.0/performance_guide_api_blackwell.html)
明确区分SM1030的tcgen05路径，并排除其integer tcgen05。逐operand/kind、block scale、
tile/layout、target与编译器核对[PTX支持范围](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)；
不能拿通用Blackwell峰值或FP4宣传数字当成任意算子的可达上限。

跨架构复用数学契约、数据流、shape/路由案例、诊断问题和验收方法；重新实现并验证layout、
指令lowering、同步及资源模型。tile/stage/warp数量、binary和性能结论不直接迁移。

## 按热点类型设计单因素候选

| 热点 | 优先分析 | 合法候选与复验重点 |
| --- | --- | --- |
| Dense/Grouped GEMM | 实际M/N/K、per-expert M直方图、padding、转置/重排/量化开销 | tile/stage、producer分工、布局、分组/尾部调度；同时看spill与完整wrapper |
| Attention | q/chunk/KV长度、prefix状态、mask/LSE、index/压缩池与真实流量 | tile/分块/流水、减少冗余搬运；不改窗口、丢token或改变共享KV语义 |
| Norm/reduction/activation | 访存事务、向量化、launch、消减误差与依赖 | 合并读写/epilogue；核实极值、尾部、reduction order和后续等待 |
| Router/pack/quantize | tie-break、top-k、对齐、scale/round/clamp和临时空间 | 避免重复layout转换；保留路由分布、metadata和异常输入契约 |
| Dispatch/combine/通信 | global/local工作量、rank到达、pack与wire量、publication/ack | 单独研究融合/流水；不能仅优化expert GEMM却改掉路由/通信语义 |
| 多Stream/fusion | event/collective顺序、buffer生命周期、SM/HBM资源竞争 | 先证明依赖正确，再测并行；overlap变大不自动等于forward变快 |

每次只改一个预声明轴。把多个实现、开关、精度、graph模式一起改，无法通过原假设检验。
occupancy不是越高越好；更多stage可能减少等待，也可能增加shared/register/TMEM压力和spill。
kernel-only可以辅助归因，正式局部门禁计**完整operator wrapper**，包含必要pack/convert/launch/清理。
跨wrapper融合需重新定义两边共同的compound契约并重新捕获，不能拿不同边界的数字比较。

## 精度与功能验收

先冻结reference、atol/rtol、逐元素判断和relative L2界，绑定到case/模型/量化格式。
外部验证器计算每元素abs_error<=atol+rtol*abs(reference)，报告bad_elements、
nonfinite_elements、checked_elements、max_abs_error和relative_l2。
不能用“最大误差<=atol+rtol×全张量最大值”替代逐元素检查；L2也不能掩盖局部坏点。
不预设一个适用于全部FP8/FP4/BF16/TF32算子的统一容差。

至少覆盖真实捕获shape、奇数/tail/边界、零/极值、scale边界、空/偏斜expert、重复destination、
多次调用/state清理、非默认Stream；启用graph时还要capture/replay和异步生命周期测试。
数学、dtype、量化scale、mask、router tie语义或KV格式改变，属于新任务/实验组，不能偷偷放宽标准。

每个父版本和候选都要 **precision-before → native全部层次 → precision-after**，
使用可验证的coordinator clock与区间。两次精度覆盖全部条件、operator case和forward rank，
以及声明的stream/graph/repeat/edge模式。外部报告的真实性仍由验证器负责。
服务正式请求逐条比对task中预先冻结的reference first-token ID，不能在候选报告里改expected值。
Tokenizer/template之后实际长度、完整SSE、全部请求ID、终止与Cache owner计数均参与验收。

## 三层收益与容量复验

1. **Operator**：逐shape wrapper样本，按捕获call_count形成每rank加权work，不跨rank相加成延迟。
   保留单case变化与慢shape；native不得违背匹配物理资源下界，经验参考被超过不等于违反物理界。
2. **Prefill**：统一coordinated边界、all-stream/all-rank、完整相关chunk，P50/P95与global new tokens/s。
   更改head、只测最快chunk、未加入相关Stream或改变token工作量均无效。
3. **Service**：完整cohort+drain，TTFT P50/P95、新prefill/逻辑input/request三种吞吐。
   Cache命中不算作同量新计算；failed/incomplete SSE、首token不一致、未入账/静默重试不能通过。
4. **KV与长上下文**：逐rank全生命周期峰值与non-KV峰值必须保留原选型的每副本驻留slots。
   non-KV+原slots×每完整上下文Cache需求+reserve不能超设备预算；记录实际接纳且无换出/重算。
   原部署目标长度（含输出预留）的质量与满驻留服务SLO还要由外部报告重新验收，不能只测短输入。

默认5个独立配对轮次，P95每轮至少200样本，跨轮CV<=5%。预声明gain_conditions是收益目标，
其余当前campaign条件仍须非退化；默认operator/forward/service最低收益为5%/3%/3%，退化界2%。
这些是可调整工程起点，不是硬件理论值。每个配对轮都过线才算收益，输出最差配对gain；
它不是置信区间或统计显著性声明。没有挑最好轮、只看总体均值或只挑有利shape的入口。

operator通过但forward失败，保留局部收益并拒绝；forward通过但服务失败，同样拒绝。
未调用该算子的条件须有固定捕获的not_applicable证明，仍保留服务/forward非退化检查；
不能为凑覆盖伪造shape或零时长benchmark。新prefill=0时不宣称新计算吞吐提升。
长上下文的质量/SLO若未实测，只能保留未知，不能把显存预算公式当作2Mi质量与吞吐证明。

通过候选重新采集同口径timeline/counters，检查原假设和热点迁移，保留未解释项及下一轮attack_plan。
接受的是限定矩阵内的证据；没有自动部署。服务未达标或根因未明时继续分析，不推进基线。

## 可运行入口与契约

~~~bash
ifa kernel-demo --architecture moore-s5000 --output runs/kernel-s5000-001
ifa kernel-demo --architecture nvidia-sm90 --output runs/kernel-sm90-001
ifa kernel-demo --architecture nvidia-sm103 --output runs/kernel-sm103-001
ifa kernel-task workspaces/hotspot/contract.json --output runs/kernel-task-001
ifa kernel-evaluate workspaces/hotspot/bundle.json --output runs/kernel-eval-001
~~~

演示input/contract.json与input/evaluation.json可作为schema示例，全部数值、
编译产物与验证声明都虚构；synthetic即使三层通过也recommended_candidate=null，绝不认证后端。
任务模板不填真实命令时只用于设计；演示的SYNTHETIC-NOT-EXECUTABLE命令不会运行。

| 输入 | 约定 |
| --- | --- |
| contract | kind、固定comparison、parent、完整service矩阵、所有rank、toolchain、validation_modes、策略和reviewable commands |
| deployment_bundle/selection | 原始部署bundle与原选型结果，离线重放并一致核对；每副本qualified_replica_slots用于保留容量 |
| conditions | 与matrix_source中的service workload_id/spec精确一致；原始performance bundle、diagnosis、case、reference与request manifest |
| operator case | rank/device/operator/type/signature、shape/dtype/layout/量化/路由、call_count、捕获源及冻结精度标准 |
| revision record | parent/revision/patch与SHA256、编译binary/resource报告、内存/上下文报告、前后精度、配对各轮三层native、修改后profile/diagnosis |
| precision | 精确绑定task_digest/revision/patch、case/forward reference digest、每scope/rank输出检查与顺序区间 |
| profile | 与第二模块相同的raw bundle，candidate多绑定revision/task_digest/patch；diagnosis须有完整工具与定量假设检验 |

task_digest包含规范化任务和全部固定来源SHA256。源文件、reference、矩阵、部署或上限改变会改变
fingerprint，旧候选证据不能复用。评估按报告声明与schema核对，不解析所有厂商二进制或替代数值验证器。
未来executor须验证实际输入、命令、代码/编译产物与GPU执行的对应关系；布尔声明不能认证自己。
输入身份/格式错误拒绝；坏baseline停止比较，坏candidate记录拒绝后继续；缺分析时不进入候选验收。
保留input/manifest、kernel.task/evaluation.json、docs/draft/plan、candidates.jsonl、报告及JSONL事件，
拒绝覆盖已有目录。v1 replay与分布式独立契约保持原行为，不因新增模块而放宽旧门禁。
