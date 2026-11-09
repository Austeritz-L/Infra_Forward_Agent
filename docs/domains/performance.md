# 2. 性能统计、资源上限与瓶颈攻坚

目标：对 [prefill 测试矩阵](testing.md) 中每个条件，回答“主要时间在哪里、哪类资源约束、
离适用上限多远、先攻哪一处、用什么实验验证”。只列耗时排行榜不构成瓶颈定位。
入口：`configs/performance/prefill.json`、`performance/`、`agents/performance.py`。
当前实现导入式统计、时间线、资源模型与调查任务输出；真实采集和因果归因仍待接入。
只研究 prefill，服务输出1token；不做decode/TPOT分析。

## 三种上限必须分开

| 层次 | 用途 | 必须记录 | 能否作为物理上限 |
| --- | --- | --- | --- |
| architectural_peak | 约束理论最大速率、给出条件化乐观界 | 精确SKU、频率、ISA、dtype、稀疏/稠密、FMA/byte计数 | 仅在适用性与工作量成立时 |
| sustained_microbenchmark | 同机、同访问方式/规模下实际持续能力 | 设备/工具链/clock/cache、规模曲线、精度、重复样本、原始报告 | 否，可能被更好的实现超过 |
| matched_reference | 同shape/layout/routing/topology的成熟实现 | 输入/调用边界一致、实现版本、正确性、完整计时 | 否，只是当前可比较参考 |

S5000、SM90、SM103分别校准；不在代码填宣传TFLOPS、显存带宽或假定支持的指令。
峰值接近只能说明所建模型接近某个资源约束，不能单凭它证明唯一瓶颈。
参考实现被超过并非“违反物理”；若理论模型要求时间大于原生计时，则先查计数、时钟、范围、
dtype/稀疏与单位，不据此宣称不可能的收益。缺参考不是零机会，应列为高优先级测量任务。

## 各计算类型如何建立上限

| 类型 | 主要资源假设 | 必须做的校准/证据 | 典型攻坚方向 |
| --- | --- | --- | --- |
| Dense GEMM | Tensor/MMA、HBM/L2、tile效率 | 精度/ISA对应的M/N/K扫描，执行FLOPs和padding | 小M、尾块、数据复用、layout与流水 |
| Grouped/Expert GEMM | Tensor、group规模、路由不均、同步 | 服务捕获的每expert计数/权重/padding，各rank负载 | group排程、负载均衡、padding/dispatch成本 |
| Attention | Tensor+SFU、KV流量、shared/L2、依赖 | 实际q/context/chunk/head、KV格式、稀疏/压缩路径 | KV访问/解量化、块选择、融合、同步 |
| Norm/激活/Reduction | Vector/SFU、HBM、归约依赖、launch | 对应元素数、dtype、reduction width的参考曲线 | 融合减少读写/launch，vector化，归约布局 |
| Quantize/Pack/Scale | Vector/整数、scale归约、搬运 | group/scale/layout/packing一致 | 减少中间转换、搬运与重复scale |
| Gather/Scatter/KV搬运 | 真实事务量、L2命中、访问延迟、原子/TLB | index局部性、对齐、访问粒度；连续copy不能直接代替随机访问上限 | 页访问、合并、减少放大/元数据 |
| Router/TopK/Sort | 整数/Vector/原子、launch、分布 | 真实路由和分段/排序规模，空/偏斜expert | 合并pass、prefix/permute、原子冲突 |
| Collective/P2P | 链路、消息启动、pack/unpack、rank到达 | 同group/ranks/大小/dtype/算法/链路的延迟-带宽曲线 | 小消息合并、分片、放置、等待和融合 |
| Host启动/同步/调度 | CPU发射、runtime锁、event/collective依赖 | CPU/GPU correlation、队列、graph replay、线程栈 | 减少发射/同步，调整依赖；不能靠GPU峰值解释 |

这些是待检验的资源集合，不是“看到算子名字就选定瓶颈”。整数/atomic/SFU可用匹配op/s或
inst/s；寄存器、occupancy、spill和依赖延迟属于额外约束，不能硬转成FLOP/s。
Attention/稀疏/MLA/MoE的工作量依赖实际路径，不能一律套dense全序列公式。

## 模型与统计公式

对同一调用/step，先明确实际执行量F（包括padding）、所需/实测字节Q、SFU/整数工作量等。
执行FLOPs与有用FLOPs分开，FMA计一次或两次必须固定；counter实际流量与算法最少字节分开。

```text
t_tensor = F_tensor / P_tensor
t_hbm    = Q_hbm / B_hbm
t_l2     = Q_l2 / B_l2
t_link   = Q_link / B_link
t_model  = max(t_tensor, t_vector, t_sfu, t_hbm, t_l2, t_shared, t_link, ...)
```

潜在重叠的资源时间取max，不直接相加。若是有证据的串行阶段/DAG，可逐阶段建界再沿依赖组合；
当前程序未实现DAG组合。Roofline的F/Q与P/B用于检查理想计算/带宽平衡，低带宽利用率本身
不能证明带宽饱和，可能是依赖、随机访问、occupancy、指令发射或同步限制。

程序输出每资源amount/rate、参考时间、参考达成率、原始依据与主导模型资源（接近max的候选）。
默认达成率≥80%进入“减少工作/流量”调查，低于此进入“实现效率差距”调查；它是可调分流阈值，
不是架构定律或确认结论。native默认至少5样本、CV≤5%，不足/不稳不输出机会比例。
所有quantity必须引用artifact、计数依据和compatibility_key；必须与上限的资源、单位、环境、
架构、synthetic/measured类别及ISA/dtype/layout/shape/topology计数约定一致。

对同一全局prefill step，每rank汇总同资源工作量，使用该rank资源速率得到局部下界。
所有rank都有匹配物理界时，可报告乐观资源界：

```text
t_step_resource_lower = max_r(t_rank_resource_lower[r])
tokens_per_s_upper   = global_new_tokens / t_step_resource_lower
```

它不是把rank计时相加，不需要假设host墙钟对齐；但只覆盖所建资源、同一固定step，
还缺CPU/通信依赖/启动等约束，不能直接当真实服务可达吞吐。程序明确保持service_ceiling未知。
整个服务的上限还需固定长度/命中/到达分布，按物理资源汇总每有效新增token的需求，加入CPU、
链路和排队约束，并用多条件负载扫描验证。不能从单个最快GEMM推导服务上限。

## 采集协议：五次独立实验

1. **Native**：按测试矩阵做精度→无profiler计时→精度。保留逐请求、逐step、各rank和算子原始
   样本；固定输入/Cache/版本/shape/routing，完整保留失败与尾部排空。正式结果只来自此通道。
2. **Timeline**：另外采集有代表性的all-rank prefill窗口，CPU launch、GPU kernel、copy、
   Stream/event、collective、graph节点与request→microbatch→layer→operator关联都要保留。
   只抓TP0不能确认慢rank/全局关键路径；长测需稳态/尾部/异常代表窗口，不假设一小段代表全程。
3. **Counters**：对热点signature小范围采样，保留请求的metric清单、可用/缺失列表、原始report、
   可搜索导出、工具版本、replay/cache控制和计数器语义。不把“本报告已保存字段”称为全套指标。
4. **Calibration**：同一自有设备、clock/温度/访问方式/工作集/消息拓扑条件下测持续能力和成熟
   参考，先后做精度，保留规模曲线与重复样本。不同采集器的byte/operation口径先做copy/算术校准。
5. **Validation**：一次改变一个主要因素，观察原假设预言的计数器/暴露变化，再无profiler配对
   forward/service A/B。精度、所有必测条件非退化和指定目标收益通过，才记录已验证收益。

NV：Nsight Systems/PyTorch导出timeline，Nsight Compute按安装版本查询支持的metrics。
SM90与SM103独立核实，不照搬metric或ISA支持。S5000：先核实安装的MUSA trace/MCU与导出
能力；不凭NV支持推断MUSA可用。当前没有实现自动启动这些工具或解析其二进制report。
Profiler replay/序列化/时钟/cache设置会扰动并发。计数器采集与native timing是独立实验，
quantity/native时间计算是条件模型，不是同一时刻实测带宽。采集扰动需单独报告，不能用profile
中的总时长代替正式吞吐。所有GPU采集/校准需先确认归属，不自动恢复旧任务或抢占生产服务。

## 算子时间与占比

每个测试条件单独分析，键包含workload/Cache、phase、microbatch/chunk、operator、signature、
layer、rank/device、stream和run/capture。不能平均不同长度、命中和架构的加速比。

| 输出 | 定义 | 不能解释为 |
| --- | --- | --- |
| work_us / work_share | 同组GPU事件duration之和 / 所纳入全部GPU事件duration之和 | forward墙钟占比 |
| union_us / wall_coverage | 同组区间并集 / 本rank/device的forward窗口 | 与其他组互斥的占比 |
| exclusive_us | 仅该组GPU事件存在，其他组没有事件的区间 | 因果关键路径、确定能省掉的时间 |
| gpu_busy_union_us | 所有GPU事件区间并集 | SM活跃周期利用率/有效算术占用率 |
| gpu_event_count | kernel/copy等事件数 | 框架operator调用次数 |
| operator_invocations | 显式invocation/correlation去重后次数 | 没映射时猜测的调用数 |

全重叠但关键的producer可能exclusive=0，仍进入overlapped_hotspots依赖复核；不能直接判无价值。
分组保留各层累计work，实际时间分位数是GPU事件而非整个框架op wrapper；完整wrapper延迟
另取无profiler调用样本。融合kernel与多个框架算子需显式映射，不能靠名称自动拆分。
边界裁剪的GPU事件显式计数，未裁剪只表示未检测到跨边界事件，不证明marker覆盖了全部异步尾部；
缺rank或已发生裁剪的capture只进入补证据队列。各rank本地窗口长度可比较，但不证明全局慢rank因果。

## 通信与多Stream overlap

每rank/device独立计算C（计算）、N（通信）、M（copy/memory）区间并集：

```text
busy = |C ∪ N ∪ M|; idle = |forward_window| - busy
overlap_CN = |C ∩ N|
comm_without_compute = |N| - |C ∩ N|
comm_overlap_fraction = |C ∩ N| / |N|
```

另外统计各Stream busy并集、至少两个不同Stream活跃的时间并集及peak active streams。
同一Stream内多事件不是多Stream重叠。通信名字启发式分类与显式分类都标来源，自定义dispatch/
combine需明确标注；代码没有假设NCCL以外的通信会自动被正确识别。

collective长可能来自晚到rank/producer等待、barrier、自旋、pack/unpack，不能全归为网络搬运。
payload bytes、实际wire bytes、算法带宽和bus bandwidth必须用相同定义比较；小消息优先看
启动/同步延迟曲线，不拿大包峰值套用。Overlap增大也可能竞争HBM/SM并让forward变慢。
测试并行前需确认数据/collective顺序、event与buffer生命周期；软件时间重叠不能证明依赖正确。

## 瓶颈定位与攻坚流程

1. 验证同条件输入、Cache、精度、完整时间窗口与实际负载；异常/未完成窗口不作为有效收益。
2. 将服务路径与forward分开观察。取成对request/step的CPU/queue/transfer证据，不用TTFT P50
   减forward P50计算“调度开销”；这些分量可能重叠，分位数也不能这样相减。
3. 看all-rank暴露、GPU gaps、rank间差异和Stream/collective到达。先区分host供给、算子、
   内存、通信、等待；时间线不完整时保留未知。
4. 对热点绑定实际shape/ISA/路由，建立三层上限和原始counter。区分“实际流量大”与“带宽已饱和”，
   “计算少”与“发射/依赖限制”，以及“资源上限接近”与“还有实现差距”。
5. 输出四类任务：measure_first、kernel_efficiency、work_or_traffic_reduction、wait_and_overlap；
   全重叠热点另保留依赖复核。优先级考虑暴露和证据缺口，不把未知热点机会设为0。
6. 实现差距队列使用exclusive_us×(1-t_model/t_native)作调查提示。它不是收益预测，不跨rank相加，
   不跨capture/条件相加；接近上限时转向减少FLOPs/bytes/重复工作或改变消息/排布。
7. 用单因素干预验证预言：改变工作集/访问局部性、M/group分布、消息大小/到达、发射数量或
   合法的调度依赖；不要随便关闭同步造成数值错误，也不要改多个框架配置后归因一个算子。
8. 只有完整服务/forward A/B复验才确认收益，然后重新采集检查瓶颈迁移。

有证据的串行关键份额f和局部加速s才可用Amdahl：1/(1-f+f/s)。work_share或exclusive_us/forward
不能直接代替f。跨host关键路径、CPU/collective因果归因和自动干预未实现，因此当前所有
confirmed_bottleneck=false、service_speedup_prediction=null。这是证据边界，不是宣称无法优化。

## 代码、输入与输出

- `performance/statistics.py`：同条件正有限样本、线性插值P50/P95/P99、CV、尾样本资格；服务
  cohort含全部尝试。timed load保留完整发压+drain分母，不能将20分钟测试裁成成功请求的短窗口。
- `performance/timeline.py`：按rank/device/组的work、union、exclusive、通信交集和不同Stream活跃。
- `performance/ceilings.py`：匹配引用、资源时间模型、物理/经验界分离；超界/零工作/不稳样本不填机会。
- `performance/report.py`：同identity的native与trace/counter源绑定、原始artifact快照/SHA256、调查队列。

```bash
ifa performance-demo --output runs/performance-demo-001
ifa performance runs/performance-demo-001/input/performance.json --output runs/performance-replay-001
```

演示全部synthetic，没有NV或S5000实测上限。可在演示input副本中查看完整schema：
identity必须包括kind/architecture/environment/workload_id，native文件与之完全相同。
一份bundle对应同一条件/shape口径，可有多个capture；traces每capture每rank一个选定forward。
Native forward需coordinated_prefill_forward、all_streams_joined、rank_coverage、boundary_id和固定
global_new_tokens_per_sample；算子native需operator_invocation、rank/device/operator/compute_type/signature。
Service需fixed_work_drained或timed_offer_drained；每request单调clock_id相同、sent/first/terminal_us、
完整状态、1输出token、input/cached/new（normalized owner计数满足cached+new=input）。
分布式Locust worker未对齐时钟不能直接合并绝对时间；先单worker分析或提供经验证的归一化。
Timed cohort另有offered_window.started_us/finished_us/completed；主吞吐分母覆盖该完整窗口与drain，
同时保留原始首发至最后终止时间，不将短成功片段当整轮结果。
可选step_resource_budgets使用同一step_contract、global_new_tokens及各rank完整资源量，native
scope为prefill_rank_interval。预算步数/全局token不一致时拒绝合成step上限。

Trace沿用Chrome JSON并支持args.ifa_operator、ifa_compute_type、ifa_signature、ifa_invocation、
ifa_layer；无映射时明确标kernel_name_fallback。原v1的analyze/replay命令保持兼容。
Ceiling含basis/rate/unit/source、architecture/environment/kind/compatibility_key；quantity含resource、
amount/unit/amount_basis、ceiling_id、source和相同compatibility_key。同资源先聚合再建模，
字节统一为byte，不能把GB/s或GiB/s原值不经换算填入。所有source为bundle目录内相对文件，
保留report/export/公式来源；输入规范化的真实性仍依赖产物生成方，当前不自动验证counter二进制。

输出performance.analysis.json、attack-plan.json、report.md、events.jsonl和input快照/manifest。
输出目录不得已存在。每条结论都有来源、未知项与下一步；生成报告不等于测试验收/自动优化完成。

## 参考与实现边界

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：聚焦窗口、标记、CPU/GPU timeline。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：metrics、replay/cache控制、采集开销、资源/roofline分析。
- [HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis)：按rank的计算/通信/idle与overlap分析方向。

没有集成上游源码，也没有启动硬件采集。经验上“流量减少但完整forward未加速”、重叠算子不能
相加、缺失metric不能补0等原则保留；历史结论不作为当前硬件上限或已确认瓶颈。


v0.7的[独立Trace比较](traces.md)复用本模块时间并集语义，并对齐两版的实际工作和采集协议。
mixed/unknown事件保留总busy；component_split_known=false时，C/N值仅描述已知分类子集，
不能将未分解的融合通信理解为0。新比较器将对应分量delta设为null。
