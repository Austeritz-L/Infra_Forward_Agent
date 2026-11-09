# 5. 优化前后 Trace 对齐、差异与瓶颈迁移

v0.7实现独立trace_comparison/、ifa trace-compare与合成演示。输入为规范化Chrome complete-event
JSON/JSON.gz及显式比较manifest，输出逐条件/窗口/rank的差异、迁移假设、补证据任务和来源快照。
当前不自动采集或解析nsys-rep、ncu-rep、mcu-rep、Perfetto protobuf，也不执行优化/部署。
设计配置：[configs/traces/prefill.json](../../configs/traces/prefill.json)。

核心原则：**先证明比较的是同一份逻辑工作，再解释时间与占比；收益由native验收判定。**
时间线的sum、union、exclusive不同，单个慢kernel不一定是关键路径，通信占比上升不一定变慢。

## 前四模块如何进入对比

| 模块 | 固定/提供的证据 | 本模块的处理 |
| --- | --- | --- |
| 测试矩阵 | 当前campaign全部service workload ID/spec、输入/Cache/协议、forward/operator绑定 | conditions必须覆盖整个声明矩阵；未提供窗口的条件明确不充分，不筛掉后汇总成功 |
| 性能分析 | 原始all-rank timeline、work/union/exclusive、matched资源参考与counter | 复用相同时间并集语义；未知资源交回第二模块，不从trace时长反推硬件上限 |
| 部署选型 | 架构、固定物理设备、rank放置、TP/EP/PP/DP/CP、Cache格式/容量 | 固定放置才能直接做rank-local差异；拓扑改变需新的分布工作归一化 |
| 算子流程 | parent/revision/patch、原始task/evaluation bundle、精度与三层native报告 | 可选重新验证原始bundle；不只读取一个accepted=true字段，不放宽原门禁 |

native链接复用prepare_task、validate_record和compare_records：重放部署证据、深度分析、
前后精度、5轮算子/forward/服务验收、长上下文质量/SLO和KV预算检查。
还核对两版revision/patch、实际部署、完整条件矩阵、profile/native边界与global new tokens。
这仍是导入报告的确定性验收，不是对真实GPU执行或报告真实性的认证。

可在没有native链接时分析trace；输出仅为profile观察/假设，不能确认精度或服务收益。
有native通过也不代表已找到因果关键路径，不能将某个算子delta认作全部收益来源。

## 可比性的分层检查

| 检查层 | 必须相同或有证明的内容 | 不满足时 |
| --- | --- | --- |
| 身份 | synthetic/measured、模型/权重、架构、镜像/runtime、tokenizer/dataset、全局workload、Cache格式 | incomparable，不计算数值差异 |
| 部署 | 全部rank与物理device、topology；不能将同名rank当成同一份工作 | incomparable，交第三模块；保留两边单独观察 |
| 场景/阶段 | 同prefill grain、边界/head、实际输入/复用/新token、request set与chunk plan | incomparable；不按同一长度标签强配 |
| 采集 | 相同timeline工具/版本/options、replay/cache控制、选择策略；有开销对照与完整性证明 | 协议不同不直接比较；缺证明标partial |
| 语义 | operator/type/signature/layer或有证明的fusion/rename映射 | 未匹配行保留，不填0，不据子集判断迁移 |
| 时间 | 每rank/device本地有效窗口、全相关Stream、无截断/丢失/重复/未支持GPU phase | partial，保留单边值但关闭正式delta/汇总 |

字段不全、来源文件缺失、rank缺失或unknown不等于数值0。两边环境都填unknown也不能证明可比。
GPU事件重放pass不能与timeline pass混比，未采到某个Stream不能声称该Stream工作消失。
S5000/SM90/SM103各自做同架构before/after；不将跨厂商时长或counter原值当作同条件优化收益。

当前直接差异计算要求相同topology与物理放置。改变TP/EP/PP/DP/CP后，单rank的token、层、
专家与通信职责可能变化，即使rank数相同也不能自动配对。第三模块仍可独立评价固定资源下的
服务/容量；本模块只保留单边trace，直到有明确工作分片归一化adapter。它不自动除TP或乘EP。

## 从窗口到算子的对齐顺序

1. **先选逻辑窗口**：manifest声明condition、logical_step、request_manifest_digest、
   chunk_plan_digest、global_new_tokens、actual_cached_tokens、scope和head_boundary。
   phase必须prefill。step、单chunk与完整多chunk请求是不同粒度，不能互换。
2. **再选实际range**：每rank提供marker名与selector，例如ifa_step；同一文件有多个forward
   时必须明确选择，不能默认取第一个或最快窗口。输入保留实际run-local ID与映射来源。
   使用ifa_step选择窗口时，GPU事件也须绑定该step；未绑定/其他in-flight step标partial，
   不能将后台工作算作当前请求的算子退化。
3. **归零、不拉伸**：只减去该rank窗口起点，保留真实us宽度；绝对起始时间可不同。
   不动态时间规整，不将两条timeline缩放成同长，不以第N个kernel或launch序号强行配对。
4. **按语义分组**：默认键为operator、compute_type、signature、layer，四项必须明确。
   signature应包含实际shape/dtype/layout/量化/路由语义，不把实现tile参数误当数学shape。
5. **按逻辑调用配对**：ifa_logical_invocation应由layer/chunk/callsite/实例等稳定语义生成。
   两边ID集合相同才计算逐调用差异；GPU事件的本地ifa_invocation不自动跨run配对。

原始GPU事件完整保留，normalized.events.jsonl记录相对时间、rank/physical device/Stream、
语义元数据与side/pair ID。它是本项目的归一化表，不冒充可直接导入的Perfetto二进制。
绝对时钟和原始窗口仍可在输入与capture记录中审计，不能把不同host的相对零点当成同时到达。

### Fusion、拆分、重命名与真正消失

一份semantic_mappings记录一个共同逻辑单元，例如baseline的Norm+Copy对应candidate的融合wrapper。
两边使用精确selector列表，附verified、semantic_contract与source。允许一对多、多对一，
但**一个GPU事件只能归属一个canonical group**，冲突直接报错，防止重复计数。
“名字像”或“数学应该相同”不构成证明；输出/精度/状态生命周期由第四模块及外部验证器核实。

基线存在而candidate没有匹配事件时，默认unmatched，而非0us或无限加速。
只有显式verified语义映射、该side的absence_verified及完整采集证明，才以0记录已消除的工作。
缺少logical invocation时只能比较共同单元的work/union等统计，不构造逐调用wrapper收益。

融合compute+communication等无法从一个kernel边界拆开的情况用ifa_kind=mixed标记。
总busy与逻辑单元仍可比较；C/N分量、通信暴露和相应交集置null，需要子阶段插桩/counter补充。
不能把同一个融合kernel完整时间同时加给compute与communication。

## 统计口径：绝对时间与占比一起看

对rank/device窗口W、逻辑单元g，C/N/M分别是明确分类的compute/communication/memory事件集合：

~~~text
event work_g = sum(duration of GPU events in g)
work_share_g = work_g / sum(all selected GPU event durations)
coverage_g = |union(g)| / |W|
exclusive_g = |union(g) minus union(all other logical units)|
busy = |C union N union M union mixed/unknown events|
idle = |W| - busy
overlap_CN = |C intersect N|
comm_without_compute = |N minus C|
~~~

work可能超过窗口；work_share分母是累计GPU事件时间，不能当作墙钟份额。
coverage可以相互重叠，不要求加起来100%。exclusive是其他逻辑单元未覆盖的区间，
不是kernel self-time、可删除时间或因果critical path。PDL/自旋/consumer等待可包含在kernel duration内，
需要source、launch/event和依赖证据继续拆解。
comm_without_compute仍可能与copy重叠，不能称为全部裸露时间或纯链路时间。

每行同时保留before/after、candidate-minus-baseline、相对变化；占比额外报**百分点**。
baseline=0时relative_change=null，不输出无穷大。新增/删除/未匹配行都保留对应状态和原始值。
逐logical invocation报GPU事件work/union/span与相对位置；span包含间隙，不等于native wrapper时长。

描述性汇总按condition/rank分别进行，默认至少3对预声明窗口，输出中位数和配对delta范围。
不混输入长度/并发/Cache，不跨rank相加，不挑好窗口，也不将3个profile窗口当P95验收。
任一声明pair不完整时，该条件正式汇总和迁移判定都关闭；完整pair的单边记录仍可审阅。
同一文件/同一选中窗口被多个pair ID复用会被识别，不计作重复证据。

## 通信与多Stream对比

collective对齐键包括稳定逻辑collective ID、group、op、dtype与logical payload。
ID要区分layer/chunk/dispatch/combine/实例，不能把不同group或payload的事件混成一次调用。
group/payload改变时，单次collective的ratio关闭，另保留每版实际通信总量与窗口观察。
它可能是优化造成的流量变化，也可能是不同工作；须结合相同global workload的来源判断。

每rank分别报collective work/union、C/N交集及comm_without_compute。
ifa_wire_tx_bytes仅在ifa_wire_byte_convention=event_tx_bytes时按事件求和；
它是每rank该事件实际发送字节，不能重复填整个collective的总量。logical payload、wire TX、
发送+接收、algbw、busbw与链路带宽不混用。缺wire计数保留null。
collective duration可能包含pack、晚到rank、barrier和自旋，pure_wire_time与跨rank到达偏斜默认null。

Stream ID只在本次rank/device内有意义，baseline Stream 0不自动对应candidate Stream 0。
每rank显式给stream_roles，将真实Stream映射到compute/communication/copy等逻辑角色；
多对一时先求角色并集，再算角色间交集。未匹配角色保留unknown，不补0。
同时计算至少两个不同物理Stream活跃的时间、peak active streams和逐角色busy/intersection。
同Stream内多个事件不构成多Stream overlap；[CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
可使同Stream相关kernel出现重叠，也不能据此新增Stream。
这些都是时间重叠，不是SM occupancy，也不证明并行没有资源竞争。

本版不做跨host时钟校准或依赖DAG。即使两版各自归零，也不能计算全局collective arrival、
拼接全局critical path或将rank-local窗口和当成forward耗时。保留这些未知项作为下一步补证据任务。

## 瓶颈迁移的判定流程

1. 检查所有条件/声明pair的可比性与完整性；任意丢rank、截断、未知分组先补证据。
2. 每rank按共同语义单元的exclusive exposure寻找主导者，默认要求各窗口主导者一致。
   所有未匹配单元都必须先解决，不能删除它们后宣布迁移。
3. 主导单元变化时，同时看新主导者的绝对work、exclusive与share；默认变化界为max(2us,10%)。
4. 对应分类：

| 状态 | 含义 | 下一步 |
| --- | --- | --- |
| absolute_work_increase_hypothesis | 新主导单元绝对GPU事件work增加 | 检查调用数/shape/padding、真实流量和资源参考 |
| exposure_increase_with_stable_work | work稳定，但与其他单元重叠减少、暴露增加 | 检查producer/consumer、rank到达、Stream依赖与竞争 |
| relative_share_only | work/exclusive没有明显增加，仅占比/主导排序上升 | 不判为变慢；检查其他热点优化后剩余上限 |
| same_dominant_unit | 主导者仍相同 | 对照counter/资源模型与native结果决定下一轮 |
| unstable/insufficient/unmatched | 覆盖或匹配不足 | 采集更多对应窗口、补映射，保留unknown |

例：GEMM缩短后，通信550us保持不变，而exclusive从250us增至450us。通信占比与暴露上升，
说明下一轮值得调查通信/依赖；它没有证明网络更慢，也没有证明450us都能通过overlap消除。
如果通信绝对时间与暴露都未变，只是分母变小，则归为relative_share_only。

next-tasks将缺采集、native验收、资源模型与依赖调查分开。原有反证、精度和服务门禁继续有效；
新热点只能作为第四模块的分析输入，不能跳过深度分析直接改代码。
所有confirmed_bottleneck=false、causal_critical_path=null，service_speedup_from_trace=null。

## 可运行入口、证据与限制

~~~bash
ifa trace-demo --architecture moore-s5000 --output runs/trace-demo-s5000-001
ifa trace-demo --architecture nvidia-sm90 --output runs/trace-demo-sm90-001
ifa trace-demo --architecture nvidia-sm103 --output runs/trace-demo-sm103-001
ifa trace-compare workspaces/trace-pair/bundle.json --output runs/trace-compare-001
~~~

演示与原始schema见input/trace-comparison.json：所有硬件、timeline和native声明均为synthetic，
不代表三种架构的实测或收益。已有输出目录拒绝覆盖。

| bundle字段 | 内容 |
| --- | --- |
| kind / sides | 两版comparison、revision、patch SHA256、placement和topology |
| matrix_source / conditions | 当前service矩阵与完整condition spec |
| pairs | 独立pair ID、condition ID、两边window/protocol/identity与每rank trace path/selector/clock/stream_roles |
| semantic_mappings | 有来源的精确selector列表、共同语义契约、可选absence证明 |
| native | 可选原始kernel_bundle与candidate_id；重新验证前四模块证据 |
| policy | 最少窗口对、主导稳定比例、绝对/相对变化界；不是收益验收阈值 |

输出trace.comparison.json、normalized.events.jsonl、bottleneck-migration.json、next-tasks.json、
report.md、events.jsonl和input/SHA256 manifest。保留原始来源、每对问题、未匹配行及单边值。
形状/时序metadata和来源声明仍由产物生成方负责；hash绑定材料，不认证硬件执行。

ifa plan新增trace.contract.json；trace_comparison.bundles可按架构指定比较bundle，阶段状态为
needs_offline_validation，规划不会运行比较。旧pairs仍可记录文件，但两份raw trace不足以替代manifest；
没有bundle/pair或禁用时保持skipped，不阻塞其他模块。原replay/precision/selection门禁保持不变。

参考[Perfetto Trace Processor](https://perfetto.dev/docs/analysis/trace-processor)的规范化/查询分离设计、
[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)的focused capture与annotation。
未集成其执行后端。vendor binary导入、Perfetto SQL、大规模流式处理、时钟校准、CPU/collective完整
依赖图与因果关键路径仍需后续adapter；本版以可复核的小型Chrome窗口为目标。
