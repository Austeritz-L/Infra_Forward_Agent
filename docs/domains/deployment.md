# 3. 分布式部署选择：长上下文容量、性能与平衡

v0.5提供 `deployment/` 离线预算与证据选型、`ifa deployment`、合成演示及长上下文规划档。
它消费明确列举的候选，不自动穷举拓扑、探测机器、部署模型或执行硬件验收。
S5000、SM90、SM103使用同一契约，各自提供能力和布局证据；没有预填显存、带宽或支持结论。

## 先区分四种“长度/容量”

1. **模型原生长度**：固定checkpoint配置和模型声明的范围。
2. **实际可用单请求长度**：模型语义/位置编码、runtime限制、长上下文质量、每副本最紧张rank
   的显存同时满足；更改context/RoPE参数不能自动证明质量和容量。
3. **单副本KV token池**：多个活动请求占用的归一化逻辑token总池，不能当作单请求上限。
4. **全服务容量**：独立副本各自可容纳的完整请求数之和；DP空闲显存不能拼给一条请求。

本仓库把 **2,097,152 token（2Mi）作为容量规划目标**，另保留十进制2,000,000的区别。
总长度=tokenizer/chat template之后实际prompt token+输出预留；当前prefill规范输出1 token，
因此2Mi档prompt=2,097,151。增加输出预算时必须重算，不能悄悄挤占KV余量。

不能统一假设“主流模型已经支持2M”。2026-11-09核对的
[DeepSeek-V4.1-Flash配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)
给出 `text_config.max_position_embeddings=1048576`；
[MiniMax M3.1 Flash官方说明](https://code.minimax.io/tools/m3.1-flash)给出1M窗口，未在此核实其本地
checkpoint精确token数。二者均不能作为本仓库已支持单请求2Mi的证据。
[context-evidence.json](../../configs/deployment/context-evidence.json)记录日期和未知项，只是来源观察；
真实实验必须保存本地权重配置及revision，模型与后端分别验收。
超过原生长度必须另有exact-context extension+quality证据，否则不推荐。
可规划超出已知长度的压力场景，但runner应标记blocked/inconclusive，不能截断后报告通过。

## 三种目标的共同门槛与选择规则

固定物理设备/链路、模型语义、权重与KV格式、镜像/runtime、全局输入集、Cache准备及计时边界。
允许改变明确声明的拓扑/放置，不能把加卡、换量化、缩短输入混作拓扑收益。
权重或KV量化需建立新实验组，经过长上下文质量/精度验收，再做显式跨组研究。

共同门槛：后端支持组合 → exact-context质量/精度 → 分析预算 → 全rank生命周期峰值/接纳
→ precision-before → native service/forward → precision-after → 每条件SLO和重复稳定性。
外部执行器产出报告；离线命令检查身份、覆盖、字段及数字一致性，不能证明报告真实。

| 目标 | 约束通过后首先比较 | 次级规则 | 调查顺序 |
| --- | --- | --- | --- |
| capacity | 已实测同时驻留的独立完整长请求数 | 吞吐、延迟、稳定ID | 去KV复制、存储分片、权重放置、临时buffer，最后增加独立副本 |
| performance | 同条件归一化新prefill吞吐；可切换延迟得分 | 延迟、容量、稳定ID | 低通信/低气泡放置、足够计算并行、再扩大DP |
| balanced | 容量/吞吐/延迟Pareto筛选，再加权几何评分 | 同分按稳定ID | 保留折中点，按业务容量偏好调整权重 |

capacity仍须满足性能下限，performance仍须装下完整长请求。
所有目标都受 `min_context_slots`、全部条件吞吐下限、TTFT和forward延迟上限约束。
容量使用实测接纳数这个**已验证下界**；分析预测最大slot数仅用于下一轮探边界。
资格文件的capacity_condition_id必须指向完整目标长度的cold-miss服务条件，客户端并发至少达到
报告的总slot数，该条件也须通过全部SLO。另需峰值记录证明全部slot同时活动驻留，未靠淘汰、
换出、retraction或重算维持容量。排队的客户端数量不能代替活动驻留证明。

条件j吞吐使用各独立轮中位数；延迟使用各轮所选分位数最大值。
`T=exp(Σwj log(tpsj/fixed_reference_tpsj)/Σwj)`，
`L=exp(Σwj log(fixed_reference_ttftj/ttftj)/Σwj)`。
reference与权重须测试前冻结，不能将不同长度/命中条件原始TPS相加。
balanced默认 `(capacity/max_capacity)^0.5 × (T/max_T)^0.3 × (L/max_L)^0.2`，
权重可改，程序按权重和归一化指数。max来自合格候选全集，新增候选要重算排名。
排名不建立统计显著收益；误差区间重叠时保留近似等效方案，增加配对A/B再变更。

## TP / EP / PP / DP / CP 的比较口径

| 方案 | 权重与持久Cache | 通信/调度代价 | 小显存长上下文判断 |
| --- | --- | --- | --- |
| TP | 分片部分权重；GQA可按本地KV head分片，MQA/MLA可能复制 | 高频AR/RS/AG，组大小/消息量决定延迟 | 读取实际head/复制布局；不能全KV直接除TP |
| EP | 分片专家，共享/attention权重仍可能复制；不保证KV分片 | dispatch/combine AllToAll、偏斜、padding | 专家显存节省可让位KV；额外buffer和冗余专家须扣除 |
| PP | 按实际层分片权重与Cache，取最紧张stage | 激活传输、气泡、chunk/microbatch调度 | 可降低单卡负担，但stage不均衡及单请求延迟须实测 |
| DP | 每个独立请求副本复制权重与自身Cache池 | 请求路由/排队；内部attention-DP等组不一定是独立副本 | 每副本先单独装得下；不能救一条装不下的2Mi请求 |
| CP / PCP / DCP | 分prefill工作或持久KV序列，两者不等价；固定state/index可能仍复制 | attention交换/归约、ring/AG、到达偏斜、临时buffer | 只认可实际持久分片；计算均摊不等于长期KV缩小 |

**拓扑名称不是乘法公式。** TP、EP、attention-DP、DCP等可能复用同一组rank。
`axes`描述意图，`replicas[].ranks[]`描述一次请求实际经过的物理rank，后者决定预算。
每候选覆盖同一物理设备集合，每台设备恰映射一个rank；当前不建模多进程共卡。
一张卡被多个组使用不能重复计容量；PP阶段也必须归入同一个请求副本。

[vLLM版本化CP说明](https://docs.vllm.ai/en/v0.20.1/serving/context_parallel_deployment/)
是布局调查入口，不能据此宣布MUSA或任意NV模型组合已支持。
[vLLM Ascend设计说明](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html)
展示PCP/DCP与部分Cache仍复制的情况；这是特定后端例子，各后端需独立落实。
模型、量化、PP/CP/graph组合是否合法，由固定runtime revision的能力证明给出。

## 按物理池与最紧张rank核算显存

每rank列互斥字节预算，未知不能填0；来源在 `layout_source` 留存：

`Bcache,r=HBM_r−weights−runtime−graphs−communication_buffers
           −prefill_workspace_peak−other−retained_prefix−safety_reserve`。

- weights含共享/专家/量化scale/冗余专家；通信融合和overlap双buffer也计入。
- workspace取目标长度、计划并发和全部生命周期（启动/JIT/graph capture、prefill、清理/恢复）
  的保守峰值；扩大并发后会增长则重建预算。不能只读idle后free memory。
- runtime是额外管理开销；Cache page/scale元数据在Cache公式计，避免重复。
- retained_prefix是目标活动请求之外仍保留的前缀，不与活动Cache重复扣。
- 设备总量、进程占用、allocator allocated/reserved和峰值都保留原始记录，不能重复相加。
- fragmentation_fraction=f表示分配空间不可有效使用比例，需求按`ceil(raw/(1−f))`膨胀。
  另留安全余量；0值须有测量/协议依据。

普通注意力本地唯一物理池：
`entry_bytes=ceil(local_kv_heads×(dK+dV)×bits/8)+scale_bytes+entry_metadata_bytes`。
`pool_bytes=copies×ceil(local_retained_tokens/page_tokens)
             ×(page_tokens×entry_bytes+page_metadata_bytes)`。
copies计唯一实际存储层/池份数；共享Cache逻辑层不重复计，TP复制也不能漏。
`local_retained_tokens=ceil(retained_tokens/effective_storage_shards)`，持久分片核实后才允许除。
窗口/压缩需原模型语义及runtime证据；压缩尾部、index、scale、recurrent/replay/draft分别列池。
packed格式允许明确每entry含scale字节数，**不推断复杂模型公式**。
DeepSeek-V4.1配置存在压缩和跨层共享，不能套全层普通GQA公式；应导入真实物理池。
recurrent/replay固定state以bytes_per_request计，不因开CP就自动除。

副本容量 `Cr=min_rank floor(Bcache,r/cache_bytes_per_full_context,r)`，
再受runtime归一化token池和实际接纳约束。服务总容量 `Σ副本Cr`。
混合长度驻留/动态调度不由这个slot公式预测，须对应混合服务流量另验。

手算仅解释普通未压缩KV：80层、8个KV head、dK=dV=128、BF16，
每token `80×8×256×2=327,680 bytes`；2Mi完整上下文需要 **640 GiB** 唯一KV。
无复制铺到8卡仍是80 GiB/卡，未计权重/workspace；8张24GiB卡无法仅换拓扑装下。
CP能去某些布局的复制并改变分布，不能凭空缩小唯一数据。
FP8主体理论减半还要加scale/元数据，质量/内核支持另验，不是现成可用结论。

90% prefix命中减少新prefill计算，完整上下文仍须可供attention访问。
共享物理前缀可提高特定会话容量，但不能保证独立长请求容量；本版保守按独立完整请求验。
Chunked prefill主要限制临时workspace，通常不减少全长KV；缩chunk不是2Mi容量证明。

## 小显存机器的选型流程

1. 只读核实卡容量/链路、设备归属和模型配置，从256Ki→512Ki→1Mi→2Mi逐档规划。
   不支持档位明确阻断，不能跳过后声称全矩阵通过。
2. 按head-shard、复制MLA、共享/压缩池、固定state等实际布局建模。
   先算每请求副本是否装下一条完整请求，再算多请求，不先铺DP。
3. 高速互联域内试支持的存储分片、TP/EP；有复制优先去复制。跨节点另计通信/接纳代价。
   PP按实际层/专家/Cache分布核算，避免平均显存掩盖爆卡rank。
4. 按需限制大graph池、chunk workspace及额外overlap buffer，重测性能/峰值。
   可接受小chunk/少并发仍须达预设SLO，不能自动改变本次输入契约。
5. KV量化建立新固定格式组，验证长距离检索/多位置/真实任务质量及精度前后门禁。
   不悄悄改窗口、丢token或改注意力语义。
6. host/storage前缀保留恢复只计流量/时间和分层命中，不加到活动HBM容量。
   真正活动卸载/流式attention需证明HBM峰值有界、兼容性和PCIe/网络代价；
   本版未建模这种协议，保留待验证，不给容量折扣。
7. 所有副本仍装不下时返回infeasible或证据不足，后续需要更多资源、已验证的新Cache格式
   或真正活动卸载，不能合并DP数字。

从C1/C2/C4起步后，按预测容量逐步扩展并发/驻留探边界，并在比较前冻结各候选共用的条件矩阵。
例如要认证12个slot，必须补充对应满驻留服务SLO及峰值证据；默认C4不能认证12-slot服务容量。
拓扑候选只做列举/淘汰/排名；下一轮新增哪些组合、增大哪些并发仍由用户/未来搜索器决定。

参考[vLLM KV offload说明](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md)
和[SGLang HiCache说明](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)，
实际版本支持组合、可恢复前缀和活动序列驻留边界要逐项核实。

## 通信、吞吐与延迟评估

固定**全局**新token、序列与路由分布，另报每rank有效/padded token、消息量和最慢rank。
不能固定per-GPU batch再随卡组改变全局工作量；EP需专家负载和dispatch/combine证据。
定义m为全局token数、p为并行组rank数、H为hidden维度、b为每元素字节、K为top-k。
同global工作、理想均匀路由、相同字节宽度的ring估计（每rank发送量，接收另列）：
单次TP AR每rank约`2(p−1)/p×mHb`；EP dispatch约`(p−1)/p×(m/p)×KHb`，
combine同量，二者之和/一次AR为`K/p`。这是特定代数条件；padding、量化、融合、算法、
额外collective和偏斜都会改流量，不能据此断言服务吞吐。

逐phase保留logical payload、wire bytes、group/algorithm/link、带宽参考、rank持续时间、
到达偏斜/等待、计算重叠和额外buffer。helper只接受tx_wire_bytes_per_rank；
wire_bytes是每rank发送字节，参考速率必须匹配该方向，不能与发送+接收总量混用。
参考时间 `messages×startup_us+wire_bytes/bandwidth_bytes_per_s`，缺参考保留unknown。
它不是服务延迟；busbw、algbw、链路单/双向带宽不能互换，不能累加重叠phase当总耗时。
继续用[性能模块](performance.md)的逐rank并集、暴露与依赖调查定位瓶颈。

服务：logical input token/s、新prefill token/s、request/s分开，分母含全部提交、失败与最终drain。
Locust固定并发与开放到达率不能混比；稳态保留完整offer窗口，失败/超时/不完整SSE不合格。
forward：coordinated all-rank/all-stream prefill边界，包含同请求全部chunk，不选最快chunk。
算子：同场景forward捕获shape/layout/routing，结果回完整服务复验。
TTFT含排队/调度/前处理，不等同GPU forward。只生成1token，不验decode/TPOT。

每条件至少3个匹配独立配对轮次，round ID和全局条件相同。外部执行器负责交替/随机顺序、
精度在计时前后、真实并发与Cache证明。离线gate要求每轮吞吐/延迟SLO都过线，
吞吐和延迟跨轮CV不超预设值（默认3%）。P95每轮≥200样本，P99≥1000；可显式选P50，
但不能声称尾延迟通过。超长样本成本高时先做少量容量/质量资格，尾延迟保留inconclusive。

## 可运行入口与输入契约

```bash
ifa plan configs/campaigns/long-context-capacity.json --output runs/long-plan-001
ifa deployment-demo --output runs/deployment-demo-001
# 复制演示input到独立工作区，替换证据后导入：
ifa deployment workspaces/deployment/bundle.json --output runs/deployment-001
```

长上下文档24 workload=4长度×3并发×2 Cache，三层72条/架构、三个架构216条。
默认仍44 workload/132条/架构；全tiers152 workload/456条/架构。勾选tier不会启动作业。
模型或机器不支持时保留阻断记录。

JSON使用schema_version=1，字段可复制合成演示的input/deployment.json：

| 字段 | 内容/校验 |
| --- | --- |
| kind / comparison | synthetic或measured；架构/模型/权重/镜像/runtime/tokenizer/dataset/global_workload/cache_format齐全 |
| target | 总长度、输出预留、原生长度与留存模型配置来源 |
| devices | 唯一物理设备ID、memory_bytes及来源；每候选同集合 |
| policy / conditions | 权重/最低容量、round IDs、CV/分位数；条件长度/全局并发/Cache/冻结reference/SLO |
| axes / replicas | TP/EP/PP/DP/CP意图；副本及完整rank预算、唯一Cache池和layout来源 |
| qualification | 同身份质量/前后精度/顺序/后端组合、exact-context扩展/runtime限制、全rank峰值/副本slots及原始来源 |
| measurements | 每条件每轮native service与coordinated forward；service沿用第二模块原始请求schema |
| communication | 可选，有来源的逐phase wire流量/匹配参考，不替代native计时 |

identity包括kind、完整comparison、candidate_id、layout_signature；测量另加condition和round_id。
condition的forward_boundary_id固定完整prefill计时范围；资格文件还须绑定全部qualified_condition_ids
和qualified_round_ids，不能拿某一轮的精度声明覆盖新轮次。Cache全命中且新prefill为0的条件
应单独做Cache-only研究，不能纳入本模块新prefill吞吐评分。
`deployment.selection.layout_signature`对axes、replicas、devices、target做SHA256，
防止复用其他放置的证明；service/forward身份须精确匹配。原始来源和验收字段仍是生产者声明，
SHA256只绑定材料，不认证真实性。应由可信外部验证器产出，不能手改kind当实测。
memory还需simultaneous_active_contexts_verified、no_eviction_or_recompute与capacity_condition_id，
将容量证明与满驻留服务条件绑定。reported_context_slots只是输入报告值，全部门槛通过后才赋予
verified_context_slots并参与推荐。
service额外要求actual_cache_profile_verified和actual_concurrency_verified；cold-miss逐请求检查cached=0，
50%/90%/全命中按owner归一化计数独立检查token加权命中，cache_fraction_tolerance默认0.02，
页很大时应在条件中预先冻结符合测试契约的容差。其他分层准备仍需外部验证，布尔字段不代替证明。
逐条件输出新prefill/逻辑input/request三种吞吐和实测命中范围，不将logical TPS当成新计算吞吐。

缺qualification可预算但不推荐；缺SLO/轮次/质量/峰值输出拒绝原因，身份矛盾或格式非法报错。
输出deployment.selection.json、report.md、事件流、原始input与SHA256 manifest，拒绝覆盖旧目录。
含逐rank各池预算、副本瓶颈、三目标排名与Pareto集合。synthetic永远recommendations=null，仅演示。
measured合格候选可以成为离线推荐，**不会部署/替换生产**。
原v1 replay仍拒绝拓扑变化；新模块是固定资源的独立契约。ifa plan仅记录任务，选型需单独调用。
