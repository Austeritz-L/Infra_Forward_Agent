# 1. Prefill 场景测试矩阵与验收设计

配置：`configs/scenarios/prefill-service.json`，suite schema v2；展开：`testing/matrix.py`。
总控额外输出 `testing.contract.json`。当前实现规范校验、矩阵展开与三层关联；HTTP/Locust runner、
捕获器、统计器与运行时验收器尚未接入。本文是拟执行规范，阈值为可调整的工程初始建议。

## 三层范围与指标

本轮只研究 prefill，服务统一 temperature=0、max_new_tokens=1，保留首 token 与完整结束证据。
不验收 decode forward、TPOT、ITL 或 decode token/s；单输出 token 的 TPOT 为 N/A。
服务仍含 logits/head、采样、HTTP/router/排队/传输；TTFT 不等于纯 prefill forward。

| 层 | 主指标 | 边界与精度 |
| --- | --- | --- |
| Service | TTFT P50/P95、logical input token/s、new prefill token/s、request/s | 发送到首语义 token；完整消费协议终止；配对首 token ID 全量一致 |
| Prefill forward | step wall P50/P95、实际新增 token/s、各 rank interval | 实际 prefill/extend microbatch，加入全部相关 Stream；固定 head 边界，全 rank 前后精度 |
| Operator | 单调用 P50/P95、服务调用权重累计 work、数值误差 | 相同 shape/layout/量化/路由/状态；kernel-only 和完整 wrapper 分开，容差绑定 reference |

并发是客户端在途上限，不等于 scheduler batch。输入长度是模板和分词后的完整 prompt，
不等于 Cache 命中后的 query 长度。冷算表示可复用 prefix/KV 冷，权重/编译器/执行器已经热身。
不同模型/tokenizer 使用不同输入 manifest；同模型跨架构复用同一份输入身份与测试口径。

## 场景族

K=1024 token。下表数量是单架构 service 条件数；每组还派生对应的 forward/operator 条目。

| ID / tier | 场景 | 输入 | 并发或到达方式 | Cache | 条件数 |
| --- | --- | --- | --- | --- | --- |
| S00 smoke | 快速正确性/链路 | 2K/8K | C1/C4 | miss/90% | 8 |
| S01 core | 固定长度冷算 | 2/4/6/8/16/32K | C1/C16 | miss | 12 |
| S02 core | 等长控制命中 | 4/16/32K | C1/C16 | 50%/90%/完整前缀热 | 18 |
| S03 core | Locust 固定工作量历史回放 | 会话按长度和轮次入桶 | C1/C12/C16，spawn2 | miss/自然历史 | 6 |
| S04 extended | 并发扩展 | 8/32K | C2/C4/C8/C32/C64 | miss | 10 |
| S05 extended | 长短混合公平性 | 2/8/32K，权重50/30/20 | C4/C16/C32 | miss/90% | 6 |
| S06 extended | 历史交错方式对照 | 32/96K | C16，间隔0/250ms | miss | 4 |
| S07 soak | Locust 20分钟稳态 | 冻结会话与游标 | C12/C32 | 自然历史 | 2 |
| S08 stress | 阶梯与回落 | 8/32K | C1→4→12→16→32→16→4→1 | miss/90% | 4 |
| S09 stress | 突发与恢复 | 2/32K | C16/C32/C64 bursts | miss/90% | 12 |
| S10 stress | 到达率/排队拐点 | 8/32K | 冻结基线RPS的0.5/0.8/1/1.2倍 | miss | 8 |
| S11 stress | 工作集淘汰 | 8/32K | C4/C16，容量比0.5/1.2/2 | 淘汰抖动 | 12 |
| S12 optional | host恢复 | 16/32K | C1/C16 | device冷、host热 | 4 |
| S13 optional | storage恢复 | 16/32K | C1/C16 | device/host冷、storage热 | 4 |
| S14 optional | context/显存边界 | 64/96/128K | C1/C4/C16 | miss/90% | 18 |
| S15 long_context | 长上下文容量/性能 | 总长度256Ki/512Ki/1Mi/2Mi，prompt减1 | C1/C2/C4 | miss/90% | 24 |

默认 smoke+core：44 workload × 3层 = 132记录/架构；三个架构共396记录。全部tiers：
152 workload、456记录/架构（包含显式long_context档）。smoke只判功能，不具备正式性能资格；S08–S11为characterization，
超载拒绝与容量边界单独报告，不冒充正常条件的qualification成功。其余重型tier显式开启。

历史会话按512–8K、8K–32K、32K–128K桶记录实际长度，保留turn/tool结构，不为落桶截断消息。
超出模型context的条件明确blocked/not_applicable。S05逐长度桶报告尾延迟与完成率，防止总体
均值掩盖长请求饥饿。资源、context与Cache能力必须在执行前核实。

long_context总长度包含1 token输出预留，prompt为262143/524287/1048575/2097151。
这是规划目标，须先通过[分布式长度/容量资格](deployment.md)，再执行冷算/命中条件并绑定
实际prefill与算子；未经捕获不推断shape，不能仅勾选tier就宣称模型支持2Mi。

## Cache准备与证明

| 状态 | 准备 | 验收依据 |
| --- | --- | --- |
| cold-miss | JIT热身后drain，清理/隔离全部可复用层；wave内独立前缀/namespace | owner侧实际cached=0；发送过flush不算证明 |
| partial-50/90 | 完整请求IDs和长度不变，只预热页对齐的目标前缀 | 实际比率容差max(2个百分点,一页/prompt长度) |
| warm-full-prefix | 相同prompt再发，冻结namespace | 记录logits/head末尾重算及page尾部；无调用的forward/operator须有证据标N/A |
| natural-history | 固定且互斥的用户历史，禁止数据回绕 | 测真实token加权命中，按session/turn/长度分层，不预设90% |
| eviction-churn | 先测可复用KV容量，再构建旋转工作集 | 命中、淘汰、重载、内存、排队和恢复 |
| host/storage | 明确device/host/storage冷热与ready/publication | P/prefill owner分层恢复计数与输出一致性；客户端或D侧元数据不单独作证 |

优先相同token IDs+Cache预置/namespace配对。无法隔离时可用等长、词表合法、冻结身份的fresh
prompt bank，但须报告输入变化和路由分布限制。不能额外加128 token后声称只改变命中状态。
不要在in-flight wave中途全局flush。观测不全为inconclusive，无法控制为blocked。
按实际owner聚合sum(cached)/sum(cached+new)，去掉TP/PP、chunk、request重复；包含式tier
计数不可相加。逻辑输入、复用量、新增有效计算量、recompute/padding分别记录。

## 冷算与Locust协议

冷算：固定请求manifest、输出1token、最多8轮准备；最近3轮elapsed的max-min/mean<3%且
无新增JIT才正式5轮，每轮64请求，按配置C发压。每轮drain后重建Cache状态，A/B交错运行，
冻结请求、顺序、采样、镜像/权重、拓扑、其他服务参数并记录环境漂移。250ms交错单列S06。

Locust：继承确定性分组、逐轮增长前缀、C12/spawn2和20分钟路径。本轮输出改为1token，
下一轮仍用数据集assistant历史，明确是历史回放，不是模型输出驱动的对话。
每用户单在途请求、wait_time=0；users不是RPS，constant pacing也不是固定到达率。
S03每用户200请求×5轮，A/B全局request/session/turn manifest完全相同；多worker分组必须
全局唯一，不能各自从相同文件和uid=0重放。日志身份在推进游标前冻结；CSV/SSE/服务端room
使用同一关联ID。数据用尽停止或判无效，不能无声回绕引入额外命中。

S07先120s热身排除，再完整1200s发压，最多300s drain。时间不足、中途设备错误、无有效完成、
drain未完均不是完整成功。分别报告offered-window和正式请求cohort的首发到最后终止窗口，
完整入账offered/admitted/completed/rejected/failed/unfinished；不只保留成功片段。
固定时长A/B推进到不同轮次，故主因果对比采用S03；S07用于稳定性，解释吞吐前先对齐命中和
长度/turn分布。S08过渡段不混plateau，S09保留恢复段，S10需外部到达计划+有界队列，
RPS在baseline校准后冻结。客户端CPU/连接/调度迟到单报，不自动归给GPU服务。

## 指标分母

- TTFT：单调时钟发送至首个有token/text/token-count进展的语义事件，排除header/空SSE。
- E2E：至协议规定的成功终止并完整消费，即使1token也不强行等同TTFT。
- logical input token/s：成功cohort完整prompt总token/同cohort完整时间，包含命中前缀。
- new prefill token/s：关联的实际新增有效token/同窗口时间，recompute/padding另列。
- request/s：完整成功请求/明确时间，同时保留失败与unfinished；timed load另报offered速率。
- forward wall：同一实际step入口到全部相关Stream完成，head边界固定；各rank本地event区间
  分别报，max-rank仅为慢rank指标，不在未校时下构造跨host全局关键路径。
- operator work：标明kernel-only或wrapper，按原服务call count/shape权重累计，不平均加速比。
  若算子占比分母是累计GPU work必须明示，它不是forward墙钟占比。

命中前缀导致的逻辑吞吐上升不能直接称为计算吞吐提升；每长度/并发/Cache/架构独立成表。

## 初始验收标准（设计建议，可调）

| 项目 | 标准 |
| --- | --- |
| 完整性 | qualification零HTTP/transport/SSE/parse错误、零缺终止、finish reason合法、输出1token；零未入账/静默重试 |
| 精度 | 同请求首token全量一致；forward/operator有模型/dtype/量化专属reference容差，全rank计时前后检查 |
| 正式运行 | 无新JIT、profiler/debug同步、fault/watchdog、意外重启、OOM/retraction；身份与工作量可比 |
| Cache | 满足上述真实owner/页对齐/分层证据，不依据预期标签判定 |
| 样本 | 固定工作量配对正式5轮；soak至少1完整20分钟，仅稳定性证据，收益需独立配对复测；每条件P95至少200请求、P99至少1000，否则尾指标inconclusive |
| 噪声 | 正式轮吞吐CV≤5%，超标调查/重测，不挑较好轮次 |
| 服务非退化 | TTFT P50/P95各退化≤5%；logical input和new prefill token/s各退化≤3%，以相同Cache/负载为前提 |
| forward非退化 | P50 latency≤3%、P95≤5%、新增token/s退化≤3%；要求同源捕获契约 |
| 算子非退化 | 服务权重累计work退化≤3%、关键shape P95≤5%；合成均匀路由仅诊断 |
| 优化晋级 | 指定目标收益≥3%，配对95%区间下界>0；所有必测条件精度与非退化通过，不要求每条件都提升3% |
| 绝对SLO | 按条件填写TTFT上限/吞吐下限；目前null，只能做相对回归判定，不能声称生产SLO通过 |

样本下限是工程门槛，不保证统计显著性；区间按轮次/会话重采样，不能把相关token当独立样本。
缺容差/Cache/捕获证据为inconclusive；确实不适用为not_applicable，执行前不满足条件为blocked；
明确错误/退化为fail。必测缺失不能用“完成的项目都pass”合成整体通过。
超载characterization预期拒绝不放宽正常qualification的零错误规则。

## 服务→forward→算子绑定

workload_id派生三个scope；forward/operator保持requires_service_capture。规划ID不是实际输入
digest。真实manifest还需request/session/turn/worker、tokenizer/template/weights/image/revision、
各层cached/new、输出token、时间戳；step/layer/rank、microbatch成员、query/context、chunk、
cu_seqlens/position、KV page/index、dtype/scale/layout、Stream依赖与workspace生命周期。
MoE另外保留router indices/weights、各rank expert counts、padding、dispatch/combine和调用权重。

口径示例：16K输入、50%命中、C16。假设一个chunked step实际只选4条请求，query长度为
[2048,1024,4096,1024]，本step新增tokens=8192，不是C16×16K=262144，也不必是4×8K。
Expert GEMM的M还取决于路由、padding和rank放置；算子用捕获值，不手工猜测。
人工边界/空expert/偏斜/非默认Stream集另列。归因捕获与正式计时分开，重新准备同一Cache状态。
算子候选通过后仍需回灌同workload_id的forward和服务复验。

## 历史方法与参考

只读核对了本地M3.1的cold_p_only.py、strict_client.py、固定工作量Locust wrapper、SSE observer
和pd-conv-hit-stress-kit。继承固定工作量、多轮、真实历史与严格流检查的方法，不继承历史收益。
旧加长盐头、游标提前推进、缺SSE完整验收和无声回绕不作为新默认。没有复制旧脚本/数据/
节点/凭据，没有恢复旧任务，本次没有实际发压。

- [Locust用户、等待、响应验证](https://docs.locust.io/en/stable/writing-a-locustfile.html)：闭环与catch_response，SSE完整性额外校验。
- [Locust分布式](https://docs.locust.io/en/stable/running-distributed.html)：master/worker分工，测试另行保证全局会话分片。
- [SGLang benchmark分层](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)：HTTP/调度器/静态ModelRunner分开，静态batch不冒充动态服务批次。

原evidence schema v1与replay gate保持兼容。本suite v2只规范设计，尚未实现runtime gate，
不会把service指标自动塞进旧forward_ms/tokens_per_s字段。
