# 证据格式与真实接入

先运行 `ifa fixtures --output workspaces/inputs`，获得完整可编辑的 v1 输入示例。
`evidence.json` 是 manifest；相对路径均以该文件目录为基准，不能逃逸该目录。
导入时复制产物并生成 SHA256 清单。该清单保证后续可核对输入内容，不证明外部测量真实可靠。

## Evidence bundle

| 字段 | 约定 |
| --- | --- |
| schema_version | 当前为 1 |
| kind | synthetic 或 measured；两类不可比较 |
| id / revision | 候选标识及被测代码 revision；真实 runner 应附 patch hash |
| environment | image_digest、hardware、runtime、profiler、topology（必须有 ranks） |
| workload | 模型、权重与输入指纹、dtype、seed、forward 边界及其他语义约束 |
| traces | 每个 rank 一个 `{path, rank, marker}`，选定单个 forward |
| benchmark | 不带 profiler 的 benchmark JSON 路径 |
| precision.before / after | 两个 precision JSON 路径 |

v1 比较时 environment 与 workload 必须完全一致。image_digest 应是镜像内容摘要；硬件应含
设备型号/数量、固件、频率策略，runtime 应含框架及驱动版本；topology 应含 TP/EP/PP/DP 与
rank 到设备/节点的映射。当前读取器仅检查这些顶层字段存在，内容真实性依赖产物生成方。
模型名和 dtype 不足以描述量化，真实任务还应保留 config/index、tensor dtype、scale/group/layout。

Benchmark 报告包含 `completed=true, profiled=false, errors=[], warmup_iterations>=1`、
`clock_id, started_ns, finished_ns` 和 `conditions`。
每个 condition 包含唯一 `id`、非空 `spec` 和至少 5 个默认样本：

```json
{
  "id": "decode-c16",
  "spec": {"input_length": 1024, "output_length": 128, "concurrency": 16, "phase": "decode"},
  "samples": [{"forward_ms": 8.0, "tokens_per_s": 2000.0}]
}
```

上例仅展示样本形状，单样本不满足默认门禁。测量器应定义同步边界、每样本 step 数、有效 token
计数方法、热身、图捕获、prefix/cache 状态、路由分布和计时器开销。`tokens_per_s` 是测量器
直接报告的有效 token/秒，不由本工具从平均延迟猜算，不混入 padding token 或其他请求的计数。
forward span 内的多 Stream 工作必须被边界同步覆盖，不能只对某一个 Stream 计时。

Precision 报告包含：

- `passed, outputs_finite, contract_equal` 必须为 true，`errors` 必须为空。
- `validation_contract` 标识精度参考与容差；具体输出、逐 tensor/rank 差异报告应由 runner 留档。
  前后检查、基线和候选必须使用同一个 contract。
- `ranks, conditions` 必须覆盖完整矩阵。
- `clock_id` 与 benchmark 相同，`started_ns, finished_ns` 必须满足前检、计时、后检的顺序。

这些布尔值是外部验证器的声明。v0.1 不重放模型，也不检查数值 tensor 或验证布尔值是否属实。
下一个 live runner 必须把声明绑定到真实验证程序、revision、运行身份和原始输出。
候选可接受性不等于 checkpoint 精度、服务稳定性或请求级质量均已验收。

## 默认选择策略

默认每条件至少 5 个样本；延迟和吞吐样本均为正且有限，CV ≤ 5%。
使用样本中位数比较：

```text
latency_gain    = 1 - candidate_forward_ms / baseline_forward_ms
throughput_gain = candidate_tokens_per_s / baseline_tokens_per_s - 1
```

每个条件的主目标收益须 ≥ 3%，两个指标任一退化不得超过 1%。
主目标可配置 latency 或 throughput。少量稳定样本与 CV 阈值属于工程筛选，
不构成统计显著性证明；正式收益报告应增加交错 A/B、重复批次、分位数及置信区间。
timeout、错误、缺样本、NaN、零时间、精度失败不得作为有效候选。

## Trace 输入

只支持 Chrome JSON 或 gzip JSON 的 `ph=X` 完整事件；`ts`/`dur` 单位为微秒。
`displayTimeUnit` 只影响显示，不改变时间戳单位。
GPU category 白名单为 `kernel, gpu_kernel, musa_kernel, gpu_memcpy, gpu_memset`。
CPU `aten::*`/runtime range 不作为 GPU 工作。

每个 rank 必须有一个名字与 `marker` 相同的 forward complete event，默认名字 `forward`。
多个 step 需先挑一个导出或使用唯一名字标记；不自动选择首个 step。
`args.rank` 优先于 trace spec 中的 rank；`args.device` 优先于 pid，stream 优先用 `args.stream`。
多进程使用 pid 回退可能分裂同一物理设备，正式 adapter 应填写真实设备映射。
`args.ifa_kind` 可显式指定 compute、communication、memory。
未给 hint 时，仅以 NCCL/MCCL/RCCL 名字猜测通信；自定义 dispatch/combine 需要显式标注。
MCU/NCU 表格、Nsys SQLite 与 Perfetto protobuf 需要专门转换器，当前不直接接受。
大 trace 当前全量读入内存，后续用 Perfetto/HTA 后端扩展。

## 首个真实任务如何接入

1. 填写 [任务契约模板](../templates/task/task.json)，固定一个镜像与一个 prefill/decode 工作负载。
2. 在已确认归属的隔离设备上运行精度检查，再收集不带 profiler 的完整 forward 样本，再复查精度。
3. 另外采集一个代表性 forward 的全 rank trace，补齐分类与边界，避免 profiler 开销污染收益。
4. 将真实报告整理为 bundle，kind 改为 measured，填入实测环境与权重身份。
5. 从生成的 replay.json 改为真实 bundle 路径；先验收基线，再逐个导入候选。
6. 检查各条件 decision、trace 分析和原始证据后，再决定是否做服务验收。
