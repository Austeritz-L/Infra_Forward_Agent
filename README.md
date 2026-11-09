# Infra Forward Agent

**面向多架构的综合性能 Agent：规范测试、统计分析、分布式选型、算子优化与可选 trace 对比。**

当前目标架构登记为 **Moore S5000、NVIDIA SM90、NVIDIA SM103**，后续通过架构档案扩展。
五个模块共享场景、环境、证据与验收规则；通信融合、多 Stream overlap 与 KV 容量贯穿其中。
总体结构与用户控制入口见 [综合框架](docs/framework.md)。

Infra Forward Agent 希望把 Infra 工程师的优化工作组织成一个持续闭环：

```text
固定镜像与 workload → 基线验收 → 采集 trace → 分析 → 提出可证伪假设
        ↑                                              ↓
重新定位瓶颈 ← 保留通过验收的候选 ← 精度/性能验证 ← 隔离实现候选
```

项目借鉴 [Kernel Design Agents](https://github.com/NVlabs/kda) 的任务契约与实验工作区，
将优化对象扩展到完整 forward。分析方法参考
[HTA](https://github.com/facebookresearch/HolisticTraceAnalysis) 的时间拆解与通信重叠分析；
大规模 trace 的 SQL 分析后端计划参考 [Perfetto](https://perfetto.dev/docs/analysis/getting-started)。
参考关系、版本与实现边界见 [references](docs/references.md)。

## v0.3：Prefill 测试规范、五模块规划与可执行回放基线

新增 `ifa plan`：展开测试矩阵，为每个架构生成 preflight 和五模块任务，
记录依赖、所需证据、用户经验引用及预期输出。它只做规划，不执行 GPU 作业或调用模型。

| 模块 | 细化入口 |
| --- | --- |
| 规范测试条目、服务多场景覆盖 | [Testing](docs/domains/testing.md) |
| prefill/decode forward、算子占比与性能 | [Performance](docs/domains/performance.md) |
| 分布式部署：Cache 容量 / 性能 / 平衡 | [Deployment](docs/domains/deployment.md) |
| 不同架构的指令、资源与算子优化 | [Kernel](docs/domains/kernel.md) |
| trace 对比、瓶颈迁移与分析（如有） | [Traces](docs/domains/traces.md) |

保留原有 **artifact replay**：导入已有 trace、精度报告和 benchmark，执行有预算的多轮验收，
生成下一轮任务，并以 JSONL 流式输出进度。运行时仅依赖 Python 标准库。

| 能力 | 当前状态 |
| --- | --- |
| Chrome / PyTorch 导出的完整事件 JSON、JSON.gz 导入 | 已实现；需选择单个 forward 窗口 |
| 按 rank/device 拆解热点、busy/idle、通信与计算时间重叠 | 已实现；启发式调查线索 |
| 算子 / 通信 / overlap / 分布式任务生成 | 已实现；分布式任务显式标记缺证据 |
| 精度前后验收、逐条件 forward 延迟/吞吐比较、噪声门禁 | 已实现；消费外部验证器报告 |
| 候选谱系、证据快照及 SHA256、候选淘汰、基线前进 | 已实现；仅选择证据，不部署候选 |
| 流式事件、实验报告、CPU 演示与 CI | 已实现 |
| 五模块任务规划、有限场景矩阵、多架构档案注册 | 已实现；不执行硬件作业 |
| 用户经验目录、案例模板与任务引用 | 已实现；自动检索/应用待接入 |
| Prefill 聚焦的测试 suite v2、15 场景族、服务→forward→算子绑定与验收契约 | 规范与规划已实现；真实发压、捕获、统计和 runtime gate 待接入 |
| 自动抓取 Nsys / NCU / MCU、SSH / 容器执行 | 待实现；已有适配协议 |
| LLM 规划、代码生成、KDA 执行桥接 | 待实现；已有任务模板与交接契约 |
| 因果关键路径、完整跨 rank DAG、KV 容量实测 | 待实现 |

**演示中的 trace 和性能数值全部为合成输入，不代表 NV、S5000 或模型的实测收益。**
现阶段不需要模型 API Key，也不会连接 GPU 主机或修改服务。

## 快速运行

```bash
git clone https://github.com/Austeritz-L/Infra_Forward_Agent.git
cd Infra_Forward_Agent
python3 -m venv .venv              # Python >= 3.10
source .venv/bin/activate
python -m pip install -e '.[dev]'

ifa architectures
ifa plan configs/campaigns/multi-architecture.json --output runs/campaign-plan-001
ifa demo --output runs/demo-001
python -m unittest discover -s tests -v
ruff check .
```

默认 suite 仅研究 prefill，统一输出 1 token，排除 decode/TPOT 验收。
smoke+core 共 44 组服务工作负载，各自关联 forward/operator，合计 132 条规划记录/架构、
三个架构共 396 条，生成 18 个阶段任务（含 preflight）。
规划输出包含 campaign.plan.json、scenario.matrix.json、testing.contract.json、events.jsonl 和 plan.md。
`execution_ready=false` 与 `needs_adapter` 表示真实 runner 尚未接入；未提供 trace pair 时跳过对比。

完整 [测试矩阵与验收规范](docs/domains/testing.md) 包含冷算、Locust 固定工作量、20 分钟稳态、
命中控制、长短混合、突发/阶梯/到达率、淘汰抖动与可选分层恢复。用户通过
`configs/scenarios/prefill-service.json` 的 enabled_tiers、workloads 与 contracts 修改场景、Cache、
发压协议、指标和初始验收门槛。全部 tiers 共 128 组 workload，384 条记录/架构，不会默认执行。

演示包含两个 rank、C1/C16 两个条件和三轮候选：算子候选通过后通信成为新热点，
overlap 候选继续通过，最后一个候选因精度失败被拒绝。所有轮次均保留。
输出目录必须不存在，避免覆盖旧实验。

```text
runs/demo-001/
  config.json                    # 本次策略与迭代预算
  evidence/*/                    # 原始输入快照 + manifest.json
  baseline.analysis.json
  iteration.001.analysis.json    # 通过候选后重新分析
  tasks.000.json                 # 下一轮优化任务
  decision.001.json              # 每个条件的指标与验收原因
  events.jsonl                   # 顺序编号的流式事件
  summary.json
  report.md
```

编辑输入和单独检查：

```bash
ifa fixtures --output workspaces/replay-inputs
ifa run workspaces/replay-inputs/replay.json --output runs/replay-001
ifa analyze workspaces/replay-inputs/baseline/rank-0.trace.json --rank 0
ifa compare workspaces/replay-inputs/baseline/evidence.json \
  workspaces/replay-inputs/kernel-candidate/evidence.json
```

`compare` 返回 0 表示通过、2 表示淘汰、1 表示命令/输入错误。
`run` 返回 0 表示流程完成，候选可以全部被淘汰；自动化应同时检查 `summary.json`。
输入格式和实测接入步骤见 [evidence contract](docs/evidence-contract.md)。

## 仓库结构

```text
src/infra_forward_agent/
  campaign.py               # 多架构总控规划与五模块依赖
  scenarios.py              # service/forward/operator 场景矩阵
  testing/                  # prefill suite v2 校验与同源三层测试规范
  agents/                   # testing/performance/deployment/kernel/traces 五个逻辑职责
  architectures/            # 可扩展注册表 + S5000/SM90/SM103 档案
  contracts.py              # 证据、策略、决策契约
  trace.py                  # Chrome complete-event 归一化
  analysis.py               # 时间并集、热点、调查任务
  evaluation.py             # 精度/可比性/延迟/吞吐验收
  engine.py                 # 有预算的 replay 闭环
  store.py                  # 实验快照与流式事件
  adapters/protocols.py     # Profiler / Optimizer / Executor 扩展接口
  cli.py / demo.py          # 命令行与合成演示
configs/campaigns/          # 架构列表、容量/性能目标、经验与 trace 引用
configs/scenarios/          # 用户可编辑的场景 suite
configs/backends/           # 厂商后端集成声明；不代表已完成接入
docs/                      # 架构、证据、路线图、研究来源
playbooks/                 # 四个优化方向的调查与验证方法
prompts/                   # coding agent 的任务起点
templates/                 # 单项优化实验的工作区模板
knowledge/                 # 用户实测案例入口；当前仅引用，未自动检索
integrations/kda/          # 将热点任务交给 KDA 的接口约定
tests/                     # CPU 上验证分析与验收逻辑
```

## 范围与原则

- 主要评价对象是固定 workload 下的**完整 forward 延迟与有效 token 吞吐**。算子 microbenchmark
  用于归因；服务 TTFT/TPOT、端到端请求吞吐单独验收，不能互相替代。
- 框架版本、模型语义、权重量化和业务流量固定。算子替换、通信融合、TP/EP/PP/DP 放置、
  KV 容量和 Stream 调度是探索对象。每个实验声明允许变化的轴。
- 先精度，再不带 profiler 的计时，再精度。保留所有 rank、条件和失败记录。
- 累计 kernel 时间不等于 forward 墙钟时间；时间重叠不等于依赖正确，也不能证明因果关键路径。
- 控制面保持硬件中立，后端能力逐项实测。S5000 的 MUSA/MCU 与 NVIDIA 的 CUDA/Nsys/NCU
  分开接入，不能把 CUDA 支持直接当作 MUSA 支持。
- 历史 S5000 部署和优化工作只提供待导入的方法与案例。旧作业不会因本仓库创建而恢复。

请从 [framework](docs/framework.md)、[architecture](docs/architecture.md)、[roadmap](docs/roadmap.md) 和
[contributing](CONTRIBUTING.md) 继续。新代码采用 [MIT](LICENSE) 许可；未包含第三方实现。
