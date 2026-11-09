# Contributing

先阅读 README.md、AGENTS.md 与 docs/evidence-contract.md。
改动应对应一个明确的问题、可复现输入和可检验的结果。

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check .
python -m build
```

添加后端时实现 adapters/protocols.py 中的接口，并标注 implemented / planned / unsupported。
在隔离工作区保留契约、patch、完整 rank 精度结果、未带 profiler 的 benchmark 和原始 trace。
CPU CI 验证控制逻辑；真实硬件验收单独提交环境摘要、运行命令和证据路径。
不要将合成 fixture、kernel microbenchmark 或历史结果写成当前 forward 收益。

提交 PR 时说明具体触发条件、行为变化、验证结果与已知限制。仓库只收小型去敏 fixture；
大型 trace 和私有日志使用外部存储，引用校验摘要。不要提交主机凭证、用户输入或权重。
