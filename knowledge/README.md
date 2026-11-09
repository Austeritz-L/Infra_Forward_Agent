# 用户经验与案例入口

长期通用规则放 AGENTS.md，诊断方法放 playbooks，具体实测与失败案例放这里。
建议按 `cases/<architecture>/<topic>.md` 组织，每条经验必须说明适用条件、证据与反例。
从 templates/knowledge/case.md 复制开始；不要把历史结论当作当前硬件的实测或资源预约。

campaign 的 knowledge_paths 可引用本目录。当前 planner 仅记录这些引用，KernelAgent 将其放入
后续任务输入；尚未实现全文检索、自动提示词注入、规则解析或自动修改决策。
没有新增任何历史实测案例，避免在未核对原始证据时迁移结论。
