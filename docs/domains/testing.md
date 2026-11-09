# 1. 规范测试条目

入口：configs/scenarios/baseline.json；实现位置：scenarios.py、agents/testing.py。
当前可规范 scope/phase、展开有限的场景轴并生成待执行条目。

后续由用户细化：服务类型、prefill/decode/混合、长度与并发、prefix/cache 状态、路由分布、
流式输出、持续负载、错误/超时、输出一致性、各场景的通过标准。
每条测试应有固定输入与环境身份、cache 准备、热身、同步边界、样本数和清理步骤。
service/forward/operator 三层验收独立；跳过和失败必须保留原因，不能静默遗漏。

待接：service/forward/operator runner、reference 校验器、服务覆盖报告。
