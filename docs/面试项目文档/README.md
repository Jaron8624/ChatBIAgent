# NexusMind 面试项目文档

> 语义驱动的多智能体 ChatBI 与数据洞察系统。本文档将项目拆成可独立讲述的工程主题；主线始终是：**能确定的，不交给概率**。

## 文档地图

| 模块 | 文档 | 面试一句话 |
|---|---|---|
| 语义与路由 | [01-意图识别与ChatBI路由](./01-意图识别与ChatBI路由/README.md) | 规则、LLM、历史模式三路融合，将问题分到数据查询、趋势、对比、根因和 Schema 咨询 |
| 工具协议 | [02-MCP工具协议与发现](./02-MCP工具协议与发现/README.md) | 自研 JSON-RPC 2.0 工具协议，ChatBI 模块与外部工具统一接入 |
| 复合查询 | [03-ReAct循环-Schema到SQL取证](./03-ReAct循环-Schema到SQL取证/README.md) | Schema → SQL → 观察 → 再查询，用签名去重和上限控制循环 |
| 多智能体 | [04-多Agent协作-规划与并行](./04-多Agent协作-规划与并行/README.md) | SOTA 规划、flash 执行、Coordinator 并行拆解，按复杂度分级成本 |
| 上下文 | [05-动态上下文-ContextStore延迟加载](./05-动态上下文-ContextStore延迟加载/README.md) | ContextStore 将截断变成可检索的延迟加载 |
| 知识检索 | [06-RAG与Schema混合检索](./06-RAG与Schema混合检索/README.md) | 父子块 + Dense/BM25 + RRF + Rerank，并复用于 DDL/指标/SQL Schema 检索 |
| 记忆 | [07-三级分析记忆与口径复用](./07-三级分析记忆与口径复用/README.md) | 分层长期记忆保存指标口径偏好、分析事件和操作偏好 |
| 编排 | [08-LangGraph编排-确定性优先](./08-LangGraph编排-确定性优先/README.md) | root cause 专属链路、NL→IR→SQL 编译优先、失败再降级 |
| 在线质量 | [09-在线质量门禁与监控](./09-在线质量门禁与监控/README.md) | p99、成功率和 SQL Guard 门禁指标驱动在线观测 |
| 评测 | [10-双层评测-过程与结果](./10-双层评测-过程与结果/README.md) | 过程层定位路径，结果层以 EX/表/行/字段验证结果 |
| 安全 | [11-SQL安全门禁与审计](./11-SQL安全门禁与审计/README.md) | 写操作拦截、AST 校验、结果合理性与审计轨迹 |
| 权限 | [12-数据权限与审计](./12-数据权限与审计/README.md) | RBAC、危险操作确认与审计追溯 |
| 自进化 | [13-BadCase闭环与Fewshot进化](./13-BadCase闭环与Fewshot进化/README.md) | Bad Case 修复必须经字段级验证后才回写 Few-shot 知识库 |
| 分析会话 | [14-多轮分析会话恢复](./14-多轮分析会话恢复/README.md) | Redis 快照保存指标、过滤条件和 SQL schema，支持追问恢复 |
| 交付 | [15-MCP部署与Docker六容器](./15-MCP部署与Docker六容器/README.md) | MCP 双传输 + Docker 六容器 + Milvus Lite/standalone 双模式 |
| 求职 | [简历模板](./简历模板/README.md) / [面试QA](./面试QA/README.md) | 可直接复用的简历段和 ChatBI 深挖问答 |

> 每篇文档均按“一句话口诀 → 配图 → 通俗讲解 → 面试问答 → 记忆口诀”结构编写，小白也能看懂，可直接背诵。配有实现过程示意图。

## 量化底座

| 评测 | 结果 |
|---|---:|
| 域内端到端集 | EX 100%（15/15） |
| 复合专项集 | EX 100%（12/12） |
| 域内泛化集 | EX 80%（8/10） |
| Spider dev 全 20 库 | EX 81.8%（81/99） |
| Spider 表命中率 / 行召回率 / 字段准确率 | 97.0% / 0.851 / 0.898 |
| 工程验证 | 34 项回归测试 |

建议先阅读 [`../../INTERVIEW_GUIDE.md`](../../INTERVIEW_GUIDE.md) 获取完整项目话术，再按上述主题准备追问。