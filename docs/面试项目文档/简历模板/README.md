# 简历模板 — NexusMind（语义驱动的多智能体 ChatBI）

> 使用说明：提供完整版与精简版两个项目经历段，均可直接粘贴；数据与口径与 `docs/INTERVIEW_GUIDE.md` 保持一致。

---

## 一、完整版（约 420 字）

### NexusMind — 语义驱动的多智能体 ChatBI 系统（个人项目）

**技术栈**：Python / FastAPI / LangGraph / React / sqlglot / Milvus / Redis / PostgreSQL / Docker Compose / MCP

- **语义层确定性编译**：自研 MetricPlanCompiler，沉淀 14 个业务指标、18 个维度和 6 类复合指标计划，自然语言先解析为 IR（指标/维度/过滤/排序/LIMIT）再确定性编译 SQL，指标口径由代码保证，不依赖 LLM 即兴生成
- **SQL Guard 三层防护**：sqlglot AST 只读校验 → 执行错误反馈自修复 → 空结果/LIMIT 截断/NULL 占比等结果合理性检查，拦截"能执行但不可信"的静默错误
- **Schema 检索与 RAG**：Domain→Table→Column 三级收敛 + DDL/业务定义/SQL 示例三语料混合检索（Dense+BM25+RRF+Rerank），控制 Prompt 长度并降低表列幻觉
- **多智能体编排**：LangGraph 实现"根因规划链 / 语义编译 / ReAct 循环 / Coordinator 并行"多路径路由；ContextStore 把超长结果归档为 ref_id，支持按需回忆，截断变成延迟加载
- **四套独立评测**：域内端到端 15 题 EX 100%、复合专项 12 题 100%、泛化 10 题 80%、Spider 20 库 99 题 EX 81.8%（zero-shot，高于 GPT-4 zero-shot 的 60-75%），并以表命中率 97%、行召回率 0.851、字段准确率 0.898 构成诊断链
- **工程闭环**：MCP JSON-RPC 2.0 工具协议 + 三级记忆 + 会话压缩恢复 + RBAC/SQL 审计；20,606 行代码、34 项回归测试、Docker 六容器一键部署

---

## 二、精简版（约 200 字）

### NexusMind — 多智能体 ChatBI 系统

**技术栈**：Python / FastAPI / LangGraph / sqlglot / Milvus / Docker

- 语义层 MetricPlanCompiler：14 指标 × 18 维度 × 6 复合计划，NL→IR→确定性 SQL，口径可审计
- SQL Guard：AST 只读校验 + 执行自修复 + 空结果/截断/NULL 合理性检查
- Schema 三级检索 + Dense/BM25/RRF/Rerank 混合检索；ContextStore 延迟加载突破上下文限制
- 四套评测：域内 15 题 EX 100%、Spider 20 库 99 题 81.8%（表命中 97%、行召回 0.851、字段准确 0.898）
- MCP 自研工具协议 + 三级记忆 + RBAC 审计，34 项回归测试

---

## 三、STAR 口述稿

**S（情境）**：业务方用自然语言问数，纯 LLM 直出 SQL 存在指标口径漂移、JOIN 错误和静默失败，很难做到可信。

**T（任务）**：构建一个把"能确定的"交给确定性编译、把"不确定的"交给带守卫的 LLM 的 ChatBI 系统，并建立可诊断的评测体系。

**A（行动）**：1) 设计 SchemaLinker 把问题解析成 IR，按置信度路由到语义编译、复合模板、根因链或 LLM fallback；2) 用 sqlglot 实现 AST 校验、执行自修复与结果合理性检查；3) 搭建四级 Schema 检索与三语料混合检索控制 Prompt 长度；4) 建立四套评测和字段级指标，用 Bad Case 闭环驱动迭代。

**R（结果）**：域内 15 题 EX 100%；Spider 20 库 99 题 EX 81.8%，表命中 97%，行召回 0.851，字段准确 0.898；34 项回归测试全通过。

---

## 四、使用建议

1. Spider 81.8% 是 LLM 直出（无语义层）的口径，域内 100% 是自建小集口径——两个数字必须成对出现，避免被追问时口径不一致。
2. 预期追问优先级：语义层 vs 纯 NL2SQL → 四套评测关系 → SQL Guard 自修复 → Spider 指标为何精简为 4 项 → AST 前缀表误拦等真实 bug。
3. 深挖答案见 `docs/INTERVIEW_GUIDE.md` 与 `../面试QA/README.md`。