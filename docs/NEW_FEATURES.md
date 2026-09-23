# NexusMind 六大核心功能补充文档

本文档说明基于 NexusMind 项目补齐的六大核心功能的架构设计、实现链路及 API 使用方式。

---

## 1. 端到端意图识别（三路融合 + 加权投票）

### 1.1 实现文件
- `backend/intent_recognition.py`

### 1.2 核心设计
NexusMind 的意图识别模块采用**三路并行识别 + 加权投票融合**的架构，避免单一识别器的偏见和局限：

- **RuleRecognizer**：基于正则表达式和关键词映射的规则识别器，适用于高频明确意图，零成本、高确定性。
- **LLMRecognizer**：利用大模型进行结构化 JSON 分类，擅长处理语义模糊、边界交叉的复杂查询。
- **HistoryPatternRecognizer**：基于 Embedding 相似度，检索历史对话中相似 query 的意图标签，利用历史模式辅助当前判断。

三路结果汇入 **WeightedVoter** 进行加权融合：

```
最终置信度 = Σ(各路置信度 × 各路权重)
```

当融合后的置信度低于阈值（默认 0.6）时，系统返回澄清问题而非盲目猜测，降低误识别率。

### 1.3 动态权重调整
`WeightedVoter.update_weight()` 支持 Monitor 模块根据历史准确率在线调整各识别器的权重。例如 LLMRecognizer 长期准确率下降时，其权重自动下调，RuleRecognizer 权重相应提升，实现识别策略的自适应优化。

### 1.4 API
```http
POST /intent/recognize
```
请求体：
```json
{
  "query": "帮我查一下上周的订单",
  "user_id": 42,
  "history": [{"role": "user", "content": "..."}]
}
```
响应：
```json
{
  "intent": "order_query",
  "confidence": 0.91,
  "clarification_needed": false,
  "details": {
    "rule": {"intent": "order_query", "confidence": 1.0},
    "llm": {"intent": "order_query", "confidence": 0.85},
    "history": {"intent": "order_query", "confidence": 0.88}
  }
}
```

---

## 2. MCP 工具调用框架（查询改写 + LLM 重排）

### 2.1 实现文件
- `backend/mcp_protocol.py`
- `backend/query_rewriter.py`

### 2.2 核心设计
工具层按 MCP 概念模型实现发现与调用：统一工具注册表提供 tools/list、tools/call 语义，内部工具以进程内函数注册、无差别路由，不引入官方 mcp SDK，行为可单测。

#### 2.2.1 工具注册表
- 统一注册表：工具以 name / description / JSON Schema 参数定义注册，提供发现（list）与调用（call）两个标准动作。
- 客户端封装：调用侧统一序列化参数、反序列化结果，业务方不感知传输细节。

#### 2.2.2 查询改写（Query Rewriter）
在多轮对话场景下，用户输入常包含指代和省略。`QueryRewriter` 基于 LLM 结合最近 N 轮历史（默认 6 轮）进行：
- **指代消解**：将"它"、"这个"等替换为具体实体。
- **省略补全**：补全省略的主语、宾语或时间范围。
- **口语化改写**：将口语化表达转为适合检索的正式 query。

输出结构化 JSON：
```json
{
  "rewritten_query": "完整改写后的查询",
  "need_rewrite": true,
  "reason": "指代消解：'它' → '上周的订单'"
}
```

#### 2.2.3 LLM 重排（LLM Reranker）
工具发现后，候选工具可能较多。`LLMReranker` 对每个候选工具让 LLM 按与 query 的相关性打出 0-10 分，并给出理由，最终按分数降序排列。相比传统向量相似度，LLM 能更好地理解工具描述中的功能语义，实现精准匹配。

### 2.3 调用链路
```
用户 query → QueryRewriter 改写 → MCPToolRouter 发现工具 → LLMReranker 重排 → 调用最优工具 → 返回结果
```

### 2.4 API
```http
GET /mcp/tools
POST /mcp/tools/call
```

---

## 3. 多轮对话记忆管理（三级记忆 + 自动压缩）

### 3.1 说明
NexusMind 原有记忆模块已实现三级记忆（语义记忆、情景记忆、程序性记忆）和自动压缩（`auto_store_memory_from_psql` 后台任务）。本次升级**完全保留原有实现**，仅通过 Monitor 模块为其增加闭环反馈能力。

### 3.2 闭环增强
Monitor 记录记忆检索的延迟和成功率，当混合检索失败时自动降级为稠密向量检索，保障可用性。

---

## 4. 多 Agent 路由编排（意图路由 + 性能路由 + 降级）

### 4.1 实现文件
- `backend/orchestrator.py`

### 4.2 核心设计
基于 **LangGraph StateGraph** 构建多 Agent 编排图：

```
[Supervisor] → 意图识别 + 路由决策
    ↓
[RAG Agent]    → 知识库检索（混合检索 + 重排序）
[Memory Agent] → 记忆检索（三级记忆）
[MCP Agent]    → 工具调用（查询改写 + LLM 重排）
[Direct Agent] → 兜底直接回答
    ↓
[Synthesize]   → 结果融合与答案生成
    ↓
[Reflection]   → LLM 质量审核（相关性 / 准确性 / 完整性 / 安全性）
    ↓
END / Retry / Fail
```

#### 4.2.1 路由策略
- **意图路由**：Supervisor 根据 `intent_recognition.py` 的识别结果，将 query 分发到最合适的 Agent。
- **性能路由**：Monitor 实时统计各 Agent 的 p99 延迟和成功率，Supervisor 优先选择性能最优的 Agent。
- **降级策略**：当某 Agent 失败或超时时，按预设降级链路执行：
  - 混合检索失败 → 稠密检索
  - 重排序失败 → 原始结果
  - RAG 失败 → 纯 LLM 直接回答
  - Agent 整体失败 → Direct Agent 兜底

#### 4.2.2 Reflection 节点
生成答案后，Reflection Agent 用 LLM 从四个维度审核答案质量，评分低于阈值则触发重试（最多 `ORCHESTRATOR_MAX_RETRIES` 次）。

### 4.3 API
```http
POST /chat/orchestrate
POST /chat/orchestrate/stream   # SSE 流式输出
```

---

## 5. Monitor 闭环监控（在线表现驱动路由权重动态调整）

### 5.1 实现文件
- `backend/monitor.py`

### 5.2 核心设计
Monitor 模块实现**在线指标采集 → 滑动窗口聚合 → 权重动态调整**的闭环：

#### 5.2.1 指标采集
每次 Agent 调用记录 `MetricRecord`：
- `agent_type`：RAG / Memory / MCP / Direct
- `latency_ms`：调用耗时
- `success`：是否成功
- `token_count`：Token 消耗
- `timestamp`：时间戳

#### 5.2.2 滑动窗口聚合
`SlidingWindowAggregator` 维护双端队列（默认 1000 条 / 300 秒），实时计算：
- p50 / p90 / p99 延迟
- 成功率
- 平均 Token 消耗
- QPS

#### 5.2.3 权重动态调整
`WeightAdjuster` 根据聚合指标按规则调整各 Agent 的调度权重：
- 成功率 < 95%：权重 × 0.5
- p99 > 3000ms：权重再 × 0.5
- p99 > 5000ms：权重降至 0.1

Supervisor 路由时按调整后的权重进行加权选择，实现"表现好的 Agent 多接单，表现差的少接单"的自愈效果。

### 5.3 API
```http
GET /monitor/metrics
POST /monitor/feedback
```

---

## 6. 端到端评测（LLM-as-Judge 四维度打分）

### 6.1 实现文件
- `backend/evaluation.py`

### 6.2 核心设计
NexusMind 采用 **LLM-as-Judge** 模式，让大模型充当评测官，对每条对话结果进行结构化评分。

#### 6.2.1 四维度评分
| 维度 | 说明 | 分值 |
|------|------|------|
| Relevance | 答案与用户问题的相关程度 | 1-10 |
| Accuracy | 事实正确性、有无幻觉 | 1-10 |
| Completeness | 是否覆盖用户问题的全部要点 | 1-10 |
| Safety | 是否存在有害、偏见、泄露隐私等内容 | 1-10 |

四个维度取平均得到 `overall_score`，≥ `EVALUATION_PASS_THRESHOLD`（默认 6.0）视为及格。

#### 6.2.2 Bad Case 归档
`BadCaseRepository` 自动将低分样本（不及格）归档到内存仓库（最大 1000 条），用于：
- 离线分析模型薄弱点
- 作为 few-shot 示例优化 prompt
- 驱动 Monitor 调整权重

### 6.3 API
```http
POST /evaluation/evaluate          # 单条评测
POST /evaluation/batch             # 批量评测
GET  /evaluation/bad-cases         # 查看 Bad Case 库
GET  /evaluation/summary           # 评测汇总统计
```

---

## 新增环境变量速查

在 `.env` 中新增以下配置：

```bash
# 多 Agent 编排
ORCHESTRATOR_MAX_RETRIES=2

# MCP 协议
MCP_SERVER_PORT=8765
MCP_CALL_TIMEOUT=30

# Monitor 监控
MONITOR_WINDOW_SIZE=1000
MONITOR_WINDOW_SECONDS=300
MONITOR_SUCCESS_RATE_THRESHOLD=0.95
MONITOR_P99_THRESHOLD_MS=3000
MONITOR_P99_CRITICAL_MS=5000

# 评测
EVALUATION_PASS_THRESHOLD=6.0
EVALUATION_MAX_BAD_CASES=1000
EVALUATION_MODEL=qwen-turbo

# 查询改写
QUERY_REWRITE_MAX_HISTORY=6
```

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                            │
│  /chat/orchestrate  /intent/recognize  /mcp/tools  ...      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator (LangGraph)                  │
│  Supervisor → RAG/Memory/MCP/Direct → Synthesize → Reflection│
└─────────────────────────────────────────────────────────────┘
       │           │            │           │
       ▼           ▼            ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Intent  │ │   RAG    │ │  Memory  │ │   MCP    │
│Recognition│ │(Hybrid+  │ │(3-Level +│ │(Query   │
│(3-Way +  │ │ Rerank)  │ │Compress) │ │Rewrite+ │
│ Weighted │ │          │ │          │ │LLM      │
│  Vote)   │ │          │ │          │ │Rerank)  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Monitor & Evaluation                    │
│  SlidingWindow → WeightAdjuster  |  LLM-as-Judge + BadCases │
└─────────────────────────────────────────────────────────────┘
```

---

---

## 7. 权限控制框架（ACL）

### 7.1 实现文件
- `backend/access_control.py`

### 7.2 核心设计
NexusMind 内置企业级权限控制，支持四级角色（admin/editor/viewer/guest）和细粒度权限项：

- **角色权限映射**：admin 拥有全部权限，editor 可读写但不能删库，viewer 只读，guest 仅能对话
- **危险操作拦截**：删除知识库、文件写入/删除等操作被标记为危险，需二次确认
- **审计日志**：所有敏感操作记录用户ID、动作、资源、结果和时间戳，支持查询和归档
- **API 级装饰器**：`@require_permission(Permission.XXX)` 直接在路由层拦截无权限请求

### 7.3 API
```http
GET  /acl/permissions          # 获取当前角色权限
GET  /acl/audit-logs           # 查询审计日志
GET  /acl/dangerous-logs       # 查询危险操作日志（admin）
POST /acl/confirm-dangerous    # 确认危险操作
```

---

## 8. 自进化机制（Self-Evolution）

### 8.1 实现文件
- `backend/self_evolution.py`

### 8.2 核心设计
系统能够从用户反馈中自动学习并进化：

1. **反馈收集**：低分评测、用户点踩、Bad Case 自动进入进化队列
2. **规则抽取**：LLM 分析反馈，抽取通用改进规则（触发条件 + 改进 prompt）
3. **去重/合并**：与现有 Skills 对比相似度，避免重复规则
4. **决策沉淀**：支持 `accept`（注册为新 Skill）、`merge`（合并到现有 Skill）、`discard`（丢弃）
5. **自动决策**：高置信度规则（≥0.8）可自动接受，无需人工审核

### 8.3 API
```http
GET  /evolution/rules            # 查看进化规则
POST /evolution/review           # 规则决策（accept/merge/discard）
POST /evolution/auto-review      # 自动接受高置信度规则
GET  /evolution/stats            # 进化统计
```

---

## 9. Skills 分层体系

### 9.1 实现文件
- `backend/skills_manager.py`

### 9.2 核心设计
Skills 是可复用的工作流模板，与 Memory 共同构成分层长期上下文：

- **Skill 定义**：名称 + 描述 + Prompt 模板 + 示例 + 分类标签
- **动态检索**：根据用户 query 关键词匹配，自动检索最相关的 Skills
- **Prompt 注入**：编排时自动将相关 Skills 注入 Synthesize 节点的上下文
- **版本快照**：支持 Skill 版本快照和回滚
- **开箱即用**：预设 5 个默认 Skills（RAG 检索、记忆召回、任务规划、代码审查、数据分析）

### 9.3 API
```http
GET  /skills                     # 列出 Skills
POST /skills                     # 创建 Skill
GET  /skills/stats               # Skills 统计
```

---

## 10. MCP 外部 Server 接入

### 10.1 实现文件
- `backend/mcp_protocol.py`（扩展）

### 10.2 核心设计
外部工具接入沿用 MCP 概念模型，新增外部工具源接入能力：

- **连接模式**：支持 stdio（子进程）和 SSE（HTTP 流）两种方式连接外部工具源
- **连接池管理**：统一管理多个外部工具源，统一发现工具和调用
- **环境变量配置**：`MCP_SERVERS=name:command:args;name2:url:https://...` 一行配置多个外部工具源
- **权限过滤**：非 admin 用户自动过滤危险工具

### 10.3 API
```http
GET  /mcp/servers                # 列出外部 Server 和工具
POST /mcp/servers/connect        # 连接所有外部 Server（admin）
```

---

## 11. 多 Agent 增强（explore/plan/general + Plan Mode）

### 11.1 实现文件
- `backend/orchestrator.py`（扩展）

### 11.2 核心设计
在原有 RAG/Memory/MCP/Direct 四 Agent 基础上，新增三种子 Agent：

- **ExploreAgent（探索型）**：广泛检索多个知识源（放宽 top_k），适用于开放式/调研型问题
- **PlanAgent（规划型）**：将复杂任务拆解为可执行子任务列表，输出带步骤标注的执行计划
- **GeneralAgent（通用型）**：直接回答兜底，集成对话历史做个性化回复

**Plan Mode（显式规划）**：
- 触发条件：query 包含"帮我/步骤/流程/计划/怎么做"等关键词，或意图识别为 complex_task
- 执行链路：Supervisor → PlanAgent → Synthesize（按步骤组织回答）
- Skills 和 Plan 计划同时注入 Synthesize 的上下文

**路由策略增强**：
```
clarify → END
plan_keywords / complex_task → PlanAgent
explore_research → ExploreAgent
knowledge_rag → RAGAgent
memory_query → MemoryAgent
ticket_handler → MCP
greeting/chitchat → GeneralAgent
```

---

## 12. 会话恢复与上下文压缩

### 12.1 实现文件
- `backend/session_recovery.py`

### 12.2 核心设计
- **自动保存**：每次对话完成后自动保存到 Redis（压缩存储，TTL 24h）
- **会话恢复**：传入 session_id 可恢复完整对话历史和状态
- **上下文压缩**：Token 超过阈值时，自动将旧消息摘要化，保留最近对话
- **会话管理**：支持列出、恢复、删除会话

### 12.3 API
```http
GET    /session/list             # 列出会话
POST   /session/resume           # 恢复会话
DELETE /session/{session_id}     # 删除会话
```

---

## 更新后的环境变量速查

```bash
# 多 Agent 编排
ORCHESTRATOR_MAX_RETRIES=2

# MCP 协议
MCP_SERVER_PORT=8765
MCP_CALL_TIMEOUT=30

# Monitor 监控
MONITOR_WINDOW_SIZE=1000
MONITOR_WINDOW_SECONDS=300
MONITOR_SUCCESS_RATE_THRESHOLD=0.95
MONITOR_P99_THRESHOLD_MS=3000
MONITOR_P99_CRITICAL_MS=5000

# 评测
EVALUATION_PASS_THRESHOLD=6.0
EVALUATION_MAX_BAD_CASES=1000
EVALUATION_MODEL=qwen-turbo

# 查询改写
QUERY_REWRITE_MAX_HISTORY=6

# 权限控制
ACL_MAX_AUDIT_LOGS=5000

# Skills
SKILLS_DIR=./skills

# 自进化
EVOLUTION_DIR=./evolution

# 会话恢复
SESSION_MAX_TOKENS=8000
SESSION_COMPRESS_THRESHOLD=6000
SESSION_TTL_SECONDS=86400

# MCP 外部 Server（示例）
MCP_SERVERS=
```

---

## 更新后的架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                            │
│  /chat/orchestrate  /intent/recognize  /skills  /session... │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│         Orchestrator (LangGraph) — Plan Mode 增强            │
│  Supervisor → Plan/Explore/RAG/Memory/MCP/General            │
│                    → Synthesize → Reflection → END/Retry     │
└─────────────────────────────────────────────────────────────┘
       │         │        │           │           │
       ▼         ▼        ▼           ▼           ▼
┌──────────┐ ┌────────┐ ┌────┐ ┌──────────┐ ┌──────────┐
│  Plan    │ │Explore │ │RAG │ │  Memory  │ │   MCP    │
│ (Task    │ │(Broad  │ │(Hybrid│ │(3-Level +│ │(Internal│
│  Break   │ │ Search)│ │+Rerank)│ │Compress) │ │+External│
│  down)   │ │        │ │      │ │          │ │ Servers) │
└──────────┘ └────────┘ └────┘ └──────────┘ └──────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Skills + Memory  分层长期上下文                  │
│  SkillsManager (可复用工作流)  |  MemoryManager (三级记忆)   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  ACL 权限控制  |  Self-Evolution 自进化  |  Session 会话恢复  │
│  Monitor 监控  |  Evaluation 评测        |  QueryRewriter    │
└─────────────────────────────────────────────────────────────┘
```

---

*文档版本：v2.0 | 最后更新：2026-08-23*
