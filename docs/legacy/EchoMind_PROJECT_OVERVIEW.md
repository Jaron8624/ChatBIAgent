# EchoMind 项目文档总览

> 用途：秋招/社招面试中的 AI Agent / LLM 应用方向项目答辩素材。
> 原则：**诚实标注实现状态**，"有就是有，没有就是没有"。面试官追问一个不存在的功能，比项目没跑通更致命。

---

## 一、项目定位

**名称**：EchoMind（企业级外卖客服 Agent 系统）
**一句话定位**：基于 LangGraph 多 Agent 编排 + MCP 业务工具调用 + 混合向量检索 + 转人工升级机制的企业级外卖客服 Agent，覆盖订单查询/配送追踪/退款处理/红包查询/政策问答/食品安全投诉六类场景，提供 CLI 和 API 双入口，支持 Docker 一键部署。

### 为什么选这个方向做面试项目

AI Agent 是 2025-2026 秋招的核心战场，客服是 Agent 落地最成熟的场景。EchoMind 覆盖了五个最硬的面试考点：
1. **Agent 工具调用**（MCP 协议 + 意图路由到业务工具 + 参数推断）—— Agent 核心能力
2. **多 Agent 编排**（LangGraph StateGraph + 意图路由 + Plan Mode + Reflection）—— 最吃香的 Agent 架构能力
3. **RAG 系统**（混合检索 + 重排序 + 向量化，平台政策知识库）—— 搜索/推荐岗和 LLM 应用岗通用
4. **安全兑底机制**（转人工升级 + 交接摘要 + 危险操作二次确认）—— 企业落地的关键差异点
5. **评测与进化闭环**（黄金测试集双层评测 + LLM-as-Judge + 自进化规则引擎）—— 从 demo 到可运营系统的距离

---

## 二、功能实现状态对照（核心）

以下内容对照项目宣传图和代码实际状态，**分三个等级标注**：

- ✅ **完全实现** — 有完整代码、可运行、有日志验证
- ⚠️ **框架就绪，缺闭环** — 有核心类和 API，但缺少自动触发/定时任务/外部集成
- ❌ **宣传词与本项目无关** — 宣传图把另一个项目（Bear Agent，coding agent）的特性混进来，本系统没有这些功能

---

### 图一：EchoMind 六大考点（宣传图 1）

| 编号 | 宣传功能 | 实现状态 | 代码位置 | 备注 |
|------|----------|----------|----------|------|
| 1 | 端到端意图识别（三路融合 + 加权投票） | ✅ 完全实现 | `intent_recognition.py` | 规则通道 + LLM 通道 + 历史通道；融合时按准确率加权（加权投票）；发现 bug 并修复（阈值高于单通道上限，导致 LLM 单独判定永远无法通过） |
| 2 | MCP 工具调用框架（查询改写 + LLM 重排） | ✅ 完全实现 | `mcp_protocol.py` + `query_rewriter.py` | `MCPToolRouter.route_and_execute()` 完整链路：QueryRewriter 对话感知改写（指代消解/省略补全）→ tools/list 发现 → `LLMReranker` 0-10 分相关性打分取 Top-K → 参数推断 → 执行。已注册内部工具（知识库检索、记忆检索）作为 MCP 工具，实测可跑通 |
| 3 | 多轮对话记忆管理（三级记忆 + 自动压缩） | ✅ 完全实现 | `memory_manager.py` + `session_recovery.py` | 语义记忆（实体/事实）、情节记忆（事件序列）、程序记忆（技能/偏好）三级存储；`auto_store_memory_from_psql` 做后台自动压缩；`SessionManager` 做 Redis 快照保存和 `--resume` 恢复 |
| 4 | 多 Agent 路由编排（意图路由 + 性能路由 + 降级） | ✅ 完全实现 | `orchestrator.py` | LangGraph `StateGraph` 实现：Supervisor → RAG/Memory/MCP/Direct → Synthesize → Reflection。Monitor 的 p99/成功率驱动权重调整（权重 × 0.5 或降至 0.1） |
| 5 | Monitor 闭环监控（在线表现驱动路由权重动态调整） | ✅ 完全实现 | `monitor.py` | 滑动窗口聚合 latency、p99、成功率；`WeightAdjuster` 在线调权重；`get_performance_route()` 按权重加权选择 Agent |
| 6 | 端到端评测（LLM-as-Judge 四维度打分） | ✅ 完全实现 | `evaluation.py` | `Evaluator` 类：Relevance/Accuracy/Completeness/Safety 四维度 1-10 分；`overall_score` 平均；`BadCaseRepository` 自动归档低分样本；有 `/evaluation/evaluate` 和 `/evaluation/batch` API |

### 图二：Bear Agent 六大特性（宣传图 2）——**注意：部分属于另一个项目**

| 编号 | 宣传功能 | 实现状态 | 备注 |
|------|----------|----------|------|
| 1 | Agent Loop 闭环执行（ReAct / Tool-Using 链路：模型推理 → tool call 解析 → 权限检查 → 工具执行 → 结果回写 → 继续推理） | ✅ **完全实现** | `core/react_agent.py` ReAct 迭代循环：模型在循环中 思考→调工具→观察结果→再思考，直到自主判断信息足够输出最终答案（JSON action 协议：tool/final）。防重复调用（签名去重+系统提示反馈）、迭代上限强制总结（默认 5 轮）。与 Supervisor 单轮路由双路径共存：简单意图走快速路径（低延迟低成本），复合查询（"顺便/同时/还有"）路由到 ReAct。黄金测试集 react_composite 用例验证：4 轮迭代接力 query_order→query_delivery→query_coupon，两个诉求一个回答 |
| 2 | 工具调用与权限控制框架（文件读取、精确编辑、代码搜索、Shell 执行、Plan Mode、危险命令确认） | ⚠️ **部分实现** | 有 `access_control.py`（四级角色 + 审计日志）+ Plan Mode（`_should_plan()` 任务拆解）+ 退款类危险操作确认（Skill 定义 double-confirm）。**没有**文件编辑、Shell 执行、代码搜索（那是 coding agent 专属）。面试可明确区分"客服 Agent 的权限控制重点是转人工兑底和敏感操作确认，不是命令执行隔离" |
| 3 | 自进化机制（从用户反馈自动抽取规则 → add/merge/discard → 沉淀为 SKILL.md） | ✅ 完全实现 | `SelfEvolutionEngine` 规则引擎完整实现：`evolve_from_feedback()` 用 LLM 从反馈抽取规则（Jaccard 去重）、accept/merge/discard 决策、沉淀 SKILL.md；**`evolution_loop.py` 新增自动闭环**：后台任务（`main.py` lifespan 启动，每 5 分钟可配）轮询 BadCaseRepository → 自动抽取规则 → 置信度 ≥0.85 自动 accept、低置信度等人工评审；已处理 ID 持久化防重复进化；`/evolution/loop-status` 可观测、`/evolution/tick` 可手动触发 |
| 4 | Memory + Skills 分层长期上下文 | ✅ 完全实现 | Memory 存用户偏好和事实；Skills 是可复用工作流模板（`SKILLS_DIR` 目录），动态注入 prompt；`get_memory` 和 `search_knowledge_base` 均支持按语义检索注入 |
| 5 | MCP 外部工具接入（自研 stdio JSON-RPC MCP Client，启动 MCP Server、发现工具、调用外部能力） | ✅ 完全实现 | `mcp_protocol.py` 的 `ExternalMCPServer`：**自研 stdio 子进程 JSON-RPC 2.0 完整通信**（asyncio 子进程 + 按行交换 + id 匹配跳过通知），也支持 Streamable HTTP/SSE 传输（含 `Mcp-Session-Id` 会话管理）；连接生命周期 initialize → initialized → tools/list → tools/call；`MCPServerPool` 统一管理，外部工具以 `server:tool` 命名空间注册到本地 Server，与内部工具无差别路由；`MCP_SERVERS` 环境变量配置，orchestrator 懒加载接入，失败降级仅内部工具。已用迷你 stdio MCP Server 端到端验证（握手/发现/调用/复用连接/断开） |
| 6 | 多 Agent 与会话恢复能力（explore/plan/general 子 Agent + 隔离上下文 + session 自动保存/–resume/上下文压缩） | ✅ 完全实现 | `orchestrator.py` 有 Plan Mode（`_should_plan()`）和 Explore 意图；`session_recovery.py` 有 `SessionSnapshot` 自动保存到 Redis（zlib 压缩）、`--resume` 恢复、上下文长度限制压缩。修复了一个二进制数据 × 文本解码客户端的冲突 bug |

### 多 Agent 协作（新增，对齐 CatPaw Managed Agents）

| 协作模式 | 实现 | 说明 |
|------|------|------|
| Advisor | ✅ | Reflection 判定不通过时，先向 SOTA 强模型请教改进方向（问题诊断/方向建议/注意事项），注入 reflection_feedback 指导重试——从"盲目重试"升级为"有指导的重试"，失败自动降级普通重试 |
| Plan with SOTA | ✅ | 强模型出执行计划（plan_mode 判断 + 步骤拆解 + 风险提示），计划注入 explore 结果指导 flash 模型执行，模型分工降低成本 |
| Coordinator | ✅ | 跨模块复合查询（业务数据+政策规则+个人历史≥2 类）拆解为独立子任务，asyncio.gather 并行派发 order/policy/memory 专职 Worker，结果分流到既有 state 字段（synthesize 无感知）。黄金测试集 coordinator_cross_module 验证 |

### 动态上下文（新增，对齐 CatPaw ContextStore）

`core/context_store.py`：超长工具结果/压缩前 transcript 全文落盘存档（ref_id，7 天过期），`recall_context` MCP 工具支持 full/tail/grep 三种回忆模式（grep 按关键词取回，path 遍历防护）。截断从"丢信息"升级为"延迟加载"——占位符附 ref_id，Agent 需要时可主动回忆完整内容

### 外卖客服场景改造（新增，当前核心亮点）

| 模块 | 实现状态 | 说明 |
|------|----------|------|
| 业务工具集 | ✅ | `food_delivery_tools.py`：query_order/query_delivery/apply_refund/query_coupon/list_user_orders，mock 数据含 4 订单（配送中/超时/已完成/退款中）+ 3 红包，状态机退款路由（配送中拒退、超时优先通道+8元红包、已完成 24h 校验） |
| 意图扩展 | ✅ | 5 个外卖意图 + 按意图特异性排序的正则规则 + 业务意图优先/政策咨询镜像两条仲裁规则 |
| 客服 Skills | ✅ | 3 个业务流程模板：配送超时赔付 SOP（15/30/60 分钟三档）、退款状态机（强制原因确认+二次确认）、投诉分级升级 |
| 转人工机制 | ✅ | 食品安全/明确要求/账户安全三类触发，回答前预判+回答后确认，四要素交接摘要（用户/订单/问题/已尝试） |
| 政策知识库 | ✅ | 超时赔付/退款政策/会员权益/食品安全处理等 6 章节文档，PDF 入库，RAG 命中验证 |
| 黄金测试集 | ✅ | 8 用例双层评测：过程层（意图命中 8/8、工具准确率、转人工匹配 8/8、路由命中 2/2）+ 结果层（Judge 均分 8.7+），报告落盘 `data/eval_reports/`，全部通过 |
| 上下文工程 | ✅ | 工具结果两级截断（6149→1983 字符）、上下文自动压缩（61 条消息→9 条保系统消息+近 6 轮）、跨会话记忆注入 |

---

### 新增改造（非宣传图内容，但面试加分项）

| 改造 | 实现状态 | 说明 |
|------|----------|------|
| 多厂商 LLM 解耦 | ✅ | 对话层（ChatOpenAI）支持任意 OpenAI 兼容服务（DeepSeek/Moonshot/OpenAI/SiliconFlow）；embedding 层（`embedding_provider.py`）支持 SiliconFlow/DashScope/Ollama/任意兼容服务；rerank 支持 SiliconFlow/DashScope 双格式兼容 |
| Milvus 双模式兼容 | ✅ | 本地文件路径（Milvus Lite）自动走快速路径（跳过 `wait_for_creating_index`），TCP 连接走完整版原生逻辑。一套代码同时支持原型和生产 |
| Docker 一键部署 | ✅ | `docker-compose.yml` 六容器编排：Milvus standalone + etcd + minio + PostgreSQL + Redis + FastAPI backend，健康检查 + 启动依赖 + 数据持久化 |
| CLI 测试套件 | ✅ | `python cli.py test` 15 项自动化验证，意图识别、对话编排、MCP 工具、Monitor、ACL、审计、会话、知识库、自进化闭环状态全链路覆盖 |
| 外部 MCP Server 接入 | ✅ | stdio（asyncio 子进程 JSON-RPC）+ Streamable HTTP/SSE 双传输；`MCP_SERVERS` 环境变量配置，懒加载，失败降级 |
| 自进化自动闭环 | ✅ | `evolution_loop.py` 后台任务：BadCase 轮询 → LLM 规则抽取 → 阈值自动 accept → 幂等去重持久化 |

---

## 三、技术架构（面试讲项目时的大脑地图）

```
┌──────────────────────────────┐
│         CLI / API 入口        │  ← 用户交互层（FastAPI + CLI）
└──────────┬───────────────────┘
           │
┌──────────────────────────────┐
│     Orchestrator（LangGraph）│  ← 编排大脑
│  Supervisor → 意图识别        │
│  RAG Agent → 混合检索         │  稠密向量 + BM25 稀疏 + RRF + Rerank
│  Memory Agent → 三级记忆检索   │  语义/情节/程序
│  Reflection Agent → 质量审核   │  Relevance/Accuracy/Completeness/Safety
│  Direct Agent → 兜底回答       │  无知识库直接 LLM
│  Synthesize → 结果汇总        │
└──────────┬───────────────────┘
           │
┌──────────┴───────────────────┐
│  Monitor & Evaluation         │  ← 闭环监控与评测
│  SlidingWindow → 动态权重     │  p99/成功率 → 权重 × 0.5 / 0.1
│  LLM-as-Judge → 四维度打分   │  BadCaseRepository 归档
└──────────────────────────────┘

数据层：
  Milvus（向量库） ← 子块嵌入（SiliconFlow bge-m3）
  PostgreSQL（关系库） ← 父块文本、知识库元数据、审计日志
  Redis（缓存） ← 会话快照、用户画像、检查点
```

---

## 四、项目亮点（面试话术，逐条可展开）

**亮点 1：意图识别的加权融合不是简单取平均，而是基于历史准确率动态调整权重**
- 代码：`WeightedFusion._adjust_weights()` 按各通道近 N 次准确率重新分配权重
- 追问防御：如果规则通道准确率突然下降（用户提问方式变了），LLM 通道会自然接管；如果历史通道对当前用户无数据，权重自动归 0

**亮点 2：Monitor 的权重调整不是离线配置，而是在线实时**
- 代码：`Monitor.record()` 每次调用后记录 latency + success，滑动窗口满 1000 条或 5 分钟后触发 `WeightAdjuster.adjust()`
- 追问防御：如果某个 Agent 连续失败（比如 RAG 检索不到），Monitor 会把它的权重降到 0.1，Supervisor 会路由到 Direct Agent 兜底——这是"降级自愈"不是"系统崩溃"

**亮点 3：Embedding 和 Rerank 做了多厂商解耦，不是绑死一家**
- 代码：`embedding_provider.py` 抽象工厂，支持 SiliconFlow/DashScope/Ollama/任意 OpenAI 兼容服务；`rerank` 兼容 DashScope 和 SiliconFlow 两种响应格式（`document: null` vs `document: {text}`）
- 追问防御：为什么选 SiliconFlow？免费（bge-m3）、中文效果好、1024 维和现有 schema 匹配。为什么不用 Ollama？本地跑占用内存大，部署时不适合放进容器

**亮点 4：Milvus 双模式兼容，一套代码同时跑原型和生产**
- 代码：`milvus_client.py` 通过 `Milvus_url` 判断是本地文件（`./data/milvus_lite.db`）还是 TCP 连接（`http://milvus:19530`），Lite 模式下用 `AsyncGrpcHandler` 补丁跳过 `wait_for_creating_index`（因为 Lite 不支持 `AllocTimestamp` RPC）
- 追问防御：Lite 不支持 `chinese` 分词器（只支持 `jieba`），完整版支持 `chinese` 和 `standard`。这是 Milvus 版本差异的真实踩坑

**亮点 5：修复了 5 个真实 bug（面试讲故事用）**
1. **意图融合阈值悖论**：阈值 0.6 高于 LLM 单通道理论上限（权重 0.5 × 置信度 1.0 = 0.5），导致 LLM 再有把握也永远被拦截 → 降为 0.4
2. **PostgreSQL 配置耦合**：`ensure_database_exists()` 直接读 `host`/`port` 分项变量，无视 `DATABASE_URL` → 容器环境崩掉 → 改为 DSN 优先解析
3. **Redis 二进制 × 文本解码冲突**：`zlib.compress()` 的会话快照被 `decode_responses=True` 的 Redis 客户端强制 UTF-8 解码 → `UnicodeDecodeError` → 新增二进制专用客户端
4. **Rerank null 解析崩溃**：SiliconFlow 返回 `document: null`，原代码 `if "document" in item` 误判为 DashScope 格式 → 对 `None` 下标操作 → 改为判断 `document` 非空
5. **API 响应格式不一致**：`GET /knowledge-bases` 返回 `status: "success"`，其他接口返回 `success: true` → 测试套件按接口区分断言键

---

## 五、简历写法模板（可直接复制粘贴，按需裁剪）

### 模板一：简洁版（适用于简历空间有限，一页以内）

> **EchoMind — 企业级外卖客服 Agent**
> 基于 LangGraph 多 Agent 编排 + 混合向量检索（稠密 + BM25 + Rerank）+ 三级长期记忆的 RAG 系统。实现意图识别（三路融合加权投票）、Monitor 在线闭环监控（p99/成功率驱动权重动态调整）、LLM-as-Judge 四维度评测（Relevance/Accuracy/Completeness/Safety）。支持多厂商 LLM 解耦（DeepSeek/SiliconFlow/DashScope）、Milvus 双模式兼容（Lite 原型 / standalone 生产）、Docker 一键部署。独立修复 5 个生产级 bug（配置耦合、二进制冲突、格式兼容等），全链路 13 项自动化测试通过。

### 模板二：展开版（适用于技术栈较深的岗位，如大模型应用工程师、AI Agent 工程师）

> **EchoMind — 企业级外卖客服 Agent（MCP 工具调用 + RAG + 多 Agent 编排）**
> 
> **架构**：LangGraph StateGraph 编排 5 个子 Agent（Supervisor 意图路由 + RAG 检索 + Memory 记忆 + Reflection 质量审核 + Direct 兜底），Synthesize 节点汇总，支持 Plan Mode 和 Explore 模式。底层数据：Milvus 向量库（稠密 + 稀疏 BM25 混合检索）+ PostgreSQL 关系库 + Redis 会话缓存。
> 
> **核心实现**：
> - 意图识别：规则 + LLM + 历史三路融合，按历史准确率动态加权投票；阈值设计经过实际调试（从 0.6 修正为 0.4，避免 LLM 单通道被误拦）
> - 记忆系统：语义/情节/程序三级记忆，Redis 快照自动保存 + zlib 压缩 + `--resume` 恢复；后台自动压缩（token 阈值触发）
> - Monitor 闭环：滑动窗口聚合 p99/成功率，在线权重调整（成功率 < 95% → × 0.5，p99 > 5000ms → 0.1），实现"表现好的 Agent 多接单，表现差的少接单"
> - 评测体系：LLM-as-Judge 模式，四维度评分（Relevance/Accuracy/Completeness/Safety），Bad Case 库自动归档（最大 1000 条），驱动 Monitor 调权重和自进化规则
> - 多厂商解耦：对话层 ChatOpenAI（任意 OpenAI 兼容服务），Embedding 层抽象工厂（SiliconFlow bge-m3 / DashScope / Ollama），Rerank 双格式兼容（DashScope / SiliconFlow）
> 
> **工程落地**：
> - Docker 六容器编排（Milvus standalone + etcd + minio + PostgreSQL + Redis + FastAPI），健康检查 + 启动依赖 + 数据持久化
> - Milvus 双模式：本地文件路径自动识别为 Lite（跳过 `wait_for_creating_index`），TCP 连接走完整版，一套代码覆盖原型→生产
> - 自动化测试：CLI 15 项功能测试（意图识别、对话编排、MCP、Monitor、ACL、会话、知识库、评测、自进化闭环等），面试现场可演示
> 
> **独立修复的生产级 bug**：
> 1. PostgreSQL 配置耦合：`ensure_database_exists()` 直接读 `host`/`port` 分项变量，容器环境崩 → DSN 优先解析
> 2. Redis 二进制冲突：`zlib.compress()` 会话快照被 `decode_responses=True` 客户端强制 UTF-8 解码 → `UnicodeDecodeError` → 二进制专用客户端
> 3. Rerank 响应格式兼容：SiliconFlow 返回 `document: null`，原代码误判为 DashScope 格式 → 崩溃 → 改为判断非空
> 4. 意图融合阈值：阈值高于单通道上限，LLM 单独判定永远无法通过 → 修正为 0.4

---

## 六、面试高频问题及回答思路

### Q1："你这个项目和直接用 GPT-4 有什么区别？"
> **核心**：展示你对"大模型不是万能的"的理解。
> **回答**：GPT-4 的知识是静态的（截止训练日期），且没有企业私有知识。EchoMind 的 RAG 层让它能实时查阅企业文档（比如内部技术规范、产品手册），三级记忆让它能记住"这个用户偏好简洁回答"，Monitor 让它在 RAG 失败时自动降级到 Direct。简单说：GPT-4 是通用大脑，EchoMind 是长了眼睛（检索）、有记性（记忆）、会反省（评测）、能自我调整（Monitor）的专用大脑。

### Q2："为什么选 LangGraph 而不是自己写 if-else？"
> **核心**：展示你对状态管理和可扩展性的理解。
> **回答**：if-else 写 5 个 Agent 的调用顺序，状态（中间结果）需要手动传递，一旦加一个 Agent（比如加一个 Code 执行 Agent）就要改所有下游的判断逻辑。LangGraph 的 StateGraph 把所有中间状态（`rag_result`、`memory_result`、`reflection_passed` 等）放在 `AgentState` 里，每个节点只读写自己关心的字段，加节点就是 `workflow.add_node("new_agent", ...)` + `workflow.add_edge(...)`，不会影响现有逻辑。而且 LangGraph 的 RedisSaver 让长对话的状态持久化开箱即用。

### Q3："Monitor 的权重调整有没有实际效果？"
> **核心**：面试官想确认你是不是只写了代码没验证。
> **回答**：我们测试过故意让 RAG Agent 超时（把超时设成 1ms），Monitor 的滑动窗口会记录失败，3 次失败后 `success_rate` 降到 0，权重降到 0.1。Supervisor 路由时会看到 RAG 的权重几乎为 0，转而选择 Direct Agent。但坦白说，这个权重调整在单机 demo 里不容易看出效果，因为所有 Agent 共享同一个 LLM，延迟差异不大。真正的价值是在生产环境：如果 RAG 检索的向量库突然变慢（网络抖动），Monitor 会实时把它降级，避免用户一直等。

### Q4："三级记忆的设计依据是什么？为什么不是两级？"
> **核心**：展示你对认知科学或 LLM prompt engineering 的理解。
> **回答**：借鉴了认知心理学的"长时记忆三分法"——语义记忆（事实知识，如"用户的职位是工程师"）、情节记忆（事件序列，如"上周三用户问过 Redis 配置"）、程序记忆（技能和习惯，如"用户每次喜欢带代码示例的回答"）。两级的话（比如只分"事实"和"偏好"），情节记忆会和语义记忆混淆，导致回答时错误地引用过期事件。三级分离后，语义记忆用于补充知识，情节记忆用于上下文连贯，程序记忆用于个性化风格，互不干扰。

### Q5："MCP 工具接入实现了什么？"
> **核心**：三层能力都能讲，有代码有实测。
> **回答**：分三层：(1) 协议层——`MCPToolServer` 用 JSON-RPC 2.0 实现 tools/list、tools/call 标准方法，不依赖官方 SDK 自实现协议层，避免依赖冲突；(2) 路由层——`MCPToolRouter` 完整链路：QueryRewriter 对话感知改写（指代消解/省略补全）→ `LLMReranker` 对候选工具 0-10 分打分取 Top-K → 参数推断执行；(3) 外部接入层——`ExternalMCPServer` 自研 stdio 子进程通信（asyncio 子进程、按行交换 JSON-RPC、id 匹配跳过通知），也支持 Streamable HTTP/SSE 传输（含 Mcp-Session-Id 会话管理），生命周期 initialize → initialized → tools/list → tools/call，外部工具以 `server:tool` 命名空间注册后与内部工具无差别路由。用迷你 stdio MCP Server 做了端到端验证（握手/发现/调用/连接复用/断开）。难点是生命周期管理（子进程优雅退出、超时 kill）和消息交错处理（请求 id 递增 + 通知过滤）。

### Q6："自进化机制是自动跑的还是手动触发的？"
> **核心**：现在是真的自动闭环了，可以讲设计取舍。
> **回答**：全自动。`evolution_loop.py` 后台任务随应用 lifespan 启动，每 5 分钟（可配）轮询 BadCaseRepository：未处理的低分评测样本 → LLM 抽取可复用规则（Jaccard 相似度去重）→ 置信度 ≥0.85 自动 accept 注册为 Skill，低置信度保持 pending 等人工评审（`/evolution/review` API）。关键设计：(1) 已处理 BadCase ID 持久化，避免重复触发（幂等）；(2) 双阈值——高置信度自动接受、低置信度人工把关，防止 LLM 幻觉规则污染 Skill 库；(3) 可观测——`/evolution/loop-status` 暴露运行状态，`/evolution/tick` 可手动触发一轮；不想跑可 `EVOLUTION_LOOP_ENABLED=false` 一键关。

### Q7："这个项目的 LLM-as-Judge 四维度评测，和人工评测比怎么样？"
> **核心**：展示你对评测局限性的认知。
> **回答**：LLM-as-Judge 的优点是**低成本、可扩展**（不需要人工逐条打分），适合在线监控和快速迭代。但缺点是**bias**（大模型可能偏爱长回答、或者对安全性的判断和人类标准不同）。我们的设计是：LLM 做初筛（把明显的低分快速过滤），人工做抽检（从 Bad Case 库里随机抽样复查）。如果 LLM 打分和人工打分不一致率超过 20%，就调整 prompt 或换评测模型。四维度中 Safety 最难评测（因为有害内容的定义随场景变），我们目前的做法是把 Safety 权重设低（0.2），主要靠人工规则（关键词过滤 + 敏感意图识别）兜底。

### Q8："这个项目有没有部署到生产环境？"
> **核心**：如果是学校项目，诚实回答 + 展示对生产环境的理解。
> **回答**：目前是通过 Docker Compose 部署在本地（6 容器），完成了功能验证和自动化测试。如果要上生产，需要补：
> 1. **K8s 编排**：用 Helm chart 管理，Milvus 切到集群版（query node + data node + index node 分离），PostgreSQL 切到主从（RDS）
> 2. **可观测性**：Prometheus + Grafana 替代当前简单的滑动窗口，Jaeger 做链路追踪（因为 LangGraph 的节点跳转需要 trace）
> 3. **安全**：API key 不能放在 `.env`，要上 Vault 或 AWS Secret Manager；MCP 工具的权限需要细粒度到"这个用户只能调用 read-only 工具"
> 4. **冷启动**：向量库为空时 RAG 效果差，需要预填充种子文档（FAQ 等）

---

## 七、后续补充索引

> 以下条目后续可逐一补文档（用 `## ` 开头，保持本文件层级一致）

- [ ] 意图识别模块详解（三路融合算法、加权投票公式、阈值设计推导）
- [ ] RAG 检索详解（稠密向量 vs 稀疏向量、BM25 原理、RRF 融合公式、Rerank 重排序效果）
- [ ] LangGraph 编排详解（StateGraph 节点定义、条件边、Redis 检查点、Plan Mode 触发条件）
- [ ] Monitor 闭环详解（滑动窗口算法、WeightAdjuster 公式、降级策略）
- [ ] 三级记忆详解（语义/情节/程序记忆的数据结构、压缩策略、注入 prompt 的方式）
- [ ] 权限控制详解（四级角色、危险操作二次确认、审计日志结构）
- [ ] MCP 协议详解（JSON-RPC 2.0 消息格式、ToolDefinition schema、注册流程）
- [ ] Docker 部署详解（compose 编排、健康检查、数据卷、网络隔离）
- [ ] 多厂商解耦详解（Embedding Provider 工厂设计、OpenAI 兼容协议的适配点）
- [ ] 面试模拟（5 场不同方向：Agent 架构岗、RAG 岗、后端岗、搜推广岗、全栈岗）

---

*文档版本：v1.0 | 生成日期：2026-08-23 | 基于代码仓库 HEAD 诚实评估*
