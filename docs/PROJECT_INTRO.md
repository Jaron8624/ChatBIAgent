# EchoMind

一个我自己花了一周多调通的企业级 ChatBI 数据分析 Agent。本意是做一个能直接 Docker 跑起来、每个模块都能拆开讲清楚的面试项目——但最后修 bug 的时间比写代码还长，也算意外收获。

## 这个项目是做什么的

一句话：对话式 BI（ChatBI）。用户问"最近三个月各品类销售额"它生成 SQL 查双库，问"为什么 11 月销售额暴增"它走根因分析规划链拆解假设并并行取证，问"数据质量有没有问题"它跑 7 条对账规则出告警报告。

不是通用聊天机器人，也不只是 RAG 问答——是**语义层 + 意图路由 + MCP 工具调用（含 SQL 守卫自修复）+ 根因分析规划链 + 数据质量对账 + 字段级评测闭环**的组合场景。数据层用 Olist 巴西电商（14747 订单）+ Chinook 数字媒体双 SQLite 数据集（ATTACH 跨库查询），真实数据、内置业务异常（黑五大促/家具差评率/卖家延迟），但协议层（MCP JSON-RPC 2.0）和编排层（LangGraph StateGraph）都是真实的。

覆盖六类分析诉求：数据查询、趋势分析、对比分析、根因分析（专属规划链）、报表生成、Schema 咨询。外加数据质量对账巡检。六大升级方向（语义层/Schema 三级检索/SQL 守卫/根因链/字段级评测/对账 Agent）详见 [CHATBI_UPGRADE.md](CHATBI_UPGRADE.md)。

## 为什么这样设计

面试项目有个矛盾：写得太简单面试官觉得你没深度，写得太复杂自己都跑不起来。我的取舍是：

- **场景真实**：客服是 Agent 落地最成熟的场景之一——意图明确可枚举、工具调用有真实业务价值、转人工是天然的兜底机制，面试时每个设计都能对应到真实业务问题
- **模块解耦**：六层目录（core/tools/rag/memory/evaluation/infra）依赖单向，每个文件只做一件事，`core/intent_recognition.py` 只管意图，不用管路由；`tools/food_delivery_tools.py` 只管业务工具，不用管编排；`evaluation/evaluation.py` 只管评测，不用管业务逻辑
- **真实可运行**：所有功能都经过黄金测试集验证（8 用例，过程层+结果层双层指标，含 ReAct/Coordinator 路由用例），不是"理论上可以这样写"

## 架构

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   FastAPI    │────▶│ LangGraph        │────▶│ LLM (DeepSeek)   │
│   + CLI      │     │ StateGraph       │     │                  │
└──────────────┘     └──────────────────┘     └──────────────────┘
                            │
       ┌────────────┬───────┼────────┬─────────────┐
       ▼            ▼       ▼        ▼             ▼
  ┌──────────┐ ┌─────────┐ ┌──────────────┐ ┌─────────┐ ┌──────────┐
  │ RAG Agent│ │Memory   │ │ MCP 业务工具  │ │ Plan    │ │ Synthesize│
  │ 政策知识库│ │Agent    │ │ 订单/配送/    │ │ Mode    │ │ +转人工判断│
  │ bge-m3   │ │ 三级记忆  │ │ 退款/红包    │ │ 任务拆解  │ │ +Skills注入│
  │ BM25+RRF │ │ Milvus  │ │ (mock数据)   │ └─────────┘ └──────────┘
  │ Rerank   │ └─────────┘ └──────────────┘
  └──────────┘
```

 Supervisor 节点根据意图选择走哪条路：查订单走 MCP 工具节点，问政策走 RAG，打招呼直接走 Direct Agent，"帮我…"开头的复杂任务先 Plan Mode 拆解（但外卖业务意图优先短路，避免查单被误劫持到规划分支——这个 bug 下面详述）。复合查询（"顺便/同时/还有"）走 ReAct 循环：模型在 思考→调工具→观察→再思考 里迭代，直到自主判断信息足够（比如"查下单到哪了顺便看看还剩多少红包"会接力 query_order→query_delivery→query_coupon 三个工具）；跨模块复合查询（业务+政策+历史≥2 类）走 Coordinator：SOTA 拆解子任务后并行派发专职 Worker。RAG 和工具结果合并进 Synthesize，生成回答前先做转人工预判（食品安全/账户安全/用户明确要求），回答后做二次确认并生成四要素交接摘要（用户/订单/问题/已尝试步骤）。Reflection 节点审视回答质量，不达标时先咨询 SOTA Advisor 拿改进方向再重试，而不是盲目重试。超长工具结果和压缩前 transcript 会存档到 ContextStore，Agent 可通过 recall_context 工具按关键词回忆——截断从"丢信息"变成"延迟加载"。

## 几个我觉得值得说的技术点

### 1. 意图识别的阈值 bug

三路融合（规则 + LLM + 历史）看起来简单，但实际跑起来发现 LLM 通道的置信度最高只有 0.5（权重 0.5 × 置信度 1.0），而整体阈值设成了 0.6。这意味着 LLM 单独判定永远过不了——这是个设计上的 bug，我把它降到了 0.4。面试官如果问"你的意图识别阈值怎么定的"，这就是真实调出来的，不是拍脑袋。

### 2. 多厂商 LLM 解耦（embedding 层）

项目最初硬编码了 DashScope，但我只有 DeepSeek 的 key。为了不每个地方都改，写了 `embedding_provider.py` 做工厂模式：`EMBEDDING_PROVIDER=siliconflow` 走 SiliconFlow，`EMBEDDING_PROVIDER=ollama` 走本地 Ollama。不只是抽象，还包含格式兼容——SiliconFlow 的 rerank 返回 `document: null`，而 DashScope 返回 `document.text`，这些细节都在 `documents_process.py` 和 `tools.py` 里做了分支处理。

### 3. 自进化闭环

`self_evolution.py` 本来只是规则引擎（从反馈抽取规则、人工评审），我补了 `evolution_loop.py` 让它真的跑起来：后台任务每 5 分钟查 Bad Case 库，低分样本交给 LLM 分析原因，抽出通用规则。高置信度（≥0.85）自动 accept，低置信度 pending 等人工看。已处理 ID 持久化到文件，重启不重复跑。这个东西面试被问到"自进化怎么保证质量"，我就讲双阈值的设计——宁可少自动、不多幻觉。

### 4. MCP 外部工具接入

工具层沿用 MCP 概念模型：内部工具统一注册，LLM 按工具描述与 query 相关性重排选工具，参数推断后调用，内外部工具无差别路由；外部工具源支持 stdio / HTTP 两种接入模式，环境变量一行配置。工具挂了走降级（仅内部工具），不阻塞主链路。

### 5. 评测驱动数据治理

最意外的收获。跑四维度评测（Relevance/Accuracy/Completeness/Safety）时发现 RAG 上下文里混着乱码（全是 `???`）。排查到底：不是提取代码的 bug，是入库前那份 PDF 生成时就丢了中文。 byte 级验证（hex 全是 0x3f）确认后，重新生成了一份正常 PDF 入库，按文件哈希跨 PostgreSQL 三张表 + Milvus 集合清理脏数据，再端到端复测确认干净。顺手还修了一个 pymilvus 升级导致的删除计数 bug（返回值从 dict 变成了 int）。

### 6. 外卖场景的路由仲裁（调试了一天的一堆 bug）

做外卖客服场景时踩的坑，都是黄金测试集跑出来的，不跑测试根本发现不了：

- **Plan Mode 劫持**：`_route_decision` 里 Plan Mode 检查在外卖意图之前，用户说"帮我查一下订单…"——"帮我"这个关键词触发了规划分支，查单请求被拆解成任务列表但永远不调订单工具。修复：外卖意图优先短路。
- **requires_human 被 LangGraph 静默丢弃**：转人工标志确实生成了（日志可见），但 `AgentState` TypedDict 里没定义这两个字段，LangGraph 直接丢掉未声明的 key，API 永远返回 false。这种"日志对但结果错"的 bug 最坑，不查返回值根本发现不了。
- **规则满置信被投票稀释**：规则通道正则完全命中"食物不新鲜"（confidence=1.0），但权重只有 0.3，加权后 0.3 低于 0.4 阈值；恰好 LLM 通道解析偶发失败返回 unknown，最终意图变成 unknown 触发澄清反问。修复：规则满置信且 LLM 不可用时短路采信；再补了业务意图优先（"超时了怎么赔付"规则判 delivery_track、LLM 判知识库咨询时采信规则，否则拿不到订单工具数据）和政策咨询镜像（"会员的配送费减免权益"规则判 RAG、LLM 判红包查询时采信规则）两条仲裁规则。
- **正则优先级**："订单 WM20260822002 超时了"同时命中"订单"（order_query）和"超时"（delivery_track），规则字典必须按意图特异性排序，高精度词在前。还有"配送费"是政策咨询信号，配送追踪的正则要用负向先行断言排除 `配送(?!费)`。

这些 bug 的共性是：**多通道融合的仲裁规则，比单一通道的准确率更重要**。每个仲裁规则都对应一个真实 bad case，不是预设的。

## 踩过的坑（选几个有代表性的）

| 问题 | 根因 | 修复 |
|------|------|------|
| `UnicodeDecodeError` on Redis session restore | zlib.compress 存 bytes，但 Redis decode_responses=True 强制 UTF-8 解码 | 新增 `get_redis_binary_client()`，decode_responses=False |
| Docker 容器连不上 PostgreSQL | `ensure_database_exists()` 直接读 `host`/`port` 分项变量，容器环境用 DATABASE_URL | 新增 DSN URL 解析，从连接字符串提取参数 |
| Milvus Lite 报 `AllocTimestamp` not implemented | Milvus Lite 是单文件模式，不支持异步 gRPC 的某些调用 | 检测 uri 是否为本地文件路径，是则跳过 `wait_for_creating_index` |
| Rerank 空指针 | SiliconFlow 返回 `document: null`，代码里 `if "document" in item` 把它当 DashScope 格式处理 | 改为 `doc and isinstance(doc, dict) and doc.get("text")` |
| 意图识别 LLM 永远过不了 | 单通道最大可能分 0.5，阈值设 0.6 | 阈值从 0.6 降到 0.4 |
| 查订单返回"系统未能返回实时状态" | Plan Mode 关键词劫持了外卖意图路由 | 外卖意图短路提前到 Plan Mode 检查之前 |
| 转人工标志 API 返回永远 false | AgentState 未声明 requires_human 字段，LangGraph 丢弃未定义 key | 补齐字段定义和初始值 |
| 评测 tool_accuracy 全为 0 | 工具结果无工具名前缀，评测用 `split(":")[0]` 拿到的是 JSON 片段 | 工具结果统一 `tool_name: result` 格式 |
| Judge 偶发 `'int' object is not subscriptable` | LLM 偶尔返回扁平 JSON `{"accuracy": 8}` 而非嵌套结构 | 解析层归一化兼容两种格式 |
| 订单号传参为空 | MCP 参数推断只匹配 query/search 键名，order_id 落到 else 分支传空串 | 按 WM\d+ 正则提取订单号 |

## 快速开始

```bash
git clone https://github.com/Dreamt0511/NexusMind.git
cd NexusMind
cp .env.example .env
# 编辑 .env 填入你的 API key（DeepSeek 或任意 OpenAI 兼容服务）
docker-compose up -d

# 等 30 秒，然后运行 CLI 测试套件
chmod +x nexusmind
./nexusmind test
```

六容器全部健康后，直接体验外卖客服：

```bash
curl -X POST http://localhost:8000/chat/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我查一下订单 WM20260823001 的状态", "user_id": 1}'

# 返回 requires_human + handoff_summary 的转人工场景
curl -X POST http://localhost:8000/chat/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"query": "我吃完外卖拉肚子了，食物不新鲜！", "user_id": 1}'
```

跑黄金测试集（8 用例双层评测，Judge 均分 8.7+，报告落盘 `data/eval_reports/`）：

```bash
docker exec nexusmind-backend python -c \
  "import asyncio; from evaluation import run_golden_eval; print(asyncio.run(run_golden_eval()))"
```

## 技术栈

- **编排**：LangGraph `StateGraph`（Supervisor → 多 Agent / ReAct / Coordinator → Synthesize → Reflection）+ 多 Agent 协作（Advisor / Plan with SOTA）+ ContextStore 动态上下文
- **检索**：Milvus（双模式兼容：Lite 原型 + standalone 生产），bge-m3 向量 + BM25 稀疏 + RRF 融合 + bge-reranker-v2-m3 重排
- **记忆**：三级记忆（语义/情节/程序），Milvus 独立集合 + Redis 会话快照
- **LLM**：DeepSeek v4（聊天）、SiliconFlow（embedding + rerank）、qwen-turbo（评测 Judge）
- **存储**：PostgreSQL（文档元数据/分块/对话历史）、Redis（会话快照）、Milvus（向量）
- **部署**：Docker Compose 六容器，带健康检查和启动依赖

## 文件结构

```
backend/
  main.py / api.py        入口层：应用启动 + 全部路由
  core/                    编排层：Agent 大脑
    orchestrator.py           LangGraph 编排图 + 转人工机制
    react_agent.py            ReAct 迭代循环（思考→工具→观察→再思考）
    collaboration.py          多 Agent 协作（Advisor/Plan with SOTA/Coordinator）
    context_store.py          动态上下文存档 + recall_context 回忆工具
    intent_recognition.py     三路融合意图识别 + 业务意图仲裁规则
    query_rewriter.py         查询改写（指代消解/省略补全）
    agent.py                  子 Agent 基类
    context_manager.py        工具结果两级截断（超长部分存档 ContextStore）
  tools/                   工具层：工具注册与路由
    mcp_protocol.py           工具注册表 + 工具路由 + 外部工具接入
    food_delivery_tools.py    外卖业务工具集（订单/配送/退款/红包，mock 数据）
    tools.py                  通用工具（知识库/记忆检索）
  rag/                     检索层
    documents_process.py      文档分块 + 混合检索 + rerank
    knowledeg_base_manager.py 知识库管理
    embedding_provider.py     Embedding 工厂（多厂商）
    milvus_client.py          Milvus 客户端（双模式兼容）
  memory/                  记忆层
    memory_manager.py         三级记忆 CRUD
    session_recovery.py       会话快照保存/恢复 + 上下文压缩
    auto_store_memory_from_psql.py  记忆沉淀后台任务
  evaluation/              评测进化层
    evaluation.py             LLM-as-Judge + 黄金测试集双层评测 + 报告落盘
    self_evolution.py         规则引擎（抽取/去重/评审）
    evolution_loop.py         自进化自动触发闭环
    monitor.py                滑动窗口监控 + 权重动态调整
  infra/                   基础设施层
    config.py / schemas.py    配置 + 数据模型
    access_control.py         ACL 四级角色 + 审计日志
    postgresql_client.py / redis_cache.py / hash_storage.py
    skills_manager.py         Skills 加载/检索/注入
  skills/                  客服业务流程模板（JSON）
    food_delivery_timeout.json    配送超时赔付 SOP
    food_delivery_refund.json     退款状态机流程
    food_delivery_complaint.json  投诉分级升级
  evolution/               自进化持久化数据（规则/去重记录）
cli.py                    CLI 工具（15 项自动化测试）
docker-compose.yml        六容器编排
docs/
  knowledge/外卖客服知识库.md   平台政策知识库（超时赔付/退款/会员/食品安全）
  PROJECT_OVERVIEW.md     详细文档（架构图/简历模板/面试 QA）
  PROJECT_INTRO.md        本文档
```

分层原则：依赖单向 `core → tools/rag/memory/evaluation → infra`，禁止反向。每层可独立替换——比如把 rag/ 换成 Elasticsearch 实现，core/ 不用改一行。

## 许可证

MIT
