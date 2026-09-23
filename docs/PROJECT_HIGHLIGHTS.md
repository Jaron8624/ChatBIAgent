# NexusMind 项目亮点总览

```mermaid
mindmap
  root((NexusMind<br/>ChatBI Agent))
    架构设计
      LangGraph 多智能体编排
        Supervisor 路由分发
        ReAct 循环迭代
        Coordinator 多 Agent 并行
        Reflection 质量审查
      FastAPI 后端 + React 前端
        8 个前端页面<br/>Chat / Audit / Eval / Monitor
        MCP JSON-RPC 2.0 自研协议层
        多厂商 LLM 解耦工厂模式
      双数据集真实数据
        Olist 巴西电商 8.9MB
        Chinook 数字媒体 984KB
        Spider 20 库 839MB
      20606 行代码 / 34 项回归测试
    ChatBI 核心引擎 12 模块
      语义层 MetricPlanCompiler
        14 个预定义指标
        18 个维度字典含 JOIN 路径
        6 类复合指标计划<br/>确定性 SQL 模板编译
        NL → IR → SQL 两阶段编译
      Schema 三级检索
        Domain → Table → Column
        动态 Few-shot SQL 知识库
        三语料混合检索 DDL+业务+SQL
      SQL 守卫
        sqlglot AST 语法校验
        执行失败自修复重试
        结果合理性检查
      根因分析规划链
        假设拆解 → 并行取证
        4 条非空证据查询
      字段级评测闭环
        行召回率 Row Recall
        单元格准确率 Cell Accuracy
        在线 Bad Case 采样
      数据质量对账 Agent
        7 条 DQ 规则
        Olist 4 告警 / Chinook 0 异常
    四套评测体系
      V5 原题集 15 题
        端到端混合链路
        EX 100% 15/15
      复合专项集 12 题
        6 计划 × 2 改写
        EX 100% 12/12
      域内泛化集 10 题
        全新表述
        EX 80% 8/10
      Spider 全 20 库 99 题
        真实 Spider SQLite 数据
        EX 78.8% 78/99 严格判等
        SL 97.0% 96/99
        Executability 100%
        行召回 0.851
        字段准确 0.898
        Avg Jaccard 0.823
    工程亮点
      评测驱动数据治理
        四维度评测发现脏数据
        byte 级验证 + 跨库清理
      自进化闭环
        Bad Case → 规则抽取
        双阈值质量保障
      可观测性
        TraceStore 全链路追踪
        异常采样 + 慢查询标记
      安全沙箱
        SQL 注入防护
        只读查询强制
        写操作拦截
```

## 架构流程图

```mermaid
graph TB
    subgraph User["用户提问"]
        Q["自然语言问题"]
    end

    subgraph Linking["Schema Linking 阶段"]
        SL["SchemaLinker<br/>NL → IR"]
        IR["SchemaLinkingIR<br/>metrics + dimensions + filters + limit"]
        Match["match_metrics_by_query<br/>指标别名匹配"]
        MatchComp["match_composite_plan<br/>复合计划识别"]
    end

    subgraph Route["路由决策"]
        Semantic["semantic_compile<br/>确定性编译"]
        Composite["composite<br/>复合计划模板"]
        LLM["llm_fallback<br/>LLM 自由生成"]
        RootCause["root_cause<br/>根因分析规划链"]
    end

    subgraph Compile["语义编译层"]
        Compiler["MetricPlanCompiler<br/>14 指标 × 18 维度"]
        Template["6 复合指标计划<br/>确定性 SQL 模板"]
        AST["ASTSQLValidator<br/>sqlglot 语法校验"]
        Guard["SQLGuardPipeline<br/>执行 → 自修复 → 合理性检查"]
    end

    subgraph Exec["执行层"]
        DB["双 SQLite 数据库<br/>Olist 8.9MB + Chinook 984KB<br/>ATTACH 跨库查询"]
    end

    subgraph Eval["评测层"]
        FieldEval["字段级评测<br/>Row Recall + Cell Accuracy"]
        Trace["TraceStore<br/>全链路追踪 + 异常采样"]
        DQ["DataQualityAgent<br/>7 条对账规则"]
    end

    Q --> SL
    SL --> IR
    IR --> Match
    IR --> MatchComp
    Match --> Route
    MatchComp --> Route
    Route -->|单指标 + 维度一致| Semantic
    Route -->|复合计划命中| Composite
    Route -->|根因/对比/为什么| RootCause
    Route -->|未命中指标| LLM
    Semantic --> Compiler
    Composite --> Template
    Compiler --> AST
    Template --> AST
    LLM --> AST
    AST -->|语法通过| Guard
    AST -->|语法错误| LLM
    Guard -->|执行成功| DB
    Guard -->|执行失败 + 自修复| DB
    RootCause --> DB
    DB --> FieldEval
    DB --> Trace
    DB --> DQ

    style Semantic fill:#4CAF50,color:#fff
    style Composite fill:#2196F3,color:#fff
    style LLM fill:#FF9800,color:#fff
    style RootCause fill:#9C27B0,color:#fff
    style DB fill:#607D8B,color:#fff
```

## 评测结果对比

```mermaid
graph LR
    subgraph Suites["四套评测"]
        S1["V5 原题集<br/>15 题<br/>━━━━━━━━━━<br/>EX: 100% (15/15)<br/>━━━━━━━━━━<br/>语义层+复合计划<br/>+LLM fallback"]
        S2["复合专项集<br/>12 题<br/>━━━━━━━━━━<br/>EX: 100% (12/12)<br/>━━━━━━━━━━<br/>6 计划 × 2 改写<br/>计划识别+确定性编译"]
        S3["域内泛化集<br/>10 题<br/>━━━━━━━━━━<br/>EX: 80% (8/10)<br/>━━━━━━━━━━<br/>全新表述<br/>2 LLM 题未跑"]
        S4["Spider 20 库<br/>99 题<br/>━━━━━━━━━━<br/>EX: 78.8% (78/99)<br/>SL: 97.0%<br/>Exec: 100%<br/>Recall: 0.851<br/>Accuracy: 0.898<br/>━━━━━━━━━━<br/>真实 Spider 数据<br/>20 个跨域数据库<br/>严格判等口径"]
    end

    subgraph Evolution["优化历程"]
        V1["V1 Baseline<br/>EX: 20.0%"] --> V2["V2 Schema检索<br/>EX: 26.7%"]
        V2 --> V3["V3 SQL契约<br/>EX: 40.0%"]
        V3 --> V4["V4 语义层<br/>EX: 46.7%"]
        V4 --> V5["V5 复合计划<br/>EX: 100%"]
        V5 --> V51["V5.1 Spider全库<br/>EX: 78.8%"]
    end

    S1 --- Evolution
    S2 --- Evolution
    S3 --- Evolution
    S4 --- Evolution

    style S1 fill:#4CAF50,color:#fff
    style S2 fill:#4CAF50,color:#fff
    style S3 fill:#FF9800,color:#fff
    style S4 fill:#2196F3,color:#fff
    style V5 fill:#4CAF50,color:#fff
    style V51 fill:#2196F3,color:#fff
```

## Spider 四项核心指标

```mermaid
graph TB
    subgraph Metrics["Spider 全库评测四项核心指标"]
        M1["EX 执行准确率（严格判等）<br/>78.8%"]
        M2["SL 表命中率<br/>97.0%"]
        M3["行召回率<br/>0.851"]
        M4["字段准确率<br/>0.898"]
    end

    subgraph Chain["评测链路"]
        C1["选对表<br/>SL 97%"]
        C2["查全行<br/>Recall 0.851"]
        C3["值准确<br/>Accuracy 0.898"]
        C4["整体对<br/>EX 81.8%"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph Detail["关键解读"]
        I1["10/20 库 EX 满分<br/>含 8 表 dog_kennels<br/>4 库 40% 主因列序+脏值"]
        I2["zero-shot 无 few-shot<br/>GPT-4 zero-shot 60-75%<br/>Spider SOTA 86-91% 用 few-shot+自我一致性<br/>另 3 题 Jaccard≥0.8 部分正确"]
    end

    M1 --- C4
    M2 --- C1
    M3 --- C2
    M4 --- C3
    C4 --- I1
    C4 --- I2

    style M1 fill:#4CAF50,color:#fff
    style M2 fill:#4CAF50,color:#fff
    style M3 fill:#4CAF50,color:#fff
    style M4 fill:#4CAF50,color:#fff
    style C4 fill:#2196F3,color:#fff
```

## 数据规模与技术栈

| 维度 | 数据 |
|------|------|
| 总代码行数 | 20,606 行（Python 6,830 + TypeScript/React + 配置） |
| ChatBI 核心模块 | 12 个 Python 模块，2,405 行 |
| 评测脚本 | 4 个，1,348 行 |
| 回归测试 | 34 项，352 行 |
| 文档 | 7 份，1,834 行 |
| 域内数据 | Olist 8.9MB（14,747 订单）+ Chinook 984KB |
| Spider 数据 | 20 个 dev 数据库，839MB 真实数据 |
| 后端 | FastAPI + LangGraph + Milvus + Redis + PostgreSQL |
| 前端 | Vite + React + TypeScript，8 个页面 |
| LLM | DeepSeek v4-flash（text2sql）+ SiliconFlow bge-m3（embedding） |
| 评测指标 | 4 项（EX / SL / 行召回率 / 字段准确率） |
| 评测套件 | 4 套（V5 原题 / 复合专项 / 域内泛化 / Spider 全库） |
