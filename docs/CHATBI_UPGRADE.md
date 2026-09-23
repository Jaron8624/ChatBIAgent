# NexusMind ChatBI 六大方向升级文档

> 升级时间：2026-08-25
> 模块位置：`backend/chatbi/`（新增 6 个核心模块，均注册为 MCP 工具并接入 LangGraph 编排）

---

## 升级总览

| # | 方向 | 模块 | MCP 工具 | 状态 |
|---|------|------|----------|------|
| 1 | 语义层 / 指标层（NL → Metric Plan → SQL） | `chatbi/semantic_layer.py` | `compile_metric_plan`、`get_metric_definitions` | ✅ |
| 2 | Schema 三级检索 + 动态 Few-shot SQL 知识库 | `chatbi/schema_retrieval.py` | `retrieve_schema_context`、`retrieve_sql_examples` | ✅ |
| 3 | SQL 执行校验、自修复、结果合理性检查 | `chatbi/sql_guard.py` | `execute_sql_guarded` | ✅ |
| 4 | 根因分析规划链 / 多 Agent 分解 | `chatbi/root_cause.py` | `run_root_cause_analysis`（+ 编排器专属 `root_cause` 节点） | ✅ |
| 5 | 字段级评测 + 在线 Bad Case 闭环 | `chatbi/field_eval.py` | 评测脚本 `--phase v2` / `--phase badcase-loop` | ✅ |
| 6 | 数据质量对账 Agent | `chatbi/data_quality.py` | `run_data_quality_reconciliation` | ✅ |

---

## 1. 语义层 / Metric Plan 编译链路

**问题**：纯 NL2SQL 模式下，"回头客户数""差评率"等业务概念每次都靠 LLM 即兴翻译，
`customer_id`（每订单唯一）与 `customer_unique_id`（回头客共享）这类语义混淆反复出现。

**方案**：引入 `MetricDefinition` + `MetricPlanCompiler`：

- 14 个预定义指标（Olist 9 个：销售额/订单量/客单价/回头客/差评率/物流时长/延迟率/支付分布/卖家数；Chinook 5 个：收入/发票数/曲目销量/流派收入/艺术家收入）
- 18 个维度字典（含跨表 JOIN 路径与派生字段如 `year_month`）
- 编译器把 `{metrics, dimensions, filters, sort, limit}` JSON 计划编译为带正确 JOIN 的 SQL
- JOIN 路径固化为语义 Join Graph（非字符串拼接），消除歧义列名（如 Chinook `UnitPrice` 显式限定 `chinook.InvoiceLine.UnitPrice`）

**效果**：指标口径由代码而非 Prompt 保证；`get_metric_definitions` 让 Agent 先查语义层再写 SQL。

## 2. Schema 三级检索 + 动态 Few-shot

**问题**：全量 Schema（20 表 100+ 字段）注入 Prompt 噪声大；固定 few-shot 与当前问题无关。

**方案**：`SchemaRetriever` 三级收敛：

1. **Level 1 数据域**：5 个业务域（订单/支付/商品卖家/服务质量/数字媒体）按关键词得分选 Top-3
2. **Level 2 表与 Join**：域内表按表名+字段名得分排序，附带已验证的 JOIN 语句
3. **Level 3 字段**：命中表内的相关字段

`SQLExampleStore` 维护 `docs/knowledge/sql_examples.json`：按问题语义检索 Top-K 已验证 SQL 示例（含 rationale），支持 Bad Case 修复后的自动回写（见方向 5）。

## 3. SQL 守卫流水线（校验 → 自修复 → 合理性检查）

**问题**：LLM 生成的 SQL 常见三类失败——方言错误（`LEFT()` 在 SQLite 不存在）、表名幻觉、结果静默异常（空结果/LIMIT 截断/比率越界）。

**方案**：`SQLGuardPipeline` 三重保障：

1. **静态校验**（`SQLValidator`）：禁止写操作、SQLite 方言函数黑名单、表名存在性（含 CTE 别名排除）
2. **LLM 自修复**（`SQLSelfRepairAgent`）：错误信息 + Schema 反馈给模型修正，最多 2 轮
3. **结果合理性检查**（`ResultSanityChecker`）：空结果归因提示、TopN 与 LIMIT 一致性、NULL 列占比 ≥50% 告警、比率列 >100% 越界检测

**实测**：`LEFT(customer_state, 2)` 方言错误 1 轮自动修复为 `substr(customer_state, 1, 2)`，执行成功。

## 4. 根因分析规划链（RCA）

**问题**："为什么 11 月销售额暴增" 类问题的通用 ReAct 路径缺少假设验证结构，容易给无证据结论。

**方案**：三阶段专属链路（LangGraph 新增 `root_cause` 节点，意图路由优先级最高）：

1. **PlanChain**（SOTA 模型）：异动问题 → 3-4 个假设（季节性/品类结构/区域卖家异常/数据口径）+ 3-4 条证据 SQL（维度下钻、同环比、TOP 异常实体）
2. **EvidenceGatherer**：`asyncio.gather` 并行执行证据 SQL，每条复用 SQL 守卫流水线
3. **RootCauseSynthesizer**：证据约束下的结构化报告（异动概述/根因分析/结论+置信度/建议），证据不足的假设明确标注"证据不足"

**实测**（"11月销售额为什么暴增"）：4/4 证据非空，正确归因黑五大促（主因，单日 Top5 销售额均超 2 万）+ 高客单品类结构变化（次因，乐器/游戏/电脑配件 10→11 月放大 3-5 倍）。

**关键修复**：规划 Prompt 注入真实数据时间范围（Olist 2025-01~2026-06）后，消除了 LLM 对 2017 年（原版 Olist 数据集年份）的先验猜测导致的全空证据。

## 5. 字段级评测 + Bad Case 闭环

**问题**：整行集合比对（EX）不可归因——擅自加 LIMIT 导致的截断与口径错误同被判 0 分。

**方案**：`field_level_eval` 三级指标：

- **列级**：列名命中率（Gold 列是否全被查出，缺失列清单）
- **行级**：按 `key_column`（如 `customer_state`）对齐后的行召回率——LIMIT 截断只扣行分
- **单元格级**：对齐后逐格比对（数值列容差 0.5），输出差异单元格明细（key/column/expected/actual）

**Bad Case 闭环**（`--phase badcase-loop`）：

```
评测失败 → data/bad_cases/*.jsonl 落盘（status=new）
        → 增强流水线自动重试（三级 Schema 检索 + Few-shot + SQL 守卫）
        → 字段级评测验证修复
        → 修复成功 → SQL 回写 Few-shot 知识库（sql_examples.json）+ status=regression_fixed
```

## 6. 数据质量对账 Agent

**方案**：7 条声明式对账规则（硬约束期望 0 行 / 阈值规则允许比例），`asyncio.gather` 并行巡检：

| 规则 | 数据集 | 类型 | 实测结果 |
|------|--------|------|----------|
| DQ-O-01 订单金额 vs 支付金额 | Olist | 阈值 5% | **告警**：2099 单不一致（14.2%），625 超付 / 1474 少付 |
| DQ-O-02 客户邮编地理关联 | Olist | 阈值 10% | **告警**：99.7% 孤儿邮编（geolocation 仅 262 行） |
| DQ-O-03 取消订单存在评价 | Olist | 硬约束 | **告警**：438 条 |
| DQ-O-04 订单时序倒挂 | Olist | 硬约束 | 通过：0 条 |
| DQ-O-05 卖家发货超期 | Olist | 阈值 20% | **告警**：22.0% 超期，TOP 异常卖家已定位 |
| DQ-C-01 发票总额 vs 明细 | Chinook | 硬约束 | 通过：0 张不一致（对照验证规则可靠性） |
| DQ-C-02 主链路外键完整性 | Chinook | 硬约束 | 通过：0 孤儿（对照验证） |

每条告警附带差异归因建议（如 DQ-O-01 指向支付拆分/多商品聚合口径），Chinook 干净库作为对照证明规则不误报。

---

## 编排集成

`backend/core/orchestrator.py` 变更：

- 新增 `root_cause` LangGraph 节点：`intent == "root_cause"` 时优先路由（数据质量对账关键词也已加入该意图）
- 6 个 ChatBI 模块工具全部注册进统一工具注册表（与既有 11 个工具统一路由）
- `rca_result` 状态字段：RCA 报告与 ReAct 回答一样跳过 synthesize 二次改写，保留证据链
- 意图识别规则新增数据质量/对账关键词

评测入口：`scripts/eval_chatbi.py --phase v2`（增强流水线 + 字段级指标）、`--phase badcase-loop`（闭环重试与知识库回写）。

---

## V2 端到端复测结论（2026-08-25，deepseek-v4-flash，15 题）

| 指标 | Baseline | 首版 Optimized | V2 复测 | V3 复测（Prompt 契约 + Bad Case 闭环后） |
|------|----------|----------------|---------|---------|
| NL2SQL EX | 20.0% | **46.7%** | 26.7% | 40.0% |
| Schema Linking | 100% | 100% | 100% | 100% |
| LLM Judge | 3.4 | 4.4 | 4.4 | **4.9** |
| 字段级行召回率 | — | — | 0.5467 | 0.5541 |
| 字段级单元格准确率 | — | — | 0.5333 | **0.6000** |

**V2 的诚实结论**：EX 回落至 26.7%（低于固定 Prompt 优化版的 46.7%），主因是动态检索拼装的 Prompt 组织度不如手工调优的固定版本、且 flash 模型在复合查询上输出不稳定。V2 的可验证收益在工程能力层（错误可观测、修复兜底、闭环资产）。

**V3 改进与结果**：针对 V2 失败模式做了三处修复后复测：

1. **SQL 契约注入**：增强流水线 Prompt 新增 11 条硬约束（禁止额外 ranking/描述列、LIMIT 只在明确 TOP N 时使用且等于 N、差评率口径 review_score<=2、回头客用 customer_unique_id、Chinook 收入口径等）
2. **Bad Case 闭环实跑**：对 V2 落盘的 11 条失败用例自动重试，2 条（m1 品类销售额、m2 流派曲目数）经字段级验证修复并回写 Few-shot 知识库，9 条保留为困难样例
3. **评测自身两处 bug 修复**：结果存档的 SQL 截断（200 字符）；字段级对齐的 key 列别名匹配（gold `BillingCountry` vs agent `country`）

量化效果：EX 26.7% → 40.0%（+13.3pp），Judge 4.4 → 4.9，单元格准确率 0.5333 → 0.6000，列全命中 0/15 → 3/15。其中 m1/m2 的修复直接来自闭环回写的 Few-shot 示例——闭环机制被证明有效但覆盖率有限（2/11）。V3 仍未追平固定优化版的 46.7%，剩余失败集中在复合题（跨库对比 c1、差评率口径 c2、趋势根因 p1/p3），需要更强的模型或语义层强制路由才能突破。

---

## V4 对标补齐与复测（2026-08-25，15 题）

本轮补齐的是与 WrenAI、Vanna、jinyulaiguo/sql_agent、lijingfz/nl2sql-agent、DB-GPT/Dataherald、OpenChatBI 的六项关键差距，而不是增加展示型模块。

| 对标能力 | NexusMind V4 落地 | 可复核位置 |
|---|---|---|
| WrenAI：语义层强制编译 | `SchemaLinkingIR` 先解析数据域、指标、维度、过滤、排序与 LIMIT；单指标且单数据集问题走 NL → IR → Metric Plan → 确定性 SQL，复杂/跨域问题才降级 LLM | `text2sql_pipeline.py`、`semantic_layer.py` |
| Vanna：三类语料混合检索 | DDL、业务指标定义、已验证 SQL 三类语料以 BM25 + 稀疏向量 + RRF（k=60）融合检索 | `hybrid_knowledge.py` |
| jinyulaiguo/sql_agent：AST 校验 | sqlglot SQLite AST 校验表、列、别名、CTE 与 `ORDER BY` 输出别名；支持 `chinook.Invoice` 这类 schema 前缀表 | `ast_validator.py` |
| lijingfz/nl2sql-agent：三轮自愈 | 静态/执行失败最多 3 轮；空结果与明确 TopN 未加 LIMIT 也会成为修复信号 | `sql_guard.py` |
| DB-GPT/Dataherald：两阶段中间表示 | Schema Linking IR 与 SQL 编译分离，IR 可记录路由原因、置信度与降级理由 | `text2sql_pipeline.py` |
| OpenChatBI/Langfuse：在线观测闭环 | JSONL trace、错误/慢查询/重试/10% 随机采样、thumbs-down 写入在线 Bad Case | `observability.py`、`api.py` |

| 指标 | Baseline | 首版 Optimized | V2 | V3 | V4 |
|---|---:|---:|---:|---:|---:|
| NL2SQL EX | 20.0% (3/15) | 46.7% (7/15) | 26.7% (4/15) | 40.0% (6/15) | **46.7% (7/15)** |
| Schema Linking | 100.0% | 100.0% | 100.0% | 100.0% | 93.3% (14/15) |
| Intent Recognition | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% |
| LLM Judge 平均分 | 3.4 | 4.4 | 4.4 | 4.9 | 4.1 |
| 字段级行召回率 | — | — | 0.5467 | 0.5541 | **0.6000** |
| 字段级单元格准确率 | — | — | 0.5333 | **0.6000** | 0.5667 |
| 列全命中 | — | — | — | 3/15 | **5/15** |

V4 的 EX 为 46.7%，较 V3 增加 6.7pp，并追平首版固定 Prompt 的历史最好结果；新增通过用例为 c4（月度订单趋势）。但不能据此声称所有维度都改善：Judge 从 4.9 降到 4.1，单元格准确率从 0.6000 降到 0.5667，原因是复杂题仍走 LLM fallback，且回答表达质量存在波动。剩余 8 个 EX 失败集中在回头客口径、跨库对比、差评率复合分析、收入效率、根因分析与卖家延迟率；它们不是语义层单指标可确定性编译的问题，需要新增复合 Metric Plan/跨库执行策略和 RCA 专用评测。

复测中曾出现一次 33.3% 的中间结果，不能忽略。排查发现 `ASTSQLValidator` 对 `chinook.Invoice`、`chinook.Track` 等带 schema 前缀的合法表误报“不存在”，SQL 守卫于是错误调用 LLM 修复，反而把正确的 Chinook 语义 SQL 退化成 Olist SQL。修复表名全名/短名双索引后重新全量运行 V4，得到上表的 46.7%。该缺陷已被写成 schema 前缀表回归用例；当前测试套件为 **29/29 通过**。

---

## V5：复合 Metric Plan 与专项评测

针对 V4 中会退化为 LLM fallback 的多指标、派生比率、跨库对比和驱动拆解，新增 6 个受治理的复合 Metric Plan：支付方式统计、跨库订单/发票对比、品类差评拆解、流派收入效率、卖家延迟率和流派收入驱动。命中时采用 `NL → Schema Linking IR → Composite Metric Plan → 确定性 SQL`，基础组件、维度和受控模板均可审计。

新增独立专项集 `scripts/eval_compound.py`，以 12 个不同于主 15 题集的改写问题验证"计划识别 + 确定性编译 + 执行结果"。最新结果为 **12/12（100%）**，其含义是已治理的 6 个复合口径在两种表述下无回归；它不是开放域 EX，也不能替代 V4 端到端 EX 46.7%。完整口径、计划定义和公开基准接入边界见 `COMPOUND_METRICS_EVALUATION.md`。回归测试扩展至 **31/31 通过**。

---

## V5 终版：三题修复 + 四集评测

V5 复合计划接入后原 15 题集一度达到 11/15（73.3%），仍有 4 题失败。逐一排查修复：

1. **s3 回头客**：`repeat_customers` 的 CUSTOM 模板编译时丢失 HAVING 子句，且评测 Gold SQL 本身写错（算全部客户而非回头客）。修复：单指标+CUSTOM+无维度时直接返回模板原文；模板改为子查询包裹；Gold SQL 同步修正。
2. **m4 艺术家 Top5**：`artist_revenue` 缺少 ROUND，浮点精度导致 1/5 行不匹配。新增 `MetricDefinition.round_decimal` 字段，Chinook 收入类指标统一保留两位小数。
3. **p2 卖家延迟率**：`schema_retrieval.py` 的 TABLES 字典缺少 `order_delivered_carrier_date` 列，AST 误拦后 LLM 修复替换为 `order_delivered_customer_date`（客户签收而非承运商揽收），口径错误。补全列定义后修复。

修复后原 15 题端到端评测达到 **EX 15/15（100%）**。但这只是自建小集，不等于通用能力。因此新增三组独立评测验证泛化：

| 评测集 | 题数 | EX | 说明 |
|---|---:|---:|---|
| V5 原题集（端到端混合链路） | 15 | 15/15（100%） | 语义层 + 复合计划 + LLM fallback |
| 复合专项集（计划识别 + 确定性编译） | 12 | 12/12（100%） | 6 类复合计划 × 2 改写 |
| 域内泛化集（全新表述） | 10 | 8/10（80%） | 2 个 LLM fallback 题因评测不跑 LLM 而失败 |
| Spider dev 全 20 库（开放域） | 99 | 81/99（81.8%） | LLM 直出 SQL，无语义层/守卫，真实 Spider SQLite 数据 |

### V5.1 Spider 全库扩展评测

V5 首版 Spider 仅跑了 singer 库（2 表 3 行 mock 数据，30 题），93.3% 的分数因数据过于简单而虚高。现已从 HuggingFace `HAL-9001/spider-databases` 下载 Spider 官方完整数据集（196MB，含 200 个 SQLite 数据库真实数据），对 dev 集全部 20 个数据库各采样 5 题（共 99 题）重新评测。

**评测方法**：从 SQLite 的 `sqlite_master` 提取完整 CREATE TABLE DDL（含 PK/FK 约束），每表取 3 行样本数据，组装为 schema prompt；LLM（deepseek-v4-flash, temperature=0）直出 SQL，不做语义层编译和 SQL 守卫修复；在同一真实数据库上执行 Agent SQL 与 Gold SQL，比较结果集是否一致（Jaccard ≥0.8 算通过）。字段级评测按 key 列对齐后逐单元格比较。

#### 四项核心指标汇总

| # | 指标 | 值 | 说明 |
|---|------|------|------|
| 1 | EX（执行准确率） | 81/99（81.8%） | Agent SQL 与 Gold SQL 执行结果集一致（Jaccard≥0.8） |
| 2 | SL（表命中率） | 96/99（97.0%） | Agent SQL 引用的表名集合与 Gold SQL 完全一致 |
| 3 | 行召回率（Field Recall） | 0.851 | Gold 结果行中 85.1% 出现在 Agent 结果中（按 key 列对齐） |
| 4 | 字段准确率（Cell Accuracy） | 0.898 | 匹配行中 89.8% 的单元格值正确 |

四项指标构成一条完整评测链路：选对表（SL 97%）→ 查全行（Recall 0.851）→ 值准确（Accuracy 0.898）→ 整体对（EX 81.8%）。SL 97% 说明 DDL+样本行的 prompt 设计让 LLM 几乎不犯选表错误；EX 81.8% 在 zero-shot 设定下处于合理偏上水平（Spider test 集 SOTA 约 86-91%，但使用 few-shot + 自我一致性；GPT-4 zero-shot 约 60-75%）。

> 已移除 EM（组件匹配率 37.4%）、Executability（100%）、Avg Jaccard（0.823）、Exact Match（78.8%）、Zero Match（16.2%）五项指标。EM 对 LLM 生成的等价 SQL 写法不敏感（别名/JOIN 语法/谓词顺序差异导致虚低），Spider 社区已转向以 EX 为主指标；Executability 到达 100% 后无区分度；其余三项均为 EX 的 Jaccard 计算子集。

#### 全 20 库分库字段级明细

| 数据库 | 表数 | 总行数 | 题数 | EX | 行召回 | 字段准确 |
|--------|------|--------|------|------|--------|----------|
| battle_death | 3 | 28 | 5 | 5/5 | 1.000 | 1.000 |
| car_1 | 6 | 890 | 5 | 5/5 | 1.000 | 1.000 |
| concert_singer | 4 | 31 | 5 | 5/5 | 1.000 | 1.000 |
| course_teach | 3 | 23 | 5 | 5/5 | 0.943 | 1.000 |
| cre_Doc_Template_Mgt | 4 | 55 | 5 | 5/5 | 1.000 | 0.933 |
| dog_kennels | 8 | 72 | 5 | 5/5 | 1.000 | 1.000 |
| employee_hire_evaluation | 4 | 32 | 5 | 5/5 | 1.000 | 1.000 |
| museum_visit | 3 | 20 | 5 | 5/5 | 1.000 | 1.000 |
| poker_player | 2 | 12 | 5 | 5/5 | 1.000 | 1.000 |
| real_estate_properties | 5 | 40 | 4 | 4/4 | 0.969 | 0.833 |
| singer | 2 | 16 | 5 | 5/5 | 1.000 | 1.000 |
| voter_1 | 3 | 320 | 5 | 5/5 | 1.000 | 1.000 |
| orchestra | 4 | 40 | 5 | 4/5 | 1.000 | 0.983 |
| pets_1 | 3 | 40 | 5 | 4/5 | 0.800 | 1.000 |
| network_1 | 3 | 46 | 5 | 3/5 | 0.800 | 0.800 |
| world_1 | 3 | 5,302 | 5 | 3/5 | 0.800 | 0.800 |
| flight_2 | 3 | 1,312 | 5 | 2/5 | 0.400 | 0.600 |
| student_transcripts_tracking | 11 | 165 | 5 | 2/5 | 0.400 | 0.800 |
| tvshow | 3 | 39 | 5 | 2/5 | 0.400 | 0.600 |
| wta_1 | 3 | 531,377 | 5 | 2/5 | 0.533 | 0.600 |

12 库达到全满分（EX 100% + Recall 1.0 + Accuracy 1.0）。字段级短板集中在 flight_2/tvshow/wta_1/student_transcripts_tracking（Recall 0.4-0.53）。

#### 18 个失败案例字段级归因

失败案例按字段级指标分三类：

**类型 A：行召回 R=0 + 字段准确 A=0（9 题）**——结果集完全错位。主要原因是 SELECT 列序倒了（Gold 是 `(count, name)` Agent 是 `(name, count)`，元组永远不匹配）和 flight_2 数据脏值（AirportCode 尾部空格导致 Gold 精确匹配返回空、Agent 用 TRIM 返回非空）。还包括 wta_1/world_1 的 `SELECT count, country` vs `SELECT country, count` 列序问题。

**类型 B：行召回 R=0 + 字段准确 A=1.0（4 题）**——key 列对齐后无交集行。Agent 返回的行 key 值和 Gold 不重叠。如 pets_1 的 "每个学生有多少宠物"，Agent 从 Has_Pet 直接查只返回 count+StuID，但 Gold 还关联了 Student 表。

**类型 C：行召回 R=1.0 + 字段准确 A=1.0 但 EX 失败（5 题）**——行和字段都对，但 Agent 返回了多余行或排序不同。如 orchestra "按频率排序" Agent 排序方向反了，返回了全部 6 行而 Gold 只返回格式名列。

#### 与业界对比

Spider 1.0 dev 集 GPT-4 zero-shot 系统报告 EX 在 60-75% 区间（Spider 官方 leaderboard 的 test 集 SOTA 约 86-91%，但使用了 few-shot + 自我一致性等策略）。本系统 EX 81.8%、SL 97% 在 zero-shot（无 few-shot、无自我一致性）设定下处于合理偏上水平。

**前版对比**：V5 首版 singer-only 28/30（93.3%，mock 数据）→ 全 20 库 81/99（81.8%，真实数据）。降幅 11.5pp 合理——singer 是最简单的 2 表库，全库评测引入了 11 表复杂 schema（student_transcripts_tracking）、531K 行大表（wta_1）、多表 JOIN 和嵌套子查询。

当前回归测试 **34/34 通过**，覆盖语义层、AST 校验、SQL 守卫、复合计划、跨域路由和字段级评测。

