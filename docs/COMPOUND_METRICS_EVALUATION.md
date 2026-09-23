# NexusMind 复合 Metric Plan 与评测体系

## 目标与边界

NexusMind 的基础语义层将单指标问题编译为确定性 SQL，例如“各州订单量”或“按流派统计曲目数”。但真实 ChatBI 的高价值问题通常还包括多指标同表聚合、派生比率、跨库对比，以及用于解释收入或质量异动的指标拆解。继续让 LLM 直接生成这类 SQL，会使指标口径、分母、去重键和 Join 基数不可审计。

本次新增 `COMPOSITE_METRIC_PLANS`。它将已治理的复合问题表达为“计划名、基础组件、数据集、维度、受控 SQL 模板”的注册表。自然语言先经过复合计划识别，命中后走 `NL → Schema Linking IR → Composite Metric Plan → 确定性 SQL`，不经过自由 SQL 生成。未命中的复杂问题仍会降级到原有 LLM/RCA 链路。

该机制只覆盖已经登记的业务口径，不应被解读为通用 Text2SQL 或通用根因分析能力。

## 已治理的复合计划

| 计划 | 解决的问题 | 核心口径 |
|---|---|---|
| `payment_method_stats` | 支付方式的订单/交易笔数与平均金额 | `COUNT(*)` 和 `AVG(payment_value)`，按 `payment_type` 聚合 |
| `state_orders_vs_invoices` | Olist 州订单量与 Chinook 发票总数对比 | Olist 分州订单数，Chinook 发票通过受控子查询附加；不依赖 LLM 跨库 Join |
| `category_bad_review_breakdown` | 品类差评率及家具与其他品类比较 | 分子 `review_score <= 2`，分母为全评价数，输出差评数、总数、比例 |
| `genre_revenue_efficiency` | 流派收入效率 | 收入 `SUM(UnitPrice * Quantity)` 除以去重曲目数 `COUNT(DISTINCT TrackId)` |
| `seller_delay_breakdown` | 卖家延迟发货率 TopN | 只统计有揽收时间的订单项，延迟条件为揽收时间晚于 `shipping_limit_date` |
| `genre_revenue_driver` | 流派收入驱动拆解 | 同时输出曲目数、售出曲目的平均单价与收入，按收入排序 |

这些计划保存在 `backend/chatbi/semantic_layer.py`；路由及 IR 记录位于 `backend/chatbi/text2sql_pipeline.py`。新增计划时必须同时定义组件、默认限制、维度、受控 SQL 模板和至少一个正反例测试。禁止将未经验证的自然语言条件直接拼接进模板。

## 评测不止 EX

原有 `scripts/eval_chatbi.py` 的 15 题集评估端到端 Text2SQL，包括结果 EX、Schema Linking、意图识别、LLM Judge 和字段级结果对齐。其当前 V4 结果为 EX 7/15（46.7%），表示系统在包含自由 LLM fallback 的混合链路上仍有明显的复合查询短板。

新增 `scripts/eval_compound.py` 是不同目的的专项集。它包含 12 个与主集不同表述的问题，覆盖六类复合计划，每类两个改写。它要求同时满足计划识别正确、确定性编译成功、执行结果与独立 Gold SQL 完全一致。报告写入 `docs/eval_compound.json`。

最新专项结果为 12/12（100.0%）。这个数字只代表“六个已治理复合口径在两种改写表述下没有回归”，不能拿来与 V4 的 46.7% 作横向加权，也不能称为开放域 EX。它的价值在于把此前会掉进 LLM fallback 的六类失败模式固定成可重复验证的产品能力。

推荐持续同时报告三组结果：端到端混合链路 EX、字段级行/单元格准确率，以及复合计划的计划识别与执行结果准确率。对于线上运行，还应加入 SQL 可执行率、AST 误拦截率、自愈成功率、P50/P95 延迟、用户采纳率与负反馈率。

## 外部公开评测集

当前仓库没有 Spider、BIRD、Spider 2.0 或 WikiSQL 的数据文件和适配器。原因是 NexusMind 使用双 SQLite 业务库及中文业务口径，直接套用英文公开集无法验证销售额、差评率、延迟率等企业指标口径。

如果需要证明开放域泛化能力，可在后续单独引入公开集适配层，并严格分开报告：Spider/BIRD 使用其官方数据库、官方执行评测脚本和公开划分；NexusMind 业务集使用本仓库的可复现 SQLite 数据、Gold SQL 和口径断言。不能把两者混为一个 EX 数字。
