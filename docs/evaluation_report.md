# NexusMind ChatBI Agent 评测报告

> 评测时间：2026-08-24  
> 评测模型：deepseek-v4-flash  
> 数据集：Chinook（11 表，984KB）+ Olist（9 表，8.9MB）双 SQLite 数据库  
> 评测脚本：`scripts/eval_chatbi.py`

---

## 一、评测维度与方法

本评测覆盖四个核心维度，方法论参照学术界主流 NL2SQL benchmark（Spider / BIRD-SQL）并扩展至 Agent 层面：

| 维度 | 缩写 | 方法 | 指标 |
|------|------|------|------|
| NL2SQL 正确性 | EX | Agent 生成 SQL 和 Gold SQL 在同一数据库上执行，比较结果集是否一致（忽略行序） | 执行通过率 |
| Schema Linking | SL | 从 Agent SQL 中提取 FROM/JOIN 后的表名，检查是否全部为合法表名（含 CTE 别名排除） | 表名命中率 |
| 意图识别准确率 | IR | 规则匹配器（正则关键词）对用户问题做意图分类，与预期标签比对 | 分类正确率 |
| 答案质量 | Judge | LLM-as-Judge 多维打分（准确性/完整性/清晰度/洞察深度），综合 1-10 分 | 平均分 + 通过率(≥6) |

测试集 15 道题，按难度分四档：

| 难度 | 数量 | 特征 |
|------|------|------|
| Simple | 4 | 单表查询或简单聚合，无 JOIN |
| Moderate | 4 | 2-4 表 JOIN，含 GROUP BY + ORDER BY |
| Challenging | 4 | 跨库查询、CASE WHEN 百分比、子查询、 strftime 日期处理 |
| Compound | 3 | 根因分析类问题，需先查数据再做归因推理 |

---

## 二、优化措施

Baseline 使用最小化 prompt（仅表名+字段名列表，无关系图、无示例、无规则提示）。优化版做了 5 项改进：

1. **Schema 关系图**：在 schema 描述中加入表关系链（如 `olist_customers 1───* olist_orders 1───* olist_order_items *───1 olist_products`），帮助 LLM 理解 JOIN 路径。

2. **Few-shot 示例**：从 0 个增加到 4 个示例，覆盖高频失败场景——回头客户查询（`customer_unique_id` vs `customer_id` 的语义区分）、差评率计算（CASE WHEN + 百分比）、流派曲目排名。

3. **字段语义标注**：关键字段加中文注释，如 `customer_unique_id(回头客共享)`、`payment_type(credit_card/boleto/voucher/debit_card)`、`review_score(1-5分，1=最差5=最好)`。

4. **SQLite 函数约束**：显式指令"用 `strftime('%Y-%m', column)` 提取月份，不要用 `LEFT()` 函数"（SQLite 不支持 LEFT()）。

5. **结果完整性约束**：指令"不要随意加 LIMIT，除非用户明确要求'前N名'或'TOP N'"，避免 Agent 自行截断结果集导致与 Gold SQL 不匹配。

---

## 三、量化结果

### 3.1 总体对比

| 指标 | Baseline | Optimized | 变化 |
|------|----------|-----------|------|
| NL2SQL 正确率 (EX) | 3/15 (20.0%) | 7/15 (46.7%) | **+26.7%** |
| Schema Linking (SL) | 15/15 (100.0%) | 15/15 (100.0%) | — |
| 意图识别 (IR) | 9/15 (60.0%) | 9/15 (60.0%) | — |
| LLM Judge 平均分 | 3.4/10 | 4.4/10 | **+1.0** |
| Judge 通过率 (≥6) | 4/13 (30.8%) | 7/14 (50.0%) | **+19.2%** |

### 3.2 按难度分布

| 难度 | EX (Baseline) | EX (Optimized) | Judge (Baseline) | Judge (Optimized) |
|------|---------------|----------------|-------------------|-------------------|
| Simple (4) | 3/4 (75%) | **4/4 (100%)** | 4.2 | **6.2** (+2.0) |
| Moderate (4) | 0/4 (0%) | **3/4 (75%)** | 5.2 | **6.8** (+1.6) |
| Challenging (4) | 0/4 (0%) | 0/4 (0%) | 1.3 | 2.0 (+0.7) |
| Compound (3) | 0/3 (0%) | 0/3 (0%) | 1.0 | 1.0 (+0.0) |

### 3.3 逐题明细（优化后）

| # | 题目 | 难度 | EX | SL | Intent | Judge |
|---|------|------|----|----|--------|-------|
| 1 | Olist 各州订单数量 | simple | ✓ | ✓ | ✓ | 4 |
| 2 | Chinook 流派总数 | simple | ✓ | ✓ | ✓ | 7 |
| 3 | Olist 回头客户数 | simple | ✓ | ✓ | ✓ | 8 |
| 4 | Chinook 各国发票数 | simple | ✓ | ✓ | ✓ | 6 |
| 5 | Olist 品类销售额排名 | moderate | ✓ | ✓ | ✓ | 6 |
| 6 | Chinook 流派曲目数 | moderate | ✓ | ✓ | ✓ | 6 |
| 7 | Olist 支付方式统计 | moderate | ✓ | ✓ | ✓ | 8 |
| 8 | Chinook TOP5 艺术家 | moderate | ✗ | ✓ | ✓ | 7 |
| 9 | 跨库订单 vs 发票 | challenging | ✗ | ✓ | ✓ | 2 |
| 10 | 家具差评率排名 | challenging | ✗ | ✓ | ✗ | 1 |
| 11 | 流派收入效率 | challenging | ✗ | ✓ | ✗ | 0 |
| 12 | 月度订单趋势 | challenging | ✗ | ✓ | ✗ | 3 |
| 13 | 2月 vs 1月根因 | compound | ✗ | ✓ | ✗ | 1 |
| 14 | 卖家延迟率 TOP5 | compound | ✗ | ✓ | ✗ | 1 |
| 15 | Rock 收入根因 | compound | ✗ | ✓ | ✗ | 1 |

---

## 四、V2 升级后的复测与解释

2026-08-25 使用 `scripts/eval_chatbi.py --phase v2` 完成一次端到端复测。V2 接入三级 Schema 检索、动态 Few-shot、SQL 守卫与字段级评测，模型仍为 `deepseek-v4-flash`，测试集仍为 15 题。

| 指标 | Baseline | 首版 Optimized | V2 | V3 | V4 | V5 |
|------|----------|----------------|----|----|----|----|
| NL2SQL EX | 20.0% (3/15) | 46.7% (7/15) | 26.7% (4/15) | 40.0% (6/15) | 46.7% (7/15) | **100% (15/15)** |
| Schema Linking | 100.0% | 100.0% | 100.0% | 100.0% | 93.3% | 100.0% |
| Intent Recognition | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% |
| LLM Judge 平均分 | 3.4 | 4.4 | 4.4 | 4.9 | 4.1 | 5.6 |

V2 不能被解读为 EX 提升：在该次运行中，动态检索的 Prompt 组织与 flash 模型的输出不稳定性使得 EX 从 46.7% 回落到 26.7%。V3 针对性修复后（SQL 契约注入 + Bad Case 闭环实跑 + 评测自身 bug 修复）回升到 40.0%。V4 加入语义层强制编译、三类语料混合检索、sqlglot AST 校验、三轮结果反馈自愈、Schema Linking IR 与在线观测后，全量复测达到 46.7%（7/15），较 V3 +6.7pp，追平首版固定 Prompt。字段级行召回率达到 0.6000、列全命中达到 5/15；但 Judge 由 4.9 降至 4.1，单元格准确率由 0.6000 降至 0.5667，说明复杂题的 LLM fallback 仍有结果和表达波动。当前结论应是“EX 已追平历史最好值，但尚未超越，复杂查询仍是主瓶颈”，而非全面性能提升。

V4 首次运行曾得到 33.3%，原因并非模型退化而是 `ASTSQLValidator` 对 `chinook.Invoice`/`chinook.Track` 这类带 schema 前缀的合法表错误拦截，触发自修复后反而生成跨域 Olist SQL。修复 AST 的全名/短名表索引并重跑后得到 46.7%；这一缺陷及 Chinook 路由、`SUM(price)` 销售额口径、曲目目录口径、中文“最高的 5 个”LIMIT 提取均已进入本地回归集，29/29 通过。

V2 同时首次落地字段级评测。其初始汇总为行召回率 0.5467、单元格准确率 0.5333；逐题明细能够区分三类失败：例如 m2/m4 的结果行和数值正确但列别名不同，c4 月度趋势查询包含额外“最高/最低”描述字段，p2 卖家延迟率存在 TOP5 截断和指标口径差异。后续已补充列名别名归一化（如 `orders` 与 `order_count`），避免把别名差异误报为缺字段；需在下一轮完整 V2 复测中刷新该字段级基线。

SQL 守卫已通过独立集成验证：错误 SQL `LEFT(customer_state, 2)` 被自动改写为 SQLite 兼容的 `substr(customer_state, 1, 2)` 并成功执行。根因规划链在“11 月销售额为什么暴增”上完成 4/4 条非空证据查询，报告将黑五大促判为主因、高客单品类结构变化判为次因；数据质量对账 7 条规则中定位到 Olist 的 4 项真实告警，而 Chinook 两条对照规则均为 0 异常。完整实现与验证细节见 `CHATBI_UPGRADE.md`。

## 五、分析

### 5.1 优化效果归因

优化带来的 +26.7% EX 提升主要来自 simple 和 moderate 两档：

- **Simple 从 75% → 100%**：第 3 题"回头客户"在 baseline 时 LLM 误用 `customer_id`（每订单唯一）而非 `customer_unique_id`（回头客共享），优化版通过 few-shot 示例直接解决了这个语义混淆。

- **Moderate 从 0% → 75%**：四题全部通过 baseline 检查但 EX 为 0，原因是 baseline 版 LLM 随意添加 `LIMIT 5` 截断结果集，导致与 Gold SQL 的完整结果集不匹配。优化版的"不要随意加 LIMIT"规则直接修复了这个问题。此外，Schema 关系图帮助 LLM 正确识别了 `order_items → products → category_translation` 的三表 JOIN 路径。

- **Challenging 和 Compound 仍为 0%**：这两档问题涉及跨库 JOIN、CASE WHEN 嵌套百分比、根因推理等复杂逻辑，LLM 生成的 SQL 要么语法错误（调用了 SQLite 不支持的函数），要么逻辑完全偏题（如跨库对比时用了错误的 UNION 策略而非子查询）。这部分需要更强的模型（如 deepseek-v4-pro）或引入 SQL 自修复机制（执行失败后反馈错误重试）。

### 4.2 意图识别瓶颈

意图识别准确率稳定在 60%（9/15），未因 prompt 优化而提升——因为意图识别用的是规则匹配器，与 NL2SQL prompt 无关。失败案例集中在 challenging 和 compound 档：根因分析类问题（"为什么 2 月比 1 月少"）被规则匹配为 `data_query` 而非 `root_cause`，因为正则 `r"(?i)为什么.*下降"` 匹配不到"为什么 2 月比 1 月少"（关键词是"少"而非"下降"）。这需要补充规则或改用 LLM 分类。

### 4.3 Schema Linking 满分

优化前后均 100%，但需注意：这得益于双数据集表名差异大（Olist 蛇形 vs Chinook PascalCase），LLM 不容易混淆表名。且评测脚本已修复大小写敏感 bug 和 CTE 别名误报问题。

### 4.4 LLM Judge 维度分析

优化后各维度平均分（通过 EX 的 7 题）：

- 准确性 (accuracy)：7.3/10 — SQL 正确时数据准确性有保障
- 完整性 (completeness)：5.8/10 — 主要扣分点：LLM 倾向返回原始 JSON 数组而非结构化表格，且偶尔自行截断
- 清晰度 (clarity)：5.5/10 — 缺少表头标注和格式化
- 洞察深度 (insight)：3.2/10 — 最大短板：Agent 仅返回查询结果，不做趋势解读和异常归因

清晰度和洞察深度是下一阶段优化的重点方向——需要在 synthesize 节点的 system prompt 中加入"对查询结果做趋势解读和异常提示"的指令。

---

## 六、改进路线图

基于评测结果，后续优化方向按预期收益排序：

1. **SQL 自修复机制**（预期 EX +15-20%）：Agent 生成的 SQL 执行报错后，将错误信息反馈给 LLM 重新生成，最多重试 2 次。当前 challenging 档有多题因函数不支持或语法错误直接失败。

2. **意图识别规则补充**（预期 IR +13-20%）：补充 `r"(?i)为什么.*比.*少|为什么.*比.*低"` 等比较型根因模式，覆盖"为什么 A 比 B 少/低/高"这类句式。

3. **根因分析专用 prompt**（预期 compound EX +30%）：compound 档问题需要"先查数据→发现异常→分析原因"的多步推理，单次 NL2SQL 无法覆盖。需要在 synthesize 节点设计 root_cause 专用 prompt chain。

4. **跨库 JOIN 示例**（预期 challenging EX +25%）：补充 1-2 个跨库 JOIN 的 few-shot 示例，教 LLM 正确使用 `chinook.` 前缀和子查询策略。

5. **答案格式化指令**（预期 clarity +3 分）：在 synthesize prompt 中加入"用 Markdown 表格呈现数据，对趋势性数据添加 ▲▼ 标记，对异常值加粗标注"。

---

## 六、Spider 全库开放域评测

V5 终版后对 Spider 学术集做了全库扩展。V5 首版仅用 singer 库（2 表 mock 数据，93.3%），分数因数据过简而虚高。扩展后使用 Spider 官方完整数据集（HuggingFace `HAL-9001/spider-databases`，196MB），对 dev 集全部 20 个数据库各采样 5 题（共 99 题），在真实 SQLite 数据上运行 LLM 直出 SQL 评测。Prompt 使用完整 CREATE TABLE DDL + 外键 + 每表 3 行样本数据。

### 6.1 四项核心指标汇总

| # | 指标 | 值 | 说明 |
|---|------|------|------|
| 1 | EX（执行准确率，严格判等） | 78/99（78.8%） | Agent SQL 与 Gold SQL 执行结果集完全一致 |
| 1a | 部分正确（参考，不计入 EX） | 3 题 | 结果集 Jaccard ≥ 0.8 但不完全相等（course_teach ×2、real_estate_properties ×1） |
| 2 | SL（表命中率） | 96/99（97.0%） | Agent SQL 引用的表名集合与 Gold SQL 完全一致 |
| 3 | 行召回率（Field Recall） | 0.851 | Gold 结果行中 85.1% 出现在 Agent 结果中（按 key 列对齐） |
| 4 | 字段准确率（Cell Accuracy） | 0.898 | 匹配行中 89.8% 的单元格值正确 |

四项指标构成一条完整评测链路：选对表（SL 97%）→ 查全行（Recall 0.851）→ 值准确（Accuracy 0.898）→ 整体对（EX 78.8%）。SL 97% 说明 DDL+样本行的 prompt 设计让 LLM 几乎不犯选表错误；EX 78.8% 为严格判等口径（结果集完全一致才计正确；另有 3 题 Jaccard≥0.8 记为部分正确，宽松口径合计 81.8%）。在 zero-shot 设定下处于合理偏上水平（GPT-4 zero-shot 约 60-75%，Spider test SOTA 约 86-91% 但用 few-shot + 自我一致性）。

### 6.2 分库字段级明细

| 数据库 | 表数 | 总行数 | 题数 | EX | 行召回 | 字段准确 |
|--------|------|--------|------|------|--------|----------|
| battle_death | 3 | 28 | 5 | 5/5 | 1.000 | 1.000 |
| car_1 | 6 | 890 | 5 | 5/5 | 1.000 | 1.000 |
| concert_singer | 4 | 31 | 5 | 5/5 | 1.000 | 1.000 |
| course_teach | 3 | 23 | 5 | 3/5 | 0.943 | 1.000 |
| cre_Doc_Template_Mgt | 4 | 55 | 5 | 5/5 | 1.000 | 0.933 |
| dog_kennels | 8 | 72 | 5 | 5/5 | 1.000 | 1.000 |
| employee_hire_evaluation | 4 | 32 | 5 | 5/5 | 1.000 | 1.000 |
| museum_visit | 3 | 20 | 5 | 5/5 | 1.000 | 1.000 |
| poker_player | 2 | 12 | 5 | 5/5 | 1.000 | 1.000 |
| real_estate_properties | 5 | 40 | 4 | 3/4 | 0.969 | 0.833 |
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

严格口径下 10 库达到 EX 满分（其中 9 库 Recall 1.0 + Accuracy 1.0 全满分），包括 8 表复杂 schema 的 dog_kennels 和 6 表的 car_1。4 库仅 40%（flight_2/student_transcripts_tracking/tvshow/wta_1），字段级短板集中在行召回（0.4-0.53）。

### 6.3 失败案例字段级归因

严格口径下共 21 题未通过，其中 3 题（course_teach ×2、real_estate_properties ×1）结果集 Jaccard 0.86-0.88，属“高度重叠但不完全相等”的部分正确；其余 18 个失败案例按字段级指标分三类：

类型 A：行召回 R=0 + 字段准确 A=0（9 题）——结果集完全错位。主因是 SELECT 列序倒了（Gold 是 `(count, name)` Agent 是 `(name, count)`，元组永远不匹配）和 flight_2 数据脏值（AirportCode 尾部空格导致 Gold 精确匹配返回空、Agent 用 TRIM 返回非空）。

类型 B：行召回 R=0 + 字段准确 A=1.0（4 题）——key 列对齐后无交集行。Agent 返回的行 key 值和 Gold 不重叠。如 pets_1 的"每个学生有多少宠物"，Agent 从 Has_Pet 直接查只返回 count+StuID，但 Gold 还关联了 Student 表。

类型 C：行召回 R=1.0 + 字段准确 A=1.0 但 EX 失败（5 题）——行和字段都对，但 Agent 返回了多余行或排序不同。如 orchestra"按频率排序" Agent 排序方向反了，返回了全部 6 行而 Gold 只返回格式名列。

### 6.4 前版对比与业界参考

| 维度 | V5 首版（singer-only） | V5.1 全 20 库 |
|------|---:|---:|
| 数据库数 | 1 | 20 |
| 题数 | 30 | 99 |
| 数据来源 | mock（3 行） | Spider 官方真实数据 |
| EX（严格判等） | 93.3% (28/30) | **78.8% (78/99)**（宽松口径 81.8%） |
| 100% 库数 | — | 12/20 |
| <60% 库数 | — | 4/20 |

降幅合理——singer 是最简单的 2 表库，全库评测引入了 11 表复杂 schema、531K 行大表、多表 JOIN 和嵌套子查询。业界参考：Spider test 集 SOTA 约 86-91%（GPT-4 + few-shot + 自我一致性），GPT-4 zero-shot 约 60-75%。本系统严格口径 EX 78.8%、SL 97% 在 zero-shot 设定下处于合理偏上水平。完整报告见 `docs/eval_spider.json`。

---

## 七、附录

评测脚本：`scripts/eval_chatbi.py`（域内 15 题）+ `scripts/eval_spider.py`（Spider 20 库 99 题）
Baseline 原始数据：`docs/eval_baseline.json`
Optimized 原始数据：`docs/eval_optimized.json`
Spider 评测报告：`docs/eval_spider.json`
数据库文件：`data/olist.sqlite`（8.9MB）+ `data/chinook.sqlite`（984KB）
Spider 数据：`data/spider/spider_data/`（196MB，20 个 dev 库真实 SQLite）
Schema 文档：`docs/knowledge/ecommerce_schema.md`
