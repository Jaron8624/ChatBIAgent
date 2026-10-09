# NexusMind 功能验证指南

本文档说明如何验证 NexusMind 的核心代码与功能模块。离线回归测试不需要连接 LLM、Redis、Milvus 或 PostgreSQL。

---

## 前置条件

1. 后端服务已启动：`python backend/main.py`（默认监听 8000 端口）
2. 环境变量已配置（`.env` 文件）
3. 依赖已安装：`pip install -r requirements.txt`（确保包含 `requests`）

---

## 快速验证：代码编译与离线回归

Windows PowerShell：

```powershell
py -3.12 -m compileall -q backend tests scripts
py -3.12 -m unittest discover -s tests -v
```

macOS / Linux：

```bash
python3 -m compileall -q backend tests scripts
python3 -m unittest discover -s tests -v
```

当前离线回归共 74 项，覆盖语义编译、SQL 只读执行、Embedding 故障时的 BM25 回退、会话归属、身份权限、Harness 门禁和在线观测记录。后端 API OpenAPI 可生成 56 个路由。

## 快速验证：端到端功能测试

```bash
cd <项目根目录>
python cli.py test
```

输出示例：
```
============================================================
🧪 NexusMind 自动化功能测试
   API: http://localhost:8000
   用户: 1 | 角色: admin
============================================================

1. 服务健康检查...
   ✅ 后端服务在线

2. 意图识别测试...
   ✅ 意图识别

3. 对话编排测试...
   ✅ 对话编排

...（共 13 项）

📊 测试结果: 13 通过 / 0 失败 / 13 总计
============================================================
🎉 所有功能测试通过！
```

---

## 逐项验证

### 1. 交互式对话（核心功能）

**非流式对话：**
```bash
python cli.py chat
```
输入问题后按回车，查看 NexusMind 的回答。输入 `quit` 退出。

**流式对话（打字机效果）：**
```bash
python cli.py chat --stream
```

**恢复已有会话：**
```bash
python cli.py chat --session-id sess_1_abc12345
```

---

### 2. 意图识别（三路融合 + 加权投票）

```bash
python cli.py intent "帮我查一下上周的订单"
python cli.py intent "公司报销流程是什么"
python cli.py intent "hi 你好"
```

验证要点：
- 第一个应识别为 `order_query`（订单查询）
- 第二个应识别为 `knowledge_rag`（知识库检索）
- 第三个应识别为 `greeting`（问候）
- 置信度应 ≥ 0.6，否则返回澄清问题

---

### 3. 知识库管理

**上传文档：**
```bash
python cli.py kb upload ./docs/公司制度.pdf --knowledge-base-id 公司制度
```

**列出知识库：**
```bash
python cli.py kb list
```

**删除知识库：**
```bash
python cli.py kb delete --knowledge-base-id 公司制度
```

验证要点：
- 上传后能在知识库中检索到相关内容
- 对话中提问文档相关问题，能基于文档回答

---

### 4. Skills 管理

**列出 Skills（应包含 5 个预设）：**
```bash
python cli.py skills list
```

**创建自定义 Skill：**
```bash
python cli.py skills create \
  --name "onboarding_guide" \
  --description "新员工入职指南" \
  --prompt "请基于公司入职文档回答新员工问题" \
  --category "hr" \
  --tags "入职,员工,HR"
```

**查看 Skills 统计：**
```bash
python cli.py skills stats
```

验证要点：
- 预设 Skills 包含 rag_search、memory_recall、task_planning、code_review、data_analysis
- 创建后 Skills 能动态注入到对话上下文中

---

### 5. 权限控制（ACL）

后端默认只接受认证中间件写入的身份上下文。若要在本地演示前端身份选择器，可在 `.env` 中显式设置 `NEXUSMIND_ALLOW_HEADER_IDENTITY=true`；部署环境保持关闭，并由可信认证中间件提供身份。

**查看当前权限：**
```bash
python cli.py acl permissions
```

**查看审计日志：**
```bash
python cli.py acl audit-logs
```

**验证角色隔离：**
```bash
# 以 guest 角色访问（无权删除知识库）
CLI_USER_ROLE=guest python cli.py kb delete --knowledge-base-id 测试
# 预期：返回 403 权限拒绝
```

验证要点：
- admin 拥有全部权限
- guest 仅能对话，无法上传/删除文档
- 危险操作（删除知识库）需要二次确认

---

### 6. MCP 工具调用

**列出 MCP 工具：**
```bash
python cli.py mcp tools
```

**调用知识库检索工具：**
```bash
python cli.py mcp call --tool-name search_knowledge_base --arguments '{"query":"报销流程"}'
```

**调用记忆检索工具：**
```bash
python cli.py mcp call --tool-name get_memory --arguments '{"query":"我的偏好"}'
```

**列出外部 MCP Server：**
```bash
python cli.py mcp servers
```

验证要点：
- tools 列表应包含 search_knowledge_base 和 get_memory
- 调用成功返回 JSON 格式结果
- 外部 Server 配置后可通过 `mcp servers` 查看

---

### 7. Monitor 闭环监控

**查看监控指标：**
```bash
python cli.py monitor
```

**提交用户反馈（点赞/点踩）：**
```bash
python cli.py feedback --session-id test_session --satisfied true
python cli.py feedback --session-id test_session --satisfied false
```

验证要点：
- 经过多次对话后，指标中应出现各 Agent 的 p50/p90/p99 延迟和成功率
- 用户反馈后，满意度统计应更新

---

### 8. 端到端评测（LLM-as-Judge）

**评测单条对话：**
```bash
python cli.py evaluate \
  --query "公司报销流程是什么" \
  --response "报销流程如下：1. 填写报销单 2. 提交审批 3. 财务审核 4. 打款"
```

**查看评测汇总：**
```bash
python cli.py eval-summary
```

**查看 Bad Case（低分样本）：**
```bash
python cli.py eval-badcases
```

验证要点：
- 返回四个维度评分（Relevance/Accuracy/Completeness/Safety）
- overall_score ≥ 6.0 为及格
- 低分样本自动归档到 Bad Case 库

---

### 9. 自进化机制

**查看进化规则：**
```bash
python cli.py evolution rules
```

**审核规则（accept/merge/discard）：**
```bash
python cli.py evolution review --rule-id rule_abc123 --decision accept
```

**查看进化统计：**
```bash
python cli.py evolution stats
```

验证要点：
- 系统运行一段时间后，应产生 pending 状态的进化规则
- 接受后的规则自动注册为 Skill
- 高置信度规则（≥0.8）可通过 `auto-review` 自动接受

---

### 10. 会话恢复与上下文压缩

**列出会话：**
```bash
python cli.py session list
```

**恢复会话：**
```bash
python cli.py session resume --session-id sess_1_abc12345
```

**删除会话：**
```bash
python cli.py session delete --session-id sess_1_abc12345
```

验证要点：
- 对话后自动保存会话到 Redis
- 恢复会话后能看到完整对话历史
- 超长对话（Token > 8000）自动压缩旧消息为摘要

---

### 11. Plan Mode（任务规划）

```bash
python cli.py chat
# 输入：帮我制定一个完整的季度销售计划，包括目标设定、资源分配、执行步骤和复盘机制
```

验证要点：
- 包含"帮我/计划/步骤/流程"等关键词时触发 Plan Mode
- 输出应先给出执行计划（步骤列表），再按步骤回答
- 可在 Monitor 指标中看到 plan 节点的延迟记录

---

### 12. Explore 模式（广泛检索）

```bash
python cli.py chat
# 输入：调研一下市场上竞品的功能对比
```

验证要点：
- 开放式/调研型问题触发 ExploreAgent
- 检索范围扩大（top_k=10），覆盖更多知识源
- 输出综合多个来源的信息

---

## 故障排查

| 问题 | 排查方法 |
|------|---------|
| `无法连接到 http://localhost:8000` | 检查后端是否已启动：`python backend/main.py` |
| `权限拒绝 403` | 检查 `CLI_USER_ROLE` 环境变量，admin 拥有全部权限 |
| `MCP 工具调用失败` | 检查 Milvus/PostgreSQL 连接是否正常 |
| `意图识别置信度低` | 检查 `.env` 中 DASHSCOPE_API_KEY 是否配置正确 |
| `会话恢复为空` | 检查 Redis 连接，会话 TTL 为 24 小时 |

---

*文档版本：v1.1 | 最后更新：2026-09-24*
