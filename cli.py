#!/usr/bin/env python3
"""
NexusMind CLI — 命令行工具

功能：
1. 交互式对话（支持流式输出 + 会话恢复）
2. 知识库管理（上传/列出/删除）
3. 企业级功能测试（Skills/ACL/Evolution/Session/Monitor/MCP/Evaluation）
4. 自动化集成测试（一键验证所有功能）

用法：
    python cli.py chat                          # 交互式对话
    python cli.py chat --stream                 # 流式对话
    python cli.py chat --session-id xxx         # 恢复会话
    python cli.py kb upload ./doc.pdf           # 上传文档
    python cli.py test                          # 一键验证所有功能

配置：
    环境变量 API_BASE_URL 指定后端地址（默认 http://localhost:8000）
    环境变量 CLI_USER_ROLE 指定角色（默认 admin）
    环境变量 CLI_USER_ID 指定用户ID（默认 1）
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Optional, Dict, Any

# 默认配置
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
USER_ROLE = os.getenv("CLI_USER_ROLE", "admin")
USER_ID = int(os.getenv("CLI_USER_ID", "1"))

def headers():
    """获取请求头（包含用户身份）"""
    return {
        "Content-Type": "application/json",
        "X-User-Role": USER_ROLE,
        "X-User-Id": str(USER_ID),
    }

def get(path: str, params: Dict = None) -> Dict:
    """GET 请求"""
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, headers=headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"[错误] 无法连接到 {API_BASE}，请确保后端已启动")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] GET {path}: {e}")
        return {}

def post(path: str, payload: Dict = None) -> Dict:
    """POST 请求"""
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, headers=headers(), json=payload or {}, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"[错误] 无法连接到 {API_BASE}，请确保后端已启动")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] POST {path}: {e}")
        return {}

def delete(path: str) -> Dict:
    """DELETE 请求"""
    url = f"{API_BASE}{path}"
    try:
        r = requests.delete(url, headers=headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"[错误] 无法连接到 {API_BASE}，请确保后端已启动")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] DELETE {path}: {e}")
        return {}

# ==================== 交互式对话 ====================

def cmd_chat(args):
    """交互式对话"""
    print(f"\n🤖 NexusMind 命令行对话 (API: {API_BASE})")
    print(f"   用户ID: {USER_ID} | 角色: {USER_ROLE}")
    if args.session_id:
        print(f"   会话ID: {args.session_id}")
    print("   输入 'quit' 或 'exit' 退出\n")

    session_id = args.session_id or f"cli_{USER_ID}_{int(time.time())}"
    messages = []

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if args.stream:
            # 流式输出
            url = f"{API_BASE}/chat/orchestrate/stream"
            payload = {
                "query": user_input,
                "user_id": USER_ID,
                "session_id": session_id,
                "messages": messages,
            }
            try:
                r = requests.post(url, headers=headers(), json=payload, stream=True, timeout=120)
                print("🤖 NexusMind: ", end="", flush=True)
                for line in r.iter_lines():
                    if line:
                        text = line.decode("utf-8")
                        if text.startswith("data: "):
                            text = text[6:]
                        if text == "[DONE]":
                            break
                        try:
                            data = json.loads(text)
                            if "content" in data:
                                print(data["content"], end="", flush=True)
                            elif "final_response" in data:
                                print(data["final_response"], end="", flush=True)
                        except json.JSONDecodeError:
                            print(text, end="", flush=True)
                print()
            except Exception as e:
                print(f"[流式错误] {e}")
        else:
            # 非流式
            payload = {
                "query": user_input,
                "user_id": USER_ID,
                "session_id": session_id,
                "messages": messages,
            }
            result = post("/chat/orchestrate", payload)
            if result.get("success", True):
                response = result.get("final_response", "无响应")
                print(f"🤖 NexusMind: {response}")
            else:
                print(f"[错误] {result.get('error', '未知错误')}")

        messages.append({"role": "user", "content": user_input})
        # 简单保存最后一条回复到 messages（非流式时）
        if not args.stream and result.get("final_response"):
            messages.append({"role": "assistant", "content": result["final_response"]})

# ==================== 知识库管理 ====================

def cmd_kb_upload(args):
    """上传文档到知识库"""
    path = args.file
    if not os.path.exists(path):
        print(f"[错误] 文件不存在: {path}")
        return

    kb_id = args.knowledge_base_id or "默认知识库"
    url = f"{API_BASE}/document_upload"

    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "application/octet-stream")}
        data = {"knowledge_base_id": kb_id, "user_id": str(USER_ID)}
        try:
            r = requests.post(url, files=files, data=data, timeout=120)
            r.raise_for_status()
            result = r.json()
            print(f"✅ 上传成功: {result.get('message', 'OK')}")
            print(f"   文件哈希: {result.get('file_hash', 'N/A')}")
            print(f"   知识库ID: {result.get('knowledge_base_id', 'N/A')}")
        except Exception as e:
            print(f"[错误] 上传失败: {e}")

def cmd_kb_list(args):
    """列出知识库"""
    result = get("/knowledge-bases", params={"user_id": USER_ID})
    if result.get("success"):
        kbs = result.get("knowledge_bases", [])
        print(f"📚 知识库列表 ({len(kbs)} 个):")
        for kb in kbs:
            print(f"   - {kb}")
    else:
        print(f"[错误] {result.get('message', '查询失败')}")

def cmd_kb_delete(args):
    """删除知识库"""
    kb_id = args.knowledge_base_id
    if not kb_id:
        print("[错误] 请指定知识库ID: --knowledge-base-id")
        return
    result = delete(f"/knowledge-bases/{kb_id}")
    print(f"{'✅' if result.get('success') else '❌'} 删除知识库: {result.get('message', 'N/A')}")

# ==================== 意图识别测试 ====================

def cmd_intent(args):
    """测试意图识别"""
    query = args.query
    result = post("/intent/recognize", {"query": query, "session_id": "cli_test"})
    print(f"🎯 意图识别结果:")
    print(f"   查询: {query}")
    print(f"   意图: {result.get('intent', 'N/A')}")
    print(f"   置信度: {result.get('confidence', 'N/A')}")
    print(f"   权重: {json.dumps(result.get('weights', {}), ensure_ascii=False, indent=4)}")
    if result.get('clarifying_question'):
        print(f"   澄清问题: {result['clarifying_question']}")

# ==================== Skills 管理 ====================

def cmd_skills_list(args):
    """列出 Skills"""
    result = get("/skills")
    if result.get("success"):
        skills = result.get("skills", [])
        print(f"🛠️ Skills 列表 ({len(skills)} 个):")
        for s in skills:
            print(f"   - [{s.get('category', 'general')}] {s.get('name', 'N/A')}: {s.get('description', 'N/A')[:60]}")
    else:
        print(f"[错误] {result}")

def cmd_skills_create(args):
    """创建 Skill"""
    if not args.name or not args.prompt:
        print("[错误] 请提供 --name 和 --prompt")
        return
    payload = {
        "name": args.name,
        "description": args.description or args.name,
        "prompt_template": args.prompt,
        "category": args.category or "general",
        "tags": (args.tags or "").split(",") if args.tags else [],
    }
    result = post("/skills", payload)
    print(f"{'✅' if result.get('success') else '❌'} 创建 Skill: {result.get('skill_id', result.get('message', 'N/A'))}")

def cmd_skills_stats(args):
    """Skills 统计"""
    result = get("/skills/stats")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ==================== 自进化管理 ====================

def cmd_evolution_rules(args):
    """查看进化规则"""
    result = get("/evolution/rules")
    if result.get("success"):
        rules = result.get("rules", [])
        print(f"🧬 进化规则 ({len(rules)} 条):")
        for r in rules:
            print(f"   [{r.get('status', 'N/A')}] {r.get('rule_id', 'N/A')}: {r.get('trigger_pattern', 'N/A')[:50]}")
    else:
        print(f"[错误] {result}")

def cmd_evolution_review(args):
    """审核进化规则"""
    if not args.rule_id or not args.decision:
        print("[错误] 请提供 --rule-id 和 --decision (accept/merge/discard)")
        return
    result = post("/evolution/review", {"rule_id": args.rule_id, "decision": args.decision})
    print(f"{'✅' if result.get('success') else '❌'} {result.get('message', 'N/A')}")

def cmd_evolution_stats(args):
    """进化统计"""
    result = get("/evolution/stats")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ==================== 会话管理 ====================

def cmd_session_list(args):
    """列出会话"""
    result = get("/session/list")
    if result.get("success"):
        sessions = result.get("sessions", [])
        print(f"💬 会话列表 ({len(sessions)} 个):")
        for s in sessions:
            print(f"   - {s.get('session_id', 'N/A')}: {s.get('summary', 'N/A')[:50]} (tokens: {s.get('token_count', 'N/A')})")
    else:
        print(f"[错误] {result}")

def cmd_session_resume(args):
    """恢复会话"""
    if not args.session_id:
        print("[错误] 请提供 --session-id")
        return
    result = post("/session/resume", {"session_id": args.session_id})
    if result.get("success"):
        print(f"✅ 会话已恢复: {result.get('session_id', 'N/A')}")
        msgs = result.get("messages", [])
        print(f"   消息数: {len(msgs)}")
        for m in msgs[-5:]:
            print(f"   [{m.get('role', 'N/A')}] {m.get('content', 'N/A')[:80]}")
    else:
        print(f"[错误] {result.get('message', 'N/A')}")

def cmd_session_delete(args):
    """删除会话"""
    if not args.session_id:
        print("[错误] 请提供 --session-id")
        return
    result = delete(f"/session/{args.session_id}")
    print(f"{'✅' if result.get('success') else '❌'} {result.get('message', 'N/A')}")

# ==================== Monitor ====================

def cmd_monitor(args):
    """查看监控指标"""
    result = get("/monitor/metrics")
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_monitor_feedback(args):
    """提交反馈"""
    result = post("/monitor/feedback", {
        "session_id": args.session_id or f"cli_{int(time.time())}",
        "satisfied": args.satisfied,
    })
    print(f"{'✅' if result.get('success') else '❌'} {result.get('message', 'N/A')}")

# ==================== MCP 工具 ====================

def cmd_mcp_tools(args):
    """列出 MCP 工具"""
    result = get("/mcp/tools")
    if result.get("success"):
        tools = result.get("tools", [])
        print(f"🔧 MCP 工具 ({len(tools)} 个):")
        for t in tools:
            print(f"   - {t.get('name', 'N/A')}: {t.get('description', 'N/A')[:60]}")
    else:
        print(f"[错误] {result}")

def cmd_mcp_call(args):
    """调用 MCP 工具"""
    if not args.tool_name:
        print("[错误] 请提供 --tool-name")
        return
    payload = {
        "name": args.tool_name,
        "arguments": json.loads(args.arguments) if args.arguments else {},
    }
    result = post("/mcp/tools/call", payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_mcp_servers(args):
    """列出 MCP 外部 Server"""
    result = get("/mcp/servers")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ==================== 评测 ====================

def cmd_evaluate(args):
    """评测对话结果"""
    payload = {
        "query": args.query,
        "response": args.response,
        "context": args.context or "",
    }
    result = post("/evaluation/evaluate", payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_eval_summary(args):
    """评测汇总"""
    result = get("/evaluation/summary")
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_eval_badcases(args):
    """Bad Case 列表"""
    result = get("/evaluation/bad-cases")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ==================== ACL 权限 ====================

def cmd_acl_permissions(args):
    """查看权限"""
    result = get("/acl/permissions")
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_acl_audit_logs(args):
    """查看审计日志"""
    result = get("/acl/audit-logs")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ==================== 一键测试 ====================

def cmd_test(args):
    """自动化测试所有功能"""
    print("=" * 60)
    print("🧪 NexusMind 自动化功能测试")
    print(f"   API: {API_BASE}")
    print(f"   用户: {USER_ID} | 角色: {USER_ROLE}")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name: str, result: Dict, key: str = "success"):
        nonlocal passed, failed
        if result.get(key, False):
            print(f"   ✅ {name}")
            passed += 1
        else:
            print(f"   ❌ {name}: {result.get('message', result)}")
            failed += 1

    # 1. 健康检查
    print("\n1. 服务健康检查...")
    try:
        r = requests.get(f"{API_BASE}/docs", timeout=5)
        check("后端服务在线", {"success": r.status_code == 200}, "success")
    except Exception as e:
        print(f"   ❌ 后端服务离线: {e}")
        failed += 1
        return

    # 2. 意图识别
    print("\n2. 意图识别测试...")
    r = post("/intent/recognize", {"query": "帮我查一下上周的订单", "session_id": "test"})
    check("意图识别", r)

    # 3. 对话编排
    print("\n3. 对话编排测试...")
    r = post("/chat/orchestrate", {"query": "你好", "user_id": USER_ID, "session_id": "test"})
    check("对话编排", r)

    # 4. MCP 工具列表
    print("\n4. MCP 工具测试...")
    r = get("/mcp/tools")
    check("MCP 工具列表", r)

    # 5. Monitor 指标
    print("\n5. Monitor 监控测试...")
    r = get("/monitor/metrics")
    check("Monitor 指标", r)

    # 6. 用户反馈
    print("\n6. 用户反馈测试...")
    r = post("/monitor/feedback", {"session_id": "test", "satisfied": True})
    check("用户反馈", r)

    # 7. Skills 列表
    print("\n7. Skills 管理测试...")
    r = get("/skills")
    check("Skills 列表", r)

    # 8. 会话列表
    print("\n8. 会话管理测试...")
    r = get("/session/list")
    check("会话列表", r)

    # 9. ACL 权限
    print("\n9. ACL 权限测试...")
    r = get("/acl/permissions")
    check("ACL 权限查询", r)

    # 10. 审计日志
    print("\n10. 审计日志测试...")
    r = get("/acl/audit-logs")
    check("审计日志", r)

    # 11. 进化规则
    print("\n11. 自进化测试...")
    r = get("/evolution/rules")
    check("进化规则列表", r)

    # 12. 评测汇总
    print("\n12. 评测系统测试...")
    r = get("/evaluation/summary")
    check("评测汇总", r)

    # 13. 知识库列表（该接口返回 status 字段而非 success）
    print("\n13. 知识库测试...")
    r = get("/knowledge-bases", params={"user_id": USER_ID})
    check("知识库列表", r, key="status")

    # 14. 自进化闭环状态
    print("\n14. 自进化闭环测试...")
    r = get("/evolution/loop-status")
    check("进化闭环状态", r)

    # 15. 进化统计
    r = get("/evolution/stats")
    check("进化统计", r)

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed} 通过 / {failed} 失败 / {passed + failed} 总计")
    print("=" * 60)

    if failed > 0:
        print("⚠️ 部分功能未通过，请检查后端日志")
        sys.exit(1)
    else:
        print("🎉 所有功能测试通过！")

# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        prog="nexusmind",
        description="NexusMind 企业级智能知识助手 — 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py chat                           # 交互式对话
  python cli.py chat --stream                # 流式对话
  python cli.py test                         # 一键验证所有功能
  python cli.py kb upload ./report.pdf       # 上传文档
  python cli.py intent "帮我查订单"          # 测试意图识别
  python cli.py skills list                  # 列出 Skills
  python cli.py monitor                      # 查看监控指标
  python cli.py session list                 # 列出会话

环境变量:
  API_BASE_URL    后端地址 (默认: http://localhost:8000)
  CLI_USER_ROLE   用户角色 (默认: admin)
  CLI_USER_ID     用户ID (默认: 1)
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # chat
    chat_parser = subparsers.add_parser("chat", help="交互式对话")
    chat_parser.add_argument("--stream", action="store_true", help="流式输出")
    chat_parser.add_argument("--session-id", help="恢复指定会话")
    chat_parser.set_defaults(func=cmd_chat)

    # test
    test_parser = subparsers.add_parser("test", help="一键验证所有功能")
    test_parser.set_defaults(func=cmd_test)

    # kb
    kb_parser = subparsers.add_parser("kb", help="知识库管理")
    kb_sub = kb_parser.add_subparsers(dest="kb_cmd", help="知识库子命令")

    kb_upload = kb_sub.add_parser("upload", help="上传文档")
    kb_upload.add_argument("file", help="文件路径")
    kb_upload.add_argument("--knowledge-base-id", help="知识库ID")
    kb_upload.set_defaults(func=cmd_kb_upload)

    kb_list = kb_sub.add_parser("list", help="列出知识库")
    kb_list.set_defaults(func=cmd_kb_list)

    kb_delete = kb_sub.add_parser("delete", help="删除知识库")
    kb_delete.add_argument("--knowledge-base-id", required=True, help="知识库ID")
    kb_delete.set_defaults(func=cmd_kb_delete)

    # intent
    intent_parser = subparsers.add_parser("intent", help="测试意图识别")
    intent_parser.add_argument("query", help="查询文本")
    intent_parser.set_defaults(func=cmd_intent)

    # skills
    skills_parser = subparsers.add_parser("skills", help="Skills 管理")
    skills_sub = skills_parser.add_subparsers(dest="skills_cmd", help="Skills 子命令")

    skills_list = skills_sub.add_parser("list", help="列出 Skills")
    skills_list.set_defaults(func=cmd_skills_list)

    skills_create = skills_sub.add_parser("create", help="创建 Skill")
    skills_create.add_argument("--name", required=True, help="Skill 名称")
    skills_create.add_argument("--description", help="描述")
    skills_create.add_argument("--prompt", required=True, help="Prompt 模板")
    skills_create.add_argument("--category", default="general", help="分类")
    skills_create.add_argument("--tags", help="标签（逗号分隔）")
    skills_create.set_defaults(func=cmd_skills_create)

    skills_stats = skills_sub.add_parser("stats", help="Skills 统计")
    skills_stats.set_defaults(func=cmd_skills_stats)

    # evolution
    evo_parser = subparsers.add_parser("evolution", help="自进化管理")
    evo_sub = evo_parser.add_subparsers(dest="evo_cmd", help="自进化子命令")

    evo_rules = evo_sub.add_parser("rules", help="查看进化规则")
    evo_rules.set_defaults(func=cmd_evolution_rules)

    evo_review = evo_sub.add_parser("review", help="审核规则")
    evo_review.add_argument("--rule-id", required=True, help="规则ID")
    evo_review.add_argument("--decision", required=True, choices=["accept", "merge", "discard"], help="决策")
    evo_review.set_defaults(func=cmd_evolution_review)

    evo_stats = evo_sub.add_parser("stats", help="进化统计")
    evo_stats.set_defaults(func=cmd_evolution_stats)

    # session
    session_parser = subparsers.add_parser("session", help="会话管理")
    session_sub = session_parser.add_subparsers(dest="session_cmd", help="会话子命令")

    session_list = session_sub.add_parser("list", help="列出会话")
    session_list.set_defaults(func=cmd_session_list)

    session_resume = session_sub.add_parser("resume", help="恢复会话")
    session_resume.add_argument("--session-id", required=True, help="会话ID")
    session_resume.set_defaults(func=cmd_session_resume)

    session_delete = session_sub.add_parser("delete", help="删除会话")
    session_delete.add_argument("--session-id", required=True, help="会话ID")
    session_delete.set_defaults(func=cmd_session_delete)

    # monitor
    monitor_parser = subparsers.add_parser("monitor", help="查看监控指标")
    monitor_parser.set_defaults(func=cmd_monitor)

    monitor_feedback = subparsers.add_parser("feedback", help="提交用户反馈")
    monitor_feedback.add_argument("--session-id", help="会话ID")
    monitor_feedback.add_argument("--satisfied", type=bool, default=True, help="是否满意")
    monitor_feedback.set_defaults(func=cmd_monitor_feedback)

    # mcp
    mcp_parser = subparsers.add_parser("mcp", help="MCP 工具管理")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_cmd", help="MCP 子命令")

    mcp_tools = mcp_sub.add_parser("tools", help="列出 MCP 工具")
    mcp_tools.set_defaults(func=cmd_mcp_tools)

    mcp_call = mcp_sub.add_parser("call", help="调用 MCP 工具")
    mcp_call.add_argument("--tool-name", required=True, help="工具名")
    mcp_call.add_argument("--arguments", help="参数（JSON 字符串）")
    mcp_call.set_defaults(func=cmd_mcp_call)

    mcp_servers = mcp_sub.add_parser("servers", help="列出 MCP 外部 Server")
    mcp_servers.set_defaults(func=cmd_mcp_servers)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="评测对话结果")
    eval_parser.add_argument("--query", required=True, help="用户问题")
    eval_parser.add_argument("--response", required=True, help="AI 回答")
    eval_parser.add_argument("--context", help="上下文")
    eval_parser.set_defaults(func=cmd_evaluate)

    eval_summary = subparsers.add_parser("eval-summary", help="评测汇总")
    eval_summary.set_defaults(func=cmd_eval_summary)

    eval_bad = subparsers.add_parser("eval-badcases", help="Bad Case 列表")
    eval_bad.set_defaults(func=cmd_eval_badcases)

    # acl
    acl_parser = subparsers.add_parser("acl", help="权限管理")
    acl_sub = acl_parser.add_subparsers(dest="acl_cmd", help="ACL 子命令")

    acl_perm = acl_sub.add_parser("permissions", help="查看权限")
    acl_perm.set_defaults(func=cmd_acl_permissions)

    acl_audit = acl_sub.add_parser("audit-logs", help="查看审计日志")
    acl_audit.set_defaults(func=cmd_acl_audit_logs)

    args = parser.parse_args()

    if not args.command:
        # 无子命令时默认进入 chat 模式，像 `cc` 一样直接对话
        class DefaultArgs:
            stream = False
            session_id = None
        cmd_chat(DefaultArgs())
        return

    args.func(args)

if __name__ == "__main__":
    main()
