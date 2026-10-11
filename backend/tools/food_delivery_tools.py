"""
外卖业务工具集 — EchoMind 企业级外卖客服 Agent

基于 Mock 数据模拟外卖平台业务系统，注册到 MCPToolServer 供 Agent 调用：
- query_order          查询外卖订单详情
- query_delivery       查询骑手位置/配送状态
- apply_refund         申请退款（敏感操作，需用户确认）
- query_coupon         查询红包/优惠券
- list_user_orders     列出用户订单

Mock 数据设计覆盖客服典型场景：
- 配送超时、骑手在途、已送达、待支付、退款处理中等状态
- 退款边界：可退（超时）、不可退（已食用超过时限）
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any

# ==================== Mock 数据 ====================

MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "WM20260823001": {
        "order_id": "WM20260823001",
        "user_id": 1,
        "user_name": "小明",
        "status": "delivering",  # 配送中
        "merchant": "川湘菜馆（望京店）",
        "items": [
            {"name": "辣椒炒肉盖饭", "quantity": 1, "price": 22.0},
            {"name": "冰镇酸梅汤", "quantity": 2, "price": 6.0},
        ],
        "total": 34.0,
        "packing_fee": 2.0,
        "delivery_fee": 3.5,
        "coupon_deduct": 5.0,
        "paid": 34.5,
        "created_at": "2026-08-23 12:02:11",
        "estimated_delivery": "2026-08-23 12:32:11",
        "courier": {"name": "王师傅", "phone": "138****5678"},
        "payment_method": "微信支付",
    },
    "WM20260822002": {
        "order_id": "WM20260822002",
        "user_id": 1,
        "user_name": "小明",
        "status": "timeout",  # 配送超时
        "merchant": "粤味烧腊（酒仙桥店）",
        "items": [
            {"name": "叉烧双拼饭", "quantity": 1, "price": 28.0},
        ],
        "total": 28.0,
        "packing_fee": 1.5,
        "delivery_fee": 3.0,
        "coupon_deduct": 0.0,
        "paid": 32.5,
        "created_at": "2026-08-22 18:31:05",
        "estimated_delivery": "2026-08-22 19:01:05",
        "actual_delivery": None,
        "delay_minutes": 25,  # 超时 25 分钟
        "courier": {"name": "李师傅", "phone": "139****2233"},
        "payment_method": "支付宝",
    },
    "WM20260821003": {
        "order_id": "WM20260821003",
        "user_id": 1,
        "user_name": "小明",
        "status": "completed",  # 已送达
        "merchant": "麦当劳（望京南店）",
        "items": [
            {"name": "巨无霸套餐", "quantity": 1, "price": 39.9},
            {"name": "麦辣鸡翅(2块)", "quantity": 1, "price": 13.5},
        ],
        "total": 53.4,
        "packing_fee": 0.0,
        "delivery_fee": 9.0,
        "coupon_deduct": 8.0,
        "paid": 54.4,
        "created_at": "2026-08-21 11:15:30",
        "estimated_delivery": "2026-08-21 11:45:30",
        "actual_delivery": "2026-08-21 11:42:08",
        "courier": {"name": "张师傅", "phone": "137****8899"},
        "payment_method": "微信支付",
    },
    "WM20260820004": {
        "order_id": "WM20260820004",
        "user_id": 1,
        "user_name": "小明",
        "status": "refunding",  # 退款处理中
        "merchant": "鲜果切（大山子店）",
        "items": [
            {"name": "现切西瓜盒(500g)", "quantity": 1, "price": 15.9},
            {"name": "阳光玫瑰葡萄盒", "quantity": 1, "price": 25.9},
        ],
        "total": 41.8,
        "packing_fee": 1.0,
        "delivery_fee": 4.0,
        "coupon_deduct": 3.0,
        "paid": 43.8,
        "created_at": "2026-08-20 15:22:47",
        "estimated_delivery": "2026-08-20 15:52:47",
        "actual_delivery": "2026-08-20 15:56:12",
        "refund": {
            "reason": "商品有质量问题（葡萄不新鲜）",
            "apply_at": "2026-08-20 16:30:00",
            "amount": 43.8,
            "status": "审核中（预计 24 小时内到账）",
        },
        "courier": {"name": "赵师傅", "phone": "136****4455"},
        "payment_method": "微信支付",
    },
}

MOCK_COUPONS: Dict[str, Dict[str, Any]] = {
    "HB001": {
        "coupon_id": "HB001",
        "user_id": 1,
        "type": "红包",
        "amount": 8.0,
        "min_order": 30.0,
        "valid_until": "2026-08-31",
        "status": "未使用",
        "scope": "全场通用",
    },
    "HB002": {
        "coupon_id": "HB002",
        "user_id": 1,
        "type": "会员红包",
        "amount": 12.0,
        "min_order": 50.0,
        "valid_until": "2026-08-25",
        "status": "未使用",
        "scope": "仅限会员商家",
    },
    "HB003": {
        "coupon_id": "HB003",
        "user_id": 1,
        "type": "配送费减免券",
        "amount": 5.0,
        "min_order": 20.0,
        "valid_until": "2026-08-23",
        "status": "已过期",
        "scope": "仅减免配送费",
    },
}

# 退款申请记录（敏感操作日志）
REFUND_LOG: list = []

STATUS_TEXT = {
    "delivering": "配送中",
    "timeout": "配送超时",
    "completed": "已送达",
    "refunding": "退款处理中",
    "pending_payment": "待支付",
    "canceled": "已取消",
}


# ==================== 工具实现 ====================

def query_order(order_id: str) -> str:
    """查询外卖订单详情"""
    order = MOCK_ORDERS.get(order_id.strip().upper())
    if not order:
        available = ", ".join(MOCK_ORDERS.keys())
        return json.dumps({
            "error": f"未找到订单 {order_id}",
            "hint": f"当前用户可用订单号: {available}",
        }, ensure_ascii=False)

    order["status_text"] = STATUS_TEXT.get(order["status"], order["status"])
    return json.dumps(order, ensure_ascii=False, indent=2)


def query_delivery(order_id: str) -> str:
    """查询骑手位置和配送状态"""
    order = MOCK_ORDERS.get(order_id.strip().upper())
    if not order:
        return json.dumps({"error": f"未找到订单 {order_id}"}, ensure_ascii=False)

    status = order["status"]
    courier = order.get("courier", {})
    base = {
        "order_id": order["order_id"],
        "status": status,
        "status_text": STATUS_TEXT.get(status, status),
        "courier": courier,
        "merchant": order["merchant"],
        "estimated_delivery": order.get("estimated_delivery"),
    }
    if status == "delivering":
        base["courier_position"] = "距离您约 1.2 公里，预计 8 分钟后送达"
        base["progress"] = "骑手已取餐，正在配送途中"
    elif status == "timeout":
        base["delay_minutes"] = order.get("delay_minutes", 0)
        base["progress"] = "订单已超时，平台正在催促骑手"
        base["compensation_hint"] = "超时 15 分钟以上可申请超时赔付（红包形式）"
    elif status == "completed":
        base["actual_delivery"] = order.get("actual_delivery")
        base["progress"] = "订单已送达完成"
    elif status == "refunding":
        base["progress"] = "订单已送达，当前有退款申请处理中"
        base["refund"] = order.get("refund")
    return json.dumps(base, ensure_ascii=False, indent=2)


def apply_refund(order_id: str, reason: str) -> str:
    """申请退款 — 敏感操作"""
    order = MOCK_ORDERS.get(order_id.strip().upper())
    if not order:
        return json.dumps({"error": f"未找到订单 {order_id}"}, ensure_ascii=False)

    if order["status"] == "refunding":
        return json.dumps({
            "error": "该订单已有退款申请正在处理中，请勿重复申请",
            "refund": order.get("refund"),
        }, ensure_ascii=False)

    if order["status"] == "delivering":
        return json.dumps({
            "error": "订单配送中，暂不支持退款。可等待送达后申请，或联系骑手协商拒收",
        }, ensure_ascii=False)

    # 已送达订单检查时效（签收后 24 小时内可退）
    if order["status"] == "completed":
        refund_record = {
            "order_id": order["order_id"],
            "reason": reason,
            "amount": order["paid"],
            "apply_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "已提交，预计 24 小时内审核",
            "eta": "审核通过后 1-3 个工作日原路退回",
        }
        REFUND_LOG.append(refund_record)
        order["status"] = "refunding"
        order["refund"] = refund_record
        return json.dumps({
            "success": True,
            "message": "退款申请已提交",
            "refund": refund_record,
        }, ensure_ascii=False)

    if order["status"] == "timeout":
        refund_record = {
            "order_id": order["order_id"],
            "reason": f"配送超时（{order.get('delay_minutes', 0)} 分钟）",
            "user_reason": reason,
            "amount": order["paid"],
            "apply_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "超时订单优先审核，预计 2 小时内处理",
            "extra_compensation": "符合超时赔付条件，将额外获得 8 元无门槛红包",
        }
        REFUND_LOG.append(refund_record)
        order["status"] = "refunding"
        order["refund"] = refund_record
        return json.dumps({
            "success": True,
            "message": "退款申请已提交（超时订单优先通道）",
            "refund": refund_record,
        }, ensure_ascii=False)

    return json.dumps({"error": f"订单状态 {order['status']} 不支持退款"}, ensure_ascii=False)


def query_coupon(user_id: int) -> str:
    """查询用户红包/优惠券"""
    coupons = [c for c in MOCK_COUPONS.values() if c["user_id"] == user_id]
    if not coupons:
        return json.dumps({"message": "当前没有可用红包"}, ensure_ascii=False)
    valid = [c for c in coupons if c["status"] == "未使用"]
    return json.dumps({
        "total": len(coupons),
        "valid_count": len(valid),
        "coupons": coupons,
    }, ensure_ascii=False, indent=2)


def list_user_orders(user_id: int) -> str:
    """列出用户所有订单"""
    orders = [o for o in MOCK_ORDERS.values() if o["user_id"] == user_id]
    if not orders:
        return json.dumps({"message": "没有查询到订单"}, ensure_ascii=False)
    summary = [{
        "order_id": o["order_id"],
        "merchant": o["merchant"],
        "status_text": STATUS_TEXT.get(o["status"], o["status"]),
        "total": o["total"],
        "created_at": o["created_at"],
    } for o in sorted(orders, key=lambda x: x["created_at"], reverse=True)]
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ==================== 注册到 MCP Server ====================

def register_food_delivery_tools(mcp_server) -> None:
    """把外卖工具注册到 MCPToolServer"""
    from tools.mcp_protocol import ToolDefinition

    async def _query_order(order_id: str) -> str:
        return query_order(order_id)

    async def _query_delivery(order_id: str) -> str:
        return query_delivery(order_id)

    async def _apply_refund(order_id: str, reason: str) -> str:
        return apply_refund(order_id, reason)

    async def _query_coupon(user_id: int = 1) -> str:
        return query_coupon(user_id)

    async def _list_user_orders(user_id: int = 1) -> str:
        return list_user_orders(user_id)

    tools = [
        ToolDefinition(
            name="query_order",
            description="根据订单号查询外卖订单详情：状态、商品、金额、骑手、支付方式等",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "外卖订单号，如 WM20260823001"},
                },
                "required": ["order_id"],
            },
            handler=_query_order,
            category="food_delivery",
        ),
        ToolDefinition(
            name="query_delivery",
            description="根据订单号查询骑手实时位置和配送进度，含超时赔付提示",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "外卖订单号"},
                },
                "required": ["order_id"],
            },
            handler=_query_delivery,
            category="food_delivery",
        ),
        ToolDefinition(
            name="apply_refund",
            description="为订单申请退款。敏感操作：必须先与用户确认订单信息和退款原因后才能调用",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "外卖订单号"},
                    "reason": {"type": "string", "description": "退款原因，需用户确认"},
                },
                "required": ["order_id", "reason"],
            },
            handler=_apply_refund,
            category="food_delivery",
        ),
        ToolDefinition(
            name="query_coupon",
            description="查询用户当前可用的红包和优惠券（金额、门槛、有效期）",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户ID", "default": 1},
                },
            },
            handler=_query_coupon,
            category="food_delivery",
        ),
        ToolDefinition(
            name="list_user_orders",
            description="列出用户的全部外卖订单（用户不知道订单号时先用这个查）",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户ID", "default": 1},
                },
            },
            handler=_list_user_orders,
            category="food_delivery",
        ),
    ]
    for tool in tools:
        mcp_server.register_tool(tool)
    return tools
