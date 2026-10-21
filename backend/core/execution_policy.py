"""Conservative per-request performance choices; never bypass access or SQL guards."""
import math
import os
import re


def enabled(name):
    return os.getenv(name, 'true').lower() in {'1', 'true', 'yes'}


def scalar_count_request(query):
    """Only a closed grammar, not a broad substring classifier, may skip models."""
    q=re.sub(r'[\s，,。？！?!、]', '', str(query or ''))
    q=re.sub(r'请(?:简短|简洁)回答', '', q)
    q=re.sub(r'^那|呢$', '', q)
    return bool(re.fullmatch(
        r'(?:统计|查询|查看|计算)?(?:Olist的?)?订单(?:总数|数量|量)(?:只看(?:delivered|canceled|shipped|approved)状态)?'
        r'|(?:统计|查询|查看|计算)?(?:Olist的?)?(?:delivered|canceled|shipped|approved)(?:状态)?的?订单(?:总数|数量|量)', q, re.I))


def verified_count_answer(query, routed, guarded):
    ir=routed.get('ir', {})
    if not scalar_count_request(query) or not routed.get('compiled'):
        return None
    if ir.get('metrics')!=['order_count'] or ir.get('dimensions') or ir.get('composite'):
        return None
    if not guarded.ok or guarded.warnings or guarded.repaired or len(guarded.rows)!=1 or len(guarded.columns)!=1:
        return None
    row=guarded.rows[0]
    if len(row)!=1 or isinstance(row[0],bool) or not isinstance(row[0],(int,float)):
        return None
    value=row[0]
    if not math.isfinite(value) or value<0 or value!=int(value):
        return None
    status=ir.get('filters',{}).get('order_status')
    mentioned=re.findall(r'\b(delivered|canceled|shipped|approved)\b',query,re.I)
    if mentioned and (len(set(s.lower() for s in mentioned))!=1 or status!=mentioned[0].lower()):
        return None
    # Do not silently render filters not covered by this response template.
    if set(ir.get('filters',{}))-{'order_status'} or isinstance(status,list):
        return None
    qualifier=f'，状态为 {status}' if status else ''
    return f'Olist 订单总数为 {int(value)} 笔{qualifier}。'


def review_before_publish(query):
    return enabled('ORCHESTRATOR_SELECTIVE_REVIEW') and bool(re.search(
        r'退款|支付|转账|删除|个人信息|隐私|执行操作|修改订单', str(query or '')))
