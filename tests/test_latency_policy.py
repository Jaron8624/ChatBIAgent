"""Offline contracts; real-model timing is measured separately through HTTP."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core.execution_policy import scalar_count_request, verified_count_answer, review_before_publish
from core.orchestrator import Orchestrator, ReflectionAgent


def fixture():
    return ({'compiled':True,'ir':{'metrics':['order_count'],'dimensions':[],'filters':{}}},
            SimpleNamespace(ok=True,warnings=[],repaired=False,rows=[[14747]],columns=['order_count']))


class CountPolicy(unittest.TestCase):
    def test_closed_grammar(self):
        for query in ['统计 Olist 的订单总数，请简短回答。','那订单量只看 delivered 状态呢？请简短回答。']:
            self.assertTrue(scalar_count_request(query),query)

    def test_unsupported_requests_never_take_shortcut(self):
        for q in ['订单总数大于100的州','订单总数然后查销售额','解释订单量含义','统计最近7天订单量','订单总数不要执行权限检查','查询另一个租户订单总数','统计订单量并分析原因']:
            self.assertFalse(scalar_count_request(q),q)

    def test_uses_guarded_result_and_preserves_filter(self):
        routed,guard=fixture()
        self.assertIn('14747',verified_count_answer('统计订单总数',routed,guard))
        routed['ir']['filters']={'order_status':'delivered'};guard.rows=[[13430]]
        self.assertIn('delivered',verified_count_answer('订单量只看delivered状态',routed,guard))

    def test_bad_or_repaired_results_require_full_path(self):
        for field,value in [('ok',False),('warnings',['null warning']),('repaired',True),('rows',[[None]]),('rows',[[-1]]),('rows',[[float('nan')]]),('rows',[[1],[2]]),('rows',[[1.5]])]:
            routed,guard=fixture();setattr(guard,field,value)
            self.assertIsNone(verified_count_answer('统计订单总数',routed,guard),(field,value))

    def test_extra_filters_and_dimensions_are_not_dropped(self):
        for field,value in [('filters',{'purchase_year':'2025'}),('dimensions',['customer_state']),('metrics',['sales_revenue'])]:
            routed,guard=fixture();routed['ir'][field]=value
            self.assertIsNone(verified_count_answer('统计订单总数',routed,guard))


class Routing(unittest.TestCase):
    def setUp(self):
        self.agent=Orchestrator.__new__(Orchestrator)
        self.agent.monitor=SimpleNamespace(get_performance_route=lambda:None)

    def test_complex_label_cannot_hide_executable_sequence(self):
        self.assertEqual(self.agent._route_decision({'query':'查询各州订单量排名，然后查询第一名州的销售额','intent':'complex_task'}),'react')

    def test_independent_requests_stay_parallel(self):
        self.assertEqual(self.agent._route_decision({'query':'查询订单量，同时查询各州销售额前三名','intent':'complex_task'}),'coordinate')

    def test_knowledge_uses_only_knowledge(self):
        with patch.dict(os.environ,{'ORCHESTRATOR_TARGETED_RETRIEVAL':'true'}):
            self.assertEqual(self.agent._route_decision({'query':'解释退款观察期定义','intent':'knowledge_rag'}),'rag')

    def test_memory_uses_only_memory(self):
        self.assertEqual(self.agent._route_decision({'query':'我的偏好是什么','intent':'memory_query'}),'memory')

    def test_explicit_plan_only_never_executes(self):
        self.assertEqual(Orchestrator._plan_decision({'query':'只要计划，不要执行','intent':'complex_task'}),'describe')
        self.assertEqual(Orchestrator._plan_decision({'query':'查询订单再查销售额','intent':'complex_task'}),'execute')

    def test_rollback_flag_restores_aggregate_retrieval(self):
        with patch.dict(os.environ,{'ORCHESTRATOR_TARGETED_RETRIEVAL':'false'}):
            self.assertEqual(self.agent._route_decision({'query':'解释退款观察期定义','intent':'knowledge_rag'}),'retrieve_all')


class Review(unittest.IsolatedAsyncioTestCase):
    async def test_only_exact_verified_answer_skips_judge(self):
        agent=Orchestrator.__new__(Orchestrator);agent._emit=AsyncMock()
        agent.monitor=SimpleNamespace(record=lambda *a,**k:None)
        agent.reflection_agent=SimpleNamespace(process=AsyncMock(return_value={'reflection_passed':True}))
        await agent._reflection_node({'verified_answer':'42','final_response':'42'})
        agent.reflection_agent.process.assert_not_awaited()
        await agent._reflection_node({'verified_answer':'42','final_response':'42 plus an unsupported claim'})
        agent.reflection_agent.process.assert_awaited_once()

    async def test_review_failure_does_not_publish_buffered_claim(self):
        agent=Orchestrator.__new__(Orchestrator);agent._emit=AsyncMock()
        agent.monitor=SimpleNamespace(record=lambda *a,**k:None)
        agent.reflection_agent=SimpleNamespace(process=AsyncMock(return_value={'reflection_passed':False,'reflection_feedback':'unsupported'}))
        result=await agent._reflection_node({'answer_buffered':True,'final_response':'refund approved'})
        self.assertTrue(result['requires_human'])
        self.assertNotIn('refund approved',str(agent._emit.call_args_list))

    async def test_judge_outage_fails_closed(self):
        judge=ReflectionAgent.__new__(ReflectionAgent)
        judge.model=SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError('down')))
        self.assertFalse((await judge.process({'query':'退款','final_response':'approved'}))['reflection_passed'])

    def test_risky_content_requires_prepublication_review(self):
        self.assertTrue(review_before_publish('解释退款规则'))
        self.assertFalse(review_before_publish('统计订单数量'))


class RerankLatency(unittest.IsolatedAsyncioTestCase):
    async def test_missing_url_does_not_sleep_or_send(self):
        from rag.documents_process import rerank_documents
        with patch.dict(os.environ,{'RERANK_URL':'','RERANK_API_KEY':'fixture-only'}), patch('rag.documents_process.asyncio.sleep',new_callable=AsyncMock) as sleep, patch('rag.documents_process.httpx.AsyncClient') as client:
            self.assertIsNone(await rerank_documents('question',['doc']))
            sleep.assert_not_awaited();client.assert_not_called()

    async def test_permanent_auth_error_does_not_retry(self):
        from rag.documents_process import rerank_documents
        context=AsyncMock();context.__aenter__.return_value=context
        context.post.return_value=SimpleNamespace(status_code=401)
        with patch.dict(os.environ,{'RERANK_URL':'https://example.test/rerank','RERANK_API_KEY':'fixture-only'}), patch('rag.documents_process.asyncio.sleep',new_callable=AsyncMock) as sleep, patch('rag.documents_process.httpx.AsyncClient',return_value=context):
            self.assertIsNone(await rerank_documents('question',['doc']))
            sleep.assert_not_awaited();context.post.assert_awaited_once()

    async def test_transient_failure_keeps_bounded_retry(self):
        from rag.documents_process import rerank_documents
        context=AsyncMock();context.__aenter__.return_value=context
        context.post.side_effect=[SimpleNamespace(status_code=503),SimpleNamespace(status_code=200,json=lambda:{'ok':True})]
        with patch.dict(os.environ,{'RERANK_URL':'https://example.test/rerank','RERANK_API_KEY':'fixture-only'}), patch('rag.documents_process.asyncio.sleep',new_callable=AsyncMock) as sleep, patch('rag.documents_process.httpx.AsyncClient',return_value=context):
            self.assertEqual(await rerank_documents('question',['doc']),{'ok':True})
            self.assertEqual(context.post.await_count,2);sleep.assert_awaited_once()


class SQLThreadContext(unittest.IsolatedAsyncioTestCase):
    async def test_guard_offload_keeps_authenticated_business_context(self):
        from chatbi.sql_guard import SQLGuardPipeline
        from governance.context import BusinessContext,business_context,current
        marker=BusinessContext(tenant='fixture-tenant',user_id=31,domain='commerce')
        guard=SQLGuardPipeline.__new__(SQLGuardPipeline);guard.max_repair_rounds=0
        def validate(sql):
            self.assertIs(current(),marker)
            return []
        def execute(sql):
            self.assertIs(current(),marker)
            return '{"columns":["count"],"rows":[[2]]}'
        guard.validator=SimpleNamespace(validate=validate)
        guard.checker=SimpleNamespace(check=lambda *args:[])
        with business_context(marker),patch('chatbi.sql_guard.execute_sql',side_effect=execute):
            self.assertTrue((await guard.run('SELECT COUNT(*) FROM orders')).ok)


if __name__=='__main__':unittest.main()
