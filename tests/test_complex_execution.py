"""Complex-query latency fixes must preserve execution and safety contracts."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from core.task_routing import plan_only
from core.orchestrator import Orchestrator
from core.react_agent import ReActAgent, ToolExecutionUncertain
from tools.mcp_protocol import ToolDefinition


class RoutingTests(unittest.TestCase):
    def test_action_specific_prohibition_does_not_disable_reads(self):
        for q in ['查销售额，不要执行退款或修改订单','查询订单，只读分析，不要执行删除操作',
                  '先查询销售额，再解释规则，不要执行支付']:
            self.assertFalse(plan_only(q),q)
            self.assertEqual(Orchestrator._plan_decision({'query':q,'intent':'complex_task'}),'execute')

    def test_explicit_plan_only_never_executes(self):
        for q in ['只要计划，不要执行','仅给出方案','请先不要实际查询数据库','只生成执行步骤']:
            self.assertTrue(plan_only(q),q)
            self.assertEqual(Orchestrator._plan_decision({'query':q,'intent':'complex_task'}),'describe')

    def test_implicit_complex_task_skips_redundant_planner(self):
        agent=object.__new__(Orchestrator)
        self.assertEqual(agent._route_decision({'query':'分析今年各月销售额和订单量，找出最大降幅月份','intent':'complex_task'}),'react')

    def test_requested_execution_plan_is_preserved(self):
        agent=object.__new__(Orchestrator)
        self.assertEqual(agent._route_decision({'query':'制定分析计划并执行月度查询','intent':'complex_task'}),'plan')


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def agent(self,*decisions):
        agent=object.__new__(ReActAgent)
        agent.model=SimpleNamespace(ainvoke=AsyncMock(side_effect=[SimpleNamespace(content=json.dumps(d) if isinstance(d,dict) else d) for d in decisions]))
        return agent

    def tool(self,side_effect='read_only'):
        return ToolDefinition('query','query',{},side_effect=side_effect)

    def action(self,n):
        return {'action':'tool','tool':'query','args':{'attempt':n}}

    async def test_two_empty_results_stop_without_third_query(self):
        agent=self.agent(self.action(1),self.action(2),'没有匹配的数据，请确认时间范围。')
        execute=AsyncMock(return_value=json.dumps({'ok':True,'rows':[]}))
        result=await agent.run('q',[self.tool()],execute)
        self.assertEqual(execute.await_count,2)
        self.assertEqual(result['termination_reason'],'no_progress')
        self.assertEqual([r['outcome'] for r in result['tool_calls']],['empty','empty'])
        self.assertIn('attempt',agent.model.ainvoke.call_args.args[0][1][1])

    async def test_business_failure_is_not_a_successful_tool(self):
        agent=self.agent(self.action(1),{'action':'final','answer':'需要修正查询'})
        result=await agent.run('q',[self.tool()],AsyncMock(return_value='{"ok":false,"warnings":["invalid column"]}'))
        self.assertFalse(result['tool_calls'][0]['ok'])
        self.assertEqual(result['tool_calls'][0]['status'],'error')
        self.assertIn('invalid column',agent.model.ainvoke.call_args.args[0][1][1])

    async def test_unknown_side_effect_failure_still_requires_reconciliation(self):
        with self.assertRaises(ToolExecutionUncertain):
            await self.agent(self.action(1)).run('q',[self.tool('unknown')],AsyncMock(return_value='{"ok":false}'))

    async def test_real_progress_resets_stagnation(self):
        agent=self.agent(self.action(1),self.action(2),self.action(3),{'action':'final','answer':'done'})
        execute=AsyncMock(side_effect=['{"rows":[]}','{"rows":[[1]]}','{"rows":[]}'])
        result=await agent.run('q',[self.tool()],execute)
        self.assertEqual(result['termination_reason'],'final')
        self.assertEqual(execute.await_count,3)

    async def test_plan_is_budgeted_reference_not_system_instruction(self):
        agent=self.agent({'action':'final','answer':'done'})
        await agent.run('q',[],AsyncMock(),plan_steps=['先查月度数据','再下钻'])
        messages=agent.model.ainvoke.call_args.args[0]
        self.assertIn('先查月度数据',messages[1][1])
        self.assertNotIn('先查月度数据',messages[0][1])

    async def test_orchestrator_forwards_existing_plan(self):
        agent=object.__new__(Orchestrator)
        agent.react_agent=SimpleNamespace(run=AsyncMock(return_value={'answer':'done','tool_calls':[],'trace':[]}))
        agent.mcp_router=SimpleNamespace(client=SimpleNamespace(list_tools=AsyncMock(return_value=[])))
        agent._load_external_mcp_pool=AsyncMock()
        agent._emit=AsyncMock()
        agent.monitor=MagicMock()
        with patch.dict(os.environ,{'ORCHESTRATOR_COMPILED_WORKERS':'false'}), \
                patch('infra.graph_checkpointer.get_graph_checkpointer_manager',return_value=SimpleNamespace(saver=None)):
            await agent._react_node({'query':'q','messages':[],'plan_steps':['existing plan']})
        self.assertEqual(agent.react_agent.run.await_args.kwargs['plan_steps'],['existing plan'])


class SQLGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_metric_context_preserves_canonical_basis_and_filters(self):
        from chatbi.semantic_layer import get_query_metric_context
        text=get_query_metric_context('Olist 2026年各月销售额和订单量')
        self.assertIn('olist_order_items_dataset', text)
        self.assertIn('字段=price', text)
        self.assertIn('默认过滤=无', text)
        self.assertNotIn('avg_order_value（', text)
        self.assertEqual(get_query_metric_context('你好'), '')

    async def test_valid_empty_result_never_changes_filters(self):
        from chatbi.sql_guard import SQLGuardPipeline
        pipeline=object.__new__(SQLGuardPipeline)
        pipeline.max_repair_rounds=2
        pipeline.validator=SimpleNamespace(validate=lambda sql: [])
        pipeline.checker=SimpleNamespace(check=lambda *args: ['查询结果为空'])
        pipeline.repair_agent=SimpleNamespace(repair=AsyncMock())
        with patch('chatbi.sql_guard.execute_sql',return_value='{"columns":["x"],"rows":[]}') as execute:
            result=await pipeline.run("SELECT x FROM t WHERE year=2018",'q')
        self.assertTrue(result.ok)
        self.assertEqual(result.rows,[])
        self.assertIn('year=2018',result.sql)
        execute.assert_called_once()
        pipeline.repair_agent.repair.assert_not_awaited()


if __name__=='__main__':unittest.main()
