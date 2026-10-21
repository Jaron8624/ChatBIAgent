import copy
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core.semantic_sequence import run_sequence
from core.react_agent import ToolExecutionUncertain
from chatbi.sql_guard import ValidationResult


class SequenceTests(unittest.IsolatedAsyncioTestCase):
    query='查询各州订单量排名，然后查询订单量第一名州的销售额'

    def router(self):
        def compile(q):
            if '订单量排名' in q:
                return {'compiled':True,'sql':'RANK_QUERY','metric_plan':{'metrics':['order_count'],'dimensions':['customer_state'],'filters':{},'sort':{'by':'order_count','order':'DESC'}}}
            state=re.search(r'([A-Z]{2})州',q).group(1)
            return {'compiled':True,'sql':f'MEASURE_{state}','metric_plan':{'metrics':['sales_revenue'],'dimensions':['customer_state'],'filters':{'customer_state':state}}}
        return SimpleNamespace(compile_if_supported=compile)

    def executor(self,tie=False):
        async def execute(sql,q):
            return ValidationResult(ok=True,sql=sql,columns=['customer_state','order_count'] if sql=='RANK_QUERY' else ['customer_state','sales_revenue'],rows=[['AA',9],['BB',9 if tie else 8]] if sql=='RANK_QUERY' else [['AA',12.34]])
        return AsyncMock(side_effect=execute)

    async def test_second_query_binds_executed_first_result(self):
        execute=self.executor();save=AsyncMock()
        with patch('core.semantic_sequence.SemanticCompilerRouter',return_value=self.router()):
            result=await run_sequence(self.query,execute,save_progress=save)
        self.assertIn('AA',result['answer']);self.assertIn('12.34',result['answer'])
        self.assertEqual([c.args[0] for c in execute.await_args_list],['RANK_QUERY','MEASURE_AA'])
        self.assertEqual(len(result['tool_calls']),2)
        self.assertEqual(save.await_count,4)

    async def test_durable_complete_steps_are_not_reexecuted(self):
        execute=self.executor();saved=None
        async def persist(value):
            nonlocal saved
            saved=copy.deepcopy(value)
        with patch('core.semantic_sequence.SemanticCompilerRouter',return_value=self.router()):
            first=await run_sequence(self.query,execute,save_progress=persist)
            execute.reset_mock()
            second=await run_sequence(self.query,execute,load_progress=AsyncMock(return_value=saved),save_progress=persist)
        self.assertEqual(first['answer'],second['answer']);execute.assert_not_awaited()

    async def test_tied_first_place_requests_clarification(self):
        execute=self.executor(tie=True)
        with patch('core.semantic_sequence.SemanticCompilerRouter',return_value=self.router()):
            result=await run_sequence(self.query,execute)
        self.assertEqual(result['termination_reason'],'needs_clarification');self.assertIn('并列',result['answer']);execute.assert_awaited_once()

    async def test_uncertain_checkpoint_blocks_replay(self):
        execute=self.executor()
        progress={'query':self.query,'mode':'semantic_sequence','tool_calls':[{'status':'started'}]}
        with patch('core.semantic_sequence.SemanticCompilerRouter',return_value=self.router()),self.assertRaises(ToolExecutionUncertain):
            await run_sequence(self.query,execute,load_progress=AsyncMock(return_value=progress))
        execute.assert_not_awaited()

    async def test_general_task_keeps_react(self):
        execute=self.executor()
        self.assertIsNone(await run_sequence('解释退款规则',execute))
        execute.assert_not_awaited()

    async def test_cached_warning_is_not_approved_on_resume(self):
        execute=self.executor()
        cached=ValidationResult(ok=True,sql='RANK_QUERY',rows=[['AA',9],['BB',8]],columns=['customer_state','order_count'],warnings=['review required'])
        progress={'query':self.query,'mode':'semantic_sequence','tool_calls':[{'status':'success','signature':'RANK_QUERY','ok':True,'result':cached.to_json()}]}
        with patch('core.semantic_sequence.SemanticCompilerRouter',return_value=self.router()),self.assertRaises(ValueError):
            await run_sequence(self.query,execute,load_progress=AsyncMock(return_value=progress))
        execute.assert_not_awaited()

    async def test_legacy_checkpoint_stays_on_original_executor(self):
        execute=self.executor()
        with patch('core.semantic_sequence.SemanticCompilerRouter',return_value=self.router()):
            self.assertIsNone(await run_sequence(self.query,execute,load_progress=AsyncMock(return_value={'query':self.query,'tool_calls':[]})))
        execute.assert_not_awaited()


if __name__=='__main__':unittest.main()
