import importlib.util
from pathlib import Path
import unittest
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock
spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).resolve().parents[1]/'scripts/benchmark_e2e.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class BenchmarkContracts(unittest.TestCase):
    def test_nearest_rank_small_sample(self):self.assertEqual(b.percentile([0,0,15],.95),15)
    def test_median_and_empty(self):
        self.assertEqual(b.percentile([4,1,3,2],.5),2)
        self.assertIsNone(b.percentile([],.95))
    def test_progress_not_answer(self):
        parser=b.SSEParser()
        self.assertEqual(parser.feed('data: {"type":"node_start"}\n\n')[0]['type'],'node_start')
    def test_fragmented_crlf_and_utf8(self):
        parser=b.SSEParser();self.assertEqual(parser.feed('data: {"type":"ans'),[])
        self.assertEqual(parser.feed('wer","content":"测试"}\r\n\r\n')[0]['content'],'测试')
    def test_failures_stay_in_denominator(self):
        row=dict(status='success',total_ms=10,first_answer_ms=5,headers_ms=1,route_match=True)
        summary=b.summarize([row,{**row,'status':'timeout','total_ms':100}],1)
        self.assertEqual(summary['success_rate'],.5)
        self.assertEqual(summary['all_attempt_elapsed']['mean_ms'],55)
        self.assertEqual(summary['successful_e2e']['mean_ms'],10)
class ApplicationFailureContracts(unittest.IsolatedAsyncioTestCase):
    async def test_fast_error_answer_is_not_a_successful_task(self):
        events=[{'type':'node_complete','node':'react_loop','error':'Error code: 402 - Insufficient Balance'},
                {'type':'answer','content':'抱歉，处理您的问题时出现内部错误，请稍后再试。'},
                {'type':'trace','spans':[]},
                {'type':'finished'}]
        async def chunks():
            for event in events:yield 'data: '+json.dumps(event)+'\n\n'
        response=SimpleNamespace(status_code=200,headers={},raise_for_status=lambda:None,aiter_text=chunks)
        context=AsyncMock();context.__aenter__.return_value=response
        client=SimpleNamespace(stream=MagicMock(return_value=context))
        row=await b.request(client,{'url':'https://example.test','token':'fixture'},b.CASES[3],'session',10)
        self.assertEqual(row['status'],'success')  # transport completed
        self.assertFalse(row['task_contract_pass'])
        self.assertTrue(row['provider_failure'])
        self.assertTrue(b.summarize([row],1)['invalid_environment'])

if __name__=='__main__':unittest.main()
