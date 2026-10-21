"""Real HTTP/SSE latency benchmark. No model or Agent execution is mocked.

Credentials are read from a private local JSON file, never recorded in reports.
Percentiles use nearest-rank; failures and timeouts stay in the denominator.
"""
from __future__ import annotations
import argparse
import asyncio
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx

ROOT=Path(__file__).resolve().parents[1]
CASES=[
 {'id':'simple_sql','query':'统计 Olist 的订单总数，请简短回答。','expected_nodes':['nl_compile']},
 {'id':'followup','seed':'统计 Olist 的订单总数，请简短回答。','query':'那订单量只看 delivered 状态呢？请简短回答。','expected_nodes':['nl_compile']},
 {'id':'rag','query':'解释星河演示电商知识库中的退款观察期定义，说明从哪一天开始计算，请简短回答。','expected_nodes':['rag_agent','retrieve_all']},
 {'id':'react','query':'查询 Olist 各州订单量排名，然后查询订单量第一名州的销售额，请简短回答。','expected_nodes':['react_loop']},
 {'id':'coordinator','query':'查询 Olist 订单量，同时查询各州销售额前三名，请简短回答。','expected_nodes':['coordinate']},
]

def percentile(values,p):
    if not values:return None
    if not 0<p<=1:raise ValueError('percentile must be in (0,1]')
    ordered=sorted(values)
    return ordered[max(0,math.ceil(len(ordered)*p)-1)]

def stats(values):
    values=[float(x) for x in values if x is not None]
    return {'n':len(values),'mean_ms':round(statistics.mean(values),3) if values else None,
      'p50_ms':round(percentile(values,.5),3) if values else None,
      'p95_ms':round(percentile(values,.95),3) if values else None,
      'min_ms':round(min(values),3) if values else None,'max_ms':round(max(values),3) if values else None,
      'stddev_ms':round(statistics.stdev(values),3) if len(values)>1 else None}

class SSEParser:
    def __init__(self):self.buffer=''
    def feed(self,text):
        self.buffer+=text
        events=[]
        while True:
            positions=[(self.buffer.find(sep),sep) for sep in ('\n\n','\r\n\r\n') if sep in self.buffer]
            if not positions:break
            pos,sep=min(positions,key=lambda x:x[0])
            frame,self.buffer=self.buffer[:pos],self.buffer[pos+len(sep):]
            data='\n'.join(line[5:].lstrip() for line in frame.splitlines() if line.startswith('data:'))
            if not data:continue
            try:event=json.loads(data)
            except ValueError:continue
            if isinstance(event,dict):events.append(event)
        if len(self.buffer)>2_000_000:raise ValueError('oversized_sse_frame')
        return events

def summarize(rows,wall_seconds):
    success=[r for r in rows if r['status']=='success']
    return {'requests':len(rows),'successful':len(success),'success_rate':len(success)/len(rows) if rows else None,
      'task_contract_passes':sum(bool(r.get('task_contract_pass')) for r in rows),
      'task_contract_rate':sum(bool(r.get('task_contract_pass')) for r in rows)/len(rows) if rows else None,
      'timeouts':sum(r['status']=='timeout' for r in rows),'errors':sum(r['status']=='error' for r in rows),
      'application_errors':sum(bool(r.get('application_error')) for r in rows),
      'invalid_environment':any(r.get('provider_failure') for r in rows),
      'all_attempt_elapsed':stats([r['total_ms'] for r in rows]),
      'successful_e2e':stats([r['total_ms'] for r in success]),
      'task_contract_pass_e2e':stats([r['total_ms'] for r in rows if r.get('task_contract_pass')]),
      'successful_first_answer':stats([r['first_answer_ms'] for r in success]),
      'successful_headers':stats([r['headers_ms'] for r in success]),
      'successful_server_reported':stats([r.get('server_ms') for r in success]),
      'route_contract_hits':sum(r.get('route_match',False) for r in rows),
      'route_contract_rate':sum(r.get('route_match',False) for r in rows)/len(rows) if rows else None,
      'wall_seconds':round(wall_seconds,3),'observed_requests_per_second':round(len(rows)/wall_seconds,5) if wall_seconds else None,
      'note':'success_rate means completed nonempty SSE response, not business correctness. task_contract_rate is a deterministic smoke check, not full answer accuracy. Closed-loop offered concurrency; first-answer is not first HTTP byte. Timeout duration is observed waiting, not completed latency.'}

async def request(client,credentials,case,session,timeout):
    start=time.perf_counter();row={'case_id':case['id'],'session_id':session,'status':'error',
      'first_answer_ms':None,'first_event_ms':None,'headers_ms':None,'trace_id':None,'server_ms':None}
    events=[];answer=[];spans={};nodes=[];finished=False
    try:
        async with asyncio.timeout(timeout):
            async with client.stream('POST',credentials['url']+'/chat/orchestrate/stream',
                headers={'Authorization':'Bearer '+credentials['token']},
                json={'query':case['query'],'session_id':session,'knowledge_base_id':'latency_fixture'}) as response:
                row['headers_ms']=(time.perf_counter()-start)*1000;row['http_status']=response.status_code
                row['trace_id']=response.headers.get('x-trace-id')
                response.raise_for_status();parser=SSEParser()
                async for text in response.aiter_text():
                    for event in parser.feed(text):
                        elapsed=(time.perf_counter()-start)*1000
                        if row['first_event_ms'] is None:row['first_event_ms']=elapsed
                        kind=event.get('type');events.append(event)
                        if kind=='answer' and str(event.get('content','')).strip():
                            if row['first_answer_ms'] is None:row['first_answer_ms']=elapsed
                            answer.append(event['content'])
                        if kind in {'node_start','node_complete'} and event.get('node'):
                            if event['node'] not in nodes:nodes.append(event['node'])
                        if kind=='intent':row['intent']=event.get('intent')
                        if kind=='error':row['sse_error']=True
                        if kind=='finished':finished=True;row['server_ms']=event.get('latency_ms')
                        for span in event.get('spans',[]):spans[span.get('spanId') or span.get('span_id') or str(len(spans))]=span
                row['status']='success' if finished and answer and not row.get('sse_error') else 'error'
                if not finished:row['error_type']='missing_terminal_event'
    except (TimeoutError,httpx.TimeoutException):row['status']='timeout';row['error_type']='request_deadline'
    except Exception as exc:row['error_type']=type(exc).__name__
    row['total_ms']=round((time.perf_counter()-start)*1000,3)
    row['nodes']=nodes;row['route_match']=any(n in nodes for n in case.get('expected_nodes',[]))
    row['answer']=''.join(answer);row['answer_chars']=len(row['answer']);row['spans']=list(spans.values())
    content=row['answer'].replace(',','').replace('，','')
    failure_text=json.dumps({'spans':row['spans'],'node_errors':[event.get('error') for event in events if event.get('error')]},ensure_ascii=False)
    row['provider_failure']=bool(re.search(r'insufficient.?balance|payment required|error code:\s*(?:401|402|403)',failure_text,re.I))
    row['application_error']=bool(row['provider_failure'] or any(event.get('error') for event in events if event.get('type')=='node_complete'))
    refused=any(word in content for word in ['无法给出','未返回任何','未能取得','还不确定你的需求','暂时无法回答','出现内部错误','未通过审核'])
    smoke=(('14747' in content) if case['id']=='simple_sql' else
           ('13430' in content) if case['id']=='followup' else
           (any(word in content for word in ['7天','七天','7 天']) and '签收' in content) if case['id']=='rag' else
           not refused)
    row['task_contract_pass']=bool(row['status']=='success' and row['route_match'] and smoke and not refused and not row['application_error'])
    row['events']=events
    return row

async def run(args):
    credentials=json.loads(args.credentials.read_text(encoding='utf8'))
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    selected=[c for c in CASES if not args.cases or c['id'] in args.cases.split(',')]
    rows=[];warmups=[];run_id=uuid.uuid4().hex[:10]
    async with httpx.AsyncClient(trust_env=False,timeout=httpx.Timeout(args.timeout,connect=10),limits=httpx.Limits(max_connections=12)) as client:
        ready=await client.get(credentials['url']+'/health/ready')
        if ready.status_code!=200:raise RuntimeError('benchmark_api_not_ready')
        if not args.skip_warmup:
            warm=await request(client,credentials,CASES[0],'bench_'+run_id+'_warm',args.timeout)
            warmups.append(warm)
            if warm.get('provider_failure'):
                (output/'invalid-environment.json').write_text(json.dumps({'status':'invalid_environment','reason':'model_billing_or_authentication_failure_during_preflight','completed_measurements':0}),encoding='utf8')
                raise RuntimeError('Model billing/authentication failure during preflight; no measured jobs started')
            if warm['status']!='success':
                (output/'preflight-failure.json').write_text(json.dumps(warm,ensure_ascii=False,indent=2),encoding='utf8')
                raise RuntimeError('real_request_preflight_failed; see preflight-failure.json')
        for concurrency in args.concurrency:
            jobs=[(case,i) for i in range(args.repeats) for case in selected]
            random.Random(20261018).shuffle(jobs)
            semaphore=asyncio.Semaphore(concurrency)
            phase_start=time.perf_counter()
            async def execute(case,repetition):
                async with semaphore:
                    session=f'bench_{run_id}_{concurrency}_{case["id"]}_{repetition}'
                    if case.get('seed'):
                        seed=await request(client,credentials,{**CASES[0],'query':case['seed']},session,args.timeout)
                        seed.update(phase='setup',concurrency=concurrency,repetition=repetition)
                        warmups.append(seed)
                        if seed['status']!='success':
                            seed={**seed,'case_id':case['id'],'status':'error','error_type':'followup_setup_failed'}
                            rows.append(seed);return
                    result=await request(client,credentials,case,session,args.timeout)
                    result.update(concurrency=concurrency,repetition=repetition,phase='measurement')
                    rows.append(result)
                    with (output/'requests.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                    if result.get('provider_failure'):
                        (output/'invalid-environment.json').write_text(json.dumps({'status':'invalid_environment','reason':'model_billing_or_authentication_failure','completed_measurements':len(rows),'failing_session':session}),encoding='utf8')
                        raise RuntimeError('Model billing/authentication failure; benchmark invalid, remaining jobs cancelled')
                    print(json.dumps({k:result.get(k) for k in ['case_id','concurrency','repetition','status','total_ms','first_answer_ms','nodes','route_match','error_type']},ensure_ascii=False),flush=True)
            await asyncio.gather(*(execute(case,i) for case,i in jobs))
            wall=time.perf_counter()-phase_start
            phase_rows=[r for r in rows if r.get('concurrency')==concurrency]
            report={'run_id':run_id,'time_local':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'concurrency':concurrency,'repeats_per_case':args.repeats,'client_timeout_seconds':args.timeout,
              'summary':summarize(phase_rows,wall),'by_scenario':{case['id']:summarize([r for r in phase_rows if r['case_id']==case['id']],wall) for case in selected},
              'setup_requests':len(warmups),'warmup_and_setup_excluded_from_latency':True,
              'throughput_includes_followup_setup_time':True,'case_definitions':selected,
              'client_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'Real model + real HTTP/SSE + PostgreSQL/Milvus; isolated synthetic knowledge and existing read-only demo business DBs. Not production load capacity or browser rendering.',
              'statistics':'nearest-rank P50/P95. All-attempt elapsed includes failures/timeouts; successful latency reported separately. Small per-scenario samples do not establish production p95.'}
            (output/f'concurrency-{concurrency}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (output/'setup-requests.json').write_text(json.dumps(warmups,ensure_ascii=False,indent=2),encoding='utf8')
    print('benchmark_completed',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--credentials',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--concurrency',type=int,nargs='+',default=[1,3])
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--cases',default='')
    parser.add_argument('--skip-warmup',action='store_true')
    args=parser.parse_args()
    if args.repeats<1 or any(c<1 or c>8 for c in args.concurrency):parser.error('invalid bounded load')
    asyncio.run(run(args))
if __name__=='__main__':main()
