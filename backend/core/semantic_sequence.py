"""Execute a bounded, governed rank-then-measure dependency without LLM guesses.

Only two independently compilable Olist metrics grouped by customer state are
eligible. All other tasks keep the general ReAct executor. Data, identities and
SQL are never supplied by the model to this path; the second binding comes from
the first executed result. Read-only calls retain durable intent and results.
"""
import copy
from datetime import datetime, timezone
import json
import re
import time
import uuid

from chatbi.semantic_layer import ALL_METRICS
from chatbi.text2sql_pipeline import SemanticCompilerRouter
from core.react_agent import ToolEvidencePersistenceError, ToolExecutionUncertain
from core.task_routing import describe_task
from infra.async_io import run_blocking


async def run_sequence(query, execute, load_progress=None, save_progress=None):
    task=describe_task(query)
    if task.dependency!='sequential' or len(task.steps)!=2:
        return None
    router=SemanticCompilerRouter()
    first=await run_blocking(router.compile_if_supported,task.steps[0])
    plan=first.get('metric_plan') or {}
    if not first.get('compiled') or plan.get('dimensions')!=['customer_state'] or len(plan.get('metrics',[]))!=1:
        return None
    metric=plan['metrics'][0]
    definition=ALL_METRICS[metric]
    if definition.dataset!='olist' or plan.get('sort')!={'by':metric,'order':'DESC'} or plan.get('limit')==1 or plan.get('filters'):
        # Need the runner-up too, so tied first places cannot be hidden.
        return None
    reference=re.compile(r'(?:'+re.escape(definition.display_name)+r')?(?:排名)?(?:第一名|榜首)(?:的)?州')
    if not reference.search(task.steps[1]):return None
    second_template=reference.sub('{state}州',task.steps[1])
    if '{state}' not in second_template:return None
    probe=await run_blocking(router.compile_if_supported,second_template.replace('{state}','ZZ'))
    second_plan=probe.get('metric_plan') or {}
    if not probe.get('compiled') or len(second_plan.get('metrics',[]))!=1 or second_plan.get('filters',{}).get('customer_state')!='ZZ':
        return None
    second_metric=second_plan['metrics'][0]
    if ALL_METRICS[second_metric].dataset!='olist':return None
    if second_plan.get('dimensions') not in ([],['customer_state']):return None

    progress=await load_progress() if load_progress else None
    if progress and progress.get('mode')!='semantic_sequence':return None
    if progress and progress.get('query')!=query:
        raise ToolExecutionUncertain('Sequence checkpoint belongs to another query')
    calls=copy.deepcopy((progress or {}).get('tool_calls',[]))
    if any(c.get('status') in {'started','uncertain'} for c in calls):
        raise ToolExecutionUncertain('Sequence has an unconfirmed tool outcome')

    async def persist():
        if save_progress:
            await save_progress({'mode':'semantic_sequence','query':query,'tool_calls':copy.deepcopy(calls),'next_iteration':len(calls)+1})

    async def run_step(compiled):
        signature=compiled['sql']
        cached=next((c for c in calls if c.get('signature')==signature and c.get('ok')),None)
        if cached:
            from chatbi.sql_guard import ValidationResult
            payload=json.loads(cached['result'])
            result=ValidationResult(ok=payload['ok'],sql=payload['sql'],rows=payload['rows'],columns=payload['columns'],warnings=payload['warnings'],repaired=payload.get('repaired',False))
            if not result.ok or result.repaired or result.warnings:
                raise ValueError('Cached dependent query needs review')
            return result
        event={'call_id':uuid.uuid4().hex,'tool':'execute_sql_guarded','args':{'sql':signature,'query':query},'signature':signature,'side_effect':'read_only','status':'started','ok':False,'started_at':datetime.now(timezone.utc).isoformat()}
        calls.append(event)
        await persist()
        started=time.perf_counter()
        try:
            result=await execute(signature,query)
            event.update(status='success' if result.ok else 'error',ok=result.ok,result=result.to_json())
        except Exception as exc:
            event.update(status='error',error=type(exc).__name__)
            await persist()
            raise
        event.update(duration_ms=(time.perf_counter()-started)*1000,finished_at=datetime.now(timezone.utc).isoformat())
        await persist()
        if not result.ok or result.repaired or result.warnings:
            raise ValueError('Dependent query failed guarded execution or needs review')
        return result

    def finished(answer,reason='final'):
        return {'answer':answer,'iterations':len(calls),'tool_calls':calls,'trace':[], 'termination_reason':reason,'semantic_sequence':True,'forced_final':False}

    first_result=await run_step(first)
    dimension_index=first_result.columns.index('customer_state')
    metric_index=first_result.columns.index(metric)
    if len(first_result.rows)<2:
        return finished('排名结果不足以核对唯一第一名，请补充查询范围。','needs_clarification')
    if first_result.rows[0][metric_index]==first_result.rows[1][metric_index]:
        return finished('查询结果存在并列第一名，请明确需要统计哪个州，或是否分别统计。','needs_clarification')
    state=str(first_result.rows[0][dimension_index])
    if not re.fullmatch('[A-Z]{2}',state):
        raise ValueError('Unexpected state binding in executed result')
    second_query=second_template.replace('{state}',state)
    second=await run_blocking(router.compile_if_supported,second_query)
    if not second.get('compiled') or (second.get('metric_plan') or {}).get('filters',{}).get('customer_state')!=state:
        raise ValueError('Dependent state binding was not preserved by compilation')
    second_result=await run_step(second)
    if len(second_result.rows)!=1:
        raise ValueError('Dependent measure must produce one row for the bound state')
    amount=second_result.rows[0][second_result.columns.index(second_metric)]
    rank='、'.join(f'{row[dimension_index]} {row[metric_index]}' for row in first_result.rows[:5])
    answer=f'各州{definition.display_name}排名前五为 {rank}。\n第一名为 {state} 州，{definition.display_name}为 {first_result.rows[0][metric_index]}；该州{ALL_METRICS[second_metric].display_name}为 {amount}。'
    return finished(answer)
