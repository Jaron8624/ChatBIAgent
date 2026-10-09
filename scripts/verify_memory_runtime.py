"""Opt-in real-service memory checks and paired retrieval ablations.

Uses configured model credentials without exporting them. Creates an isolated
PostgreSQL schema and Milvus collection; never writes production memory rows.
Reports contain synthetic test conversations, manifests and sample-level scores.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
import asyncpg
from pymilvus import AsyncMilvusClient, AnnSearchRequest, RRFRanker, MilvusClient
from langchain_core.messages import HumanMessage
from infra import postgresql_client as pg_module
from infra.llm_provider import create_chat_model
from infra.llm_usage import UsageTracker
from memory import auto_store_memory_from_psql as ingestion
from memory.ingestion_ledger import IngestionLedger, memory_write_lock
from memory.memory_manager import MemoryManager, memory_scope_user_id
from memory.management import edit_memory, MemoryConflict
from memory.context_loader import HEADER, load_memory_context
from memory.policy import cosine
from memory.session_recovery import SessionManager
from rag.embedding_provider import get_embeddings
from rag import milvus_client as milvus_module

logging.basicConfig(level=logging.WARNING)


class CachedEmbeddings:
    """Reuse vectors returned by the real API across paired variants."""
    def __init__(self, actual):
        self.actual = actual
        self.values = {}
        self.api_batches = 0

    async def aembed_documents(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.values))
        if missing:
            vectors = await self.actual.aembed_documents(missing)
            if len(vectors) != len(missing):
                raise RuntimeError('embedding_count_mismatch')
            self.api_batches += 1
            self.values.update(zip(missing, vectors))
        return [self.values[t] for t in texts]

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]


def manifest():
    files = ['backend/memory/memory_manager.py', 'backend/memory/policy.py',
             'backend/memory/ingestion_ledger.py', 'backend/memory/auto_store_memory_from_psql.py',
             'backend/memory/session_recovery.py', 'scripts/verify_memory_runtime.py']
    files += ['backend/rag/embedding_provider.py', 'backend/infra/config.py',
              'backend/memory/context_loader.py', 'backend/memory/management.py']
    return {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'source_hashes': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        'dependencies': {p: importlib.metadata.version(p) for p in
                         ['pymilvus', 'asyncpg', 'langchain-core', 'langchain-openai', 'openai', 'redis']},
        'model': os.getenv('AGENT_BASE_MODEL'), 'embedding_model': os.getenv('EMBEDDING_MODEL'),
        'temperature': 0, 'minimum_relevance': float(os.getenv('MEMORY_MIN_RELEVANCE', '.35')),
        'evidence': 'synthetic conversations; real model and embedding APIs; real local databases',
    }


async def main():
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    schema = 'memory_check_' + run_id.lower()
    collection = schema
    folder = ROOT / 'artifacts' / 'memory_runtime' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {'run_id': run_id, 'manifest': manifest(), 'schema': schema, 'collection': collection,
              'checks': [], 'ablations': [], 'status': 'running'}

    def save():
        (folder/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    def check(name, ok, **details):
        report['checks'].append(dict(name=name, passed=bool(ok), **details))
        save()
        print(json.dumps({'check': name, 'passed': bool(ok)}, ensure_ascii=False), flush=True)
        if not ok:
            raise AssertionError(name)

    pool = client = admin = None
    try:
        embeddings = CachedEmbeddings(get_embeddings())
        probe = await embeddings.aembed_query('验证记忆向量接口')
        check('real_embedding_endpoint', len(probe)>0, dimension=len(probe))
        admin = await asyncpg.connect(os.environ['DATABASE_URL'], timeout=15)
        report['postgres_version'] = await admin.fetchval('SELECT version()')
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        pool = await asyncpg.create_pool(os.environ['DATABASE_URL'], min_size=2, max_size=8,
                                        server_settings={'search_path': schema, 'timezone': 'UTC'})
        async with pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE users(user_id INT PRIMARY KEY,user_profile TEXT);
                CREATE TABLE raw_conversations(id VARCHAR(50) PRIMARY KEY,user_id INT NOT NULL,
                    tenant_id VARCHAR(160) NOT NULL,thread_id VARCHAR(50) NOT NULL,turn_id VARCHAR(64),
                    role VARCHAR(10) NOT NULL,content TEXT NOT NULL,summary_id VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                CREATE UNIQUE INDEX test_turn_identity ON raw_conversations(tenant_id,user_id,thread_id,turn_id,role)
                    WHERE turn_id IS NOT NULL;
                CREATE TABLE nexusmind_session_registry(session_id VARCHAR(128) PRIMARY KEY,
                    user_id INT NOT NULL,tenant_id VARCHAR(160) NOT NULL,status VARCHAR(16) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP);
            ''')
            await IngestionLedger(conn).ensure_schema()
        check('postgres_isolated_schema', True)
        sync = MilvusClient(uri=os.getenv('Milvus_url'), token=os.getenv('Token', 'root:Milvus'))
        report['milvus_version'] = sync.get_server_version()
        sync.close()
        client = AsyncMilvusClient(uri=os.getenv('Milvus_url'), token=os.getenv('Token', 'root:Milvus'))
        manager = MemoryManager(client, embeddings, collection, len(probe))
        await manager.init_collection()
        check('milvus_real_schema_and_indexes', True)
        pg = object.__new__(pg_module.PostgreSQLParentClient)
        pg.pool = pool

        async def get_pg():
            return pg

        async def get_milvus():
            return manager

        actual_model = create_chat_model(temperature=0, max_retries=1, timeout=60)
        report['model_outputs'] = []
        class RecordingModel:
            async def ainvoke(self, messages):
                response = await actual_model.ainvoke(messages)
                report['model_outputs'].append(str(response.content))
                save()
                return response
        model = RecordingModel()
        tracker = UsageTracker()
        # Redirect only destination resources. All storage/extraction methods are real.
        with patch.object(ingestion, 'get_postgresql_client', get_pg), \
             patch.object(ingestion, 'get_milvus_client', get_milvus), \
             patch.object(pg_module, 'get_postgresql_client', get_pg), \
             patch.object(milvus_module, 'get_milvus_client', get_milvus):
            uid, tenant = 910001, 'memory-verification'
            thread = 'check_' + uuid.uuid4().hex[:16]
            await pg.register_session_owner(thread, uid, tenant)
            await pg.add_conversation_turn(uid, tenant, thread, 'turn1',
                '请记住，我的报表默认按周汇总，金额使用人民币，报告用中文。', '已记录你的报表偏好。')
            with tracker:
                first = await ingestion.extract_and_append_memory([], model, uid, thread, tenant)
            check('real_model_extraction_and_commit', first.get('success'), result=first)
            await pg.add_conversation_turn(uid, tenant, thread, 'turn2',
                '以后改成按月汇总，不再按周汇总。金额仍然使用人民币，报告仍然用中文。', '已改为按月汇总。')
            with tracker:
                second = await ingestion.extract_and_append_memory([], model, uid, thread, tenant)
            check('real_model_preference_update', second.get('success'), result=second)
            scope = memory_scope_user_id(uid, tenant)
            records = await client.query(collection_name=collection, filter=f'user_id == {scope}',
                                         output_fields=['*'], limit=1000, consistency_level='Strong')
            report['extracted_memories'] = [{k:v for k,v in r.items() if k not in {'vector','sparse_vector'}} for r in records]
            replacements = [r for r in records if r.get('status')=='superseded']
            check('model_keys_replace_old_preference', bool(replacements),
                  retired_count=len(replacements), active_keys=[r.get('memory_key') for r in records if r.get('status')=='active'])

            # Inject interruption after real vector insertion, before source commit.
            retry_thread = 'retry_' + uuid.uuid4().hex[:16]
            await pg.register_session_owner(retry_thread, uid, tenant)
            await pg.add_conversation_turn(uid, tenant, retry_thread, 'retry1',
                '请记住，导出的文件使用CSV格式，不需要Excel格式。', '已记录导出格式。')
            original_ack = IngestionLedger.acknowledge
            async def interrupted(*args, **kwargs):
                raise RuntimeError('injected_after_vector_before_commit')
            with tracker, patch.object(IngestionLedger, 'acknowledge', interrupted):
                interrupted_result = await ingestion.extract_and_append_memory([], model, uid, retry_thread, tenant)
            before = await client.query(collection_name=collection, filter=f'user_id == {scope} and thread_id == "{retry_thread}"',
                                        output_fields=['id'], limit=1000, consistency_level='Strong')
            async with pool.acquire() as conn:
                unacked = await conn.fetchval('SELECT count(*) FROM raw_conversations WHERE thread_id=$1 AND summary_id IS NULL', retry_thread)
                pending = await IngestionLedger(conn).pending(uid, tenant)
            check('failure_keeps_sources_and_frozen_batch', not interrupted_result.get('success') and unacked==2 and bool(pending))
            # A fresh worker must not call the extraction model again for replay.
            async def forbidden(*args, **kwargs):
                raise AssertionError('frozen replay attempted another extraction')
            with patch.object(ingestion, 'extract_memories', forbidden):
                resumed = await ingestion.extract_and_append_memory([], model, uid, retry_thread, tenant)
            after = await client.query(collection_name=collection, filter=f'user_id == {scope} and thread_id == "{retry_thread}"',
                                       output_fields=['id'], limit=1000, consistency_level='Strong')
            check('fresh_worker_replay_no_duplicate_vectors', resumed.get('success') and len(before)==len(after), vector_count=len(after))
            async with memory_write_lock(pool, uid, tenant):
                try:
                    async with memory_write_lock(pool, uid, tenant):
                        raise AssertionError('second writer entered')
                except RuntimeError as error:
                    check('real_postgres_concurrent_writer_rejected', str(error)=='memory_scope_busy')
            sessions = SessionManager()
            await sessions.save_session(retry_thread, uid, [{'role':'user','content':'synthetic test'}], tenant_id=tenant)
            deleted = await sessions.delete_session(retry_thread, uid, tenant)
            async with pool.acquire() as conn:
                batch_rows = await conn.fetch('SELECT status,payload FROM nexusmind_memory_batches WHERE thread_id=$1', retry_thread)
                raw_count = await conn.fetchval('SELECT count(*) FROM raw_conversations WHERE thread_id=$1', retry_thread)
            leftovers = await client.query(collection_name=collection, filter=f'user_id == {scope} and thread_id == "{retry_thread}"',
                                           output_fields=['id'], limit=10, consistency_level='Strong')
            check('real_session_delete_removes_raw_vectors_batch_payload', deleted and raw_count==0 and not leftovers
                  and all(r['status']=='cancelled' and json.loads(r['payload'])=={} for r in batch_rows))
            with patch.object(ingestion, 'extract_memories', forbidden):
                blocked = await ingestion.extract_and_append_memory([], model, uid, retry_thread, tenant)
            check('deleted_session_cannot_be_recreated_by_worker', not blocked.get('success'))

            # Same collection, different user/tenant contains no extracted memory.
            isolated = await manager.hybrid_retrieval_memories('报表按月还是按周汇总', uid, 0, 3, 2, 1, 'other-tenant')
            check('real_tenant_filter', not any(isolated.values()))

            await retirement_retry(manager, client, check)

            await retrieval_ablation(manager, embeddings, report, check)
            await managed_memory_checks(pg, manager, client, check)
        report['model_usage'] = tracker.snapshot()
        if not report['model_usage']['cost_complete']:
            report['model_usage']['estimated_cost'] = None
        report['embedding_api_batches'] = embeddings.api_batches
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        # Do not serialize arbitrary network exceptions that might contain secrets.
        report['error_type'] = type(error).__name__
        print('FAILED: ' + type(error).__name__, flush=True)
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        if client:
            await client.close()
        if pool:
            await pool.close()
        if admin:
            await admin.close()
        print('REPORT: ' + str(folder/'report.json'), flush=True)


async def retrieval_ablation(manager, embeddings, report, check):
    uid, tenant, thread = 910002, 'memory-retrieval', 'ablation_' + uuid.uuid4().hex[:12]
    facts = [
        ('report_period', '用户的经营报表默认按月汇总，不按周汇总。'),
        ('currency', '用户的销售金额统一以人民币元展示，不换算成美元。'),
        ('export', '用户导出数据时选择CSV文件，不需要Excel文件。'),
        ('language', '用户要求报告和分析结论使用中文书写。'),
        ('timezone', '用户的数据统计时区采用北京时间，即Asia/Shanghai。'),
        ('refund', '用户计算销售额时排除已退款订单，只统计有效成交订单。'),
        ('region', '用户负责华东区域，包括上海、江苏和浙江。'),
        ('inventory', '用户要求库存预警阈值是剩余10件以下。'),
        ('latte', '用户喝咖啡时喜欢拿铁，不加糖。'),
        ('travel', '用户下一次旅行计划去云南大理，计划坐高铁。'),
    ]
    cases = [
        ('我的经营报表按什么周期统计？','report_period'), ('经营数据默认是每周还是每月汇总？','report_period'),
        ('销售金额应该用哪种货币？','currency'), ('给我输出销售额时金额单位是什么？','currency'),
        ('我希望下载什么格式的数据文件？','export'), ('查到结果后默认导出CSV还是Excel？','export'),
        ('报告应该用什么语言写？','language'), ('我的分析结论要用中文还是英文？','language'),
        ('统计跨日订单时使用哪个时区？','timezone'), ('每天的统计日期按北京时间吗？','timezone'),
        ('退款订单应该计入销售额吗？','refund'), ('我算成交金额时怎么处理已退款订单？','refund'),
        ('我负责的区域包含哪些省市？','region'), ('上海江苏浙江属于我负责的范围吗？','region'),
        ('库存剩多少件需要提醒我？','inventory'), ('我的库存预警线是多少件？','inventory'),
        ('我喝咖啡要加糖吗？','latte'), ('我喜欢什么咖啡？','latte'),
        ('我下一次准备去哪里旅行？','travel'), ('去大理的旅行打算用什么交通方式？','travel'),
        ('如何修复Docker启动失败？',None), ('Python异步任务怎么取消？',None),
        ('Milvus的HNSW索引参数应该怎样设置？',None), ('LangGraph如何实现图执行恢复？',None),
        ('量子计算中的纠缠是什么意思？',None), ('帮我解释HTTP和HTTPS的区别。',None),
        ('如何配置GitHub分支保护？',None), ('给我介绍Linux内核的调度算法。',None),
    ]
    report['retrieval_dataset'] = {'facts':facts, 'queries':cases,
        'hash':hashlib.sha256(json.dumps([facts,cases],ensure_ascii=False).encode()).hexdigest(),
        'provenance':'hand-authored synthetic diagnostic dataset; not production traffic'}
    await embeddings.aembed_documents([t for _,t in facts] + [q for q,_ in cases])
    plan = {'semantic_memory':[dict(content=t, memory_key='preference.'+k, evidence_kind='user_quote',
                importance_score=1 if k in {'latte','travel'} else .5) for k,t in facts]}
    await manager.add_memories_batch(uid, thread, plan, batch_id='ablation-dataset', tenant_id=tenant)
    scope = memory_scope_user_id(uid, tenant)
    expr = f'user_id == {scope} and memory_type == "semantic"'
    before_ids = {r['id'] for r in await manager.client.query(collection_name=manager.collection_name,
                  filter=expr,output_fields=['id'],limit=100,consistency_level='Strong')}
    variants = {'legacy_rrf_scoring':[], 'cosine_no_threshold':[], 'current_threshold_035':[],
                'optimized_threshold_040':[], 'dense_only_035':[]}
    for query, expected in cases:
        vector = await embeddings.aembed_query(query)
        reqs = [AnnSearchRequest(data=[vector], anns_field='vector',
                    param={'metric_type':'COSINE','params':{'ef':64}}, limit=6, expr=expr),
                AnnSearchRequest(data=[query], anns_field='sparse_vector', param={'metric_type':'BM25'},limit=4,expr=expr)]
        hits = (await manager.client.hybrid_search(collection_name=manager.collection_name, reqs=reqs,
                ranker=RRFRanker(), limit=3, output_fields=['*'], consistency_level='Strong'))[0]
        candidates = [dict(h['entity'], score=cosine(vector,h['entity']['vector']),score_kind='cosine') for h in hits]
        legacy = sorted(hits[:2], key=lambda h:.45*h['distance']+.25+.3*h['entity']['importance'], reverse=True)
        with patch.dict(os.environ, {'MEMORY_MIN_RELEVANCE':'.35'}):
            current = await manager.hybrid_retrieval_memories(query, uid, 0, 1, 0, 0, tenant)
        with patch.dict(os.environ, {'MEMORY_MIN_RELEVANCE':'.4'}):
            optimized = await manager.hybrid_retrieval_memories(query, uid, 0, 1, 0, 0, tenant)
        with patch.dict(os.environ, {'MEMORY_MIN_RELEVANCE':'-1'}):
            ungated = manager.get_the_top_k_memories({'semantic':candidates}, memory_configs={'semantic':{'k':1}})['semantic']
        dense = (await manager.client.search(collection_name=manager.collection_name,data=[vector],anns_field='vector',
                filter=expr,search_params={'metric_type':'COSINE','params':{'ef':64}},limit=1,
                output_fields=['*'],consistency_level='Strong'))[0]
        dense_best = dense[0]['entity'] if dense and dense[0]['distance']>=.35 else None
        predictions = {
            'legacy_rrf_scoring':legacy[0]['entity'] if legacy else None,
            'cosine_no_threshold':ungated[0] if ungated else None,
            'current_threshold_035':current.get('semantic',[None])[0] if current.get('semantic') else None,
            'optimized_threshold_040':optimized.get('semantic',[None])[0] if optimized.get('semantic') else None,
            'dense_only_035':dense_best,
        }
        sample = {'query':query,'expected':expected,'candidates':[dict(key=c.get('memory_key'),cosine=c['score']) for c in candidates], 'variants':{}}
        sample['reranked_candidates'] = [dict(key=c.get('memory_key','').removeprefix('preference.'),cosine=c['score'])
                                         for c in sorted(candidates,key=lambda x:x['score'],reverse=True)]
        for name, best in predictions.items():
            key = best.get('memory_key','').removeprefix('preference.') if best else None
            passed = key==expected
            variants[name].append((expected,passed,key))
            sample['variants'][name] = dict(predicted=key,correct=passed)
        report['ablations'].append(sample)
        print(json.dumps({'retrieval_sample':len(report['ablations']),'total':len(cases)}),flush=True)
    report['ablation_summary'] = {}
    for name, rows in variants.items():
        pos = [r for r in rows if r[0] is not None]
        neg = [r for r in rows if r[0] is None]
        report['ablation_summary'][name] = dict(samples=len(rows),correct=sum(r[1] for r in rows),
            positive_top1=sum(r[1] for r in pos)/len(pos), irrelevant_rejection=sum(r[1] for r in neg)/len(neg),
            overall=sum(r[1] for r in rows)/len(rows))
    check('real_retrieval_ablation_completed', len(report['ablations'])==len(cases))
    after_ids = {r['id'] for r in await manager.client.query(collection_name=manager.collection_name,
                 filter=expr,output_fields=['id'],limit=100,consistency_level='Strong')}
    check('retrieval_is_read_only_and_keeps_physical_ids', before_ids==after_ids)
    # Fixed alternating split. Threshold selection never sees holdout labels.
    calibration = report['ablations'][::2]
    holdout = report['ablations'][1::2]
    def predict(sample, threshold):
        best = sample['reranked_candidates'][0] if sample['reranked_candidates'] else None
        return best['key'] if best and best['cosine'] >= threshold else None
    thresholds = [.35,.4,.45,.5,.55,.6]
    selected = max(thresholds,key=lambda t:(sum(predict(s,t)==s['expected'] for s in calibration),-t))
    report['threshold_calibration'] = {
        'split':'even-index calibration; odd-index holdout; fixed before selecting threshold',
        'calibration_samples':len(calibration), 'holdout_samples':len(holdout), 'selected_threshold':selected,
        'grid_scores':{str(t):sum(predict(s,t)==s['expected'] for s in calibration) for t in thresholds},
        'holdout_correct':sum(predict(s,selected)==s['expected'] for s in holdout),
        'holdout_at_035_correct':sum(predict(s,.35)==s['expected'] for s in holdout),
        'holdout_positive_correct':sum(predict(s,selected)==s['expected'] for s in holdout if s['expected']),
        'holdout_positive_count':sum(s['expected'] is not None for s in holdout),
        'holdout_negative_correct':sum(predict(s,selected) is None for s in holdout if s['expected'] is None),
        'holdout_negative_count':sum(s['expected'] is None for s in holdout),
    }


async def managed_memory_checks(pg, manager, client, check):
    uid, tenant, thread = 910005, 'memory-management', 'manage_' + uuid.uuid4().hex[:12]
    await pg.register_session_owner(thread, uid, tenant)
    facts = {'semantic_memory': [dict(content='用户的报告使用中文。', memory_key='preference.language',
                                      evidence_kind='user_quote', importance_score=.8)]}
    await manager.add_memories_batch(uid, thread, facts, batch_id='fixture', tenant_id=tenant)
    rows = await manager.get_stable_preferences(uid, tenant)
    check('stable_preferences_without_generic_query', len(rows)==1 and rows[0]['content']=='用户的报告使用中文。')
    rid = rows[0]['id']
    check('managed_read_user_and_tenant_isolation',
          await manager.get_owned_memory(rid, uid+1, tenant) is None
          and await manager.get_owned_memory(rid, uid, 'other') is None)
    context = await load_memory_context(manager, '报告语言', uid, tenant)
    check('whole_record_context_loader', bool(context) and json.loads(context[len(HEADER):])[0]['id']==str(rid))
    with patch.object(manager, 'get_dense_vector', return_value=[]):
        lexical = await manager.hybrid_retrieval_memories('报告 中文', uid, 0, 3, 0, 0, tenant)
    check('real_bm25_without_embedding', bool(lexical.get('semantic'))
          and all(r['score_kind']=='bm25' for r in lexical['semantic']))
    try:
        await edit_memory(pg, manager, uid, tenant, str(uuid.uuid4()), 'correct', rid, 99, '用户的报告使用英文。')
    except MemoryConflict:
        check('manual_edit_optimistic_version_conflict', True)
    else:
        check('manual_edit_optimistic_version_conflict', False)
    op = str(uuid.uuid4())
    actual_write = manager.add_memories_batch
    async def interrupted(*args, **kwargs):
        await actual_write(*args, **kwargs)
        raise RuntimeError('injected_after_manual_vector_write')
    with patch.object(manager, 'add_memories_batch', interrupted):
        try:
            await edit_memory(pg, manager, uid, tenant, op, 'correct', rid, 1, '用户的报告使用英文。')
        except RuntimeError:
            pass
        else:
            raise AssertionError('manual_fault_not_injected')
    async with pg.pool.acquire() as conn:
        status = await conn.fetchval('SELECT status FROM nexusmind_memory_mutations WHERE operation_id=$1', op)
    check('manual_edit_durable_pending_after_interruption', status=='prepared')
    result = await edit_memory(pg, manager, uid, tenant, op, 'correct', rid, 1, '用户的报告使用英文。')
    latest = await manager.get_stable_preferences(uid, tenant)
    check('manual_edit_replay_latest_version', result['replayed'] and len(latest)==1
          and latest[0]['version']==2 and latest[0]['content']=='用户的报告使用英文。')
    replay = await edit_memory(pg, manager, uid, tenant, op, 'correct', rid, 1, '用户的报告使用英文。')
    check('completed_manual_edit_is_idempotent', replay['replayed'])
    delete_op = str(uuid.uuid4())
    await edit_memory(pg, manager, uid, tenant, delete_op, 'forget', latest[0]['id'], 2)
    remaining = await client.query(collection_name=manager.collection_name,
                filter=f'user_id == {memory_scope_user_id(uid, tenant)}',
                output_fields=['id'], limit=100, consistency_level='Strong')
    check('forget_all_versions_no_old_preference_resurrection', not remaining)


async def retirement_retry(manager, client, check):
    uid,tenant,thread = 910004,'memory-retirement','retire_'+uuid.uuid4().hex[:12]
    original = {'semantic_memory':[dict(content='默认按周汇总',memory_key='preference.period',
                                evidence_kind='user_quote',importance_score=.8)]}
    await manager.add_memories_batch(uid,thread,original,batch_id='old-version',tenant_id=tenant)
    candidate = {'semantic_memory':[dict(content='默认按月汇总',memory_key='preference.period',
                                 evidence_kind='user_quote',importance_score=.8)]}
    plan = await manager.resolve_conflicts(candidate,uid,tenant)
    actual_upsert = client.upsert
    async def interrupted(**kwargs):
        await actual_upsert(**kwargs)
        raise RuntimeError('injected_after_retirement_before_ack')
    with patch.object(client,'upsert',interrupted):
        try:
            await manager.add_memories_batch(uid,thread,plan,batch_id='new-version',tenant_id=tenant)
        except RuntimeError:
            pass
        else:
            raise AssertionError('retirement_fault_not_injected')
    check('retirement_retry_resolves_current_physical_ids',
          await manager.add_memories_batch(uid,thread,plan,batch_id='new-version',tenant_id=tenant))
    scope = memory_scope_user_id(uid,tenant)
    rows = await client.query(collection_name=manager.collection_name,filter=f'user_id == {scope}',
                              output_fields=['id','status','version','content'],limit=100,consistency_level='Strong')
    check('retirement_retry_keeps_one_active_version',len(rows)==2
          and len([r for r in rows if r.get('status')=='active'])==1)


if __name__ == '__main__':
    asyncio.run(main())
