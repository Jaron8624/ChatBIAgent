"""Exercise production methods with deterministic storage doubles, no SDK server."""
import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from memory.policy import batch_id, content_hash, cosine, should_extract, validated_items
from memory.ingestion_ledger import IngestionLedger, lock_key, memory_write_lock


def load_manager():
    tree = ast.parse((ROOT / 'backend/memory/memory_manager.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    scope = dict(globals(), logger=logging.getLogger(__name__), AsyncMilvusClient=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'memory_manager.py', 'exec'), scope)
    return scope['MemoryManager']


Manager = load_manager()


class PolicyTests(unittest.TestCase):
    def test_batch_identity_scope_and_order(self):
        rows = [{'id': 'a', 'content': 'hello'}, {'id': 'b', 'content': 'world'}]
        self.assertEqual(batch_id(1, 't', 's', rows), batch_id(1, 't', 's', rows[::-1]))
        self.assertNotEqual(batch_id(1, 't', 's', rows), batch_id(1, 'other', 's', rows))

    def test_quotes_require_user_source(self):
        rows = [{'id': '1', 'role': 'assistant', 'content': '喜欢周报'},
                {'id': '2', 'role': 'user', 'content': '以后改成月报'}]
        candidates = [{'content': '月报', 'source_message_ids': ['2'],
                       'source_quote': '以后改成月报', 'memory_key': 'preference.report'},
                      {'content': '周报', 'source_message_ids': ['1'],
                       'source_quote': '喜欢周报', 'memory_key': 'preference.report'}]
        actual = validated_items(candidates, rows)
        self.assertEqual(actual[0]['memory_key'], 'preference.report')
        self.assertEqual(actual[1]['memory_key'], '')
        self.assertEqual(actual[1]['evidence_kind'], 'unverified')

    def test_forged_quote_cannot_update(self):
        result = validated_items([{'content': 'x', 'source_message_ids': ['1'],
                                  'source_quote': 'fake', 'memory_key': 'preference.x'}],
                                 [{'id': '1', 'role': 'user', 'content': 'real'}])
        self.assertFalse(result[0]['memory_key'])

    def test_short_explicit_and_idle_extraction(self):
        self.assertTrue(should_extract([{'role': 'user', 'content': '记住用月报'}], 10))
        now = datetime.now(timezone.utc)
        self.assertTrue(should_extract([{'created_at': now-timedelta(seconds=901)}], 10, now=now))
        self.assertFalse(should_extract([{'created_at': now}], 10, now=now))

    def test_cosine(self):
        self.assertEqual(cosine([1, 0], [1, 0]), 1)
        self.assertEqual(cosine([0, 0], [1, 0]), 0)
        self.assertEqual(cosine([1], [1, 0]), 0)

    def test_relevance_not_popularity(self):
        manager = Manager(None, None, 'm', 2)
        data = {'semantic': [dict(id=1, score=.1, importance=1, last_access_at=time.time()),
                             dict(id=2, score=.9, importance=.5, created_at=time.time()),
                             dict(id=3, score=.99, score_kind='rrf')]}
        with patch.dict(os.environ, {'MEMORY_MIN_RELEVANCE': '.35'}):
            result = manager.get_the_top_k_memories(data, memory_configs={'semantic': {'k': 3}})
        self.assertEqual([m['id'] for m in result['semantic']], [2])

    def test_latest_key_wins_even_before_retirement(self):
        manager = Manager(None, None, 'm', 2)
        rows = [dict(id=1, memory_key='preference.x', version=1, score=.99),
                dict(id=2, memory_key='preference.x', version=2, score=.7)]
        actual = manager.get_the_top_k_memories({'semantic': rows}, memory_configs={'semantic': {'k': 3}})
        self.assertEqual([m['id'] for m in actual['semantic']], [2])


class Storage:
    def __init__(self):
        self.rows = []
        self.fail_retire = False

    async def query(self, **kw):
        expr = kw['filter']
        for field in ('ingestion_key', 'memory_key', 'content_hash'):
            marker = field + ' == '
            if marker in expr:
                value = json.JSONDecoder().raw_decode(expr.split(marker)[1])[0]
                return [dict(r) for r in self.rows if r.get(field) == value]
        return [dict(r) for r in self.rows]

    async def insert(self, **kw):
        for row in kw['data']:
            self.rows.append(dict(row, id=len(self.rows)+1))

    async def upsert(self, **kw):
        if self.fail_retire:
            raise RuntimeError('simulated interruption')
        for change in kw['data']:
            for row in self.rows:
                if row['id'] == change['id']:
                    row.update(change)


class Embeddings:
    async def aembed_documents(self, texts):
        return [[1., 0.] for _ in texts]


class VersionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Storage()
        self.manager = Manager(self.store, Embeddings(), 'm', 2)

    async def test_update_and_retry_after_partial_write(self):
        self.store.rows = [dict(id=1, memory_key='preference.report', version=1,
                                content='使用周报', status='active')]
        candidate = dict(content='使用月报', memory_key='preference.report', evidence_kind='user_quote')
        plan = await self.manager.resolve_conflicts({'semantic_memory': [candidate]}, 1)
        self.assertEqual(plan['semantic_memory'][0]['version'], 2)
        self.store.fail_retire = True
        with self.assertRaises(RuntimeError):
            await self.manager.add_memories_batch(1, 's', plan, 'summary', batch_id='batch')
        self.assertEqual(len(self.store.rows), 2)
        self.store.fail_retire = False
        self.assertTrue(await self.manager.add_memories_batch(1, 's', plan, 'summary', batch_id='batch'))
        self.assertEqual(len(self.store.rows), 2)
        self.assertEqual(self.store.rows[0]['status'], 'superseded')

    async def test_exact_duplicate_is_successful_empty_plan(self):
        self.store.rows = [dict(id=1, content='事实', content_hash=content_hash('事实'))]
        plan = await self.manager.resolve_conflicts({'semantic_memory': [{'content': '事实'}]}, 1)
        self.assertEqual(plan['semantic_memory'], [])
        self.assertTrue(await self.manager.add_memories_batch(1, 's', plan, batch_id='b'))

    async def test_similar_but_different_fact_is_not_discarded(self):
        self.store.rows = [dict(id=1, content='上海销售额100', content_hash=content_hash('上海销售额100'))]
        plan = await self.manager.resolve_conflicts({'semantic_memory': [{'content': '上海销售额200'}]}, 1)
        self.assertEqual(len(plan['semantic_memory']), 1)

    async def test_unverified_item_does_not_supersede(self):
        self.store.rows = [dict(id=1, memory_key='preference.x', content='old')]
        plan = await self.manager.resolve_conflicts({'semantic_memory': [dict(content='new',
            memory_key='preference.x', evidence_kind='unverified')]}, 1)
        self.assertNotIn('supersede_ids', plan['semantic_memory'][0])

    async def test_retirement_replay_after_auto_id_replacement(self):
        self.store.rows = [dict(id=1,memory_key='preference.report',version=1,content='周报',status='active')]
        plan = await self.manager.resolve_conflicts({'semantic_memory':[dict(content='月报',
            memory_key='preference.report',evidence_kind='user_quote')]},1)
        actual = self.store.upsert
        async def replaced_then_interrupted(**kwargs):
            await actual(**kwargs)
            self.store.rows[0]['id'] = 101
            raise RuntimeError('interrupted after physical id changed')
        with patch.object(self.store,'upsert',replaced_then_interrupted):
            with self.assertRaises(RuntimeError):
                await self.manager.add_memories_batch(1,'s',plan,batch_id='changed-id')
        self.assertTrue(await self.manager.add_memories_batch(1,'s',plan,batch_id='changed-id'))
        self.assertEqual(len(self.store.rows),2)
        self.assertEqual(len([r for r in self.store.rows if r.get('status')=='active']),1)

    async def test_read_access_does_not_rewrite_vector_rows(self):
        records = {'semantic':[dict(id=1,last_access_at=0)]}
        with patch.object(self.store,'upsert',side_effect=AssertionError('read wrote vectors')):
            await self.manager.update_memory_last_access_time(records)
        self.assertGreater(records['semantic'][0]['last_access_at'],0)


class LedgerTests(unittest.IsolatedAsyncioTestCase):
    async def test_lock_rejects_busy_scope_and_always_releases(self):
        class Connection:
            acquired = True
            released = False

            async def fetchval(self, *args):
                return self.acquired

            async def execute(self, *args):
                self.released = True

        connection = Connection()
        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield connection

        with self.assertRaises(ValueError):
            async with memory_write_lock(Pool(), 1, 't'):
                raise ValueError('worker interrupted')
        self.assertTrue(connection.released)
        connection.acquired = False
        connection.released = False
        with self.assertRaisesRegex(RuntimeError, 'memory_scope_busy'):
            async with memory_write_lock(Pool(), 1, 't'):
                self.fail('must not enter')
        self.assertFalse(connection.released)
        self.assertNotEqual(lock_key(1, 'a'), lock_key(1, 'b'))

    async def test_pending_serializes_all_sessions_for_same_user(self):
        class Connection:
            async def fetchrow(self, query, *args):
                self.query, self.args = query, args
                return {'batch_id': 'b', 'payload': '{"thread_id":"older-session"}'}
        connection = Connection()
        actual = await IngestionLedger(connection).pending(1, 't')
        self.assertEqual(actual[1]['thread_id'], 'older-session')
        self.assertEqual(connection.args, (1, 't'))
        self.assertNotIn('thread_id=$3', connection.query)

    async def test_delivery_failure_does_not_acknowledge_source(self):
        tree = ast.parse((ROOT / 'backend/memory/auto_store_memory_from_psql.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_deliver_batch')
        class Milvus:
            failed = True
            async def add_memories_batch(self, **kwargs):
                if self.failed:
                    raise RuntimeError('offline')
                return True
        milvus = Milvus()
        async def get_milvus():
            return milvus
        scope = dict(logger=logging.getLogger(__name__), get_milvus_client=get_milvus)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'delivery', 'exec'), scope)
        class Ledger:
            acknowledged = False
            failure = False
            allowed = True
            cancelled = False
            async def active(self, *args):
                return self.allowed
            async def acknowledge(self, *args):
                self.acknowledged = True
            async def failed(self, *args):
                self.failure = True
            async def cancel_session(self, *args):
                self.cancelled = True
        ledger = Ledger()
        payload = dict(user_id=1, tenant_id='t', thread_id='s', memories={},
                       summary_id='sum', message_ids=['1'], profile='', token_count=10, message_count=1)
        with self.assertRaises(RuntimeError):
            await scope['_deliver_batch']('b', payload, ledger, None)
        self.assertFalse(ledger.acknowledged)
        self.assertTrue(ledger.failure)
        milvus.failed = False
        self.assertTrue((await scope['_deliver_batch']('b', payload, ledger, None))['success'])
        self.assertTrue(ledger.acknowledged)
        ledger.allowed = False
        ledger.acknowledged = False
        self.assertFalse((await scope['_deliver_batch']('b', payload, ledger, None))['success'])
        self.assertTrue(ledger.cancelled)
        self.assertFalse(ledger.acknowledged)


if __name__ == '__main__':
    unittest.main()
