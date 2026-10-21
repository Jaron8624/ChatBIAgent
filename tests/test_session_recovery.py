"""Ownership and deletion contracts without a cache or import-order stubs."""
import copy
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from memory.session_recovery import SessionManager


class Registry:
    def __init__(self):
        self.records = {}
        self.pool = self

    def acquire(self): return self
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def fetch(self, sql, user, tenant, limit):
        return [{'session_id': sid} for sid, value in self.records.items()
                if value == (user, tenant, 'active')][:limit]

    async def get_session_registry_status(self, sid, user, tenant):
        value = self.records.get(sid)
        return None if value is None else value[2] if value[:2] == (user, tenant) else 'owner_mismatch'

    async def register_session_owner(self, sid, user, tenant):
        self.records.setdefault(sid, (user, tenant, 'active'))
        return self.records[sid] == (user, tenant, 'active')

    async def mark_session_deleted(self, sid, user, tenant):
        if await self.get_session_registry_status(sid, user, tenant) not in {'active', 'deleted'}:
            return False
        self.records[sid] = (user, tenant, 'deleted')
        return True

    async def delete_conversation_thread(self, *args): return 0


class Checkpoints:
    saver = object()
    def __init__(self): self.snapshots = {}; self.during_save = None
    async def save_session_snapshot(self, thread, snapshot, **kwargs):
        if self.during_save: await self.during_save()
        if await kwargs['is_deleted'](): return False
        self.snapshots[thread] = copy.deepcopy(snapshot)
        return True
    async def load_session_snapshot(self, thread): return copy.deepcopy(self.snapshots.get(thread))
    async def delete_thread(self, thread): self.snapshots.pop(thread, None)


class SessionOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry, self.checkpoints = Registry(), Checkpoints()
        self.sessions = object.__new__(SessionManager)
        self.sessions._initialized = True
        patches = [
            patch('infra.postgresql_client.get_postgresql_client', AsyncMock(return_value=self.registry)),
            patch('infra.graph_checkpointer.get_graph_checkpointer_manager', return_value=self.checkpoints),
            patch('infra.graph_checkpointer.checkpoint_thread_id', side_effect=lambda sid, uid, tenant: (tenant, uid, sid)),
            patch('core.context_store.context_identity', side_effect=lambda *args: nullcontext()),
            patch('core.context_store.get_context_store', return_value=SimpleNamespace(delete_session_archives=lambda: 0)),
            patch('rag.milvus_client.get_milvus_client', AsyncMock(return_value=SimpleNamespace(delete_session_memories=AsyncMock(return_value=0)))),
            patch('memory.ingestion_ledger.memory_write_lock', side_effect=lambda *args: self.registry),
            patch('memory.ingestion_ledger.IngestionLedger', return_value=SimpleNamespace(ensure_schema=AsyncMock(), cancel_session=AsyncMock())),
        ]
        for item in patches: item.start(); self.addCleanup(item.stop)
        self.assertTrue(await self.sessions.save_session('sid', 11, [{'role': 'user', 'content': 'private'}], tenant_id='team'))

    async def test_owner_and_tenant_are_enforced(self):
        self.assertIsNone(await self.sessions.resume_session('sid', 22, 'team'))
        self.assertIsNone(await self.sessions.resume_session('sid', 11, 'other'))
        self.assertEqual((await self.sessions.resume_session('sid', 11, 'team')).messages[0]['content'], 'private')

    async def test_delete_requires_owner_and_cannot_resurrect(self):
        self.assertFalse(await self.sessions.delete_session('sid', 22, 'team'))
        self.assertFalse(await self.sessions.delete_session('sid', 11, 'other'))
        self.assertTrue(await self.sessions.delete_session('sid', 11, 'team'))
        self.assertIsNone(await self.sessions.resume_session('sid', 11, 'team'))
        self.assertFalse(await self.sessions.save_session('sid', 11, [], tenant_id='team'))
        self.assertNotEqual((await self.sessions.resume_or_create('sid', 11, 'team'))[0], 'sid')

    async def test_delete_when_checkpoint_already_pruned_uses_registry(self):
        self.checkpoints.snapshots.clear()
        self.assertTrue(await self.sessions.delete_session('sid', 11, 'team'))
        self.assertEqual(self.registry.records['sid'][2], 'deleted')

    async def test_save_racing_with_delete_is_rejected(self):
        async def delete(): await self.registry.mark_session_deleted('late', 11, 'team')
        self.checkpoints.during_save = delete
        self.assertFalse(await self.sessions.save_session('late', 11, [], tenant_id='team'))
        self.assertNotIn(('team', 11, 'late'), self.checkpoints.snapshots)

    async def test_anonymous_sessions_are_not_persisted(self):
        self.assertFalse(await self.sessions.save_session('guest', 0, []))
        self.assertIsNone(await self.sessions.resume_session('sid', 0))
        self.assertEqual(await self.sessions.list_sessions(0), [])
        self.assertFalse(await self.sessions.delete_session('sid', 0))

    async def test_list_is_tenant_scoped_and_zero_limit_is_empty(self):
        self.assertEqual(await self.sessions.list_sessions(11, tenant_id='other'), [])
        self.assertEqual(await self.sessions.list_sessions(11, limit=0, tenant_id='team'), [])
        self.assertEqual([s['session_id'] for s in await self.sessions.list_sessions(11, tenant_id='team')], ['sid'])


if __name__ == '__main__': unittest.main()
