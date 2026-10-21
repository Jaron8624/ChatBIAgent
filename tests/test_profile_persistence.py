"""Profiles remain readable immediately after writes without any cache package."""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from infra.postgresql_client import PostgreSQLParentClient


class Pool:
    def __init__(self): self.values = {}
    def acquire(self): return self
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def fetchrow(self, sql, uid):
        return {'user_profile': self.values[uid]} if uid in self.values else None
    async def execute(self, sql, uid, profile): self.values[uid] = profile


class ProfileTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = object.__new__(PostgreSQLParentClient)
        self.client.pool = Pool()

    async def test_save_read_and_update_are_immediately_visible(self):
        self.assertTrue(await self.client.update_user_profile(11, 'prefer weekly'))
        self.assertEqual(await self.client.get_user_profile(11), 'prefer weekly')
        self.assertTrue(await self.client.update_user_profile(11, 'prefer monthly'))
        self.assertEqual(await self.client.get_user_profile(11), 'prefer monthly')

    async def test_users_do_not_share_profiles(self):
        await self.client.update_user_profile(11, 'private')
        self.assertIsNone(await self.client.get_user_profile(12))

    async def test_blank_update_does_not_erase_profile(self):
        await self.client.update_user_profile(11, 'keep')
        self.assertFalse(await self.client.update_user_profile(11, '  '))
        self.assertEqual(await self.client.get_user_profile(11), 'keep')

    async def test_read_reaches_database_without_cache(self):
        self.client.pool.fetchrow = AsyncMock(return_value={'user_profile': 'from database'})
        self.assertEqual(await self.client.get_user_profile(11), 'from database')
        self.client.pool.fetchrow.assert_awaited_once()


if __name__ == '__main__': unittest.main()
