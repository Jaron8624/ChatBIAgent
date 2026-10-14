"""Pure policy boundaries; durable behavior is covered by the opt-in runtime probe."""
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, AsyncMock

from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from memory.managed_store import identity, file_topic, view_path, encrypt, decrypt, materialize, cosine, terms
from memory.consolidation import parse_selection


class ManagedMemoryPolicyTests(unittest.TestCase):
    def test_identity_rejects_bool_and_anonymous(self):
        for value in (True, 0, -1, '1', None):
            with self.assertRaises(PermissionError):
                identity(value, 't')
        self.assertEqual(identity(1, 't'), (1, 't'))

    def test_virtual_paths_only(self):
        self.assertEqual(file_topic('L1/preference.currency.md'), 'preference.currency')
        for path in ('../secret.md', '/L1/a.md', 'L2/profile.md', 'L1/../x.md',
                     'L1/a\\b.md', 'L1/x.md/../../secret', 'C:/secret.md'):
            with self.assertRaises(ValueError):
                file_topic(path)

    def test_encrypted_file_view(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'MEMORY_FILES_ROOT': directory, 'MEMORY_STORE_FERNET_KEY': Fernet.generate_key().decode()}):
            store, unit = str(uuid.uuid4()), str(uuid.uuid4())
            data = encrypt({'content': '按月汇总，人民币'})
            materialize({'store_id': store, 'id': unit, 'version': 1, 'payload': data})
            path = view_path(store, unit, 1)
            self.assertTrue(path.resolve().is_relative_to(Path(directory).resolve()))
            self.assertNotIn('人民币'.encode(), path.read_bytes())
            self.assertEqual(decrypt(path.read_bytes())['content'], '按月汇总，人民币')
            with self.assertRaises(ValueError):
                view_path('../escape', unit, 1)

    def test_selection_cannot_invent_sources(self):
        self.assertEqual(parse_selection('{"selected_ids":["a"]}', {'a'}), ['a'])
        for text in ('{"selected_ids":["invented"]}', '{"selected_ids":["a","a"]}',
                     '{"selected_ids":[]}', 'not json', '{"selected_ids":"a"}'):
            with self.assertRaises((ValueError, TypeError)):
                parse_selection(text, {'a'})

    def test_similarity_and_chinese_terms(self):
        self.assertEqual(cosine([], []), 0)
        self.assertEqual(cosine([1], [1, 2]), 0)
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1)
        self.assertIn('月汇', terms('按月汇总'))


class RequestMemoryAuthorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_managed_result_does_not_fall_back(self):
        from memory.context_loader import load_request_memory_context
        service = AsyncMock()
        service.context.return_value = ''
        with patch.dict(os.environ, {'MANAGED_MEMORY_ENABLED': 'true'}), \
                patch('memory.managed_store.get_managed_memory', return_value=service):
            self.assertEqual(await load_request_memory_context('q', 1, 't', 's'), '')
        service.context.assert_awaited_once_with(1, 't', 's', 'q', token_budget=600)

    async def test_managed_failure_cannot_broaden_scope(self):
        from memory.context_loader import load_request_memory_context
        with patch.dict(os.environ, {'MANAGED_MEMORY_ENABLED': 'true'}), \
                patch('memory.managed_store.get_managed_memory', side_effect=RuntimeError('outage')):
            with self.assertRaises(RuntimeError):
                await load_request_memory_context('q', 1, 't', 's')


if __name__ == '__main__':
    unittest.main()
