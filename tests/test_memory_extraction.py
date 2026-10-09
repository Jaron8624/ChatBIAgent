"""Model-output contract tests; require the project's real Python dependencies."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from memory.auto_store_memory_from_psql import extract_memories
from rag.embedding_provider import get_embeddings


class Model:
    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(self.outputs[len(self.prompts)-1], ensure_ascii=False))


class ExtractionTests(unittest.IsolatedAsyncioTestCase):
    def test_embedding_batch_limit_configured(self):
        with patch.dict('os.environ', {'EMBEDDING_PROVIDER':'openai_compatible',
                'EMBEDDING_BASE_URL':'https://example.invalid/v1', 'EMBEDDING_MODEL':'test',
                'EMBEDDING_API_KEY':'test', 'EMBEDDING_BATCH_SIZE':'10'}):
            self.assertEqual(get_embeddings().chunk_size,10)

    def test_invalid_embedding_batch_limit_rejected(self):
        with patch.dict('os.environ', {'EMBEDDING_PROVIDER':'openai_compatible',
                'EMBEDDING_BASE_URL':'https://example.invalid/v1','EMBEDDING_MODEL':'test',
                'EMBEDDING_API_KEY':'test','EMBEDDING_BATCH_SIZE':'0'}):
            with self.assertRaises(ValueError):
                get_embeddings()

    async def test_missing_sources_retried_and_existing_keys_supplied(self):
        bad = {'semantic_memory':[{'content':'按月汇总','importance_score':.8}]}
        good = {'semantic_memory':[dict(content='按月汇总',importance_score=.8,
                source_message_ids=['1'], source_quote='改成按月汇总', memory_key='preference.report_period')]}
        model = Model([bad,good])
        actual = await extract_memories([dict(id='1',role='human',content='改成按月汇总')], model,
                    [dict(memory_key='preference.report_period',content='按周汇总')])
        self.assertEqual(len(model.prompts),2)
        self.assertIn('preference.report_period', model.prompts[0])
        self.assertEqual(actual['semantic_memory'][0]['evidence_kind'],'user_quote')

    async def test_invalid_source_twice_fails_closed(self):
        bad = {'semantic_memory':[dict(content='按月汇总',importance_score=.8,
                source_message_ids=['invented'],source_quote='改成按月汇总',memory_key='preference.report_period')]}
        model = Model([bad,bad])
        with self.assertRaisesRegex(ValueError,'invalid_provenance'):
            await extract_memories([dict(id='1',role='human',content='改成按月汇总')],model)
        self.assertEqual(len(model.prompts),2)


if __name__ == '__main__':
    unittest.main()
