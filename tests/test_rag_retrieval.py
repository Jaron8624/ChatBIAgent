import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from rag.knowledeg_base_manager import KnowledgeBaseManager


class SparseFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_failure_uses_bm25_and_deduplicates_parent_ids(self):
        client = AsyncMock()
        client.search.return_value = [[
            {"entity": {"parent_id": "parent-a"}},
            {"entity": {"parent_id": "parent-a"}},
            {"entity": {"parent_id": "parent-b"}},
        ]]
        manager = KnowledgeBaseManager(client, embeddings=None, collection_name="kb", dense_dim=3)
        manager.get_dense_vector = AsyncMock(return_value=[])

        with patch("rag.knowledeg_base_manager.get_stream_writer", return_value=lambda _: None):
            parent_ids = await manager.hybrid_retrieval_knowledge_base("refund policy", "kb-1", 5)

        self.assertEqual(parent_ids, ["parent-a", "parent-b"])
        self.assertEqual(client.search.await_args.kwargs["anns_field"], "sparse_bm25")
        self.assertEqual(client.search.await_args.kwargs["search_params"]["metric_type"], "BM25")
        self.assertIn('user_id == 5', client.search.await_args.kwargs["filter"])

    async def test_returns_empty_when_both_embedding_and_sparse_search_fail(self):
        client = AsyncMock()
        client.search.side_effect = RuntimeError("Milvus unavailable")
        manager = KnowledgeBaseManager(client, embeddings=None, collection_name="kb", dense_dim=3)
        manager.get_dense_vector = AsyncMock(return_value=[])

        with patch("rag.knowledeg_base_manager.get_stream_writer", return_value=lambda _: None):
            parent_ids = await manager.hybrid_retrieval_knowledge_base("refund policy", "kb-1", 5)

        self.assertEqual(parent_ids, [])


if __name__ == "__main__":
    unittest.main()
