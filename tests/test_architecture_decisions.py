import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from core.task_routing import describe_task, requires_loop
from harness.store import HarnessStore
from harness.models import CandidateSpec, CandidateStatus
from harness.runtime import HarnessRuntime
from infra.graph_checkpointer import checkpoint_thread_id
from cryptography.fernet import Fernet
from rag.index_contract import fingerprint, index_configuration
from rag.jobs import source_path


class TaskRoutingTests(unittest.TestCase):
    def test_business_nouns_are_not_multiple_tasks(self):
        for query in ("最近30天商品销量", "对比各品类销量", "统计订单收入与退款金额", "查看商品销量趋势"):
            with self.subTest(query=query):
                self.assertFalse(requires_loop(query))

    def test_actual_sequential_requests_are_preserved(self):
        plan = describe_task("先查订单然后分析退款")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.dependency, "sequential")

    def test_parallel_requests_are_independent(self):
        plan = describe_task("查订单数量同时统计退款金额")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.dependency, "independent")

    def test_descriptive_connector_does_not_create_tasks(self):
        self.assertFalse(requires_loop("查询同时支持线上线下的商品销量"))


class VersionTests(unittest.TestCase):
    def test_index_fingerprint_detects_same_dimension_model_change(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "model-a"}):
            a = index_configuration(1024, "knowledge")
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "model-b"}):
            b = index_configuration(1024, "knowledge")
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_index_fingerprint_detects_chunking_change(self):
        config = index_configuration(1024, "knowledge")
        first = fingerprint(config)
        config["chunking"]["child_chunk_size"] += 1
        self.assertNotEqual(first, fingerprint(config))

    def test_checkpoint_identity_survives_encryption_rotation(self):
        identity, first, second = [Fernet.generate_key().decode() for _ in range(3)]
        with patch.dict(os.environ, {"GRAPH_CHECKPOINT_IDENTITY_KEY": identity, "GRAPH_CHECKPOINT_FERNET_KEY": first}):
            a = checkpoint_thread_id("session", 7, "tenant")
        with patch.dict(os.environ, {"GRAPH_CHECKPOINT_IDENTITY_KEY": identity, "GRAPH_CHECKPOINT_FERNET_KEY": second}):
            self.assertEqual(a, checkpoint_thread_id("session", 7, "tenant"))

    def test_upload_source_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            source_path("../outside")


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = HarnessStore("sqlite:///" + str(Path(folder.name) / "test.db"))
        self.addCleanup(self.store.engine.dispose)

    def candidate(self, name, status):
        # This fixture checks channel concurrency, not scoped rule eligibility.
        c = CandidateSpec.build("agent_config", name, {"action_prompt": name})
        self.store.register_candidate(c)
        self.store.transition_candidate(c.candidate_id, CandidateStatus.EVALUATING, "test", "test")
        self.store.transition_candidate(c.candidate_id, CandidateStatus.EVALUATED, "test", "test")
        if status != CandidateStatus.EVALUATED:
            self.store.transition_candidate(c.candidate_id, status, "test", "test")
        return c

    def test_canary_can_run_alongside_active(self):
        active = self.candidate("active", CandidateStatus.ACTIVE)
        canary = self.candidate("canary", CandidateStatus.EVALUATED)
        self.store.transition_candidate(canary.candidate_id, CandidateStatus.SHADOW, "test", "test",
            release_update={"canary_id": canary.candidate_id, "policy_json": {"traffic_percent": 100, "tenant_ids": ["allowed"]}})
        runtime = HarnessRuntime(self.store)
        self.assertEqual(runtime.select("allowed", "user").candidate_id, canary.candidate_id)
        self.assertEqual(runtime.select("other", "user").candidate_id, active.candidate_id)

    def test_stale_candidate_transition_does_not_overwrite(self):
        c = self.candidate("conflict", CandidateStatus.EVALUATED)
        stale = self.store.get_candidate(c.candidate_id)
        self.store.transition_candidate(c.candidate_id, CandidateStatus.SHADOW, "test", "newer")
        with patch.object(self.store, "get_candidate", return_value=stale):
            with self.assertRaisesRegex(RuntimeError, "conflict"):
                self.store.transition_candidate(c.candidate_id, CandidateStatus.ACTIVE, "test", "stale",
                    release_update={"active_id": c.candidate_id})
        self.assertEqual(self.store.get_candidate(c.candidate_id)["status"], "shadow")
        self.assertIsNone(self.store.get_release_channel())

    def test_publication_pointer_remembers_previous(self):
        a = self.candidate("one", CandidateStatus.EVALUATED)
        self.store.transition_candidate(a.candidate_id, CandidateStatus.ACTIVE, "test", "publish",
            release_update={"active_id": a.candidate_id})
        b = self.candidate("two", CandidateStatus.EVALUATED)
        self.store.transition_candidate(b.candidate_id, CandidateStatus.ACTIVE, "test", "publish",
            release_update={"active_id": b.candidate_id})
        channel = self.store.get_release_channel()
        self.assertEqual(channel["previous_id"], a.candidate_id)
        self.assertEqual(channel["version"], 2)

    def test_stale_release_update_rolls_back_candidate_change(self):
        a = self.candidate("old", CandidateStatus.EVALUATED)
        self.store.transition_candidate(a.candidate_id, CandidateStatus.ACTIVE, "test", "publish",
            release_update={"active_id": a.candidate_id})
        with self.assertRaisesRegex(RuntimeError, "release_version_conflict"):
            self.store.transition_candidate(a.candidate_id, CandidateStatus.ROLLED_BACK, "test", "stale",
                release_update={"active_id": None, "_expected_version": 0})
        self.assertEqual(self.store.get_candidate(a.candidate_id)["status"], "active")
        self.assertEqual(self.store.get_release_channel()["active_id"], a.candidate_id)

    def test_registered_candidate_payload_cannot_change(self):
        c = self.candidate("immutable", CandidateStatus.EVALUATED)
        c.payload["action_prompt"] = "changed after evaluation"
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.register_candidate(c)


if __name__ == "__main__":
    unittest.main()
