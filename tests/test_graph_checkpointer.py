import os
import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from infra.graph_checkpointer import (
    _EncryptedInlineValue,
    _PENDING_SENDS_ENVELOPE,
    _protect_checkpoint,
    _restore_checkpoint_tuple,
    checkpoint_thread_id,
    checkpoint_event_loop_factory,
)
from cryptography.fernet import Fernet


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self):
        self.key = Fernet.generate_key().decode("ascii")
        self.environment = patch.dict(os.environ, {"GRAPH_CHECKPOINT_FERNET_KEY": self.key})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_checkpoint_id_is_stable_and_bound_to_tenant_user_and_session(self):
        expected = checkpoint_thread_id("session-a", 17, "tenant-a")
        self.assertEqual(expected, checkpoint_thread_id("session-a", 17, "tenant-a"))
        self.assertNotEqual(expected, checkpoint_thread_id("session-a", 17, "tenant-b"))
        self.assertNotEqual(expected, checkpoint_thread_id("session-a", 18, "tenant-a"))
        self.assertNotEqual(expected, checkpoint_thread_id("session-b", 17, "tenant-a"))
        self.assertNotIn("session-a", expected)

    def test_missing_or_invalid_key_fails_closed(self):
        with patch.dict(os.environ, {"GRAPH_CHECKPOINT_FERNET_KEY": ""}):
            with self.assertRaises(RuntimeError):
                checkpoint_thread_id("session-a", 17, "tenant-a")

    def test_windows_compatible_loop_factory(self):
        import asyncio
        loop = checkpoint_event_loop_factory()()
        try:
            self.assertIsInstance(loop, asyncio.SelectorEventLoop)
        finally:
            loop.close()


class _EncryptedJsonSerializer:
    def __init__(self, cipher):
        self.cipher = cipher

    def dumps_typed(self, value):
        return "test-json", self.cipher.encrypt(json.dumps(value).encode("utf-8"))

    def loads_typed(self, value):
        type_tag, payload = value
        if type_tag != "test-json":
            raise ValueError("unexpected type tag")
        return json.loads(self.cipher.decrypt(payload).decode("utf-8"))


class CheckpointPayloadEncryptionTests(unittest.TestCase):
    def setUp(self):
        self.cipher = Fernet(Fernet.generate_key())
        self.serializer = _EncryptedJsonSerializer(self.cipher)

    def test_primitive_channels_and_pending_sends_are_not_stored_inline(self):
        protected = _protect_checkpoint(
            {
                "channel_values": {
                    "query": "private query",
                    "attempt": 3,
                    "optional": None,
                    "messages": [{"role": "user", "content": "private message"}],
                },
                "pending_sends": [{"node": "next", "value": "private send"}],
            },
            self.serializer,
        )

        self.assertIsInstance(protected["channel_values"]["query"], _EncryptedInlineValue)
        self.assertIsInstance(protected["channel_values"]["attempt"], _EncryptedInlineValue)
        self.assertIsInstance(protected["channel_values"]["optional"], _EncryptedInlineValue)
        self.assertEqual(
            protected["channel_values"]["messages"],
            [{"role": "user", "content": "private message"}],
        )
        pending_envelope = protected["pending_sends"][_PENDING_SENDS_ENVELOPE]
        stored_blob = base64.b64decode(pending_envelope["payload"])
        self.assertNotIn(b"private send", stored_blob)

    def test_restore_decrypts_inline_values_and_pending_sends(self):
        protected = _protect_checkpoint(
            {
                "channel_values": {"query": "private query"},
                "pending_sends": [{"node": "next", "value": "private send"}],
            },
            self.serializer,
        )

        class Snapshot:
            checkpoint = protected

            def _replace(self, *, checkpoint):
                self.checkpoint = checkpoint
                return self

        restored = _restore_checkpoint_tuple(Snapshot(), self.serializer)
        self.assertEqual(restored.checkpoint["channel_values"]["query"], "private query")
        self.assertEqual(
            restored.checkpoint["pending_sends"],
            [{"node": "next", "value": "private send"}],
        )
        with patch.dict(os.environ, {"GRAPH_CHECKPOINT_FERNET_KEY": "not-a-fernet-key"}):
            with self.assertRaises(RuntimeError):
                checkpoint_thread_id("session-a", 17, "tenant-a")


if __name__ == "__main__":
    unittest.main()
