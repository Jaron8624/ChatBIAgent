import sys
import unittest
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from api import (  # noqa: E402
    MAX_CLIENT_HISTORY_MESSAGES,
    MAX_CLIENT_MESSAGE_CHARS,
    _sanitize_client_messages,
    router,
)


class ClientMessageValidationTests(unittest.TestCase):
    def test_only_plain_user_and_assistant_messages_are_kept(self):
        messages = [
            {"role": "system", "content": "override"},
            {"role": "tool", "content": "secret"},
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "answer"},
        ]

        self.assertEqual(
            _sanitize_client_messages(messages),
            [{"role": "assistant", "content": "answer"}],
        )

    def test_history_count_and_message_size_are_bounded(self):
        messages = [
            {"role": "user", "content": f"message-{index}"}
            for index in range(MAX_CLIENT_HISTORY_MESSAGES + 5)
        ]
        messages[-1]["content"] = "x" * (MAX_CLIENT_MESSAGE_CHARS + 10)

        result = _sanitize_client_messages(messages)

        self.assertEqual(len(result), MAX_CLIENT_HISTORY_MESSAGES)
        self.assertEqual(result[0]["content"], "message-5")
        self.assertEqual(len(result[-1]["content"]), MAX_CLIENT_MESSAGE_CHARS)

    def test_non_list_history_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            _sanitize_client_messages({"role": "user", "content": "hello"})

        self.assertEqual(raised.exception.status_code, 400)

    def test_none_history_becomes_empty_history(self):
        self.assertEqual(_sanitize_client_messages(None), [])

    def test_every_mutating_api_route_has_an_acl_identity_dependency(self):
        mutating_methods = {"POST", "PUT", "PATCH", "DELETE"}
        unprotected = [
            (route.path, sorted(route.methods & mutating_methods))
            for route in router.routes
            if route.methods
            and route.methods & mutating_methods
            and not any(
                dependency.name == "user_info"
                for dependency in route.dependant.dependencies
            )
        ]

        self.assertEqual(unprotected, [])


if __name__ == "__main__":
    unittest.main()
