import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from tools import sql_executor


class SqlExecutorReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.olist_path = root / "olist.sqlite"
        self.chinook_path = root / "chinook.sqlite"
        conn = sqlite3.connect(self.olist_path)
        try:
            conn.execute("CREATE TABLE sample (value TEXT)")
            conn.execute("INSERT INTO sample VALUES ('stable')")
            conn.commit()
        finally:
            conn.close()
        conn = sqlite3.connect(self.chinook_path)
        try:
            conn.execute("CREATE TABLE track (title TEXT)")
            conn.execute("INSERT INTO track VALUES ('track-a')")
            conn.commit()
        finally:
            conn.close()
        self.path_patches = [
            patch.object(sql_executor, "_OLIST_DB_PATH", str(self.olist_path)),
            patch.object(sql_executor, "_CHINOOK_DB_PATH", str(self.chinook_path)),
        ]
        for path_patch in self.path_patches:
            path_patch.start()

    def tearDown(self):
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.temp_dir.cleanup()

    def test_read_only_query_allows_keyword_inside_literal(self):
        result = json.loads(sql_executor.execute_sql("SELECT 'update' AS marker, value FROM sample"))
        self.assertNotIn("error", result)
        self.assertEqual(result["rows"], [["update", "stable"]])

    def test_write_statement_is_denied_by_sqlite_authorizer(self):
        result = json.loads(sql_executor.execute_sql("INSERT INTO sample VALUES ('changed')"))
        self.assertIn("error", result)
        conn = sqlite3.connect(self.olist_path)
        try:
            self.assertEqual(conn.execute("SELECT value FROM sample").fetchall(), [("stable",)])
        finally:
            conn.close()

    def test_query_cannot_attach_another_database(self):
        extra_path = Path(self.temp_dir.name) / "extra.sqlite"
        conn = sqlite3.connect(extra_path)
        conn.close()
        result = json.loads(
            sql_executor.execute_sql(f"ATTACH DATABASE '{extra_path}' AS extra")
        )
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
