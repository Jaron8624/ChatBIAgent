"""Validate scope literal escaping on an isolated real Milvus collection."""
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from rag.knowledeg_base_manager import _scope_filter


def main():
    load_dotenv(ROOT / ".env")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    collection = "rag_filter_check_" + run_id.lower()
    folder = ROOT / "artifacts" / "rag_filter_runtime" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "collection": collection, "checks": [], "status": "running",
        "evidence_scope": "Synthetic rows; real Milvus query/delete; no model accuracy claim",
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("backend/rag/knowledeg_base_manager.py", "scripts/verify_rag_filters.py")}}
    client = None
    created = False
    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        print(json.dumps(report["checks"][-1]), flush=True)
        if not passed:
            raise AssertionError(name)
    try:
        client = MilvusClient(uri=os.environ["Milvus_url"], token=os.getenv("Token", "root:Milvus"))
        report["milvus_version"] = client.get_server_version()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("user_id", DataType.INT64)
        schema.add_field("knowledge_base_id", DataType.VARCHAR, max_length=512)
        schema.add_field("file_hash", DataType.VARCHAR, max_length=512)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
        client.create_collection(collection, schema=schema, index_params=index, consistency_level="Strong")
        created = True
        malicious = '知识库" or user_id == 999 or knowledge_base_id == "'
        file_hash = 'hash" or user_id == 999 or file_hash == "'
        client.insert(collection, [
            {"id": 1, "user_id": 7, "knowledge_base_id": malicious, "file_hash": file_hash, "vector": [1., 0.]},
            {"id": 2, "user_id": 999, "knowledge_base_id": "other", "file_hash": "other", "vector": [0., 1.]},
            {"id": 3, "user_id": 8, "knowledge_base_id": malicious, "file_hash": file_hash, "vector": [1., 1.]}])
        def ids(expression):
            return {row["id"] for row in client.query(collection, filter=expression, output_fields=["id"])}
        check("malicious_knowledge_id_is_literal", ids(_scope_filter(malicious, 7)) == {1})
        check("malicious_file_hash_is_literal", ids(_scope_filter(malicious, 7, file_hash)) == {1})
        check("other_owner_excluded", ids(_scope_filter(malicious, 8)) == {3})
        client.delete(collection, filter=_scope_filter(malicious, 7, file_hash))
        check("delete_preserves_other_owners", ids("id >= 0") == {2, 3})
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        if client is not None:
            if created:
                client.drop_collection(collection)
                report["test_collection_removed"] = True
            client.close()
        path = folder / "report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(path), "status": report["status"]}))


if __name__ == "__main__":
    main()
