"""PostgreSQL-backed, encrypted persistence for LangGraph executions."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from types import SimpleNamespace

logger = logging.getLogger(__name__)

SESSION_CHECKPOINT_NS = "session"


def checkpoint_event_loop_factory(use_subprocess: bool = False):
    """Uvicorn loop factory compatible with async psycopg on Windows."""
    return asyncio.SelectorEventLoop


def checkpoint_run_thread_id(thread_id: str, run_id: str) -> str:
    """Isolate graph runs without treating a random namespace as a subgraph."""
    if not run_id or not run_id.isalnum():
        raise ValueError("无效的图执行 ID")
    return f"{thread_id}:run:{run_id}"


def session_persistence_required() -> bool:
    return os.getenv("SESSION_PERSISTENCE_REQUIRED", "true").lower() not in {"0", "false", "no"}


class GraphCheckpointBusy(RuntimeError):
    """Raised when another worker is already changing the same session graph."""


@dataclass(frozen=True)
class _EncryptedInlineValue:
    """Force primitive channel values through LangGraph's encrypted blob serializer."""

    value: Any


_PENDING_SENDS_ENVELOPE = "__nexusmind_encrypted_pending_sends_v1__"


def _protect_checkpoint(checkpoint: dict, serializer) -> dict:
    """Encrypt primitive values that PostgresSaver would otherwise inline as JSON."""
    protected = dict(checkpoint)
    channel_values = dict(checkpoint.get("channel_values") or {})
    for name, value in channel_values.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            channel_values[name] = _EncryptedInlineValue(value)
    protected["channel_values"] = channel_values

    pending_sends = checkpoint.get("pending_sends")
    if pending_sends:
        type_tag, payload = serializer.dumps_typed(pending_sends)
        protected["pending_sends"] = {
            _PENDING_SENDS_ENVELOPE: {
                "type": type_tag,
                "payload": base64.b64encode(payload).decode("ascii"),
            }
        }
    return protected


def _restore_checkpoint_tuple(checkpoint_tuple, serializer):
    """Restore encrypted channel values after the Postgres saver loads a checkpoint."""
    checkpoint = dict(checkpoint_tuple.checkpoint)
    channel_values = dict(checkpoint.get("channel_values") or {})
    for name, value in channel_values.items():
        if isinstance(value, _EncryptedInlineValue):
            channel_values[name] = value.value
    checkpoint["channel_values"] = channel_values

    pending_sends = checkpoint.get("pending_sends")
    if isinstance(pending_sends, dict) and _PENDING_SENDS_ENVELOPE in pending_sends:
        envelope = pending_sends[_PENDING_SENDS_ENVELOPE]
        payload = base64.b64decode(envelope["payload"], validate=True)
        checkpoint["pending_sends"] = serializer.loads_typed(
            (envelope["type"], payload)
        )
    return checkpoint_tuple._replace(checkpoint=checkpoint)


def checkpoint_thread_id(session_id: str, user_id: int, tenant_id: str) -> str:
    """Return a stable, opaque ID bound to the authenticated session owner."""
    if not session_id or user_id <= 0:
        raise ValueError("图执行持久化需要已登录用户和有效会话")
    key = (os.getenv("GRAPH_CHECKPOINT_IDENTITY_KEY") or os.getenv("GRAPH_CHECKPOINT_FERNET_KEY", "")).strip()
    if not key:
        raise RuntimeError("GRAPH_CHECKPOINT_FERNET_KEY 未配置")
    try:
        raw_key = base64.urlsafe_b64decode(key.encode("ascii"))
        from cryptography.fernet import Fernet
        Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise RuntimeError("GRAPH_CHECKPOINT_FERNET_KEY 格式无效") from exc
    payload = f"{tenant_id or 'default'}\0{user_id}\0{session_id}".encode("utf-8")
    digest = hmac.new(raw_key, payload, hashlib.sha256).hexdigest()
    return f"nexusmind:{digest}"


class GraphCheckpointerManager:
    """Owns the async PostgreSQL pool and LangGraph saver for the app lifetime."""

    def __init__(self) -> None:
        self.pool = None
        self.saver = None

    async def initialize(self, dsn: str | None = None) -> None:
        if self.saver is not None:
            return
        if os.name == "nt" and isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
            raise RuntimeError(
                "Windows PostgreSQL Checkpointer 需要 SelectorEventLoop。"
                "请用 python main.py，或给 uvicorn 加上 "
                "--loop infra.graph_checkpointer:checkpoint_event_loop_factory。"
            )

        dsn = dsn or os.getenv("DATABASE_URL", "").strip()
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = "postgresql://" + dsn[len("postgresql+asyncpg://"):]
        key = os.getenv("GRAPH_CHECKPOINT_FERNET_KEY", "").strip()
        if not dsn:
            raise RuntimeError("DATABASE_URL 未配置，无法启动图执行持久化")
        if not key:
            raise RuntimeError("GRAPH_CHECKPOINT_FERNET_KEY 未配置，拒绝保存明文图状态")

        try:
            from cryptography.fernet import Fernet, MultiFernet
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "图执行持久化依赖未安装，请安装 langgraph-checkpoint-postgres"
            ) from exc

        previous_keys = [value.strip() for value in os.getenv("GRAPH_CHECKPOINT_PREVIOUS_KEYS", "").split(",") if value.strip()]
        if previous_keys and not os.getenv("GRAPH_CHECKPOINT_IDENTITY_KEY"):
            raise RuntimeError("rotation_requires_stable_checkpoint_identity_key")
        cipher = MultiFernet([Fernet(value.encode("ascii")) for value in [key, *previous_keys]])

        class EncryptedCheckpointSerializer(JsonPlusSerializer):
            """Encrypt serialized checkpoint blobs and reject plaintext blob rows."""

            _prefix = "nexusmind-fernet:"

            def __init__(self) -> None:
                super().__init__(
                    allowed_msgpack_modules=(
                        (_EncryptedInlineValue.__module__, _EncryptedInlineValue.__name__),
                    )
                )

            def dumps_typed(self, value: Any) -> tuple[str, bytes]:
                type_tag, payload = super().dumps_typed(value)
                return self._prefix + type_tag, cipher.encrypt(payload)

            def loads_typed(self, value: tuple[str, bytes]) -> Any:
                type_tag, payload = value
                if not type_tag.startswith(self._prefix):
                    raise ValueError("拒绝读取未加密的 LangGraph 检查点")
                plaintext = cipher.decrypt(payload)
                return super().loads_typed((type_tag[len(self._prefix):], plaintext))

        class EncryptedAsyncPostgresSaver(AsyncPostgresSaver):
            """Protect fields that the upstream saver stores outside its serializer."""

            async def aput(self, config, checkpoint, metadata, new_versions):
                protected = _protect_checkpoint(checkpoint, self.serde)
                # Source and step are structural metadata. Drop arbitrary metadata
                # fields so user content cannot leak into PostgreSQL JSONB metadata.
                safe_metadata = {
                    key: metadata[key]
                    for key in ("source", "step")
                    if key in metadata
                }
                return await super().aput(
                    config, protected, safe_metadata, new_versions
                )

            async def aget_tuple(self, config):
                checkpoint_tuple = await super().aget_tuple(config)
                if checkpoint_tuple is None:
                    return None
                return _restore_checkpoint_tuple(checkpoint_tuple, self.serde)

            async def alist(self, config, *, filter=None, before=None, limit=None):
                async for checkpoint_tuple in super().alist(
                    config, filter=filter, before=before, limit=limit
                ):
                    yield _restore_checkpoint_tuple(checkpoint_tuple, self.serde)

        pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=max(4, int(os.getenv("GRAPH_CHECKPOINT_POOL_SIZE", "10"))),
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )
        try:
            await pool.open(wait=True)
            saver = EncryptedAsyncPostgresSaver(
                pool, serde=EncryptedCheckpointSerializer()
            )
            production = os.getenv("NEXUSMIND_ENV") == "production" and os.getenv("NEXUSMIND_SCHEMA_BOOTSTRAP") != "true"
            if not production:
                await saver.setup()
            async with pool.connection() as conn:
                if production:
                    cursor = await conn.execute("""SELECT to_regclass('checkpoints') IS NOT NULL
                        AND to_regclass('checkpoint_blobs') IS NOT NULL
                        AND to_regclass('checkpoint_writes') IS NOT NULL
                        AND to_regclass('nexusmind_checkpoint_threads') IS NOT NULL AS ready""")
                    if not (await cursor.fetchone())["ready"]:
                        raise RuntimeError("run_explicit_checkpoint_migration_first")
                else:
                    await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS nexusmind_checkpoint_threads (
                        thread_id TEXT PRIMARY KEY,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                    )
            self.pool = pool
            self.saver = saver
            logger.info("LangGraph PostgreSQL 检查点已初始化")
        except Exception:
            await pool.close()
            raise

    def require_saver(self):
        if self.saver is None:
            raise RuntimeError("LangGraph 检查点尚未初始化")
        return self.saver

    @asynccontextmanager
    async def session_lock(self, thread_id: str) -> AsyncIterator[None]:
        if self.pool is None:
            raise RuntimeError("LangGraph 检查点尚未初始化")
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked",
                    (thread_id,),
                )
            ).fetchone()
            if not row or not row["locked"]:
                raise GraphCheckpointBusy("该会话已有请求正在运行，请稍后再试")
            try:
                yield
            finally:
                await conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (thread_id,),
                )

    async def touch_thread(self, thread_id: str) -> None:
        if self.pool is None:
            raise RuntimeError("LangGraph 检查点尚未初始化")
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO nexusmind_checkpoint_threads(thread_id, updated_at)
                VALUES (%s, NOW())
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
                """,
                (thread_id,),
            )

    async def delete_thread(self, thread_id: str) -> None:
        if self.pool is None or self.saver is None:
            raise RuntimeError("LangGraph 检查点尚未初始化")
        async with self.session_lock(thread_id):
            async with self.session_lock(thread_id + ":snapshot"):
                async with self.pool.connection() as conn:
                    rows = await (await conn.execute(
                        "SELECT thread_id FROM nexusmind_checkpoint_threads WHERE starts_with(thread_id, %s)",
                        (thread_id + ":run:",),
                    )).fetchall()
                for target in [row["thread_id"] for row in rows] + [thread_id]:
                    await self.saver.adelete_thread(target)
                    async with self.pool.connection() as conn:
                        await conn.execute(
                            "DELETE FROM nexusmind_checkpoint_threads WHERE thread_id = %s",
                            (target,),
                        )

    async def save_session_snapshot(self, thread_id: str, snapshot: dict, *, is_deleted=None,
                                    expected_run_id: str | None = None) -> bool:
        """Save canonical history and the execution pointer in a fixed namespace.

        The separate snapshot lock also serializes saves with deletion without
        reacquiring the execution lock already held by the orchestrator.
        """
        from langgraph.checkpoint.base import empty_checkpoint
        saver = self.require_saver()
        async with self.session_lock(thread_id + ":snapshot"):
            if is_deleted is not None and await is_deleted():
                return False
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": SESSION_CHECKPOINT_NS}}
            previous = await saver.aget_tuple(config)
            if expected_run_id is not None:
                previous_state = ((previous.checkpoint.get("channel_values", {}).get("session_snapshot", {}).get("state", {}))
                                  if previous else {})
                if (previous_state.get("active_run_id") != expected_run_id
                        or previous_state.get("run_status") not in {"running", "interrupted"}):
                    return False
            checkpoint = empty_checkpoint()
            old_version = (previous.checkpoint.get("channel_versions", {}).get("session_snapshot")
                           if previous else None)
            version = saver.get_next_version(old_version, None)
            checkpoint["channel_values"] = {"session_snapshot": snapshot}
            checkpoint["channel_versions"] = {"session_snapshot": version}
            # Index before writing so interrupted writes remain discoverable for deletion.
            await self.touch_thread(thread_id)
            await saver.aput(config, checkpoint, {"source": "update", "step": 0}, {"session_snapshot": version})
            if is_deleted is not None and await is_deleted():
                await saver.adelete_thread(thread_id)
                return False
            return True

    async def load_session_snapshot(self, thread_id: str) -> dict | None:
        saved = await self.require_saver().aget_tuple({"configurable": {
            "thread_id": thread_id, "checkpoint_ns": SESSION_CHECKPOINT_NS,
        }})
        if saved is None:
            return None
        return saved.checkpoint.get("channel_values", {}).get("session_snapshot")

    async def thread_snapshot(self, thread_id: str, checkpoint_ns: str = ""):
        """Read persisted graph state for one session and execution namespace."""
        if self.saver is None:
            raise RuntimeError("LangGraph 检查点尚未初始化")
        saved = await self.saver.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
        )
        if saved is None:
            return SimpleNamespace(values={}, next=(), config=None, pending_writes=())
        values = dict(saved.checkpoint.get("channel_values") or {})
        # This is a saver-level view. Actual resumption uses graph.aget_state,
        # which applies task writes and the graph's scheduler to this tuple.
        next_nodes = [name[len("branch:to:"):] for name in values if name.startswith("branch:to:")]
        for send in saved.checkpoint.get("pending_sends") or ():
            node = getattr(send, "node", None)
            if node and node not in next_nodes:
                next_nodes.append(node)
        return SimpleNamespace(values=values, next=tuple(next_nodes), config=saved.config,
                               pending_writes=saved.pending_writes)

    async def prune_expired(self) -> int:
        """Delete whole graph threads after their configured idle retention."""
        if self.pool is None or self.saver is None:
            return 0
        ttl_days = max(1, int(os.getenv("GRAPH_CHECKPOINT_TTL_DAYS", "30")))
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT thread_id FROM nexusmind_checkpoint_threads
                    WHERE updated_at < NOW() - (%s * INTERVAL '1 day')
                    ORDER BY updated_at
                    LIMIT 100
                    """,
                    (ttl_days,),
                )
            ).fetchall()
        removed = 0
        for row in rows:
            try:
                await self.delete_thread(row["thread_id"])
                removed += 1
            except GraphCheckpointBusy:
                continue
        return removed

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
        self.pool = None
        self.saver = None


_manager = GraphCheckpointerManager()


def get_graph_checkpointer_manager() -> GraphCheckpointerManager:
    return _manager


async def initialize_graph_checkpointer() -> None:
    await _manager.initialize()


async def close_graph_checkpointer() -> None:
    await _manager.close()
