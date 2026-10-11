"""Durable PostgreSQL upload jobs and shared-volume source storage.

API and workers must mount the same durable UPLOAD_STORAGE_ROOT. A local
ephemeral /tmp directory is deliberately not used as the accepted source.
"""
import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS rag_upload_jobs (
 job_id TEXT PRIMARY KEY, knowledge_base_id TEXT NOT NULL, user_id INTEGER NOT NULL,
 file_hash TEXT NOT NULL, filename TEXT NOT NULL, source_name TEXT NOT NULL,
 kb_generation TIMESTAMP NOT NULL,
 state TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
 lease_token TEXT, lease_until TIMESTAMPTZ, available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 error_type TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rag_upload_jobs_ready ON rag_upload_jobs(state,available_at);
ALTER TABLE rag_upload_jobs ADD COLUMN IF NOT EXISTS source_removed_at TIMESTAMPTZ;
"""


def storage_root():
    project = Path(__file__).resolve().parents[2]
    configured = Path(os.getenv("UPLOAD_STORAGE_ROOT", str(project / "data" / "uploads")))
    root = (configured if configured.is_absolute() else project / configured).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_path(name):
    if len(name) != 32 or not name.isalnum():
        raise ValueError("invalid_upload_source")
    return storage_root() / name


async def enqueue(pg, temporary, filename, file_hash, kb, owner):
    from rag.knowledge_consistency import KnowledgeOperations, bound_client
    job_id = uuid.uuid4().hex
    source = source_path(job_id)
    await asyncio.to_thread(shutil.copyfile, temporary, source)
    try:
        # The source is fsynced before committing acceptance. Crash before the
        # DB commit can leave an orphan source, never an accepted missing file.
        with source.open("r+b") as stream:
            os.fsync(stream.fileno())
        async with KnowledgeOperations(pg, None).lock(kb, owner) as conn:
            blocked = await conn.fetchval("""SELECT 1 FROM rag_operations WHERE
                knowledge_base_id=$1 AND user_id=$2 AND file_hash='' AND kind='delete'
                AND state <> 'complete'""", kb, owner)
            if blocked:
                raise RuntimeError("knowledge_base_deleting")
            await bound_client(pg, conn).create_knowledge_base(kb, owner)
            generation = await conn.fetchval("SELECT created_at FROM knowledge_bases WHERE knowledge_base_id=$1 AND user_id=$2", kb, owner)
            async with conn.transaction():
                await conn.execute("""DELETE FROM rag_operations WHERE knowledge_base_id=$1
                    AND user_id=$2 AND file_hash='' AND kind='delete' AND state='complete'""", kb, owner)
                await conn.execute("""INSERT INTO rag_upload_jobs
                    (job_id,knowledge_base_id,user_id,file_hash,filename,source_name,kb_generation)
                    VALUES($1,$2,$3,$4,$5,$1,$6)""", job_id, kb, owner, file_hash, filename, generation)
        return job_id
    except BaseException:
        source.unlink(missing_ok=True)
        raise


class UploadWorker:
    def __init__(self, pg, processor, lease_seconds=120):
        self.pg, self.processor, self.lease_seconds = pg, processor, lease_seconds

    async def claim(self):
        token = uuid.uuid4().hex
        async with self.pg.pool.acquire() as conn:
            await conn.execute("""UPDATE rag_upload_jobs SET state='failed',error_type='LeaseExpired',updated_at=now()
                WHERE state='running' AND lease_until<now() AND attempts>=5""")
            return await conn.fetchrow("""WITH next_job AS (
                SELECT job_id FROM rag_upload_jobs WHERE
                ((state='queued' AND available_at<=now()) OR (state='running' AND lease_until<now()))
                AND attempts < 5 ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
                UPDATE rag_upload_jobs j SET state='running', attempts=attempts+1,
                lease_token=$1,lease_until=now()+$2*interval '1 second',updated_at=now()
                FROM next_job n WHERE j.job_id=n.job_id RETURNING j.*""", token, self.lease_seconds)

    async def heartbeat(self, job):
        while True:
            await asyncio.sleep(max(1, self.lease_seconds // 3))
            async with self.pg.pool.acquire() as conn:
                changed = await conn.execute("""UPDATE rag_upload_jobs SET
                    lease_until=now()+$3*interval '1 second' WHERE job_id=$1
                    AND lease_token=$2 AND state='running'""", job["job_id"], job["lease_token"], self.lease_seconds)
                if changed != "UPDATE 1":
                    raise RuntimeError("upload_lease_lost")

    async def cleanup_sources(self):
        """Delete only database-owned terminal sources after configured retention."""
        complete_days = max(1, int(os.getenv("UPLOAD_COMPLETED_RETENTION_DAYS", "30")))
        failed_days = max(1, int(os.getenv("UPLOAD_FAILED_RETENTION_DAYS", "7")))
        async with self.pg.pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch("""SELECT job_id,source_name FROM rag_upload_jobs
                    WHERE source_removed_at IS NULL AND
                    ((state='complete' AND updated_at<now()-$1*interval '1 day') OR
                     (state='failed' AND updated_at<now()-$2*interval '1 day'))
                    ORDER BY updated_at FOR UPDATE SKIP LOCKED LIMIT 100""", complete_days, failed_days)
                for row in rows:
                    source_path(row["source_name"]).unlink(missing_ok=True)
                    await conn.execute("UPDATE rag_upload_jobs SET source_removed_at=now() WHERE job_id=$1", row["job_id"])
        return len(rows)

    async def run_once(self):
        job = await self.claim()
        if job is None:
            return False
        source = source_path(job["source_name"])
        heartbeat = asyncio.create_task(self.heartbeat(job))
        try:
            async def publish_guard(conn):
                valid = await conn.fetchval("""SELECT 1 FROM rag_upload_jobs WHERE job_id=$1
                    AND lease_token=$2 AND state='running' AND lease_until>now()""",
                    job["job_id"], job["lease_token"])
                if not valid:
                    raise ValueError("upload_lease_lost")
                deleted = await conn.fetchval("""SELECT 1 FROM rag_operations WHERE
                    knowledge_base_id=$1 AND user_id=$2 AND kind='delete'
                    AND (file_hash='' OR file_hash=$3) AND updated_at >= $4""",
                    job["knowledge_base_id"], job["user_id"], job["file_hash"], job["created_at"])
                if deleted:
                    raise ValueError("upload_deleted_after_acceptance")
            with tempfile.TemporaryDirectory(prefix="nexusmind-job-") as folder:
                staged = Path(folder) / ("document" + Path(job["filename"]).suffix)
                await asyncio.to_thread(shutil.copyfile, source, staged)
                await self.processor.process_document(staged, job["filename"], job["file_hash"],
                    job["knowledge_base_id"], job["user_id"], expected_generation=job["kb_generation"],
                    publish_guard=publish_guard)
            async with self.pg.pool.acquire() as conn:
                changed = await conn.execute("""UPDATE rag_upload_jobs SET state='complete',
                    error_type=NULL,lease_until=NULL,updated_at=now() WHERE job_id=$1
                    AND lease_token=$2 AND state='running'""", job["job_id"], job["lease_token"])
            if changed == "UPDATE 1":
                if os.getenv("UPLOAD_RETAIN_COMPLETED", "true").lower() != "true":
                    source.unlink(missing_ok=True)
        except asyncio.CancelledError:
            # Keep source and lease. A different worker can recover it after expiry.
            raise
        except Exception as exc:
            terminal = isinstance(exc, (ValueError, FileNotFoundError)) or job["attempts"] >= 5
            async with self.pg.pool.acquire() as conn:
                await conn.execute("""UPDATE rag_upload_jobs SET state=$3,error_type=$4,
                    available_at=now()+interval '30 seconds',lease_until=NULL,updated_at=now()
                    WHERE job_id=$1 AND lease_token=$2 AND state='running'""",
                    job["job_id"], job["lease_token"], "failed" if terminal else "queued", type(exc).__name__)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        return True

    async def run(self):
        last_cleanup = 0
        while True:
            if asyncio.get_running_loop().time() - last_cleanup > 300:
                await self.cleanup_sources()
                last_cleanup = asyncio.get_running_loop().time()
            if not await self.run_once():
                await asyncio.sleep(2)
