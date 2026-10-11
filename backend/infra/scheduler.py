"""Non-blocking single-executor guard for periodic maintenance."""
from contextlib import asynccontextmanager


@asynccontextmanager
async def maintenance_leader(name):
    from infra.postgresql_client import get_postgresql_client
    pg = await get_postgresql_client()
    async with pg.pool.acquire() as conn:
        acquired = await conn.fetchval("SELECT pg_try_advisory_lock(hashtextextended($1,0))", "maintenance:" + name)
        try:
            yield acquired
        finally:
            if acquired:
                await conn.execute("SELECT pg_advisory_unlock(hashtextextended($1,0))", "maintenance:" + name)
