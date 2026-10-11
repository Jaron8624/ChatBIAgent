"""Bounded adapter for legacy synchronous repositories on online paths."""
import asyncio

_limit = asyncio.Semaphore(8)


async def run_blocking(function, *args, **kwargs):
    async with _limit:
        return await asyncio.to_thread(function, *args, **kwargs)
