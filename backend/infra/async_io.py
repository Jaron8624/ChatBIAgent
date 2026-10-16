"""Separate bounded executors; cancellation retains capacity until work stops."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from weakref import WeakKeyDictionary

_pools = {
    "request": ThreadPoolExecutor(8, thread_name_prefix="request-io"),
    "tool": ThreadPoolExecutor(8, thread_name_prefix="agent-tool"),
    "maintenance": ThreadPoolExecutor(2, thread_name_prefix="maintenance-io"),
}
_limits = WeakKeyDictionary()


async def _run(pool, function, args, kwargs):
    loop = asyncio.get_running_loop()
    limits = _limits.setdefault(loop, {})
    limit = limits.setdefault(pool, asyncio.Semaphore(_pools[pool]._max_workers))
    await limit.acquire()
    try:
        future = _pools[pool].submit(
            copy_context().run, partial(function, *args, **kwargs)
        )
    except BaseException:
        limit.release()
        raise

    def finished(_):
        try:
            loop.call_soon_threadsafe(limit.release)
        except RuntimeError:  # the owning event loop has already shut down
            pass

    future.add_done_callback(finished)
    return await asyncio.wrap_future(future, loop=loop)


async def run_blocking(function, *args, **kwargs):
    return await _run("request", function, args, kwargs)


async def run_tool(function, *args, **kwargs):
    return await _run("tool", function, args, kwargs)


async def run_maintenance(function, *args, **kwargs):
    return await _run("maintenance", function, args, kwargs)
