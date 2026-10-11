"""Run as: python -m rag.upload_worker (from backend or with PYTHONPATH)."""
import asyncio
import os
from dotenv import load_dotenv


async def main():
    load_dotenv()
    from infra.postgresql_client import get_postgresql_client
    from rag.milvus_client import get_milvus_client
    from rag.documents_process import DocumentProcessor
    from rag.jobs import UploadWorker
    from infra.hash_storage import HashStorage
    pg = await get_postgresql_client()
    milvus = None
    tasks = []
    try:
        milvus = await get_milvus_client()
        if os.getenv("MAINTENANCE_ENABLED", "false").lower() == "true":
            from main import check_and_store_memory, prune_graph_checkpoints
            from infra.graph_checkpointer import initialize_graph_checkpointer
            tasks.append(asyncio.create_task(check_and_store_memory()))
            if os.getenv("GRAPH_CHECKPOINT_FERNET_KEY"):
                await initialize_graph_checkpointer()
                tasks.append(asyncio.create_task(prune_graph_checkpoints()))
            from evaluation.evolution_loop import get_evolution_loop
            if os.getenv("EVOLUTION_LOOP_ENABLED", "false").lower() == "true":
                tasks.append(asyncio.create_task(get_evolution_loop().run()))
        await UploadWorker(pg, DocumentProcessor(HashStorage())).run()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        from infra.graph_checkpointer import close_graph_checkpointer
        await close_graph_checkpointer()
        if milvus is not None:
            await milvus.close()
        await pg.close()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
