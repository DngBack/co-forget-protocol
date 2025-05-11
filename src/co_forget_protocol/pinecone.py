"""Pinecone manager for the Co-Forget Protocol."""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
from pinecone import Pinecone, IndexEmbed

from .config import PineconeConfig
from .memory import MemoryQuota, MemoryBatch


class PineconeManager:
    """Manages Pinecone operations and index."""

    def __init__(self, config: PineconeConfig):
        """Initialize Pinecone manager."""
        self.config = config
        self.pc = Pinecone(api_key=config.api_key)
        self._ensure_index()
        self.index = self.pc.Index(config.index_name)

    def _ensure_index(self) -> None:
        """Ensure the Pinecone index exists."""
        if not self.pc.has_index(self.config.index_name):
            self.pc.create_index_for_model(
                name=self.config.index_name,
                cloud=self.config.cloud,
                region=self.config.region,
                embed=IndexEmbed(
                    model=self.config.model, field_map=self.config.field_map
                ),
            )

    async def upsert_records(
        self, namespace: str, records: List[Dict], quota: Optional[MemoryQuota] = None
    ) -> None:
        """Upsert records to Pinecone."""
        if not records:
            return

        if quota and not all(quota.can_add_memory() for _ in records):
            logger.warning(f"Memory quota exceeded in namespace {namespace}")
            return

        try:
            await asyncio.to_thread(
                self.index.upsert_records, namespace=namespace, records=records
            )
            if quota:
                for _ in records:
                    quota.increment()
            logger.info(f"Upserted {len(records)} records to {namespace}")
        except Exception as e:
            logger.error(f"Error upserting records: {str(e)}")
            raise

    async def search(
        self, namespace: str, query: str, top_k: int = 5, min_score: float = 0.7
    ) -> List[Dict]:
        """Search for similar records."""
        try:
            results = await asyncio.to_thread(
                self.index.search,
                namespace=namespace,
                query={
                    "top_k": top_k,
                    "inputs": {"text": query},
                    "score_threshold": min_score,
                },
            )
            return results.get("matches", [])
        except Exception as e:
            logger.error(f"Error searching records: {str(e)}")
            raise

    async def delete_records(
        self, namespace: str, ids: List[str], quota: Optional[MemoryQuota] = None
    ) -> None:
        """Delete records from Pinecone."""
        if not ids:
            return

        try:
            await asyncio.to_thread(self.index.delete, namespace=namespace, ids=ids)
            if quota:
                for _ in ids:
                    quota.decrement()
            logger.info(f"Deleted {len(ids)} records from {namespace}")
        except Exception as e:
            logger.error(f"Error deleting records: {str(e)}")
            raise

    async def list_records(self, namespace: str) -> List[str]:
        """List all record IDs in a namespace."""
        try:
            ids = await asyncio.to_thread(self.index.list, namespace=namespace)
            return ids if ids else []
        except Exception as e:
            logger.error(f"Error listing records: {str(e)}")
            raise

    async def fetch_records(self, namespace: str, ids: List[str]) -> Dict[str, Any]:
        """Fetch records by IDs."""
        if not ids:
            return {}

        try:
            vectors = await asyncio.to_thread(
                self.index.fetch, namespace=namespace, ids=ids
            )
            return vectors.get("vectors", {})
        except Exception as e:
            logger.error(f"Error fetching records: {str(e)}")
            raise
