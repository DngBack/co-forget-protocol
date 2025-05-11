"""CrewAI tools for the Co-Forget Protocol."""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
from crewai.tools.base_tool import BaseTool
from crewai_tools import SerperDevTool

from .pinecone import PineconeManager
from .memory import MemoryQuota, MemoryBatch, MemoryScorer


class PineconeUpsertTool(BaseTool):
    """Tool for upserting memories to Pinecone."""

    name = "Pinecone Upsert"
    description = "Upsert a memory to Pinecone"

    def __init__(
        self,
        pinecone: PineconeManager,
        namespace: str,
        quota: Optional[MemoryQuota] = None,
    ):
        """Initialize the tool."""
        super().__init__(name=self.name, description=self.description)
        self.pinecone = pinecone
        self.namespace = namespace
        self.quota = quota
        self.batch = MemoryBatch()

    async def _run_async(self, text: str, metadata: Dict) -> str:
        """Run the tool asynchronously."""
        if not self.quota or self.quota.can_add_memory():
            try:
                if "timestamp" not in metadata:
                    metadata["timestamp"] = asyncio.get_event_loop().time()

                records = self.batch.add(text, metadata)
                if records:
                    await self.pinecone.upsert_records(
                        self.namespace, records, self.quota
                    )
                return "Memory queued for upsert"
            except Exception as e:
                logger.error(f"Error upserting memory: {str(e)}")
                raise
        return "Memory quota exceeded"

    def _run(self, text: str, metadata: Dict) -> str:
        """Run the tool."""
        return asyncio.run(self._run_async(text, metadata))


class PineconeRetrieveTool(BaseTool):
    """Tool for retrieving memories from Pinecone."""

    name = "Pinecone Retrieve"
    description = "Retrieve similar memories from Pinecone"

    def __init__(
        self,
        pinecone: PineconeManager,
        namespace: str,
        scorer: Optional[MemoryScorer] = None,
    ):
        """Initialize the tool."""
        super().__init__(name=self.name, description=self.description)
        self.pinecone = pinecone
        self.namespace = namespace
        self.scorer = scorer or MemoryScorer()

    async def _run_async(
        self, query: str, top_k: int = 5, min_score: float = 0.7
    ) -> List[Dict]:
        """Run the tool asynchronously."""
        try:
            matches = await self.pinecone.search(
                self.namespace, query, top_k, min_score
            )

            scored_matches = []
            for match in matches:
                metadata = match.get("metadata", {})
                timestamp = metadata.get("timestamp", asyncio.get_event_loop().time())
                score = self.scorer.calculate_score(timestamp)

                if not self.scorer.should_remove(score):
                    scored_matches.append({**match, "memory_score": score})

            logger.info(f"Retrieved {len(scored_matches)} memories")
            return scored_matches
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            raise

    def _run(self, query: str, top_k: int = 5, min_score: float = 0.7) -> List[Dict]:
        """Run the tool."""
        return asyncio.run(self._run_async(query, top_k, min_score))


class PineconeDeleteTool(BaseTool):
    """Tool for deleting memories from Pinecone."""

    name = "Pinecone Delete"
    description = "Delete memories from Pinecone"

    def __init__(
        self,
        pinecone: PineconeManager,
        namespace: str,
        quota: Optional[MemoryQuota] = None,
    ):
        """Initialize the tool."""
        super().__init__(name=self.name, description=self.description)
        self.pinecone = pinecone
        self.namespace = namespace
        self.quota = quota

    async def _run_async(self, memory_ids: List[str]) -> str:
        """Run the tool asynchronously."""
        if not memory_ids:
            return "No memories to delete"

        try:
            await self.pinecone.delete_records(self.namespace, memory_ids, self.quota)
            return f"Deleted memories: {memory_ids}"
        except Exception as e:
            logger.error(f"Error deleting memories: {str(e)}")
            raise

    def _run(self, memory_ids: List[str]) -> str:
        """Run the tool."""
        return asyncio.run(self._run_async(memory_ids))


def create_tools(
    pinecone: PineconeManager,
    memory_quota: Optional[MemoryQuota] = None,
    baseline_quota: Optional[MemoryQuota] = None,
) -> Dict[str, List[BaseTool]]:
    """Create all tools for the protocol."""
    memory_tools = [
        PineconeUpsertTool(pinecone, "memories", memory_quota),
        PineconeRetrieveTool(pinecone, "memories"),
        PineconeDeleteTool(pinecone, "memories", memory_quota),
        SerperDevTool(),
    ]

    baseline_tools = [
        PineconeUpsertTool(pinecone, "baseline_memories", baseline_quota),
        PineconeRetrieveTool(pinecone, "baseline_memories"),
        PineconeDeleteTool(pinecone, "baseline_memories", baseline_quota),
        SerperDevTool(),
    ]

    manager_tools = [
        PineconeRetrieveTool(pinecone, "memories"),
        PineconeDeleteTool(pinecone, "memories", memory_quota),
    ]

    coordinator_tools = [
        PineconeRetrieveTool(pinecone, "proposals"),
        PineconeDeleteTool(pinecone, "proposals"),
        PineconeDeleteTool(pinecone, "memories", memory_quota),
    ]

    return {
        "memory": memory_tools,
        "baseline": baseline_tools,
        "manager": manager_tools,
        "coordinator": coordinator_tools,
    }
