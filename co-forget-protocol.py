import os
import time
import math
import asyncio
import logging
from typing import Sequence, List, Dict, Optional, Union, Any
from datetime import datetime
from pinecone import Pinecone, IndexEmbed
from crewai import Agent, Crew, Task, Process
from crewai.agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai_tools import SerperDevTool

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Memory Management Classes
class MemoryQuota:
    def __init__(self, max_memories: int):
        self.max_memories = max_memories
        self.current_count = 0

    def can_add_memory(self) -> bool:
        return self.current_count < self.max_memories

    def increment(self):
        self.current_count += 1

    def decrement(self):
        self.current_count = max(0, self.current_count - 1)


class MemoryScorer:
    @staticmethod
    def calculate_score(timestamp: float, importance: float = 1.0) -> float:
        time_decay = math.exp(-(time.time() - timestamp) / 10)
        return time_decay * importance

    @staticmethod
    def should_remove(score: float, threshold: float = 0.3) -> bool:
        return score < threshold


class MemoryBatch:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.records: List[dict] = []

    def add(self, text: str, metadata: dict):
        self.records.append({"text": text, "metadata": metadata})
        if len(self.records) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> List[dict]:
        if not self.records:
            return []
        records = self.records.copy()
        self.records.clear()
        return records


# Pinecone Setup
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = "co-forgetting-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed=IndexEmbed(model="all-MiniLM-L6-v2", field_map={"text": "text"}),
    )
index = pc.Index(index_name)


# Custom Tools
class PineconeUpsertTool(BaseTool):
    name = "Pinecone Upsert"
    description = "Upsert a memory to Pinecone"

    def __init__(self, pc, index_name, namespace, quota: Optional[MemoryQuota] = None):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace
        self.quota = quota or MemoryQuota(max_memories=10000)
        self.batch = MemoryBatch()

    def _run(self, text: str, metadata: dict) -> str:
        if not self.quota.can_add_memory():
            logger.warning(f"Memory quota exceeded in namespace {self.namespace}")
            return "Memory quota exceeded"

        try:
            # Add timestamp if not present
            if "timestamp" not in metadata:
                metadata["timestamp"] = time.time()

            # Add to batch
            records = self.batch.add(text, metadata)
            if records:
                self.index.upsert_records(self.namespace, records)
                self.quota.increment()
                logger.info(
                    f"Batch upserted {len(records)} memories to {self.namespace}"
                )

            return "Memory queued for upsert"
        except Exception as e:
            logger.error(f"Error upserting memory: {str(e)}")
            raise

    def flush(self) -> str:
        """Force flush any remaining records in the batch"""
        records = self.batch.flush()
        if records:
            self.index.upsert_records(self.namespace, records)
            self.quota.increment()
            logger.info(f"Flushed {len(records)} memories to {self.namespace}")
        return f"Flushed {len(records)} memories"


class PineconeRetrieveTool(BaseTool):
    name = "Pinecone Retrieve"
    description = "Retrieve similar memories from Pinecone"

    def __init__(self, pc, index_name, namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def _run(self, query: str, top_k: int = 5, min_score: float = 0.7) -> list:
        try:
            results = self.index.search(
                self.namespace,
                {
                    "top_k": top_k,
                    "inputs": {"text": query},
                    "score_threshold": min_score,
                },
            )
            matches = results.get("matches", [])

            # Score and filter memories
            current_time = time.time()
            scored_matches = []
            for match in matches:
                metadata = match.get("metadata", {})
                timestamp = metadata.get("timestamp", current_time)
                score = MemoryScorer.calculate_score(timestamp)

                if not MemoryScorer.should_remove(score):
                    scored_matches.append({**match, "memory_score": score})

            logger.info(
                f"Retrieved {len(scored_matches)} memories from {self.namespace}"
            )
            return scored_matches
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            raise


class PineconeListMemoriesTool(BaseTool):
    name = "Pinecone List Memories"
    description = "List all memory IDs in Pinecone"

    def __init__(self, pc, index_name, namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def _run(self) -> list:
        ids = self.index.list(namespace=self.namespace)
        return ids if ids else []


class PineconeFetchMemoriesTool(BaseTool):
    name = "Pinecone Fetch Memories"
    description = "Fetch memories by IDs from Pinecone"

    def __init__(self, pc, index_name, namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def _run(self, ids: list) -> dict:
        if not ids:
            return {}
        vectors = self.index.fetch(ids=ids, namespace=self.namespace)
        return vectors.get("vectors", {})


class PineconeProposeRemovalTool(BaseTool):
    name = "Pinecone Propose Removal"
    description = "Propose a memory for removal"

    def __init__(self, pc, index_name, proposals_namespace, agent_id):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.proposals_namespace = proposals_namespace
        self.agent_id = agent_id
        self.batch = MemoryBatch()

    async def _run_async(self, memory_id: str, score: float) -> str:
        try:
            text = "proposal"
            metadata = {
                "memory_id": memory_id,
                "agent_id": self.agent_id,
                "score": score,
                "timestamp": time.time(),
            }
            records = self.batch.add(text, metadata)
            if records:
                await asyncio.to_thread(
                    self.index.upsert_records, self.proposals_namespace, records
                )
                logger.info(
                    f"Proposed removal of memory {memory_id} with score {score}"
                )
            return f"Proposed to remove memory {memory_id}"
        except Exception as e:
            logger.error(f"Error proposing removal: {str(e)}")
            raise

    def _run(self, memory_id: str, score: float) -> str:
        return asyncio.run(self._run_async(memory_id, score))

    async def flush_async(self) -> str:
        records = self.batch.flush()
        if records:
            await asyncio.to_thread(
                self.index.upsert_records, self.proposals_namespace, records
            )
        return f"Flushed {len(records)} proposals"


class PineconeRetrieveProposalsTool(BaseTool):
    name = "Pinecone Retrieve Proposals"
    description = "Retrieve all proposed memories for removal"

    def __init__(self, pc, index_name, proposals_namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.proposals_namespace = proposals_namespace

    async def _run_async(self) -> List[Dict]:
        try:
            ids = await asyncio.to_thread(
                self.index.list, namespace=self.proposals_namespace
            )
            if not ids:
                return []

            vectors = await asyncio.to_thread(
                self.index.fetch, ids=ids, namespace=self.proposals_namespace
            )

            proposals = []
            for vec in vectors.get("vectors", {}).values():
                metadata = vec.get("metadata", {})
                # Group proposals by memory_id
                memory_id = metadata.get("memory_id")
                if memory_id:
                    proposals.append(
                        {
                            "memory_id": memory_id,
                            "agent_id": metadata.get("agent_id"),
                            "score": metadata.get("score", 0),
                            "timestamp": metadata.get("timestamp", 0),
                        }
                    )

            logger.info(f"Retrieved {len(proposals)} proposals")
            return proposals
        except Exception as e:
            logger.error(f"Error retrieving proposals: {str(e)}")
            raise

    def _run(self) -> List[Dict]:
        return asyncio.run(self._run_async())


class PineconeDeleteMemoriesTool(BaseTool):
    name = "Pinecone Delete Memories"
    description = "Delete specified memories from Pinecone"

    def __init__(self, pc, index_name, namespace, quota: Optional[MemoryQuota] = None):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace
        self.quota = quota

    async def _run_async(self, memory_ids: List[str]) -> str:
        if not memory_ids:
            return "No memories to delete"

        try:
            await asyncio.to_thread(
                self.index.delete, ids=memory_ids, namespace=self.namespace
            )

            if self.quota:
                for _ in memory_ids:
                    self.quota.decrement()

            logger.info(f"Deleted {len(memory_ids)} memories from {self.namespace}")
            return f"Deleted memories: {memory_ids}"
        except Exception as e:
            logger.error(f"Error deleting memories: {str(e)}")
            raise

    def _run(self, memory_ids: List[str]) -> str:
        return asyncio.run(self._run_async(memory_ids))


# Agents
memory_quota = MemoryQuota(max_memories=10000)
baseline_quota = MemoryQuota(max_memories=10000)

memory_managers = [
    Agent(
        role="Memory Manager",
        goal="Manage shared memory by proposing outdated memories for removal",
        backstory="You maintain the relevance of shared memory.",
        tools=[
            PineconeListMemoriesTool(pc, index_name, "memories"),
            PineconeFetchMemoriesTool(pc, index_name, "memories"),
            PineconeProposeRemovalTool(pc, index_name, "proposals", f"manager_{i}"),
        ],
        verbose=True,
    )
    for i in range(3)
]

task_performer = Agent(
    role="Task Performer",
    goal="Answer questions using shared memory or web search",
    backstory="You excel at using shared knowledge to answer queries.",
    tools=[
        PineconeUpsertTool(pc, index_name, "memories", quota=memory_quota),
        PineconeRetrieveTool(pc, index_name, "memories"),
        SerperDevTool(),
    ],
    verbose=True,
)

coordinator = Agent(
    role="Coordinator",
    goal="Coordinate pruning by collecting proposals and removing memories",
    backstory="You oversee efficient memory management.",
    tools=[
        PineconeRetrieveProposalsTool(pc, index_name, "proposals"),
        PineconeDeleteMemoriesTool(pc, index_name, "memories", quota=memory_quota),
        PineconeDeleteMemoriesTool(pc, index_name, "proposals"),
    ],
    verbose=True,
)

# Tasks
question_task = Task(
    description="Answer the question using shared memory. If not found, use web search and store the answer in memory with current timestamp and agent ID.",
    agent=task_performer,
    expected_output="A concise answer to the question",
)

propose_task = Task(
    description="List all memory IDs, fetch memories, calculate decay score D(t) = exp(-(current_time - timestamp) / 10). Propose memories with D(t) < 0.3 for removal.",
    agent=None,  # Assigned dynamically
    expected_output="List of proposed memory IDs",
)

coordinate_task = Task(
    description="Retrieve all proposals, group by memory ID, count votes. Remove memories with 2+ votes, clear proposals namespace.",
    agent=coordinator,
    expected_output="List of removed memory IDs",
)

# Baseline Agent (for comparison)
baseline_agent = Agent(
    role="Baseline Agent",
    goal="Answer questions and manage memory independently",
    backstory="You operate without collaborative pruning.",
    tools=[
        PineconeUpsertTool(pc, index_name, "baseline_memories", quota=baseline_quota),
        PineconeRetrieveTool(pc, index_name, "baseline_memories"),
        PineconeListMemoriesTool(pc, index_name, "baseline_memories"),
        PineconeFetchMemoriesTool(pc, index_name, "baseline_memories"),
        PineconeDeleteMemoriesTool(
            pc, index_name, "baseline_memories", quota=baseline_quota
        ),
        SerperDevTool(),
    ],
    verbose=True,
)

baseline_question_task = Task(
    description="Answer the question using baseline memory. If not found, use web search and store in memory. Prune memories with D(t) < 0.3 independently.",
    agent=baseline_agent,
    expected_output="A concise answer to the question",
)


# Main Execution
async def process_question(
    question: str, agent: BaseAgent, is_baseline: bool = False
) -> Any:
    crew = Crew(
        agents=[agent],
        tasks=[
            Task(
                description=f"Answer: {question}",
                agent=agent,
                expected_output="A concise answer",
            )
        ],
        process=Process.sequential,
    )
    result = await asyncio.to_thread(crew.kickoff)
    return str(result)  # Convert CrewOutput to string


async def prune_memories(agents: Sequence[BaseAgent], coordinator: BaseAgent):
    # Get proposals from all memory managers
    propose_tasks = []
    for agent in agents:
        if isinstance(agent, Agent) and agent.role == "Memory Manager":
            task = Task(
                description=propose_task.description,
                agent=agent,
                expected_output=propose_task.expected_output,
            )
            propose_tasks.append(task)

    # Execute proposals and coordination
    pruning_crew = Crew(
        agents=list(agents) + [coordinator],
        tasks=propose_tasks + [coordinate_task],
        process=Process.sequential,
    )
    await asyncio.to_thread(pruning_crew.kickoff)


async def main_async():
    questions = ["Who is the CEO of Tesla?", "What is the capital of France?"]
    protocol_answers = []
    baseline_answers = []

    for i, question in enumerate(questions):
        # Process questions concurrently
        protocol_answer, baseline_answer = await asyncio.gather(
            process_question(question, task_performer),
            process_question(question, baseline_agent, is_baseline=True),
        )

        protocol_answers.append(protocol_answer)
        baseline_answers.append(baseline_answer)

        # Prune after every 2 questions
        if (i + 1) % 2 == 0:
            await asyncio.gather(
                prune_memories(memory_managers, coordinator),
                process_question(
                    "", baseline_agent, is_baseline=True
                ),  # Baseline pruning
            )

            # Update memory counts
            protocol_stats = await asyncio.to_thread(index.describe_index_stats)
            baseline_stats = await asyncio.to_thread(index.describe_index_stats)

            protocol_memory_count = (
                protocol_stats.get("namespaces", {})
                .get("memories", {})
                .get("vector_count", 0)
            )
            baseline_memory_count = (
                baseline_stats.get("namespaces", {})
                .get("baseline_memories", {})
                .get("vector_count", 0)
            )

            logger.info(f"Protocol Memory Count: {protocol_memory_count}")
            logger.info(f"Baseline Memory Count: {baseline_memory_count}")

    return protocol_answers, baseline_answers


def main():
    protocol_answers, baseline_answers = asyncio.run(main_async())
    print("Protocol Answers:", protocol_answers)
    print("Baseline Answers:", baseline_answers)


if __name__ == "__main__":
    main()
