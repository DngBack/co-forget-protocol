"""Main execution module for the Co-Forget Protocol."""

import asyncio
from typing import List, Dict, Any, Tuple, Optional, Set
from loguru import logger
from crewai import Crew, Process, Task

from .config import Settings
from .memory import MemoryQuota, MemoryManager
from .pinecone import PineconeManager
from .agents import create_agents, create_tasks, AgentDict
from .pbft import PBFTCoordinator, PBFTService
from .voting import LLMVoter


class Protocol:
    """Co-Forget Protocol implementation."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the protocol."""
        self.settings = settings or Settings()
        self.pinecone = PineconeManager(self.settings.pinecone)
        self.memory_quota = MemoryQuota(self.settings.memory.max_memories)
        self.baseline_quota = MemoryQuota(self.settings.memory.max_memories)

        # Initialize memory managers with SQLite and caching
        self.memory_managers = [
            MemoryManager(
                db_path=f"memories_{i}.db",
                cache_size=self.settings.memory.cache_size,
                batch_size=self.settings.memory.batch_size,
            )
            for i in range(self.settings.agent.num_memory_managers)
        ]

        # Initialize LLM voter
        self.voter = LLMVoter(
            model_name=self.settings.agent.llm_model,
            relevance_threshold=self.settings.agent.relevance_threshold,
        )

        # Initialize PBFT coordinator
        self.pbft_coordinator = PBFTCoordinator(
            num_agents=self.settings.agent.num_memory_managers,
            max_faulty=self.settings.agent.max_faulty,
        )

        # Create agents with new components
        self.agents = create_agents(
            self.pinecone,
            self.memory_quota,
            self.baseline_quota,
            self.memory_managers,
            self.voter,
            self.settings.agent.verbose,
        )
        self.tasks = create_tasks(self.agents)

    async def process_question(
        self, question: str, agent: Any, is_baseline: bool = False
    ) -> str:
        """Process a single question."""
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
        return str(result)

    async def prune_memories(self) -> None:
        """Prune memories using PBFT consensus."""
        # Get all memory IDs
        memory_ids: Set[str] = set()
        for manager in self.memory_managers:
            memory_ids.update(manager.get_all_memory_ids())

        # Run PBFT consensus
        to_delete = await self.pbft_coordinator.run_consensus(
            memory_ids, [manager.agent_id for manager in self.memory_managers]
        )

        # Delete memories that reached consensus
        for memory_id in to_delete:
            for manager in self.memory_managers:
                manager.delete_memory(memory_id)
            self.memory_quota.decrement()

        # Flush any pending batch operations
        for manager in self.memory_managers:
            manager.flush()

        logger.info(f"Deleted {len(to_delete)} memories through consensus")

    async def run(
        self, questions: List[str], prune_interval: int = 2
    ) -> Tuple[List[str], List[str]]:
        """Run the protocol on a list of questions."""
        protocol_answers = []
        baseline_answers = []

        for i, question in enumerate(questions):
            # Process questions concurrently
            protocol_answer, baseline_answer = await asyncio.gather(
                self.process_question(question, self.agents["task_performer"]),
                self.process_question(
                    question, self.agents["baseline"], is_baseline=True
                ),
            )

            protocol_answers.append(protocol_answer)
            baseline_answers.append(baseline_answer)

            # Prune after every N questions
            if (i + 1) % prune_interval == 0:
                await asyncio.gather(
                    self.prune_memories(),
                    self.process_question(
                        "", self.agents["baseline"], is_baseline=True
                    ),  # Baseline pruning
                )

                # Update memory counts
                protocol_stats = await asyncio.to_thread(
                    self.pinecone.index.describe_index_stats
                )
                baseline_stats = await asyncio.to_thread(
                    self.pinecone.index.describe_index_stats
                )

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


async def main_async(questions: List[str]) -> Tuple[List[str], List[str]]:
    """Run the protocol asynchronously."""
    protocol = Protocol()
    return await protocol.run(questions)


def main(questions: Optional[List[str]] = None) -> None:
    """Run the protocol."""
    if questions is None:
        questions = [
            "Who is the CEO of Tesla?",
            "What is the capital of France?",
        ]

    protocol_answers, baseline_answers = asyncio.run(main_async(questions))
    print("Protocol Answers:", protocol_answers)
    print("Baseline Answers:", baseline_answers)


if __name__ == "__main__":
    main()
