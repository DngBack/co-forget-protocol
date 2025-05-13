"""CrewAI agents for the Co-Forget Protocol."""

from typing import List, Dict, Optional, Union, Sequence, cast, Any
from crewai import Agent, Task
from crewai.agent import BaseAgent
from crewai.tools import BaseTool
import math
import time

from .tools import create_tools
from .pinecone import PineconeManager
from .memory import MemoryQuota, MemoryManager
from .voting import LLMVoter


class MemoryManagerAgent(Agent):
    """Custom agent for memory management with LLM voting."""

    def __init__(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        tools: List[BaseTool],
        memory_manager: MemoryManager,
        voter: LLMVoter,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the memory manager agent."""
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            verbose=verbose,
            **kwargs,
        )
        self.memory_manager = memory_manager
        self.voter = voter

    async def propose_forgetting(
        self, memory_ids: List[str], current_context: str
    ) -> Dict[str, float]:
        """Propose memories for forgetting using LLM voting."""
        proposals = {}
        for memory_id in memory_ids:
            memory = self.memory_manager.get_memory(memory_id)
            if not memory:
                continue

            # Calculate decay score
            decay_score = math.exp(
                -(time.time() - memory["timestamp"]) / self.memory_manager.decay_factor
            )

            # Get LLM vote
            vote, confidence = self.voter.vote(
                memory["text"], memory["metadata"], current_context, decay_score
            )

            if vote == "forget":
                proposals[memory_id] = confidence

        return proposals


AgentDict = Dict[str, Union[Sequence[BaseAgent], BaseAgent]]


def create_agents(
    pinecone: PineconeManager,
    memory_quota: Optional[MemoryQuota] = None,
    baseline_quota: Optional[MemoryQuota] = None,
    memory_managers: Optional[List[MemoryManager]] = None,
    voter: Optional[LLMVoter] = None,
    verbose: bool = True,
) -> AgentDict:
    """Create all agents for the protocol."""
    tools = create_tools(pinecone, memory_quota, baseline_quota)

    if not memory_managers or not voter:
        raise ValueError("Memory managers and voter must be provided")

    # Create memory managers
    memory_managers_agents = [
        MemoryManagerAgent(
            role="Memory Manager",
            goal="Manage shared memory by proposing outdated memories for removal",
            backstory="You maintain the relevance of shared memory using LLM-based voting.",
            tools=tools["manager"],
            verbose=verbose,
            memory_manager=manager,
            voter=voter,
        )
        for manager in memory_managers
    ]

    # Create task performer
    task_performer = Agent(
        role="Task Performer",
        goal="Answer questions using shared memory or web search",
        backstory="You excel at using shared knowledge to answer queries.",
        tools=tools["memory"],
        verbose=verbose,
    )

    # Create coordinator
    coordinator = Agent(
        role="Coordinator",
        goal="Coordinate pruning by collecting proposals and removing memories",
        backstory="You oversee efficient memory management using PBFT consensus.",
        tools=tools["coordinator"],
        verbose=verbose,
    )

    # Create baseline agent
    baseline_agent = Agent(
        role="Baseline Agent",
        goal="Answer questions and manage memory independently",
        backstory="You operate without collaborative pruning.",
        tools=tools["baseline"],
        verbose=verbose,
    )

    return {
        "memory_managers": memory_managers_agents,
        "task_performer": task_performer,
        "coordinator": coordinator,
        "baseline": baseline_agent,
    }


def create_tasks(agents: AgentDict) -> Dict[str, Task]:
    """Create tasks for the protocol."""
    # Cast agents to their specific types
    task_performer = cast(BaseAgent, agents["task_performer"])
    coordinator = cast(BaseAgent, agents["coordinator"])
    baseline = cast(BaseAgent, agents["baseline"])

    question_task = Task(
        description=(
            "Answer the question using shared memory. If not found, use web search "
            "and store the answer in memory with current timestamp and agent ID."
        ),
        agent=task_performer,
        expected_output="A concise answer to the question",
    )

    propose_task = Task(
        description=(
            "List all memory IDs, fetch memories, calculate decay score "
            "D(t) = exp(-(current_time - timestamp) / 10). "
            "Use LLM to evaluate memory relevance. "
            "Propose memories with D(t) < 0.3 or low relevance for removal."
        ),
        agent=None,  # Assigned dynamically
        expected_output="List of proposed memory IDs with confidence scores",
    )

    coordinate_task = Task(
        description=(
            "Run PBFT consensus on memory proposals. "
            "Collect votes from all agents, verify signatures, "
            "and commit memory deletions when consensus is reached."
        ),
        agent=coordinator,
        expected_output="List of removed memory IDs",
    )

    baseline_task = Task(
        description=(
            "Answer the question using baseline memory. If not found, use web search "
            "and store in memory. Prune memories with D(t) < 0.3 independently."
        ),
        agent=baseline,
        expected_output="A concise answer to the question",
    )

    return {
        "question": question_task,
        "propose": propose_task,
        "coordinate": coordinate_task,
        "baseline": baseline_task,
    }
