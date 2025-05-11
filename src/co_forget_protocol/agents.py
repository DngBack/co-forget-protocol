"""CrewAI agents for the Co-Forget Protocol."""

from typing import List, Dict, Optional, Union, Sequence, cast
from crewai import Agent, Task
from crewai.agent import BaseAgent

from .tools import create_tools
from .pinecone import PineconeManager
from .memory import MemoryQuota


AgentDict = Dict[str, Union[Sequence[BaseAgent], BaseAgent]]


def create_agents(
    pinecone: PineconeManager,
    memory_quota: Optional[MemoryQuota] = None,
    baseline_quota: Optional[MemoryQuota] = None,
    num_managers: int = 3,
    verbose: bool = True,
) -> AgentDict:
    """Create all agents for the protocol."""
    tools = create_tools(pinecone, memory_quota, baseline_quota)

    # Create memory managers
    memory_managers = [
        Agent(
            role="Memory Manager",
            goal="Manage shared memory by proposing outdated memories for removal",
            backstory="You maintain the relevance of shared memory.",
            tools=tools["manager"],
            verbose=verbose,
        )
        for _ in range(num_managers)
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
        backstory="You oversee efficient memory management.",
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
        "memory_managers": memory_managers,
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
            "Propose memories with D(t) < 0.3 for removal."
        ),
        agent=None,  # Assigned dynamically
        expected_output="List of proposed memory IDs",
    )

    coordinate_task = Task(
        description=(
            "Retrieve all proposals, group by memory ID, count votes. "
            "Remove memories with 2+ votes, clear proposals namespace."
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
