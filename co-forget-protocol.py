import os
import time
import math
from pinecone import Pinecone, IndexEmbed
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool
from crewai.tools.base_tool import BaseTool

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

    def __init__(self, pc, index_name, namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def _run(self, text: str, metadata: dict) -> str:
        record = {"text": text, "metadata": metadata}
        self.index.upsert_records(self.namespace, [record])
        return "Memory upserted"


class PineconeRetrieveTool(BaseTool):
    name = "Pinecone Retrieve"
    description = "Retrieve similar memories from Pinecone"

    def __init__(self, pc, index_name, namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def _run(self, query: str, top_k: int = 5) -> list:
        results = self.index.search(
            self.namespace, {"top_k": top_k, "inputs": {"text": query}}
        )
        return results.get("matches", [])


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

    def _run(self, memory_id: str) -> str:
        text = "proposal"
        metadata = {"memory_id": memory_id, "agent_id": self.agent_id}
        record = {"text": text, "metadata": metadata}
        self.index.upsert_records(self.proposals_namespace, [record])
        return f"Proposed to remove memory {memory_id}"


class PineconeRetrieveProposalsTool(BaseTool):
    name = "Pinecone Retrieve Proposals"
    description = "Retrieve all proposed memories for removal"

    def __init__(self, pc, index_name, proposals_namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.proposals_namespace = proposals_namespace

    def _run(self) -> list:
        ids = self.index.list(namespace=self.proposals_namespace)
        if not ids:
            return []
        vectors = self.index.fetch(ids=ids, namespace=self.proposals_namespace)
        return [vec["metadata"] for vec in vectors.get("vectors", {}).values()]


class PineconeDeleteMemoriesTool(BaseTool):
    name = "Pinecone Delete Memories"
    description = "Delete specified memories from Pinecone"

    def __init__(self, pc, index_name, namespace):
        super().__init__(name=self.name, description=self.description)
        self.pc = pc
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def _run(self, memory_ids: list) -> str:
        if memory_ids:
            self.index.delete(ids=memory_ids, namespace=self.namespace)
        return f"Deleted memories: {memory_ids}"


# Agents
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
        PineconeUpsertTool(pc, index_name, "memories"),
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
        PineconeDeleteMemoriesTool(pc, index_name, "memories"),
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
        PineconeUpsertTool(pc, index_name, "baseline_memories"),
        PineconeRetrieveTool(pc, index_name, "baseline_memories"),
        PineconeListMemoriesTool(pc, index_name, "baseline_memories"),
        PineconeFetchMemoriesTool(pc, index_name, "baseline_memories"),
        PineconeDeleteMemoriesTool(pc, index_name, "baseline_memories"),
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
def main():
    questions = ["Who is the CEO of Tesla?", "What is the capital of France?"]
    protocol_answers = []
    baseline_answers = []
    protocol_memory_count = 0
    baseline_memory_count = 0

    for i, question in enumerate(questions):
        # Protocol execution
        crew = Crew(
            agents=[task_performer],
            tasks=[
                Task(
                    description=f"Answer: {question}",
                    agent=task_performer,
                    expected_output="A concise answer",
                )
            ],
            process=Process.sequential,
        )
        answer = crew.kickoff()
        protocol_answers.append(answer)

        # Baseline execution
        baseline_crew = Crew(
            agents=[baseline_agent],
            tasks=[
                Task(
                    description=f"Answer: {question}",
                    agent=baseline_agent,
                    expected_output="A concise answer",
                )
            ],
            process=Process.sequential,
        )
        baseline_answer = baseline_crew.kickoff()
        baseline_answers.append(baseline_answer)

        # Pruning after every 2 questions
        if (i + 1) % 2 == 0:
            # Protocol pruning
            propose_tasks = [
                Task(
                    description=propose_task.description,
                    agent=agent,
                    expected_output=propose_task.expected_output,
                )
                for agent in memory_managers
            ]
            pruning_crew = Crew(
                agents=memory_managers + [coordinator],
                tasks=propose_tasks + [coordinate_task],
                process=Process.sequential,
            )
            pruning_crew.kickoff()

            # Baseline pruning
            baseline_prune_task = Task(
                description="List memories, calculate D(t) = exp(-(current_time - timestamp) / 10), delete those with D(t) < 0.3.",
                agent=baseline_agent,
                expected_output="List of deleted memory IDs",
            )
            baseline_crew = Crew(
                agents=[baseline_agent],
                tasks=[baseline_prune_task],
                process=Process.sequential,
            )
            baseline_crew.kickoff()

        # Update memory counts
        protocol_stats = index.describe_index_stats()
        baseline_stats = index.describe_index_stats()
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

    # Output results
    print("Protocol Answers:", protocol_answers)
    print("Baseline Answers:", baseline_answers)
    print(f"Protocol Memory Count: {protocol_memory_count}")
    print(f"Baseline Memory Count: {baseline_memory_count}")


if __name__ == "__main__":
    main()
