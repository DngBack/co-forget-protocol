"""Co-Forget Protocol: A collaborative memory management system."""

__version__ = "0.1.0"

from .config import Settings
from .memory import MemoryQuota, MemoryScorer, MemoryBatch
from .agents import create_agents
from .tools import create_tools
from .pinecone import PineconeManager

__all__ = [
    "Settings",
    "MemoryQuota",
    "MemoryScorer",
    "MemoryBatch",
    "create_agents",
    "create_tools",
    "PineconeManager",
]
