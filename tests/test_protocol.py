"""Tests for the Co-Forget Protocol."""

import pytest
import asyncio
import time
from typing import List, Dict, Set
from unittest.mock import Mock, patch, AsyncMock

from co_forget_protocol.main import Protocol
from co_forget_protocol.config import (
    Settings,
    PineconeConfig,
    MemoryConfig,
    AgentConfig,
)
from co_forget_protocol.memory import MemoryManager, MemoryQuota
from co_forget_protocol.pbft import PBFTCoordinator, PBFTMessage
from co_forget_protocol.voting import LLMVoter


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        pinecone=PineconeConfig(
            api_key="test_key", environment="test-env", index_name="test-index"
        ),
        memory=MemoryConfig(max_memories=100, cache_size=10, batch_size=5),
        agent=AgentConfig(
            num_memory_managers=3,
            llm_model="distilbert-base-uncased",
            relevance_threshold=0.7,
            max_faulty=1,
        ),
    )


@pytest.fixture
def mock_pinecone():
    """Create mock Pinecone manager."""
    with patch("co_forget_protocol.pinecone.PineconeManager") as mock:
        mock.index = Mock()
        mock.index.describe_index_stats.return_value = {
            "namespaces": {
                "memories": {"vector_count": 0},
                "baseline_memories": {"vector_count": 0},
            }
        }
        yield mock


@pytest.fixture
def memory_manager():
    """Create test memory manager."""
    return MemoryManager(
        db_path=":memory:",  # Use in-memory SQLite
        cache_size=10,
        batch_size=5,
    )


@pytest.fixture
def voter():
    """Create test LLM voter."""
    with patch("co_forget_protocol.voting.LLMVoter") as mock:
        mock.vote.return_value = ("forget", 0.8)
        mock._check_relevance.return_value = (False, 0.3)
        yield mock


@pytest.fixture
def coordinator():
    """Create test PBFT coordinator."""
    return PBFTCoordinator(num_agents=3, max_faulty=1)


@pytest.mark.asyncio
async def test_protocol_initialization(settings, mock_pinecone):
    """Test protocol initialization."""
    protocol = Protocol(settings)
    assert protocol.settings == settings
    assert len(protocol.memory_managers) == settings.agent.num_memory_managers
    assert isinstance(protocol.voter, LLMVoter)
    assert isinstance(protocol.pbft_coordinator, PBFTCoordinator)


@pytest.mark.asyncio
async def test_memory_storage(memory_manager):
    """Test memory storage and retrieval."""
    # Store memory
    memory_id = memory_manager.store_memory(
        "Test memory", {"importance": 0.8, "source": "test"}
    )
    assert memory_id is not None

    # Retrieve memory
    memory = memory_manager.get_memory(memory_id)
    assert memory is not None
    assert memory["text"] == "Test memory"
    assert memory["metadata"]["importance"] == 0.8

    # Check cache
    assert memory_id in memory_manager.memory_cache
    assert memory_id in memory_manager.metadata_cache


@pytest.mark.asyncio
async def test_memory_batching(memory_manager):
    """Test memory batching."""
    # Add memories to batch
    for i in range(7):  # More than batch_size
        memory_manager.store_memory(f"Memory {i}", {"batch_test": True})

    # Check that batch was flushed
    assert len(memory_manager.batch.records) < memory_manager.batch_size


@pytest.mark.asyncio
async def test_pbft_consensus(coordinator):
    """Test PBFT consensus process."""
    memory_ids = {"mem1", "mem2", "mem3"}
    agents = ["agent1", "agent2", "agent3"]

    # Mock agent responses
    with patch.object(
        coordinator, "pre_prepare", new_callable=AsyncMock
    ) as mock_prepare:
        mock_prepare.return_value = True
        with patch.object(
            coordinator, "prepare", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = True
            with patch.object(
                coordinator, "commit", new_callable=AsyncMock
            ) as mock_commit:
                mock_commit.return_value = True

                # Run consensus
                result = await coordinator.run_consensus(memory_ids, agents)
                assert result == memory_ids  # All memories should be deleted


@pytest.mark.asyncio
async def test_llm_voting(voter):
    """Test LLM-based voting."""
    # Test low decay score
    vote, confidence = voter.vote(
        "Old memory", {"timestamp": 0}, "Current context", decay_score=0.05
    )
    assert vote == "forget"
    assert confidence == 1.0

    # Test high decay score
    vote, confidence = voter.vote(
        "Recent memory", {"timestamp": time.time()}, "Current context", decay_score=0.95
    )
    assert vote == "keep"
    assert confidence == 1.0

    # Test borderline case
    vote, confidence = voter.vote(
        "Borderline memory",
        {"timestamp": time.time() - 3600},
        "Current context",
        decay_score=0.5,
    )
    assert vote in ["forget", "keep"]
    assert 0 <= confidence <= 1


@pytest.mark.asyncio
async def test_protocol_run(
    settings, mock_pinecone, memory_manager, voter, coordinator
):
    """Test full protocol run."""
    protocol = Protocol(settings)

    # Mock necessary components
    protocol.pinecone = mock_pinecone
    protocol.memory_managers = [memory_manager]
    protocol.voter = voter
    protocol.pbft_coordinator = coordinator

    # Test questions
    questions = ["What is the capital of France?", "Who is the CEO of Tesla?"]

    # Run protocol
    protocol_answers, baseline_answers = await protocol.run(questions)

    assert len(protocol_answers) == len(questions)
    assert len(baseline_answers) == len(questions)

    # Check memory pruning
    await protocol.prune_memories()
    assert len(memory_manager.get_all_memory_ids()) == 0


@pytest.mark.asyncio
async def test_fault_tolerance(coordinator):
    """Test PBFT fault tolerance."""
    memory_ids = {"mem1", "mem2"}
    agents = ["agent1", "agent2", "agent3", "agent4"]  # 4 agents, 1 faulty

    # Mock one agent as faulty
    with patch.object(
        coordinator, "pre_prepare", new_callable=AsyncMock
    ) as mock_prepare:
        mock_prepare.side_effect = [True, True, False, True]  # One agent fails
        with patch.object(
            coordinator, "prepare", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.side_effect = [True, True, False, True]
            with patch.object(
                coordinator, "commit", new_callable=AsyncMock
            ) as mock_commit:
                mock_commit.side_effect = [True, True, False, True]

                # Run consensus
                result = await coordinator.run_consensus(memory_ids, agents)
                assert result == memory_ids  # Should still reach consensus


@pytest.mark.asyncio
async def test_memory_quota(settings):
    """Test memory quota management."""
    quota = MemoryQuota(max_memories=5)

    # Test increment
    for _ in range(3):
        quota.increment()
    assert quota.current_count == 3
    assert quota.can_add_memory()

    # Test decrement
    quota.decrement()
    assert quota.current_count == 2

    # Test max limit
    for _ in range(4):
        quota.increment()
    assert quota.current_count == 5
    assert not quota.can_add_memory()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
