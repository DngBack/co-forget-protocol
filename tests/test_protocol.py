"""Tests for the Co-Forget Protocol."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from co_forget_protocol import Protocol, Settings
from co_forget_protocol.memory import MemoryQuota
from co_forget_protocol.pinecone import PineconeManager


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    return Settings(
        pinecone=Settings.PineconeConfig(
            api_key="test-key",
            index_name="test-index",
        ),
        memory=Settings.MemoryConfig(
            max_memories=100,
            batch_size=10,
        ),
        agent=Settings.AgentConfig(
            num_memory_managers=2,
            verbose=False,
        ),
    )


@pytest.fixture
def mock_pinecone():
    """Create mock Pinecone manager."""
    manager = MagicMock(spec=PineconeManager)
    manager.index = MagicMock()
    manager.index.describe_index_stats = AsyncMock(
        return_value={
            "namespaces": {
                "memories": {"vector_count": 5},
                "baseline_memories": {"vector_count": 3},
            }
        }
    )
    return manager


@pytest.fixture
def protocol(mock_settings, mock_pinecone):
    """Create a protocol instance with mocks."""
    with patch("co_forget_protocol.main.PineconeManager", return_value=mock_pinecone):
        return Protocol(settings=mock_settings)


@pytest.mark.asyncio
async def test_process_question(protocol):
    """Test processing a question."""
    question = "What is the capital of France?"
    mock_crew = MagicMock()
    mock_crew.kickoff = AsyncMock(return_value="Paris")

    with patch("co_forget_protocol.main.Crew", return_value=mock_crew):
        result = await protocol.process_question(
            question, protocol.agents["task_performer"]
        )
        assert result == "Paris"


@pytest.mark.asyncio
async def test_prune_memories(protocol):
    """Test pruning memories."""
    mock_crew = MagicMock()
    mock_crew.kickoff = AsyncMock()

    with patch("co_forget_protocol.main.Crew", return_value=mock_crew):
        await protocol.prune_memories()
        mock_crew.kickoff.assert_called_once()


@pytest.mark.asyncio
async def test_run_protocol(protocol):
    """Test running the protocol."""
    questions = ["What is the capital of France?", "Who is the CEO of Tesla?"]
    mock_crew = MagicMock()
    mock_crew.kickoff = AsyncMock(side_effect=["Paris", "Elon Musk"])

    with patch("co_forget_protocol.main.Crew", return_value=mock_crew):
        protocol_answers, baseline_answers = await protocol.run(questions)
        assert protocol_answers == ["Paris", "Elon Musk"]
        assert baseline_answers == ["Paris", "Elon Musk"]
        assert mock_crew.kickoff.call_count == 4  # 2 questions * 2 agents


def test_memory_quota():
    """Test memory quota functionality."""
    quota = MemoryQuota(max_memories=3)
    assert quota.can_add_memory()
    assert quota.current_count == 0

    quota.increment()
    assert quota.current_count == 1
    assert quota.can_add_memory()

    quota.increment()
    quota.increment()
    assert quota.current_count == 3
    assert not quota.can_add_memory()

    quota.decrement()
    assert quota.current_count == 2
    assert quota.can_add_memory()

    quota.decrement()
    quota.decrement()
    assert quota.current_count == 0
    assert quota.can_add_memory()
