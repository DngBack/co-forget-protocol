# Co-Forget Protocol

A fault-tolerant, distributed memory management system using PBFT consensus and LLM-based voting.

## Features

- **Distributed Memory Management**: Multiple agents collaboratively manage shared memory
- **Fault-Tolerant Consensus**: PBFT implementation for reliable memory pruning
- **Smart Memory Voting**: LLM-based relevance scoring for intelligent memory retention
- **Efficient Storage**: SQLite with caching and batching for optimal performance
- **Unique Memory Tracking**: UUID-based memory identification

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/co-forget-protocol.git
cd co-forget-protocol
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate gRPC code:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. src/co_forget_protocol/pbft.proto
```

4. Set up environment variables:

```bash
# Create .env file
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=memories
```

## Usage

Basic usage:

```python
from co_forget_protocol import Protocol, Settings
from co_forget_protocol.config import PineconeConfig

# Initialize with custom settings
settings = Settings(
    pinecone=PineconeConfig(
        api_key="your_api_key",
        environment="us-west1-gcp",
        index_name="memories"
    ),
    memory=MemoryConfig(
        max_memories=10000,
        cache_size=100,
        batch_size=100
    ),
    agent=AgentConfig(
        num_memory_managers=3,
        llm_model="distilbert-base-uncased",
        relevance_threshold=0.7,
        max_faulty=1
    )
)

# Create protocol instance
protocol = Protocol(settings)

# Run with questions
questions = [
    "What is the capital of France?",
    "Who is the CEO of Tesla?",
    "What is the population of Tokyo?"
]

# Get answers and compare with baseline
protocol_answers, baseline_answers = await protocol.run(questions)

# Print results
print("Protocol Answers:", protocol_answers)
print("Baseline Answers:", baseline_answers)
```

## Architecture

### Components

1. **Memory Management**

   - SQLite database for persistent storage
   - LRU cache for frequently accessed memories
   - TTL cache for metadata
   - Batch processing for efficient writes

2. **PBFT Consensus**

   - Pre-prepare, prepare, and commit phases
   - Fault tolerance up to f faulty agents in 3f+1 system
   - gRPC for distributed communication

3. **LLM Voting**

   - DistilBERT for semantic similarity
   - Text classification for relevance scoring
   - Combined decay and relevance scoring

4. **Agents**
   - Memory Managers: Handle memory operations and voting
   - Task Performer: Answers questions using shared memory
   - Coordinator: Manages PBFT consensus
   - Baseline Agent: Independent memory management

### Configuration

Key configuration options in `Settings`:

```python
class Settings:
    pinecone: PineconeConfig  # Pinecone vector database settings
    memory: MemoryConfig      # Memory management settings
    agent: AgentConfig        # Agent behavior settings
```

## Testing

Run tests:

```bash
pytest tests/
```

## Performance

- Memory retrieval: O(1) with cache, O(log n) with SQLite
- Consensus: O(n) where n is number of agents
- LLM voting: O(1) per memory with GPU acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
