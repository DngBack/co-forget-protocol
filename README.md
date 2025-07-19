# Co-Forget Protocol: Implementation Repository

[![arXiv](https://img.shields.io/badge/arXiv-2506.17338-b31b1b.svg)](https://arxiv.org/abs/2506.17338)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the **official implementation** of the research paper:

**"PBFT-Backed Semantic Voting for Multi-Agent Memory Pruning"**  
*Duong Bach*  
arXiv:2506.17338 [cs.DC], 2025

## Abstract

The proliferation of multi-agent systems (MAS) in complex, dynamic environments necessitates robust and efficient mechanisms for managing shared knowledge. This implementation addresses the critical challenge of ensuring that distributed memories remain synchronized, relevant, and free from outdated data through a novel Co-Forgetting Protocol.

## Key Research Contributions

- **Context-Aware Semantic Voting**: Lightweight DistilBERT-based relevance assessment for memory items
- **Multi-Scale Temporal Decay**: Sophisticated aging functions across different time horizons
- **PBFT Consensus Mechanism**: Byzantine fault-tolerant memory pruning decisions (tolerates up to f faulty agents in 3f+1 systems)
- **Experimental Validation**: Demonstrated 52% memory reduction, 88% voting accuracy, 92% consensus success rate

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/DngBack/co-forget-protocol.git
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

### Running the Experiments

Reproduce the paper's experimental results:

```python
from co_forget_protocol import Protocol, Settings
from co_forget_protocol.config import PineconeConfig, MemoryConfig, AgentConfig

# Configure experimental setup
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
        num_memory_managers=3,  # 3f+1 = 4 agents (tolerates 1 Byzantine)
        llm_model="distilbert-base-uncased",
        relevance_threshold=0.7,
        max_faulty=1
    )
)

# Initialize protocol
protocol = Protocol(settings)

# Run experimental evaluation
questions = [
    "What is the capital of France?",
    "Who is the CEO of Tesla?", 
    "What is the population of Tokyo?"
]

# Execute protocol vs baseline comparison
protocol_answers, baseline_answers = await protocol.run(questions)
print("Protocol Results:", protocol_answers)
print("Baseline Results:", baseline_answers)
```

## Implementation Architecture

This implementation follows the paper's three-component design:

### 1. Semantic Voting Module (`voting.py`)

- **DistilBERT Integration**: Lightweight transformer for semantic similarity assessment
- **Relevance Scoring**: Context-aware memory importance evaluation
- **Text Classification**: Binary relevance decisions with configurable thresholds

### 2. Multi-Scale Temporal Decay (`memory.py`)

- **Time-Based Decay**: Exponential decay functions across multiple time horizons
- **Access Frequency**: LRU-based importance weighting
- **Combined Scoring**: Unified relevance + temporal decay metric

### 3. PBFT Consensus (`pbft.py`)

- **Three-Phase Protocol**: Pre-prepare, prepare, commit phases
- **Byzantine Fault Tolerance**: Handles up to f malicious agents in 3f+1 system  
- **gRPC Communication**: Efficient inter-agent message passing
- **Message Authentication**: Cryptographic signatures for message integrity

### Storage and Infrastructure

- **Vector Database**: Pinecone for embedding storage and similarity search
- **Metadata Storage**: SQLite for local memory metadata management
- **Caching Layer**: Multi-level caching (LRU + TTL) for performance optimization
- **Batch Processing**: Efficient bulk operations for memory updates

## Experimental Results (Paper Reproduction)

The implementation achieves the following performance metrics as reported in the paper:

| Metric | Paper Result | Implementation Status |
|--------|--------------|----------------------|
| Memory Footprint Reduction | 52% over 500 epochs | ✅ Reproduced |
| Voting Accuracy | 88% vs human benchmarks | ✅ Reproduced |
| PBFT Consensus Success Rate | 92% under Byzantine conditions | ✅ Reproduced |
| Cache Hit Rate | 82% for memory access | ✅ Reproduced |

### Running Benchmarks

```bash
# Run full experimental suite
python exp/co-forget-protocol.py

# Run specific test scenarios
pytest tests/test_protocol.py -v

# Performance profiling
python -m cProfile exp/co-forget-protocol.py
```

## Configuration

Key configuration classes mirror the paper's experimental setup:

```python
class Settings:
    pinecone: PineconeConfig      # Vector database configuration
    memory: MemoryConfig          # Memory management parameters  
    agent: AgentConfig           # Multi-agent system settings

class AgentConfig:
    num_memory_managers: int = 3  # Number of memory management agents
    llm_model: str = "distilbert-base-uncased"  # Semantic model
    relevance_threshold: float = 0.7  # Voting threshold
    max_faulty: int = 1          # Byzantine fault tolerance (f parameter)

class MemoryConfig:
    max_memories: int = 10000    # Maximum memory capacity
    cache_size: int = 100        # LRU cache size
    batch_size: int = 100        # Batch processing size
    decay_rate: float = 0.95     # Temporal decay parameter
```

## Testing and Validation

```bash
# Run unit tests
pytest tests/ -v

# Test PBFT consensus under Byzantine conditions
pytest tests/test_protocol.py::test_byzantine_tolerance -v

# Validate semantic voting accuracy
pytest tests/test_protocol.py::test_voting_accuracy -v

# Performance benchmarks
python tests/benchmark.py
```

## Paper Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{bach2025pbft,
  title={PBFT-Backed Semantic Voting for Multi-Agent Memory Pruning},
  author={Bach, Duong},
  journal={arXiv preprint arXiv:2506.17338},
  year={2025}
}
```

## Contributing

This is an academic implementation. Contributions that:

1. Improve experimental reproducibility
2. Add new evaluation metrics  
3. Optimize performance while maintaining accuracy
4. Extend to new domains/datasets

are welcome. Please follow the paper's methodology and maintain experimental rigor.

## Performance Considerations

- **Memory Requirements**: ~2GB RAM for full experimental setup
- **Compute**: GPU recommended for DistilBERT inference (CPU compatible)
- **Network**: gRPC requires stable network for multi-agent communication
- **Storage**: ~1GB for full experimental dataset and vector embeddings

## Troubleshooting

### Common Issues

1. **gRPC Connection Errors**: Ensure all agents can communicate on specified ports
2. **Pinecone API Limits**: Monitor API quota and implement rate limiting
3. **Memory Overflow**: Adjust `max_memories` and `cache_size` for available RAM
4. **Byzantine Behavior**: Verify `max_faulty < (num_agents - 1) / 3`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed PBFT logging
protocol = Protocol(settings, debug=True)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related Work

- [Practical Byzantine Fault Tolerance](https://pmg.csail.mit.edu/papers/osdi99.pdf) - Castro & Liskov, 1999
- [DistilBERT](https://arxiv.org/abs/1910.01108) - Sanh et al., 2019  
- [Multi-Agent Memory Systems](https://arxiv.org/abs/2104.07154) - Recent surveys in distributed AI

## Contact

For questions about this implementation or the research paper:

- **Author**: Duong Bach  
- **Repository**: [https://github.com/DngBack/co-forget-protocol](https://github.com/DngBack/co-forget-protocol)
- **Paper**: [arXiv:2506.17338](https://arxiv.org/abs/2506.17338)

---

**Note**: This is a research implementation. For production use, additional security hardening and optimization may be required.
