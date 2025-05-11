"""Memory management classes for the Co-Forget Protocol."""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class MemoryQuota:
    """Manages memory quota limits."""

    max_memories: int
    current_count: int = 0

    def can_add_memory(self) -> bool:
        """Check if a new memory can be added."""
        return self.current_count < self.max_memories

    def increment(self) -> None:
        """Increment the memory count."""
        self.current_count += 1

    def decrement(self) -> None:
        """Decrement the memory count."""
        self.current_count = max(0, self.current_count - 1)


@dataclass
class MemoryScorer:
    """Scores memories based on time decay."""

    decay_factor: float = 10.0
    removal_threshold: float = 0.3

    def calculate_score(self, timestamp: float, importance: float = 1.0) -> float:
        """Calculate memory score based on time decay."""
        time_decay = math.exp(-(time.time() - timestamp) / self.decay_factor)
        return time_decay * importance

    def should_remove(self, score: float) -> bool:
        """Determine if a memory should be removed based on score."""
        return score < self.removal_threshold


@dataclass
class MemoryBatch:
    """Batches memory records for efficient processing."""

    batch_size: int = 100
    records: List[Dict] = field(default_factory=list)

    def add(self, text: str, metadata: Dict) -> Optional[List[Dict]]:
        """Add a memory record to the batch."""
        self.records.append({"text": text, "metadata": metadata})
        if len(self.records) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> List[Dict]:
        """Flush the current batch of records."""
        if not self.records:
            return []
        records = self.records.copy()
        self.records.clear()
        return records
