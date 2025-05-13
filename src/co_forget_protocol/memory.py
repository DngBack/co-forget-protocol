"""Memory management classes for the Co-Forget Protocol."""

import time
import math
import uuid
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from pathlib import Path
from cachetools import LRUCache, TTLCache


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


@dataclass
class MemoryManager:
    """Manages memory storage and retrieval with SQLite tracking and caching."""

    db_path: str = "memories.db"
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decay_factor: float = 10.0
    removal_threshold: float = 0.3
    cache_size: int = 100
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 100

    def __post_init__(self):
        """Initialize SQLite database and caches."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                timestamp REAL,
                text TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

        # Initialize caches
        self.memory_cache = LRUCache(maxsize=self.cache_size)
        self.metadata_cache = TTLCache(maxsize=self.cache_size, ttl=self.cache_ttl)
        self.batch = MemoryBatch(batch_size=self.batch_size)

    def store_memory(self, text: str, metadata: Dict) -> str:
        """Store a memory with unique ID and track in SQLite."""
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        metadata["t_last"] = timestamp

        # Add to batch
        batch_result = self.batch.add(text, metadata)
        if batch_result:
            self._flush_batch(batch_result)

        # Store in SQLite
        self.cursor.execute(
            "INSERT INTO memories (id, agent_id, timestamp, text, metadata) VALUES (?, ?, ?, ?, ?)",
            (memory_id, self.agent_id, timestamp, text, str(metadata)),
        )
        self.conn.commit()

        # Update caches
        self.memory_cache[memory_id] = {"text": text, "metadata": metadata}
        self.metadata_cache[memory_id] = timestamp

        return memory_id

    def _flush_batch(self, records: List[Dict]) -> None:
        """Flush a batch of records to the database."""
        for record in records:
            self.cursor.execute(
                "INSERT INTO memories (id, agent_id, timestamp, text, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    self.agent_id,
                    time.time(),
                    record["text"],
                    str(record["metadata"]),
                ),
            )
        self.conn.commit()

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Retrieve memory details with caching."""
        # Check memory cache first
        if memory_id in self.memory_cache:
            return self.memory_cache[memory_id]

        # Check metadata cache
        if memory_id in self.metadata_cache:
            # Only fetch metadata from SQLite
            self.cursor.execute(
                "SELECT text, metadata FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = self.cursor.fetchone()
            if row:
                result = {
                    "text": row[0],
                    "metadata": eval(row[1]),
                }
                self.memory_cache[memory_id] = result
                return result

        # Full database fetch
        self.cursor.execute(
            "SELECT text, metadata, timestamp FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = self.cursor.fetchone()
        if not row:
            return None

        result = {
            "text": row[0],
            "metadata": eval(row[1]),
            "timestamp": row[2],
        }

        # Update caches
        self.memory_cache[memory_id] = result
        self.metadata_cache[memory_id] = row[2]

        return result

    def get_all_memory_ids(self) -> Set[str]:
        """Get all memory IDs from SQLite."""
        self.cursor.execute("SELECT id FROM memories")
        return {row[0] for row in self.cursor.fetchall()}

    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from SQLite and caches."""
        try:
            self.cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()

            # Clear caches
            self.memory_cache.pop(memory_id, None)
            self.metadata_cache.pop(memory_id, None)

            return True
        except sqlite3.Error:
            return False

    def flush(self) -> None:
        """Flush any pending batch operations."""
        if self.batch.records:
            self._flush_batch(self.batch.flush())

    def __del__(self):
        """Cleanup database connection and flush any pending operations."""
        self.flush()
        if hasattr(self, "conn"):
            self.conn.close()
