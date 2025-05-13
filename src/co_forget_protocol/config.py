"""Configuration classes for the Co-Forget Protocol."""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PineconeConfig(BaseModel):
    """Pinecone configuration."""

    api_key: str
    environment: str = "us-west1-gcp"
    index_name: str = "memories"


class MemoryConfig(BaseModel):
    """Memory management configuration."""

    max_memories: int = 10000
    batch_size: int = 100
    decay_factor: float = 10.0
    removal_threshold: float = 0.3
    cache_size: int = 100
    cache_ttl: int = 3600  # 1 hour


class AgentConfig(BaseModel):
    """Agent configuration."""

    num_memory_managers: int = 3
    verbose: bool = True
    llm_model: str = "distilbert-base-uncased"
    relevance_threshold: float = 0.7
    max_faulty: int = Field(
        default=1, description="Maximum number of faulty agents allowed"
    )


class Settings(BaseSettings):
    """Global settings for the protocol."""

    pinecone: PineconeConfig
    memory: MemoryConfig = MemoryConfig()
    agent: AgentConfig = AgentConfig()
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
