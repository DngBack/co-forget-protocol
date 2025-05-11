"""Configuration settings for the Co-Forget Protocol."""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PineconeConfig(BaseModel):
    """Pinecone configuration settings."""

    api_key: str = Field(default="", env="PINECONE_API_KEY")
    index_name: str = "co-forgetting-index"
    cloud: str = "aws"
    region: str = "us-east-1"
    model: str = "all-MiniLM-L6-v2"
    field_map: dict = {"text": "text"}


class MemoryConfig(BaseModel):
    """Memory management configuration."""

    max_memories: int = 10000
    batch_size: int = 100
    decay_factor: float = 10.0
    removal_threshold: float = 0.3


class AgentConfig(BaseModel):
    """Agent configuration settings."""

    num_memory_managers: int = 3
    verbose: bool = True


class Settings(BaseSettings):
    """Main settings for the Co-Forget Protocol."""

    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
