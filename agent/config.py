"""Env-driven configuration for the agent.

All settings are read from environment variables (loaded from .env via
python-dotenv if present). No external API keys are required: Neo4j runs
locally over bolt, embeddings come from a local sentence-transformers
model, and the LLM is Gemma 4 E4B loaded through HuggingFace transformers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Best-effort .env load; tolerate absence.
load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AgentConfig:
    # Neo4j (local, bolt-based; no external service)
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: _env("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "changeme"))
    neo4j_database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))

    # Gemma 4 E4B
    gemma_model_id: str = field(
        default_factory=lambda: _env("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
    )
    gemma_dtype: str = field(default_factory=lambda: _env("GEMMA_DTYPE", "bfloat16"))
    gemma_device: str = field(default_factory=lambda: _env("GEMMA_DEVICE", "auto"))
    gemma_max_new_tokens: int = field(
        default_factory=lambda: _env_int("GEMMA_MAX_NEW_TOKENS", 2048)
    )

    # Embeddings (local sentence-transformers; NAMS embedding provider string)
    embedding_model: str = field(
        default_factory=lambda: _env(
            "NAMS_EMBEDDING", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Filesystem layout for game snapshot images
    image_dir: Path = field(
        default_factory=lambda: Path(
            _env("AGENT_IMAGE_DIR", "memory_images")
        )
    )

    # Game defaults
    game_size: int = field(default_factory=lambda: _env_int("GAME_SIZE", 224))
    max_solve_steps: int = field(default_factory=lambda: _env_int("MAX_SOLVE_STEPS", 200))

    # How many of the most-recent session messages to always thread into the
    # mode-1 prompt verbatim (by recency, independent of the semantic search).
    # NAMS' get_context is pure similarity search with a threshold, so recent
    # moves are not guaranteed to be recalled; this recency window guarantees the
    # agent always sees at least this many of its latest questions/moves so it
    # has reliable "what did I just do" continuity mid-turn.
    recent_messages_window: int = field(
        default_factory=lambda: _env_int("RECENT_MESSAGES_WINDOW", 7)
    )

    # HuggingFace token (optional; some Gemma weights are gated)
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("HF_TOKEN") or None
    )

    @property
    def neo4j_settings_dict(self) -> dict:
        """kwargs for NAMS ``MemorySettings(neo4j=...)``."""
        return {
            "uri": self.neo4j_uri,
            "username": self.neo4j_username,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
        }


CONFIG = AgentConfig()
