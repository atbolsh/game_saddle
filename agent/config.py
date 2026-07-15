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


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


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

    # Sampling. Google's standardized recommendation for Gemma 4 E4B (model
    # card / HF card) is temperature=1.0, top_p=0.95, top_k=64 across all use
    # cases. Sampling (vs. the old greedy do_sample=False) also breaks the
    # degenerate fixed point where a near-identical prompt deterministically
    # reproduces the exact same move + reasoning sentence forever. Set
    # GEMMA_DO_SAMPLE=0 to restore deterministic greedy decoding.
    gemma_do_sample: bool = field(
        default_factory=lambda: _env_bool("GEMMA_DO_SAMPLE", True)
    )
    gemma_temperature: float = field(
        default_factory=lambda: _env_float("GEMMA_TEMPERATURE", 1.0)
    )
    gemma_top_p: float = field(
        default_factory=lambda: _env_float("GEMMA_TOP_P", 0.95)
    )
    gemma_top_k: int = field(
        default_factory=lambda: _env_int("GEMMA_TOP_K", 64)
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

    # Reflection (generative-agents style, arXiv:2304.03442). Every *applied*
    # move accrues ``reflection_points_per_move`` importance points; when the
    # running total reaches ``reflection_threshold`` mid-turn, the agent pauses
    # to reflect (no move that step: it re-examines the current frame and its
    # recent moves, then the reflection is fed into subsequent prompts) and the
    # total resets. Defaults: 5 points/move, threshold 150 -> reflect every 30
    # moves, i.e. after at most a 180-degree turn (one rotation is pi/30 = 6
    # degrees) if the agent is stuck spinning.
    reflection_points_per_move: int = field(
        default_factory=lambda: _env_int("REFLECTION_POINTS_PER_MOVE", 5)
    )
    reflection_threshold: int = field(
        default_factory=lambda: _env_int("REFLECTION_THRESHOLD", 150)
    )

    # Debrief (mode 4). ``debrief_max_frames`` recent snapshots are attached to
    # each debrief question by default (the newest is the session's current
    # state); the model can additionally pull up any recorded step's frames via
    # its [SHOW <n>] tool, capped at ``debrief_max_tool_calls`` fetches per
    # ask() turn.
    debrief_max_frames: int = field(
        default_factory=lambda: _env_int("DEBRIEF_MAX_FRAMES", 3)
    )
    debrief_max_tool_calls: int = field(
        default_factory=lambda: _env_int("DEBRIEF_MAX_TOOL_CALLS", 6)
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
