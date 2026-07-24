"""Env-driven configuration for the agent.

All settings are read from environment variables (loaded from .env via
python-dotenv if present). No external API keys are required: Neo4j runs
locally over bolt, embeddings come from a local sentence-transformers
model, and the LLM (any agent.model.MODEL_REGISTRY entry; Gemma 4 E4B by
default) is loaded through HuggingFace transformers.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env into os.environ (HF_TOKEN, Neo4j creds, ...). Path is
# anchored to this file so it does not depend on the process cwd -- notebooks,
# ``python -m agent.runner``, and scripts invoked from other directories all see
# the same values. Tolerates a missing file (e.g. a fresh clone before
# ``cp .env.example .env``).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ------------------------------------------------------- third-party noise
# Targeted suppression of KNOWN-benign third-party warnings. Every filter is
# pinned to an exact message so that new, potentially meaningful warnings
# still surface (per the no-fuzzy-fallbacks rule: silence nothing broadly).

# NAMS's sentence-transformers wrapper calls a renamed accessor
# (get_sentence_embedding_dimension); upstream's to fix, fires on every
# client connect.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The `get_sentence_embedding_dimension` method has been renamed",
)
# huggingface_hub deprecation raised from inside transformers' snapshot
# download; nothing in this repo passes resume_download.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The `resume_download` argument is deprecated",
)


class _DropNeo4jDeprecationNotifications(logging.Filter):
    """Drop the Neo4j driver's DEPRECATION server notifications only.

    NAMS still calls the deprecated ``db.index.vector.queryNodes`` procedure,
    so every semantic retrieval spams four WARNING lines through the
    ``neo4j.notifications`` logger -- upstream's to fix, pure noise here.
    Other notification classes stay visible on purpose: the UNRECOGNIZED
    label/property warnings are exactly the guard this repo wants against
    typo'd Cypher schemas (they also fire, benignly and only once, on a
    fresh/empty DB before any message exists).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "'DEPRECATION'" not in record.getMessage()


logging.getLogger("neo4j.notifications").addFilter(
    _DropNeo4jDeprecationNotifications()
)


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


def _env_chain(*keys: str) -> str | None:
    """First non-empty value among the given env vars, or None. Used for the
    MODEL_* names with their legacy GEMMA_* fallbacks, so existing .env files
    keep working after the model layer went multi-model."""
    for k in keys:
        v = os.environ.get(k)
        if v is not None and v.strip() != "":
            return v
    return None


def _env_chain_float(*keys: str) -> float | None:
    v = _env_chain(*keys)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_chain_int(*keys: str) -> int | None:
    v = _env_chain(*keys)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_chain_bool(*keys: str) -> bool | None:
    v = _env_chain(*keys)
    if v is None:
        return None
    return v.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class AgentConfig:
    # Neo4j (local, bolt-based; no external service)
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: _env("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "changeme"))
    neo4j_database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))

    # Model selection. ``model_key`` picks an entry from
    # agent.model.MODEL_REGISTRY (see MODEL_CANDIDATES.md for the lineup);
    # the notebooks can switch models at runtime via their dropdown.
    model_key: str = field(
        default_factory=lambda: _env("MODEL_KEY", "gemma-4-e4b")
    )
    model_dtype: str = field(
        default_factory=lambda: _env_chain("MODEL_DTYPE", "GEMMA_DTYPE") or "bfloat16"
    )
    model_device: str = field(
        default_factory=lambda: _env_chain("MODEL_DEVICE", "GEMMA_DEVICE") or "auto"
    )
    max_new_tokens: int = field(
        default_factory=lambda: _env_chain_int(
            "MODEL_MAX_NEW_TOKENS", "GEMMA_MAX_NEW_TOKENS"
        ) or 2048
    )

    # Sampling. Sampling (vs. greedy do_sample=False) breaks the degenerate
    # fixed point where a near-identical prompt deterministically reproduces
    # the exact same move + reasoning sentence forever; set MODEL_DO_SAMPLE=0
    # to force deterministic greedy decoding everywhere. All four knobs are
    # OPTIONAL overrides: when unset (the default) each model's per-spec
    # defaults apply (e.g. Gemma's 1.0/0.95/64, Kimi Thinking's 0.8,
    # Step3-VL's vendor-demonstrated greedy decoding), falling back to
    # sampling on / the model's own generation_config. Setting MODEL_DO_SAMPLE
    # or MODEL_TEMPERATURE etc. forces one value for EVERY model.
    do_sample: bool | None = field(
        default_factory=lambda: _env_chain_bool("MODEL_DO_SAMPLE", "GEMMA_DO_SAMPLE")
    )
    temperature: float | None = field(
        default_factory=lambda: _env_chain_float(
            "MODEL_TEMPERATURE", "GEMMA_TEMPERATURE"
        )
    )
    top_p: float | None = field(
        default_factory=lambda: _env_chain_float("MODEL_TOP_P", "GEMMA_TOP_P")
    )
    top_k: int | None = field(
        default_factory=lambda: _env_chain_int("MODEL_TOP_K", "GEMMA_TOP_K")
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
    # Native render size in pixels. 768x768 is deliberate: Gemma 4's image
    # processor (default 280-soft-token budget, patch 16, pooling 3) resizes
    # square inputs to at most 768x768 anyway -- rendering natively at that
    # size feeds it real detail instead of an upscaled 224px frame, at ZERO
    # change in token count (256 soft tokens per frame either way).
    game_size: int = field(default_factory=lambda: _env_int("GAME_SIZE", 768))
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

    # Debrief (mode 4). The context always carries exactly ONE frame -- the
    # one the player saw at the current message; the model moves the cursor
    # via its [SHOW <n>] / [NEXT] / [BACK] tools, capped at
    # ``debrief_max_tool_calls`` moves per ask() turn ([SEARCH] and
    # [WRITE_TIP] calls share the same budget).
    debrief_max_tool_calls: int = field(
        default_factory=lambda: _env_int("DEBRIEF_MAX_TOOL_CALLS", 64)
    )

    # Agent-initiated [SEARCH <query>] memory searches. Outside debrief (which
    # has its own shared tool budget above), a single turn may run at most
    # ``memory_search_max_calls`` searches; each search returns at most
    # ``memory_search_top_k`` results per memory tier.
    memory_search_max_calls: int = field(
        default_factory=lambda: _env_int("MEMORY_SEARCH_MAX_CALLS", 3)
    )
    memory_search_top_k: int = field(
        default_factory=lambda: _env_int("MEMORY_SEARCH_TOP_K", 5)
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
