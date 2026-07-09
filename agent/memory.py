"""NAMS (Neo4j Agent Memory) client factory and per-mode helpers.

Self-hosted **bolt** backend only -- no NAMS API key, no external DB.
Embeddings are local via ``sentence-transformers``. We deliberately do not
pass an ``llm=`` to NAMS, so no LLM provider key is required; the long-term
semantic model is seeded manually via :func:`build_semantic_model`.

The three memory tiers map cleanly onto the agent's needs:

  * short_term  -> the conversation (user questions + assistant moves)
  * reasoning   -> per-move reasoning traces (thought + tool_call = action)
  * long_term   -> the small semantic model of the game (entities + tips)

Game images and the full Settings dict are stored on ``GameSnapshot`` nodes
(see :mod:`agent.image_store`) linked to the corresponding ``Message``.
The Settings dict is **never** injected into the agent prompt in mode 1.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from pydantic import SecretStr

from .config import AgentConfig, CONFIG

logger = logging.getLogger(__name__)


def make_memory_settings(cfg: AgentConfig | None = None):
    """Build a NAMS ``MemorySettings`` for the bolt backend."""
    from neo4j_agent_memory import ExtractionConfig, MemorySettings  # local import for cost

    cfg = cfg or CONFIG
    # NAMS' default extraction pipeline has an optional "LLM fallback" stage
    # that defaults to the ``openai`` provider. With ``llm=None`` (our design)
    # NAMS still *tries* to build it, fails to find an adapter, and logs a
    # noisy "LLM extractor not available, skipping ... provider 'openai'"
    # warning on every connect. We never want a cloud LLM here (Gemma 4 E4B is
    # the only model, and it does generation, not NAMS extraction), so disable
    # the fallback explicitly: extraction stays fully local (spaCy / GLiNER /
    # sentence-transformers) and the warning goes away -- no openai/litellm
    # extra needs installing.
    return MemorySettings(
        backend="bolt",
        neo4j={
            "uri": cfg.neo4j_uri,
            "username": cfg.neo4j_username,
            "password": SecretStr(cfg.neo4j_password),
            "database": cfg.neo4j_database,
        },
        embedding=cfg.embedding_model,
        # No llm= -> we skip LLM-driven entity extraction and add entities
        # manually. This keeps the whole system local / API-key-free.
        extraction=ExtractionConfig(enable_llm_fallback=False),
    )


async def connect(cfg: AgentConfig | None = None):
    """Create and connect a NAMS ``MemoryClient``. Caller is responsible for
    closing it (use ``async with``)."""
    from neo4j_agent_memory import MemoryClient

    settings = make_memory_settings(cfg)
    client = MemoryClient(settings)
    await client.connect()
    return client


def new_session_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------- short_term

async def add_user_question(
    client: Any, session_id: str, text: str, snapshot_id: str | None = None
) -> Any:
    meta = {"kind": "game_question"}
    if snapshot_id:
        meta["snapshot_id"] = snapshot_id
    return await client.short_term.add_message(
        session_id=session_id, role="user", content=text, metadata=meta
    )


async def add_assistant_message(
    client: Any,
    session_id: str,
    content: str,
    kind: str = "move",
    snapshot_id: str | None = None,
) -> Any:
    meta = {"kind": kind}
    if snapshot_id:
        meta["snapshot_id"] = snapshot_id
    return await client.short_term.add_message(
        session_id=session_id, role="assistant", content=content, metadata=meta
    )


# ---------------------------------------------------------------- reasoning

async def start_move_trace(client: Any, session_id: str, task: str, triggered_by_message_id: str | None = None) -> Any:
    kwargs: dict[str, Any] = {"session_id": session_id, "task": task}
    if triggered_by_message_id:
        # NAMS hands back UUID objects; normalise to the stored string form so
        # any downstream bolt parameter binding works.
        kwargs["triggered_by_message_id"] = str(triggered_by_message_id)
    return await client.reasoning.start_trace(**kwargs)


async def record_move_trace(
    client: Any,
    trace: Any,
    thought: str,
    action: str,
    gold_collected: int,
) -> None:
    """Add a reasoning step + tool call to an in-progress trace."""
    step = await client.reasoning.add_step(trace.id, thought=thought)
    await client.reasoning.record_tool_call(
        step.id,
        action,  # tool name = the action (CLOCK/ANTICLOCK/FORWARD)
        {"action": action},
        {"gold_collected": int(gold_collected)},
    )


async def complete_move_trace(
    client: Any, trace: Any, outcome: str, success: bool = True
) -> None:
    await client.reasoning.complete_trace(trace.id, outcome=outcome, success=success)


# ------------------------------------------------------------------ context

_SETTINGS_LEAK_KEYS = ("settings_json", "settings", "walls", "gold", "agent_x", "agent_y", "direction")


def _strip_settings(obj: Any) -> Any:
    """Recursively remove any settings-leaking keys from a context payload
    before it reaches the model in mode 1. Defensive: NAMS' ``get_context``
    does not include ``GameSnapshot`` properties, but we filter explicitly."""
    if isinstance(obj, dict):
        return {
            k: _strip_settings(v)
            for k, v in obj.items()
            if k not in _SETTINGS_LEAK_KEYS
        }
    if isinstance(obj, list):
        return [_strip_settings(v) for v in obj]
    return obj


_SETTINGS_TEXT_RE = re.compile(
    r'"(settings_json|settings|walls|gold|agent_x|agent_y|direction)"\s*:\s*[^,}\]]+',
    re.IGNORECASE,
)


def _strip_settings_from_text(text: str) -> str:
    return _SETTINGS_TEXT_RE.sub('"<redacted>"', text)


async def get_game_context(client: Any, session_id: str, query: str) -> str:
    """Wrapper around ``client.get_context`` that scrubs any settings-leaking
    fields before they reach the model in mode 1."""
    ctx = await client.get_context(query=query, session_id=session_id)
    if isinstance(ctx, str):
        return _strip_settings_from_text(ctx)
    cleaned = _strip_settings(ctx)
    # NAMS may return a structured object; stringify for the prompt.
    import json as _json

    try:
        return _json.dumps(cleaned, default=str, indent=2)
    except Exception:
        return str(cleaned)


# ---------------------------------------------------------- semantic model

_SEMANTIC_MODEL_ENTITIES = [
    ("Agent", "PERSON", "The green circle controlled by the player; has a red eye showing its facing direction."),
    ("Gold", "OBJECT", "Small yellow circle the agent must collect. Bare levels have exactly one."),
    ("BoundaryWall", "OBJECT", "The four fixed walls framing the play area. Always present."),
    ("DiscreteGame", "SYSTEM", "The 2D discrete game engine: 224x224 board, agent + gold + walls."),
    ("Direction", "ATTRIBUTE", "The agent's facing angle in radians, 0..2pi, measured CCW from +x."),
]

_SEMANTIC_MODEL_PREFERENCES = [
    ("controls", "Available moves are CLOCK (turn clockwise), ANTICLOCK (turn counter-clockwise), and FORWARD (advance one step). One CLOCK/ANTICLOCK step is pi/30 radians; one FORWARD step is up to 1/16 of the board."),
    ("geometry", "The board is the unit square [0,1]x[0,1]. All coordinates are normalised; agent_r ~ 0.05, gold_r ~ 1/64."),
    ("goal", "Collect the gold piece. In bare levels there is exactly one gold piece; the game ends for the agent once it is eaten (overlap of agent and gold circles)."),
    ("tip_distance", "Tip: the agent does not need to know its exact coordinates. Use the visual angle between the agent's red eye and the gold to decide CLOCK vs ANTICLOCK, then FORWARD."),
    ("tip_facing", "Tip: if the gold is roughly in front of the agent's eye, FORWARD is the best move. If it is to the right, CLOCK until it is centered. If to the left, ANTICLOCK."),
    ("tip_overshoot", "Tip: only FORWARD moves the agent (up to 1/16 of the board per step); CLOCK and ANTICLOCK merely rotate it in place and never move it, so they cannot collect gold on their own. FORWARD can overshoot the gold, so aim carefully with CLOCK/ANTICLOCK first, then step FORWARD."),
]

# Hardwired relationships between the seeded entities, as (subject, predicate,
# object) triples. We write them as *direct* Neo4j edges between the Entity
# nodes (see :func:`add_semantic_relationships`), not as NAMS ``Fact`` nodes:
# ``long_term.add_fact`` only creates a standalone Fact node with the endpoints
# as string properties -- it does NOT link the Entity nodes, so no edge ever
# shows up in the graph. A direct MERGE gives a real, labeled edge.
_SEMANTIC_MODEL_RELATIONSHIPS = [
    ("Agent", "collects", "Gold"),
    ("Agent", "has", "Direction"),
    ("Agent", "bounded_by", "BoundaryWall"),
    ("DiscreteGame", "contains", "Agent"),
    ("DiscreteGame", "contains", "Gold"),
    ("DiscreteGame", "contains", "BoundaryWall"),
]


async def add_semantic_relationships(client: Any) -> dict[str, int]:
    """Create direct edges between the seeded Entity nodes (see
    ``_SEMANTIC_MODEL_RELATIONSHIPS``) via bolt write-Cypher.

    Each relationship is a real ``(:Entity)-[:PREDICATE]->(:Entity)`` edge
    (relationship type = the predicate upper-cased), so it is immediately
    visible in the visualization. ``MERGE`` makes this idempotent -- safe to
    run repeatedly against an already-seeded graph without wiping, and without
    touching or duplicating the Entity / Preference nodes.

    Also removes any redundant standalone ``Fact`` nodes left by an earlier
    ``add_fact``-based version of this seed (matched by the same triple), so
    the graph doesn't carry duplicate, edge-less clutter.
    """
    n_rel = 0
    for subject, predicate, obj in _SEMANTIC_MODEL_RELATIONSHIPS:
        # Predicates come from the fixed list above (no injection risk); use the
        # predicate as the edge TYPE so the viz shows a meaningful label.
        rel_type = predicate.upper()
        try:
            await client.graph.execute_write(
                f"MATCH (a:Entity {{name: $s}}), (b:Entity {{name: $o}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b) "
                f"SET r.predicate = $p",
                {"s": subject, "o": obj, "p": predicate},
            )
            n_rel += 1
        except Exception as exc:  # pragma: no cover - best-effort seed
            logger.warning(
                "relationship %s-[%s]->%s failed: %s", subject, predicate, obj, exc
            )
        # Clean up the edge-less Fact node from the previous approach, if any.
        try:
            await client.graph.execute_write(
                "MATCH (f:Fact {subject: $s, predicate: $p, object: $o}) "
                "DETACH DELETE f",
                {"s": subject, "o": obj, "p": predicate},
            )
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("Fact cleanup for %s-%s-%s failed: %s", subject, predicate, obj, exc)

    logger.info("Added %d semantic relationships (direct edges).", n_rel)
    return {"relationships": n_rel}


async def build_semantic_model(client: Any) -> dict[str, int]:
    """Seed the long-term memory graph with a small description of the game
    and a handful of tips. Idempotent-ish: re-running adds the same entities
    again (NAMS performs entity resolution / dedup on its side).

    Returns counts of how many entities/preferences were attempted.
    """
    n_ent = 0
    for name, etype, desc in _SEMANTIC_MODEL_ENTITIES:
        try:
            await client.long_term.add_entity(name, etype, description=desc)
            n_ent += 1
        except TypeError:
            # Older NAMS versions may not accept ``description``.
            await client.long_term.add_entity(name, etype)
            n_ent += 1
        except Exception as exc:  # pragma: no cover - best-effort seed
            logger.warning("add_entity(%s) failed: %s", name, exc)

    n_pref = 0
    for category, preference in _SEMANTIC_MODEL_PREFERENCES:
        try:
            await client.long_term.add_preference(
                category=category, preference=preference
            )
            n_pref += 1
        except Exception as exc:  # pragma: no cover - best-effort seed
            logger.warning("add_preference(%s) failed: %s", category, exc)

    rel_counts = await add_semantic_relationships(client)
    n_rel = rel_counts.get("relationships", 0)

    logger.info(
        "Seeded semantic model: %d entities, %d preferences, %d relationships.",
        n_ent, n_pref, n_rel,
    )
    return {"entities": n_ent, "preferences": n_pref, "relationships": n_rel}
