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
from . import run_logging

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
    #
    # The spaCy + GLiNER stages ARE enabled (NAMS defaults) so entities are
    # auto-discovered from messages during runs. Those two stages need their
    # packages AND model weights installed -- run ``scripts/setup_env.sh`` (not
    # just ``pip install -r requirements.txt``), or every stored message logs
    # "Stage 'SpacyEntityExtractor'/'GLiNEREntityExtractor' failed".
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
#
# One reasoning trace per *turn* (not per move). The trace's ``task`` is the
# user's instruction/goal (e.g. the question that started the turn); each move
# the agent makes becomes a ``ReasoningStep`` within that single trace, with the
# move recorded as a tool call. This matches NAMS' model -- a trace is one
# reasoning episode toward a goal -- and makes traces meaningful to both the
# in-play ``get_similar_traces`` recall and the mode-3 self-evaluation.

async def start_turn_trace(
    client: Any, session_id: str, task: str, triggered_by_message_id: str | None = None
) -> Any:
    """Open ONE trace for the whole turn. ``task`` is the goal (the user's
    instruction/question), NOT a mechanical action -- the moves are the steps."""
    kwargs: dict[str, Any] = {"session_id": session_id, "task": task}
    if triggered_by_message_id:
        # NAMS hands back UUID objects; normalise to the stored string form so
        # any downstream bolt parameter binding works.
        kwargs["triggered_by_message_id"] = str(triggered_by_message_id)
    return await client.reasoning.start_trace(**kwargs)


async def add_reasoning_step(
    client: Any,
    trace: Any,
    thought: str,
    action: str | None = None,
    gold_collected: int = 0,
) -> None:
    """Record one step of the turn's reasoning: the model's ``thought`` (its raw
    output for this generation) and, if it emitted a move, that move as a tool
    call. ``action=None`` captures a non-move generation (e.g. the final reply
    that ends the turn), which is still part of the reasoning episode."""
    if trace is None:
        return
    step = await client.reasoning.add_step(trace.id, thought=thought)
    if action:
        # NAMS ``record_tool_call`` takes only (step_id, tool_name, arguments) as
        # positionals; the tool output is the keyword-only ``result=``.
        await client.reasoning.record_tool_call(
            step.id,
            action,  # tool name = the action (CLOCK/ANTICLOCK/FORWARD)
            {"action": action},
            result={"gold_collected": int(gold_collected)},
        )


async def complete_turn_trace(
    client: Any, trace: Any, outcome: str, success: bool = False
) -> None:
    if trace is None:
        return
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


def _message_timestamp(m: Any) -> Any:
    """Best-effort timestamp accessor -- NAMS' Message dataclass uses
    ``timestamp``; some code paths expose ``created_at``."""
    return getattr(m, "timestamp", None) or getattr(m, "created_at", None)


def _format_message_line(m: Any) -> str:
    role = getattr(m, "role", "?")
    role = role.value if hasattr(role, "value") else str(role)
    content = _strip_settings_from_text(str(getattr(m, "content", "")))
    return f"[{role}] {content}"


async def get_recent_messages(client: Any, session_id: str, window: int) -> str:
    """Return the last ``window`` messages of this session by **recency**
    (chronological, most-recent last), independent of semantic similarity.

    NAMS' ``get_context`` recall is pure vector similarity (and its built-in
    "Recent Conversation" section is ordered ASC, i.e. actually the OLDEST
    messages), so recent moves are not guaranteed to surface. This queries the
    session's message tail directly -- ``ORDER BY timestamp DESC LIMIT window``
    over the ``(:Conversation {session_id})-[:HAS_MESSAGE]->(:Message)`` chain
    -- which stays constant-time as the conversation grows. Falls back to
    fetching the whole conversation and slicing the tail if the direct query
    fails (e.g. a NAMS schema change). Settings are stripped for the mode-1
    privacy invariant. Returns "" if there are none / on error.
    """
    if window <= 0:
        return ""
    lines: list[str] = []
    try:
        rows = await client.query.cypher(
            "MATCH (c:Conversation {session_id: $sid})-[:HAS_MESSAGE]->(m:Message) "
            "RETURN m.role AS role, m.content AS content "
            "ORDER BY m.timestamp DESC LIMIT $window",
            {"sid": session_id, "window": window},
        )
        # DESC gave us newest-first; reverse to chronological, most-recent last.
        for r in reversed(list(rows or [])):
            d = dict(r)
            content = _strip_settings_from_text(str(d.get("content", "")))
            lines.append(f"[{d.get('role', '?')}] {content}")
    except Exception as exc:
        logger.debug("direct recent-message query failed (%s); falling back", exc)
        lines = await _recent_messages_fallback(client, session_id, window)
    result = "\n".join(lines)
    run_logging.log_db_retrieval(
        function="get_recent_messages",
        arguments={"session_id": session_id, "window": window},
        result=result,
    )
    return result


async def _recent_messages_fallback(
    client: Any, session_id: str, window: int
) -> list[str]:
    """Slow path: fetch the WHOLE conversation via the NAMS API and slice the
    tail. limit=None is deliberate -- NAMS orders ASC, so a small limit would
    return the OLDEST messages instead of the most recent ones."""
    msgs: list[Any] = []
    try:
        convo = await client.short_term.get_conversation(session_id=session_id, limit=None)
        msgs = list(getattr(convo, "messages", None) or [])
    except Exception as exc:  # pragma: no cover - fall back to get_messages
        logger.debug("get_conversation failed (%s); trying get_messages", exc)
        try:
            msgs = list(
                await client.short_term.get_messages(session_id=session_id, limit=100000)
                or []
            )
        except Exception as exc2:
            logger.warning("recent-message fetch failed: %s", exc2)
            return []
    if not msgs:
        return []
    # Defensive: ensure chronological order before taking the tail.
    try:
        if all(_message_timestamp(m) is not None for m in msgs):
            msgs = sorted(msgs, key=_message_timestamp)
    except Exception:  # pragma: no cover - heterogeneous timestamps
        pass
    return [_format_message_line(m) for m in msgs[-window:]]


async def retrieve_context(client: Any, query: str, session_id: str) -> Any:
    """Thin wrapper over NAMS ``client.get_context`` (the semantic search across
    all memory tiers) that logs the retrieval. Returns the raw context object /
    string exactly as NAMS provides it -- callers do their own scrubbing."""
    ctx = await client.get_context(query=query, session_id=session_id)
    if isinstance(ctx, str):
        result = ctx
    else:
        import json as _json

        try:
            result = _json.dumps(ctx, default=str, indent=2)
        except Exception:
            result = str(ctx)
    run_logging.log_db_retrieval(
        function="client.get_context",
        arguments={"query": query, "session_id": session_id},
        result=result,
    )
    return ctx


# NAMS ``get_context`` embeds its own "### Recent Conversation" section in the
# formatted context it returns. We already inject the session's recent messages
# explicitly (see :func:`get_recent_messages`), so keeping NAMS's copy puts the
# same recent moves into the prompt twice. The intended split is: the recency
# window is the ONE channel for recent moves; the semantic block carries only
# general memories (relevant past messages, entities, preferences, traces).
# This regex removes NAMS's recent-conversation subsection (up to the next
# heading), leaving the rest of its output intact.
_NAMS_RECENT_CONVO_RE = re.compile(
    r"### Recent Conversation\n.*?(?=\n#{2,3} |\Z)", re.DOTALL
)


def strip_nams_recent_conversation(text: str) -> str:
    out = _NAMS_RECENT_CONVO_RE.sub("", text)
    # If that emptied the whole '## Conversation History' section, drop its
    # now-dangling header too.
    out = re.sub(r"## Conversation History\s*(?=\n## |\Z)", "", out)
    return out.strip("\n")


async def get_game_context(
    client: Any, session_id: str, query: str, recent_window: int = 0
) -> str:
    """Build the mode-1 memory context: a **recency** window of the last
    ``recent_window`` session messages (always included, verbatim) followed by
    the general **semantic** search across all memory tiers (messages, entities,
    preferences, reasoning traces). Both are scrubbed of any settings-leaking
    fields before they reach the model.

    Passing ``recent_window=0`` reproduces the old behaviour (semantic only).
    """
    ctx = await retrieve_context(client, query=query, session_id=session_id)
    if isinstance(ctx, str):
        semantic = _strip_settings_from_text(ctx)
    else:
        cleaned = _strip_settings(ctx)
        # NAMS may return a structured object; stringify for the prompt.
        import json as _json

        try:
            semantic = _json.dumps(cleaned, default=str, indent=2)
        except Exception:
            semantic = str(cleaned)
    # Recent moves belong to the recency window below, not the semantic block.
    semantic = strip_nams_recent_conversation(semantic)

    recent = await get_recent_messages(client, session_id, recent_window)

    parts: list[str] = []
    if recent:
        parts.append(
            "Recent conversation (most recent last -- your latest questions and "
            "moves this session, in order):\n" + recent
        )
    if semantic and semantic.strip() not in ("", "{}", "[]"):
        parts.append(
            "Relevant memories (semantic search across messages, entities, "
            "preferences, and past reasoning):\n" + semantic
        )
    return "\n\n".join(parts)


async def get_semantic_model(client: Any) -> str:
    """Return the full curated long-term semantic model (all ``Entity`` +
    ``Preference`` nodes) as a formatted string.

    The general ``get_context`` recall is thresholded similarity search, so it
    only returns a *subset* of long-term memory. The privileged modes (2/3) that
    should reason/judge against the complete rubric want the whole thing, so we
    read it directly. Best-effort: returns "" on error."""
    ent_rows: list[Any] = []
    pref_rows: list[Any] = []
    try:
        ent_rows = await client.graph.execute_write(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, "
            "e.description AS description ORDER BY e.name",
            {},
        ) or []
    except Exception as exc:  # pragma: no cover - best-effort read
        logger.debug("semantic-model entity fetch failed: %s", exc)
    try:
        pref_rows = await client.graph.execute_write(
            "MATCH (p:Preference) RETURN p.category AS category, "
            "p.preference AS preference ORDER BY p.category",
            {},
        ) or []
    except Exception as exc:  # pragma: no cover - best-effort read
        logger.debug("semantic-model preference fetch failed: %s", exc)

    lines: list[str] = []
    if ent_rows:
        lines.append("Entities:")
        for r in ent_rows:
            d = dict(r)
            lines.append(
                f"  - {d.get('name')} ({d.get('type')}): {d.get('description')}"
            )
    if pref_rows:
        lines.append("Preferences / tips:")
        for r in pref_rows:
            d = dict(r)
            lines.append(f"  - [{d.get('category')}] {d.get('preference')}")
    result = "\n".join(lines)
    run_logging.log_db_retrieval(
        function="get_semantic_model",
        arguments={"entities": len(ent_rows), "preferences": len(pref_rows)},
        result=result,
    )
    return result


# -------------------------------------------------------------------- DB dump

async def dump_database_to_file(
    client: Any, path: Any, include_embeddings: bool = False
) -> dict[str, Any]:
    """Dump the whole graph (all nodes + relationships) to ``path`` as a JSON
    ``.dump`` file, over the **live** bolt connection (no need to stop Neo4j, so
    it is safe to call from a running notebook session).

    This is a *logical* snapshot for inspection/analysis -- distinct from the
    native binary ``neo4j-admin database dump`` that ``scripts/neo4j_db.sh save``
    produces (which requires stopping the DB and is only loadable by
    ``neo4j-admin``). Embedding vectors are dropped by default (huge, not
    human-useful); pass ``include_embeddings=True`` to keep them.
    """
    from pathlib import Path as _Path

    def _clean(props: Any) -> dict:
        d = dict(props or {})
        if not include_embeddings:
            d = {k: v for k, v in d.items() if "embedding" not in k.lower()}
        return d

    node_rows = await client.graph.execute_write(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, "
        "properties(n) AS props",
        {},
    ) or []
    rel_rows = await client.graph.execute_write(
        "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, type(r) AS type, "
        "elementId(a) AS start, elementId(b) AS end, properties(r) AS props",
        {},
    ) or []

    nodes = []
    for r in node_rows:
        d = dict(r)
        nodes.append({
            "id": d.get("id"),
            "labels": d.get("labels"),
            "properties": _clean(d.get("props")),
        })
    relationships = []
    for r in rel_rows:
        d = dict(r)
        relationships.append({
            "id": d.get("id"),
            "type": d.get("type"),
            "start": d.get("start"),
            "end": d.get("end"),
            "properties": _clean(d.get("props")),
        })

    import datetime as _dt
    import json as _json

    payload = {
        "meta": {
            "dumped_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "include_embeddings": include_embeddings,
        },
        "nodes": nodes,
        "relationships": relationships,
    }
    out_path = _Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(payload, default=str, indent=2), encoding="utf-8")
    logger.info(
        "DB dump written: %s (%d nodes, %d relationships)",
        out_path, len(nodes), len(relationships),
    )
    return {
        "path": str(out_path),
        "nodes": len(nodes),
        "relationships": len(relationships),
    }


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
