"""The three agent modes:

  * :func:`mode_game`     -- game-playing / question answering (mode 1)
  * :func:`mode_discuss`  -- open-ended discussion with full memory access (mode 2)
  * :func:`mode_self_eval`-- self-evaluation over a Conversation + traces (mode 3)

All modes are async and share one NAMS ``MemoryClient`` and one
``Gemma4E4B`` instance. Mode 1 never lets the Settings dict reach the model;
mode 3 is the only mode where Settings JSON is included in the prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import AgentConfig, CONFIG
from . import game_io
from . import image_store
from . import memory as mem
from .model import get_model

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_GAME = (
    "You are an agent playing a 2D discrete game on a 64x64 board. You see "
    "the current game screen as an image. The board is the unit square "
    "framed by four boundary walls; there is exactly one gold piece (small "
    "yellow circle). You are the green circle with a red eye showing the "
    "direction you are facing. Your goal is to collect the gold.\n\n"
    "Available moves: CLOCK (turn clockwise by pi/30), ANTICLOCK (turn "
    "counter-clockwise by pi/30), FORWARD (advance up to 1/16 of the board "
    "in the facing direction).\n\n"
    "When asked to make a move, reply with exactly one move keyword on its "
    "own line (CLOCK, ANTICLOCK, or FORWARD). You may add a brief one-line "
    "reason first. When asked a question about the screen, answer it in "
    "plain prose without making a move. When asked to 'solve the game', "
    "emit one move at a time; the harness will re-render the screen and "
    "call you again until the gold is eaten."
)

SYSTEM_PROMPT_DISCUSS = (
    "You are an agent that plays a 2D discrete game and is also able to "
    "discuss it openly with the user. You have full access to your memory "
    "database (conversations, the game's semantic model, and your past "
    "reasoning traces). Be concise and helpful. You are not seeing a live "
    "game screen in this mode; rely on memory and the user's description."
)

SYSTEM_PROMPT_EVAL = (
    "You are evaluating how well an earlier instance of yourself played a "
    "2D discrete game. You are given the full Conversation (user questions "
    "and assistant moves), the reasoning traces recorded for each move, "
    "and the underlying game Settings at each step (which the player did "
    "NOT have access to at the time). Be specific and critical. Output a "
    "structured verdict with: overall_score (0-10), strengths, weaknesses, "
    "and per-move notes where relevant."
)


# --------------------------------------------------------------- mode 1

def _build_game_messages(system: str, image_path: str, context: str, question: str) -> list[dict]:
    user_text = []
    if context:
        user_text.append({"type": "text", "text": f"Memory context:\n{context}"})
    user_text.append({"type": "image", "url": image_path})
    user_text.append(
        {
            "type": "text",
            "text": f"Question / instruction: {question}",
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_text},
    ]


async def _record_turn(
    client: Any,
    session_id: str,
    cfg: AgentConfig,
    game: Any,
    question: str,
    raw_out: str,
    action: str | None,
    gold_collected: int,
    snapshot_before_id: str,
    snapshot_before_path: str,
) -> dict[str, Any]:
    """Persist one mode-1 turn to NAMS: user message + assistant message +
    reasoning trace + before/after GameSnapshot nodes."""
    # 1. User message linked to the 'before' snapshot.
    user_msg = await mem.add_user_question(
        client, session_id, question, snapshot_id=snapshot_before_id
    )
    await image_store.link_snapshot_to_message(
        client, user_msg.id, snapshot_before_id, role="before"
    )

    # 2. Assistant message (the raw model output; if it was a move, also tag it).
    assistant_content = raw_out if raw_out else (action or "")
    assistant_msg = await mem.add_assistant_message(
        client,
        session_id,
        assistant_content,
        kind="move" if action else "answer",
    )

    # 3. Reasoning trace for the move (if any).
    trace = None
    if action:
        trace = await mem.start_move_trace(
            client, session_id, task=f"apply {action}", triggered_by_message_id=user_msg.id
        )
        await mem.record_move_trace(client, trace, thought=raw_out, action=action, gold_collected=gold_collected)
        await mem.complete_move_trace(
            client, trace,
            outcome=f"applied {action}; gold_collected={gold_collected}",
            success=(gold_collected > 0) or True,
        )

    # 4. 'After' snapshot (only if a move was actually applied).
    snapshot_after_id = None
    snapshot_after_path = None
    if action:
        snapshot_after_id = image_store.snapshot_id()
        settings_after = game_io.game_to_settings_dict(game)
        snapshot_after_path, _ = await image_store.store_snapshot(
            client, session_id, snapshot_after_id, game, settings_after, cfg=cfg,
            label="after",
            extra={"action": action, "gold_collected": gold_collected},
        )
        await image_store.link_snapshot_to_message(
            client, assistant_msg.id, snapshot_after_id, role="after"
        )

    return {
        "user_msg_id": user_msg.id,
        "assistant_msg_id": assistant_msg.id,
        "trace_id": trace.id if trace else None,
        "snapshot_before_id": snapshot_before_id,
        "snapshot_before_path": snapshot_before_path,
        "snapshot_after_id": snapshot_after_id,
        "snapshot_after_path": snapshot_after_path,
        "action": action,
        "gold_collected": gold_collected,
    }


async def mode_game(
    client: Any,
    session_id: str,
    question: str,
    solve: bool = False,
    cfg: AgentConfig | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Mode 1: game-playing / question answering.

    If ``solve`` is True, loop moves until the gold is eaten (or the step
    cap is hit), re-rendering the screen and recomputing context each step.
    Otherwise, take a single turn: the model either answers the question or
    emits one move.
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)
    max_steps = max_steps or cfg.max_solve_steps

    game = game_io.new_bare_game(gameSize=cfg.game_size)
    turns: list[dict[str, Any]] = []

    steps = 0
    # Capture the initial 'before' snapshot for the first turn.
    snapshot_before_id = image_store.snapshot_id()
    settings_before = game_io.game_to_settings_dict(game)
    snapshot_before_path, _ = await image_store.store_snapshot(
        client, session_id, snapshot_before_id, game, settings_before, cfg=cfg,
        label="before",
        extra={"step": steps},
    )

    while True:
        # Recompute context with the current image each step.
        ctx = await mem.get_game_context(client, session_id, query=question)
        messages = _build_game_messages(SYSTEM_PROMPT_GAME, snapshot_before_path, ctx, question)
        raw = model.generate(messages)
        action = game_io.parse_action(raw)

        if action:
            gold_collected = game_io.apply_action(game, action)
        else:
            gold_collected = 0

        turn = await _record_turn(
            client, session_id, cfg, game, question, raw, action,
            gold_collected, snapshot_before_id, snapshot_before_path,
        )
        turns.append(turn)
        steps += 1
        logger.info("step %d: action=%s gold=%d remaining=%d",
                    steps, action, gold_collected, game_io.gold_remaining(game))

        # If no move was produced, this is a Q&A turn -- stop here.
        if not action:
            break

        # Set up the 'before' snapshot for the next iteration (the current
        # 'after' state becomes the next 'before').
        snapshot_before_id = image_store.snapshot_id()
        settings_before = game_io.game_to_settings_dict(game)
        snapshot_before_path, _ = await image_store.store_snapshot(
            client, session_id, snapshot_before_id, game, settings_before, cfg=cfg,
            label="before",
            extra={"step": steps},
        )

        # Stop conditions.
        if not solve:
            break
        if game_io.gold_remaining(game) == 0:
            logger.info("Gold eaten after %d steps.", steps)
            break
        if steps >= max_steps:
            logger.warning("Hit max_steps=%d without eating the gold.", max_steps)
            break

    return {"session_id": session_id, "turns": turns, "steps": steps,
            "gold_remaining": game_io.gold_remaining(game)}


# --------------------------------------------------------------- mode 2

async def mode_discuss(
    client: Any,
    session_id: str,
    user_text: str,
    cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    """Mode 2: open-ended discussion with full memory access.

    Records the user message, retrieves full context (no settings stripping
    -- this mode is the bootstrap/evaluation channel and is allowed to see
    everything NAMS returns), generates a response, and records the
    assistant message.
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)

    await client.short_term.add_message(
        session_id=session_id, role="user", content=user_text,
        metadata={"kind": "discussion"},
    )
    ctx = await client.get_context(query=user_text, session_id=session_id)
    ctx_text = ctx if isinstance(ctx, str) else json.dumps(ctx, default=str, indent=2)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DISCUSS}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"Memory context:\n{ctx_text}"},
            {"type": "text", "text": f"User: {user_text}"},
        ]},
    ]
    reply = model.generate(messages)
    await client.short_term.add_message(
        session_id=session_id, role="assistant", content=reply,
        metadata={"kind": "discussion"},
    )
    return {"session_id": session_id, "reply": reply}


# --------------------------------------------------------------- mode 3

def _format_session_for_eval(
    messages_with_snaps: list[dict[str, Any]],
    traces: list[dict[str, Any]] | None,
) -> str:
    """Build a text dump of the Conversation + snapshots + traces for the
    evaluator. The Settings JSON on each snapshot IS included here."""
    lines = ["# Conversation"]
    for i, row in enumerate(messages_with_snaps):
        m = row.get("message") or {}
        snaps = row.get("snapshots") or []
        role = m.get("role", "?")
        content = m.get("content", "")
        meta = m.get("metadata") or {}
        lines.append(f"## Turn {i} -- {role} (kind={meta.get('kind')})")
        lines.append(f"content: {content}")
        for s in snaps:
            lines.append(f"  snapshot id={s.get('id')} label={s.get('label')}")
            lines.append(f"    path={s.get('path')} "
                         f"size={s.get('width')}x{s.get('height')}")
            lines.append(f"    settings_json={s.get('settings_json')}")
    if traces:
        lines.append("\n# Reasoning traces")
        for t in traces:
            lines.append(f"## Trace {t.get('id')} task={t.get('task')}")
            for k, v in t.items():
                if k in ("id", "task"):
                    continue
                lines.append(f"  {k}: {v}")
    return "\n".join(lines)


async def mode_self_eval(
    client: Any,
    session_id: str,
    cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    """Mode 3: self-evaluate a recorded session.

    Pulls the Conversation (with linked GameSnapshot nodes incl. their
    Settings JSON), pulls the reasoning traces for the session, feeds them
    to Gemma 4 E4B with the evaluator system prompt, then appends the
    verdict to the SAME Conversation as an assistant message
    (metadata.kind='self_evaluation') and records a new reasoning trace
    capturing the evaluation reasoning.
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)

    messages_with_snaps = await image_store.fetch_messages_with_snapshots(client, session_id)
    traces = await _fetch_session_traces(client, session_id)
    dump = _format_session_for_eval(messages_with_snaps, traces)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_EVAL}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"Session id: {session_id}\n\n{dump}"},
        ]},
    ]
    verdict = model.generate(messages, max_new_tokens=cfg.gemma_max_new_tokens)

    # Append the verdict to the same Conversation.
    eval_msg = await client.short_term.add_message(
        session_id=session_id, role="assistant", content=verdict,
        metadata={"kind": "self_evaluation", "evaluated_session": session_id},
    )

    # Record a reasoning trace for the evaluation itself.
    trace = await client.reasoning.start_trace(
        session_id, task=f"self-evaluation of session {session_id}",
        triggered_by_message_id=eval_msg.id,
    )
    step = await client.reasoning.add_step(trace.id, thought=verdict)
    await client.reasoning.record_tool_call(
        step.id, "self_evaluate", {"session_id": session_id},
        {"verdict_length": len(verdict)},
    )
    await client.reasoning.complete_trace(
        trace.id, outcome="self-evaluation appended to conversation", success=True,
    )

    return {
        "session_id": session_id,
        "verdict": verdict,
        "eval_message_id": eval_msg.id,
        "trace_id": trace.id,
    }


async def _fetch_session_traces(client: Any, session_id: str) -> list[dict[str, Any]]:
    """Best-effort retrieval of reasoning traces for a session.

    Tries ``client.reasoning.search_traces`` first; falls back to a Cypher
    match on ``(:Trace)`` linked to messages in the session.
    """
    traces: list[dict[str, Any]] = []
    try:
        found = await client.reasoning.search_traces(session_id, limit=100)  # type: ignore[arg-type]
        traces = [dict(t) for t in (found or [])]
        if traces:
            return traces
    except Exception as exc:  # pragma: no cover
        logger.debug("search_traces failed: %s", exc)
    try:
        rows = await client.query.cypher(
            "MATCH (t:Trace {session_id: $sid}) RETURN t ORDER BY t.created_at ASC",
            {"sid": session_id},
        )
        traces = [dict(r["t"]) for r in rows if "t" in r]
    except Exception as exc:  # pragma: no cover
        logger.debug("Cypher trace fetch failed: %s", exc)
    return traces
