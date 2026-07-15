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
import re
from typing import Any

from .config import AgentConfig, CONFIG
from . import game_io
from . import image_store
from . import memory as mem
from . import run_logging
from .model import get_model

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_GAME = (
    "You are an agent playing a 2D discrete game on a 224x224 board. You see "
    "the current game screen as an image. The board is the unit square "
    "framed by four boundary walls; there is exactly one gold piece (small "
    "yellow circle). You are the green circle with a red eye showing the "
    "direction you are facing. Your goal is to collect the gold.\n\n"
    "You make moves by emitting exactly one of these move tokens:\n"
    "  [CLOCK]     - turn clockwise by pi/30 (rotate in place)\n"
    "  [ANTICLOCK] - turn counter-clockwise by pi/30 (rotate in place)\n"
    "  [FORWARD]   - advance up to 1/16 of the board in the facing direction\n"
    "Only [FORWARD] moves you; [CLOCK] and [ANTICLOCK] only rotate you.\n\n"
    "IMPORTANT: ONLY those exact bracketed tokens trigger a move. Talking about "
    "moving in prose (e.g. writing the word 'forward') does NOTHING. You may "
    "reason in plain prose as much as you like; nothing happens until you emit "
    "a bracketed move token, so emit one only when you truly intend that move.\n\n"
    "The instant you emit a move token it is executed, the screen is "
    "re-rendered, and you are shown the updated screen and asked for your next "
    "move. So a turn is a sequence of moves -- reason, emit one move token, see "
    "the result, repeat -- e.g. [CLOCK] [CLOCK] [FORWARD] [FORWARD] ... to "
    "navigate to the gold. Making a move does NOT end your turn.\n\n"
    "To END your turn, simply finish your reply WITHOUT emitting a move token "
    "(just stop normally). Do this once you have collected the gold or wish to "
    "stop. If the user asks a question about the screen rather than asking you "
    "to play, answer in prose with no move token (which likewise ends the turn).\n\n"
    "NOTE: the default mode is to try to solve the game (get the gold), which is "
    "what the user usually asks for. However, you may instead be asked questions "
    "about the game; in that case answer in prose, and avoid using move tokens "
    "unless it is very apparent that the user is asking you to play.\n\n"
    "HOW TO PLAY -- take this reasoning step before every move. First REASON, in "
    "a sentence or two, about where the gold is relative to your red eye: the eye "
    "points in the direction you face, and [FORWARD] sends you straight that way. "
    "Then choose your move from that reasoning:\n"
    "  - If your eye is pointing roughly at the gold, emit [FORWARD].\n"
    "  - Otherwise, aim your eye at the gold: emit [CLOCK] or [ANTICLOCK], then "
    "check the re-rendered screen to see which way your eye swung, and keep "
    "rotating that way (or reverse if you overshoot) until your eye lines up with "
    "the gold -- then emit [FORWARD].\n"
    "Always state this reasoning before the move token; reasoning first, then a "
    "single move, is how you decide well.\n\n"
    "DO NOT just copy prior observations from your memories. Make sure you "
    "evaluate whether you are facing the gold *right now*. Your memories "
    "describe PAST screens; every move changes the screen, so re-derive where "
    "your red eye points and where the gold is from the CURRENT image before "
    "every single move."
)

SYSTEM_PROMPT_REFLECT = (
    "You are an agent playing a 2D discrete game on a 224x224 board. The board "
    "is the unit square framed by four boundary walls; there is exactly one "
    "gold piece (small yellow circle). You are the green circle with a red eye "
    "showing the direction you are facing.\n\n"
    "This is NOT a move request. This is a REFLECTION pause: you have made many "
    "moves without collecting the gold, so you must stop, look at the CURRENT "
    "screen with fresh eyes, and re-examine your plan before continuing. Your "
    "recent reasoning may have been repeating itself without checking the "
    "screen; assume nothing you previously said is still true.\n\n"
    "Study the CURRENT image and answer, concretely and honestly:\n"
    "1. Am I *certain* that I was never facing the gold at any point during my "
    "recent moves? Each rotation is 6 degrees, so 30 rotations sweep 180 "
    "degrees -- could my eye have swept past the gold without me noticing?\n"
    "2. Am I possibly facing it now? Describe where the red eye points and "
    "where the gold is in the CURRENT image, not from memory.\n"
    "3. Am I still turning in the right direction? Would reversing direction, "
    "or simply going FORWARD, get my eye onto the gold faster?\n\n"
    "Do NOT emit any move token ([CLOCK]/[ANTICLOCK]/[FORWARD]) in this reply "
    "-- it would not be executed. End with the single move you intend to make "
    "next and why; you will act on it when the next screen is shown."
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

# How many of the most-recent move tokens to fold into the memory-retrieval
# query. Move tokens are the agent's own outputs (not privileged Settings), so
# this keeps the mode-1 "no coordinates" invariant intact.
_RECENT_ACTIONS_FOR_QUERY = 5


def _retrieval_query(
    question: str,
    step: int,
    recent_actions: list[str] | None,
    gold_remaining: int,
) -> str:
    """Build a *situational* memory-retrieval query for the current step.

    NAMS recall is pure vector similarity, so querying with the static user
    instruction returns roughly the same thing every step and rarely surfaces
    the reasoning traces / tips relevant to the move at hand. We enrich the query
    with the current decision context -- recent moves and gold progress -- to
    bias similarity toward "what should I do *now*". This drives the whole
    ``get_context`` fan-out (messages, entities, preferences, and the
    ``get_similar_traces`` reasoning recall). Exact coordinates are deliberately
    excluded to preserve the mode-1 privacy invariant.
    """
    parts: list[str] = []
    if question and question.strip():
        parts.append(question.strip())
    status = "gold collected" if gold_remaining == 0 else "gold not yet collected"
    if step == 0:
        # Ambiguous at step 0 (could be a question or the first move); stay close
        # to the instruction and just note progress.
        parts.append(status)
    else:
        # step > 0 is only reached after a move was made -> definitely gameplay.
        if recent_actions:
            parts.append("recent moves: " + ", ".join(recent_actions))
        parts.append(
            f"step {step}; {status}; choosing the next move to reach the gold"
        )
    return " | ".join(parts) if parts else question


def _recent_actions(turns: list[dict], limit: int = _RECENT_ACTIONS_FOR_QUERY) -> list[str]:
    """The last ``limit`` non-empty actions from a list of step/turn result dicts."""
    return [t["action"] for t in turns if t.get("action")][-limit:]


def _build_game_messages(
    system: str,
    image_path: str,
    context: str,
    question: str,
    reflection: str | None = None,
) -> list[dict]:
    user_text = []
    if context:
        user_text.append({"type": "text", "text": f"Memory context:\n{context}"})
    if reflection:
        user_text.append(
            {
                "type": "text",
                "text": (
                    "Your latest self-reflection (you wrote this after pausing "
                    "to re-examine the board; trust it over older, repetitive "
                    "memories):\n" + reflection
                ),
            }
        )
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


# ------------------------------------------------------------- reflection
#
# Generative-agents style reflection (arXiv:2304.03442): every applied move is
# worth ``cfg.reflection_points_per_move`` importance points, and when the
# running total reaches ``cfg.reflection_threshold`` the agent pauses to
# reflect instead of blindly continuing. With the defaults (5 points/move,
# threshold 150) a stuck agent reflects every 30 moves -- exactly one 180-degree
# sweep of rotations -- forcing it to ask whether it rotated past the gold.

def _summarize_actions(actions: list[str]) -> str:
    """Run-length compress a move list: CLOCK x30 reads better than 30 lines."""
    if not actions:
        return "(no moves yet)"
    runs: list[str] = []
    current, count = actions[0], 0
    for a in actions:
        if a == current:
            count += 1
        else:
            runs.append(f"{current} x{count}")
            current, count = a, 1
    runs.append(f"{current} x{count}")
    return ", ".join(runs)


def build_reflection_messages(
    image_path: str, question: str, actions: list[str]
) -> list[dict]:
    """Prompt for one reflection pause: the CURRENT frame + a compressed record
    of every move made this turn, under the reflection system prompt. Memory
    context is deliberately omitted -- the whole point is to break the loop of
    re-reading (and re-copying) stale observations."""
    n = len(actions)
    user_text = (
        f"Original instruction: {question}\n\n"
        f"You have made {n} move(s) this turn without collecting the gold: "
        f"{_summarize_actions(actions)}.\n\n"
        "This is the CURRENT screen. Reflect now: answer the three questions "
        "from your instructions based on what you actually see in this image."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_REFLECT}]},
        {"role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": user_text},
        ]},
    ]


async def persist_reflection(
    client: Any, session_id: str, trace: Any, text: str
) -> None:
    """Store a reflection in memory: as an assistant message (so it enters the
    recency window and semantic search like any other memory, per the paper)
    and as a reasoning step on the turn trace (action=None -- no move made)."""
    content = f"(reflection) {text}"
    await mem.add_assistant_message(client, session_id, content, kind="reflection")
    await mem.add_reasoning_step(client, trace, thought=content, action=None)


async def _record_step(
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
    *,
    include_user_message: bool = True,
) -> dict[str, Any]:
    """Persist the *message + snapshot* side of one mode-1 step to NAMS:
    (optional) user message + assistant message + before/after GameSnapshot
    nodes. The reasoning trace is NOT handled here -- it spans the whole turn
    and is managed by the caller (one trace per turn; see
    :func:`agent.memory.start_turn_trace`).

    In the multi-move loop a single user instruction drives many agent moves, so
    only the first step records the user message (``include_user_message=True``).
    """
    # 1. User message linked to the 'before' snapshot (first step only).
    user_msg = None
    if include_user_message:
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

    # On continuation steps there is no fresh user turn, so attach the 'before'
    # frame (the exact image the model acted on) to the assistant move instead.
    if not include_user_message:
        await image_store.link_snapshot_to_message(
            client, assistant_msg.id, snapshot_before_id, role="before"
        )

    # 3. 'After' snapshot (only if a move was actually applied).
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
        "user_msg_id": str(user_msg.id) if user_msg else None,
        "assistant_msg_id": str(assistant_msg.id),
        "snapshot_before_id": snapshot_before_id,
        "snapshot_before_path": snapshot_before_path,
        "snapshot_after_id": snapshot_after_id,
        "snapshot_after_path": snapshot_after_path,
        "action": action,
        "gold_collected": gold_collected,
    }


def _turn_trace_outcome(
    turns: list[dict[str, Any]], gold_remaining: int
) -> tuple[str, bool]:
    """Summarise a finished turn into an ``(outcome, success)`` pair for the
    trace. Success is the turn-level objective: for a turn that made moves it
    means the gold was collected; a pure Q&A turn (no moves) is a success by
    virtue of having answered."""
    made_moves = any(t.get("action") for t in turns)
    solved = gold_remaining == 0
    n = len(turns)
    if made_moves:
        actions = [t["action"] for t in turns if t.get("action")]
        outcome = (
            f"turn ended after {n} step(s); actions={actions}; "
            f"solved={solved}; gold_remaining={gold_remaining}"
        )
        return outcome, solved
    return f"answered without moving after {n} step(s)", True


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
    # Reflection bookkeeping (see the "reflection" section above): points accrue
    # per applied move; at the threshold the agent reflects and the total resets.
    reflection_points = 0
    last_reflection: str | None = None
    # One reasoning trace for the whole turn; opened after the first step's user
    # message exists (so it can be triggered_by that message) and completed once
    # the loop ends.
    trace: Any = None
    # Capture the initial 'before' snapshot for the first turn.
    snapshot_before_id = image_store.snapshot_id()
    settings_before = game_io.game_to_settings_dict(game)
    snapshot_before_path, _ = await image_store.store_snapshot(
        client, session_id, snapshot_before_id, game, settings_before, cfg=cfg,
        label="before",
        extra={"step": steps},
    )

    try:
        while True:
            # Recompute context with the current image each step: recency window
            # of the latest messages + a general semantic search across tiers,
            # queried with the *situational* state (recent moves + gold progress)
            # rather than the static instruction, so trace/tip recall is relevant
            # to the move at hand.
            query = _retrieval_query(
                question, steps, _recent_actions(turns), game_io.gold_remaining(game)
            )
            ctx = await mem.get_game_context(
                client, session_id, query=query,
                recent_window=cfg.recent_messages_window,
            )
            messages = _build_game_messages(
                SYSTEM_PROMPT_GAME, snapshot_before_path, ctx, question,
                reflection=last_reflection,
            )
            raw = model.generate(messages, stop_strings=game_io.MOVE_STOP_STRINGS)
            action = game_io.parse_action(raw)

            if action:
                gold_collected = game_io.apply_action(game, action)
                reflection_points += cfg.reflection_points_per_move
            else:
                gold_collected = 0

            turn = await _record_step(
                client, session_id, cfg, game, question, raw, action,
                gold_collected, snapshot_before_id, snapshot_before_path,
                include_user_message=(steps == 0),
            )
            turns.append(turn)

            # Open the turn trace once the first user message exists; then record
            # this generation (and every later one) as a step within it.
            if steps == 0:
                trace = await mem.start_turn_trace(
                    client, session_id, task=question,
                    triggered_by_message_id=turn["user_msg_id"],
                )
            await mem.add_reasoning_step(
                client, trace, thought=raw, action=action, gold_collected=gold_collected
            )

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

            # Reflection pause: enough importance points have accrued (default:
            # 30 moves, i.e. up to a 180-degree sweep of rotations). Generate a
            # reflection on the freshly rendered frame (no move this step),
            # persist it, and feed it into every subsequent prompt.
            if reflection_points >= cfg.reflection_threshold:
                actions_so_far = [t["action"] for t in turns if t.get("action")]
                refl_messages = build_reflection_messages(
                    snapshot_before_path, question, actions_so_far
                )
                last_reflection = model.generate(
                    refl_messages, max_new_tokens=cfg.gemma_max_new_tokens
                ).strip()
                await persist_reflection(client, session_id, trace, last_reflection)
                reflection_points = 0
                logger.info("step %d: reflection pause: %s", steps, last_reflection)
    finally:
        gold_remaining = game_io.gold_remaining(game)
        outcome, success = _turn_trace_outcome(turns, gold_remaining)
        await mem.complete_turn_trace(client, trace, outcome=outcome, success=success)

    return {"session_id": session_id, "turns": turns, "steps": steps,
            "gold_remaining": game_io.gold_remaining(game),
            "trace_id": str(trace.id) if trace else None, "success": success}


# --------------------------------------------------------------- mode 2

async def mode_discuss(
    client: Any,
    session_id: str,
    user_text: str,
    cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    """Mode 2: open-ended discussion with full memory access.

    Like mode 1 it combines a **recency** window of the latest messages (so the
    chat has reliable turn-to-turn continuity, which pure similarity search does
    not guarantee) with the general **semantic** search across all memory tiers.
    Unlike mode 1 there is NO settings stripping -- this mode is the
    bootstrap/evaluation channel and is allowed to see everything NAMS returns.
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)

    # Retrieve BEFORE storing the current message: the recency window should show
    # the PRIOR turns, and the semantic search should not just echo back the
    # message we are about to answer.
    recent = await mem.get_recent_messages(client, session_id, cfg.recent_messages_window)
    ctx = await mem.retrieve_context(client, query=user_text, session_id=session_id)
    ctx_text = ctx if isinstance(ctx, str) else json.dumps(ctx, default=str, indent=2)
    # Same channel separation as mode 1: our recency window is the one source
    # of recent messages; drop NAMS' built-in "Recent Conversation" section
    # (which is ordered ASC and actually contains the OLDEST messages anyway).
    ctx_text = mem.strip_nams_recent_conversation(ctx_text)

    context_parts: list[str] = []
    if recent:
        context_parts.append("Recent conversation (most recent last):\n" + recent)
    if ctx_text and ctx_text.strip() not in ("", "{}", "[]"):
        context_parts.append(
            "Relevant memories (semantic search across all tiers):\n" + ctx_text
        )
    context_block = "\n\n".join(context_parts) if context_parts else "(no prior context)"

    # Now record the user message (after retrieval, so it isn't self-retrieved).
    await client.short_term.add_message(
        session_id=session_id, role="user", content=user_text,
        metadata={"kind": "discussion"},
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DISCUSS}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"Memory context:\n{context_block}"},
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
    # Sanity check: every recorded turn opens a reasoning trace, so a session
    # that has assistant messages but zero traces means trace retrieval (or
    # recording) is broken -- warn loudly rather than silently evaluating a
    # session dump with its "# Reasoning traces" section missing.
    has_assistant_msgs = any(
        (row.get("message") or {}).get("role") == "assistant"
        for row in messages_with_snaps
    )
    if has_assistant_msgs and not traces:
        logger.warning(
            "Session %s has assistant messages but no reasoning traces were "
            "retrieved; the self-evaluation will run without traces. Trace "
            "recording or retrieval is likely broken.",
            session_id,
        )
    dump = _format_session_for_eval(messages_with_snaps, traces)
    # The full long-term semantic model = the intended rules/strategy. Give it to
    # the evaluator as the rubric to judge the recorded play against.
    semantic_model = await mem.get_semantic_model(client)

    sections: list[str] = [f"Session id: {session_id}"]
    if semantic_model:
        sections.append(
            "# Game semantic model (intended rules & strategy -- the standard to "
            "judge the play against)\n" + semantic_model
        )
    sections.append(dump)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_EVAL}]},
        {"role": "user", "content": [
            {"type": "text", "text": "\n\n".join(sections)},
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
    """Retrieve the session's reasoning traces.

    The exact Cypher match on ``(:ReasoningTrace {session_id})`` is the ONE
    primary path: it either returns exactly this session's traces or fails
    loudly. ``client.reasoning.search_traces`` is deliberately NOT tried first
    -- it is a *semantic similarity* search whose first argument is a free-text
    query, so passing a session id returns whatever traces happen to embed
    nearest to a hex string, possibly from other sessions. A non-empty-but-
    wrong result from it would mask a broken primary query ("sometimes works"
    fuzziness); we only fall back to it if the exact query itself errors, and
    we log a WARNING so the degradation is visible, never silent.
    """
    traces: list[dict[str, Any]] = []
    try:
        rows = await client.query.cypher(
            # NAMS labels traces ``ReasoningTrace`` (with ``started_at``), not
            # ``Trace``/``created_at`` -- an old query using those always
            # returned nothing.
            "MATCH (t:ReasoningTrace {session_id: $sid}) "
            "RETURN t ORDER BY t.started_at ASC",
            {"sid": session_id},
        )
        traces = [dict(r["t"]) for r in rows if "t" in r]
    except Exception as exc:
        logger.warning(
            "Exact trace query failed (%s); falling back to semantic "
            "search_traces -- results may include other sessions' traces.",
            exc,
        )
        try:
            found = await client.reasoning.search_traces(session_id, limit=100)  # type: ignore[arg-type]
            traces = [dict(t) for t in (found or [])]
        except Exception as exc2:  # pragma: no cover
            logger.warning("search_traces fallback also failed: %s", exc2)
    run_logging.log_db_retrieval(
        function="_fetch_session_traces",
        arguments={"session_id": session_id},
        result=traces,
    )
    return traces


# --------------------------------------------------------------- mode 4
#
# Privileged interactive debrief: a chat over a recorded play conversation with
# full access to ground truth (snapshot images + exact Settings JSON), plus a
# [SHOW <n>] tool to pull up any recorded step's frames. See
# :class:`agent.debrief.DebriefSession` for the session/loop machinery.

SYSTEM_PROMPT_DEBRIEF = (
    "You are a privileged game analyst reviewing a recorded session of a 2D "
    "discrete game. The board is the unit square [0,1]x[0,1] framed by four "
    "boundary walls; the agent is the green circle with a red eye showing its "
    "facing direction; the gold is a small yellow circle. During play the "
    "agent saw ONLY the screen image. You, the analyst, additionally see the "
    "exact game Settings at each recorded step (coordinates, angles, walls), "
    "and you discuss the session openly with the user. Be precise and "
    "quantitative; compute angles and distances from the settings rather than "
    "eyeballing the image whenever settings are available.\n\n"
    "GEOMETRY (verified against the renderer -- trust this over intuition):\n"
    "  - 'direction' is theta in radians, kept in [0, 2*pi). The board's y "
    "axis points DOWNWARD on screen.\n"
    "  - The eye points along (cos theta, sin theta): theta=0 faces right, "
    "pi/2 faces down-screen, pi faces left, 3*pi/2 faces up-screen. "
    "Increasing theta looks CLOCKWISE in the image.\n"
    "  - The [CLOCK] move INCREASES theta by pi/30 (6 degrees) and looks "
    "clockwise; [ANTICLOCK] decreases theta.\n"
    "  - The agent faces the gold when theta ~= atan2(gold_y - agent_y, "
    "gold_x - agent_x) mod 2*pi. NO sign flip: both angles live in the same "
    "y-down convention.\n"
    "  - One [FORWARD] advances up to 1/16 of the board along "
    "(cos theta, sin theta).\n\n"
    "TOOL: you may inspect any recorded step. To do so, end your reply with "
    "the token [SHOW n] where n is the step number from the move list (e.g. "
    "[SHOW 42]). That step's screen image(s) and exact settings will be shown "
    "to you, and you will be asked to continue. Emit at most one [SHOW n] per "
    "reply, at the very end, with nothing after it. When you have what you "
    "need, reply normally without a [SHOW n] token to finish your answer.\n\n"
    "You may also be asked to produce a final structured verdict on the play "
    "(overall_score 0-10, strengths, weaknesses, per-move notes); do so only "
    "when asked."
)

# One lenient matcher shared by the generation-time stop criteria and the
# parser, so stopping and parsing can never disagree. The prompt teaches only
# the canonical [SHOW 42], but the model is a small LLM and will occasionally
# mangle the call ([SHOW(42)], [SHOW: 42], [42 SHOW], ...). We accept any
# SINGLE bracket pair containing SHOW and a number, in either order, with
# arbitrary junk in between; the first number inside the brackets is the step.
# [^\[\]] confines the match to one bracket pair, so stray brackets or prose
# can neither trigger a stop nor corrupt a parse; an incomplete mangle (no
# closing bracket / missing SHOW / missing number) matches nothing and the
# reply simply ends the turn. Case-insensitivity is inline ((?i)) so the SAME
# pattern string drives both this module's parser and the generation-time
# RegexStopCriteria.
SHOW_CALL_PATTERN = r"(?i)\[(?=[^\[\]]*?SHOW)[^\[\]]*?(\d+)[^\[\]]*?\]"
SHOW_CALL_RE = re.compile(SHOW_CALL_PATTERN)


def parse_show_call(text: str) -> tuple[int | None, str]:
    """Return ``(step, text)`` for the FIRST [SHOW n] tool call in ``text``,
    truncating the text right after the call (anything beyond it is
    model-hallucinated tool output -- discard it). ``(None, text)`` if no
    complete call is present."""
    m = SHOW_CALL_RE.search(text)
    if not m:
        return None, text
    return int(m.group(1)), text[: m.end()]


def build_debrief_messages(
    move_listing: str,
    recent: str,
    frames: list[dict[str, Any]],
    question: str,
) -> list[dict]:
    """Assemble one debrief generation's prompt.

    ``move_listing``: indexed move list of the analyzed play session (defines
    the step numbers [SHOW n] refers to). ``recent``: recency window of the
    DEBRIEF conversation (unscrubbed -- this mode is privileged). ``frames``:
    dicts with ``path`` (image file), ``caption`` and ``settings_json``, in
    display order. ``question``: the user text or continuation nudge.
    """
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Recorded moves of the play session under analysis "
                    "(step numbers for [SHOW n]):\n" + move_listing,
        }
    ]
    if recent:
        user_content.append(
            {
                "type": "text",
                "text": "Debrief conversation so far (most recent last):\n" + recent,
            }
        )
    for f in frames:
        user_content.append({"type": "text", "text": f["caption"]})
        user_content.append({"type": "image", "url": f["path"]})
        if f.get("settings_json"):
            user_content.append(
                {"type": "text", "text": "Exact settings for this frame:\n" + f["settings_json"]}
            )
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DEBRIEF}]},
        {"role": "user", "content": user_content},
    ]


async def persist_debrief_verdict(
    client: Any,
    play_session_id: str,
    debrief_session_id: str,
    model: Any,
    cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    """Distill the debrief conversation into a mode-3-format structured verdict
    and store it on the ANALYZED play conversation: assistant message with
    ``kind='self_evaluation'`` plus a reasoning trace -- the exact convention
    :func:`mode_self_eval` uses, so downstream tooling treats both identically.
    """
    cfg = cfg or CONFIG
    # The whole debrief conversation (large window = effectively all of it);
    # unscrubbed -- the verdict is privileged, like mode 3.
    transcript = await mem.get_recent_messages(
        client, debrief_session_id, window=10000, scrub=False
    )
    if not transcript:
        raise ValueError(
            f"Debrief conversation {debrief_session_id!r} is empty; "
            "nothing to distill into a verdict."
        )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_EVAL}]},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": (
                    f"Session id: {play_session_id}\n\n"
                    "The following is a privileged debrief conversation about "
                    "the recorded session (the analyst saw the screens and the "
                    "exact settings). Distill it into your structured verdict "
                    "on the PLAY -- overall_score (0-10), strengths, "
                    "weaknesses, and per-move notes where relevant.\n\n"
                    "# Debrief conversation\n" + transcript
                ),
            },
        ]},
    ]
    verdict = model.generate(messages, max_new_tokens=cfg.gemma_max_new_tokens)

    eval_msg = await client.short_term.add_message(
        session_id=play_session_id, role="assistant", content=verdict,
        metadata={
            "kind": "self_evaluation",
            "evaluated_session": play_session_id,
            "debrief_session": debrief_session_id,
        },
    )
    trace = await client.reasoning.start_trace(
        play_session_id,
        task=f"debrief-distilled self-evaluation of session {play_session_id}",
        triggered_by_message_id=eval_msg.id,
    )
    step = await client.reasoning.add_step(trace.id, thought=verdict)
    await client.reasoning.record_tool_call(
        step.id,
        "save_self_eval",
        {"play_session_id": play_session_id, "debrief_session_id": debrief_session_id},
        result={"verdict_length": len(verdict)},
    )
    await client.reasoning.complete_trace(
        trace.id, outcome="debrief verdict appended to play conversation", success=True,
    )
    return {
        "play_session_id": play_session_id,
        "debrief_session_id": debrief_session_id,
        "verdict": verdict,
        "eval_message_id": str(eval_msg.id),
        "trace_id": str(trace.id),
    }
