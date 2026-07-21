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


# ------------------------------------------------------------ prompt blocks
#
# System prompts are COMPOSED from the named blocks below (see
# .cursor/rules/prompt-composition.mdc): a concept shared by two or more modes
# lives in exactly ONE block, and each SYSTEM_PROMPT_* is a "\n\n".join of
# blocks plus at most a short mode-specific paragraph. Fixing a fact in its
# block fixes every mode that uses it.

_SENT_GAME_INTRO = "You are an agent playing a 2D discrete game on a square board."
_SENT_GAME_SCREEN = "You see the current game screen as an image."
_SENT_GAME_WORLD = (
    "The board is the unit square framed by four boundary walls; there is "
    "exactly one gold piece (small yellow circle). You are the green circle "
    "with a red eye showing the direction you are facing."
)

_BLOCK_GAME_INTRO = " ".join([
    _SENT_GAME_INTRO,
    _SENT_GAME_SCREEN,
    _SENT_GAME_WORLD,
    "Your goal is to collect the gold.",
])

_BLOCK_MOVE_TOKENS = (
    "You make moves by emitting exactly one of these move tokens:\n"
    "  [CLOCK]     - turn clockwise by pi/30 (rotate in place)\n"
    "  [ANTICLOCK] - turn counter-clockwise by pi/30 (rotate in place)\n"
    "  [FORWARD]   - advance up to 1/16 of the board in the facing direction\n"
    "Only [FORWARD] moves you; [CLOCK] and [ANTICLOCK] only rotate you.\n\n"
    "IMPORTANT: ONLY those exact bracketed tokens trigger a move. Talking about "
    "moving in prose (e.g. writing the word 'forward') does NOTHING. You may "
    "reason in plain prose as much as you like; nothing happens until you emit "
    "a bracketed move token, so emit one only when you truly intend that move."
)

_BLOCK_MULTI_MOVE_TURN = (
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
    "unless it is very apparent that the user is asking you to play."
)

_BLOCK_HOW_TO_PLAY = (
    "HOW TO PLAY -- take these steps before every move. START your reply with "
    "one structured observation line, read fresh off the CURRENT screen, in "
    "exactly this form:\n"
    "  OBS: I am at <where on the board>; my eye points toward <clock "
    "direction, e.g. 4 o'clock>; the gold is at <where on the board>, toward "
    "<clock direction> of me.\n"
    "(Clock directions are as seen on screen: 12 o'clock is up-screen, 3 is "
    "right, 6 is down, 9 is left.) Then REASON, in a sentence or two, about "
    "where the gold is relative to your red eye: the eye points in the "
    "direction you face, and [FORWARD] sends you straight that way. Then "
    "choose your move from that reasoning:\n"
    "  - If your eye is pointing roughly at the gold, emit [FORWARD].\n"
    "  - Otherwise, aim your eye at the gold: emit [CLOCK] or [ANTICLOCK], then "
    "check the re-rendered screen to see which way your eye swung, and keep "
    "rotating that way (or reverse if you overshoot) until your eye lines up with "
    "the gold -- then emit [FORWARD].\n"
    "Always state the observation and reasoning before the move token; looking "
    "first, then reasoning, then a single move, is how you decide well."
)

_BLOCK_CURRENT_SCREEN = (
    "DO NOT just copy prior observations from your memories. Make sure you "
    "evaluate whether you are facing the gold *right now*. Your memories "
    "describe PAST screens; every move changes the screen, so re-derive where "
    "your red eye points and where the gold is from the CURRENT image before "
    "every single move."
)

# The one statement of the aim-tolerance rule. Shown verbatim to the player
# (all player-prompt variants) and quoted verbatim to every reviewer via
# _BLOCK_AIM_TOLERANCE_REVIEW, so player and judge always share one wording.
_BLOCK_AIM_TOLERANCE = (
    "AIM TOLERANCE: it can be hard to tell your exact facing direction from "
    "the screen, so do not demand pixel-perfect aim. If your best estimate "
    "is that the gold lies within about 45 degrees of your facing direction, "
    "[FORWARD] is a good move -- step forward and re-assess on the new "
    "screen. Reserve [CLOCK]/[ANTICLOCK] fine-tuning for when the gold is "
    "clearly off to one side or behind you."
)

_BLOCK_AIM_TOLERANCE_REVIEW = (
    "The player's instructions included this aim-tolerance rule, quoted "
    "verbatim:\n\"" + _BLOCK_AIM_TOLERANCE + "\"\n"
    "Judge the play against it: a [FORWARD] emitted while the gold was "
    "within roughly 45 degrees of the facing direction follows instructions "
    "and must not be penalized as imprecise aim; conversely, long "
    "rotate-only fine-tuning inside that tolerance goes against them."
)


def _search_tool_block(scope_note: str) -> str:
    """The [SEARCH <query>] tool description, shared by every mode that gets
    the tool; ``scope_note`` is the one sentence that varies by mode (what the
    search covers)."""
    return (
        "MEMORY SEARCH: you can search your memory on your own. To do so, end "
        "your reply with exactly one search token:\n"
        "  [SEARCH <query>] - e.g. [SEARCH tips about facing the gold]\n"
        + scope_note + " "
        "The results will be placed in your context and you will be asked to "
        "continue your reply. Emit at most one search token per reply, at the "
        "very end, with nothing after it. Searching is never a substitute for "
        "looking at the current information in front of you."
    )


_SEARCH_SCOPE_PLAY = (
    "The search covers your long-term knowledge of the game (entities and "
    "saved tips) and your past reasoning."
)
_SEARCH_SCOPE_FULL = (
    "The search covers your long-term knowledge of the game (entities and "
    "saved tips), your past reasoning, and past conversation messages."
)

_BLOCK_TIP_TOOL = (
    "SAVING TIPS: you can permanently save a one-line tip to long-term memory "
    "-- it becomes retrievable in every future mode, including live play. To "
    "save one, put the exact tip on its own line in the form 'TIP: <one line>' "
    "and end your reply with the token [WRITE_TIP].\n"
    "The tool itself is tightly controlled. You MAY suggest a tip on your own "
    "initiative if you feel strongly that a lesson is worth keeping -- but a "
    "suggestion is just prose: propose the EXACT one-line wording and ask the "
    "user whether to save it. The final decision always belongs to the user, "
    "who may accept, reject, or reword it. Only after the user has approved "
    "that exact wording in this conversation do you emit the TIP: line "
    "followed by [WRITE_TIP]. Never emit [WRITE_TIP] without such an explicit "
    "approval; if the wording changed after the approval, ask again before "
    "saving."
)


# --------------------------------------------------------- composed prompts

SYSTEM_PROMPT_GAME = "\n\n".join([
    _BLOCK_GAME_INTRO,
    _BLOCK_MOVE_TOKENS,
    _BLOCK_MULTI_MOVE_TURN,
    _BLOCK_HOW_TO_PLAY,
    _BLOCK_AIM_TOLERANCE,
    _BLOCK_CURRENT_SCREEN,
    _search_tool_block(_SEARCH_SCOPE_PLAY),
])

SYSTEM_PROMPT_REFLECT = "\n\n".join([
    " ".join([_SENT_GAME_INTRO, _SENT_GAME_WORLD]),
    (
        "This is NOT a move request. This is a REFLECTION pause: you have made many "
        "moves without collecting the gold, so you must stop, look at the CURRENT "
        "screen with fresh eyes, and re-examine your plan before continuing. Your "
        "recent reasoning may have been repeating itself without checking the "
        "screen; assume nothing you previously said is still true."
    ),
    (
        "Study the CURRENT image and answer, concretely and honestly:\n"
        "1. Am I *certain* that I was never facing the gold at any point during my "
        "recent moves? Each rotation is 6 degrees, so 30 rotations sweep 180 "
        "degrees -- could my eye have swept past the gold without me noticing?\n"
        "2. Am I possibly facing it now? Describe where the red eye points and "
        "where the gold is in the CURRENT image, not from memory.\n"
        "3. Am I still turning in the right direction? Would reversing direction, "
        "or simply going FORWARD, get my eye onto the gold faster?"
    ),
    _BLOCK_AIM_TOLERANCE,
    (
        "Do NOT emit any move token ([CLOCK]/[ANTICLOCK]/[FORWARD]) in this reply "
        "-- it would not be executed. End with the single move you intend to make "
        "next and why; you will act on it when the next screen is shown."
    ),
])

SYSTEM_PROMPT_DISCUSS = "\n\n".join([
    (
        "You are an agent that plays a 2D discrete game and is also able to "
        "discuss it openly with the user. You have full access to your memory "
        "database (conversations, the game's semantic model, and your past "
        "reasoning traces). Be concise and helpful. You are not seeing a live "
        "game screen in this mode; rely on memory and the user's description."
    ),
    _search_tool_block(_SEARCH_SCOPE_FULL),
])

SYSTEM_PROMPT_EVAL = "\n\n".join([
    (
        "You are evaluating how well an earlier instance of yourself played a "
        "2D discrete game. You are given the full Conversation (user questions "
        "and assistant moves), the reasoning traces recorded for each move, "
        "and the underlying game Settings at each step (which the player did "
        "NOT have access to at the time). Be specific and critical. Output a "
        "structured verdict with: overall_score (0-10), strengths, weaknesses, "
        "and per-move notes where relevant."
    ),
    _BLOCK_AIM_TOLERANCE_REVIEW,
])


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
    search_results: str | None = None,
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
    if search_results:
        user_text.append(
            {
                "type": "text",
                "text": (
                    "Results of the memory search(es) you ran while composing "
                    "this reply:\n" + search_results
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
    """Run-length compress a move list into 'CLOCKx61 FORWARDx1 ...' -- the ONE
    canonical way a move trace is presented to the model, in every mode."""
    if not actions:
        return "(no moves yet)"
    runs: list[str] = []
    current, count = actions[0], 0
    for a in actions:
        if a == current:
            count += 1
        else:
            runs.append(f"{current}x{count}")
            current, count = a, 1
    runs.append(f"{current}x{count}")
    return " ".join(runs)


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
            f"turn ended after {n} step(s); "
            f"actions: {_summarize_actions(actions)}; "
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

            # Inner [SEARCH] loop: the model may spend up to
            # cfg.memory_search_max_calls searches before this step's move /
            # answer. Each search's results are fed back and the step is
            # regenerated; searches are recorded on the turn trace below
            # (after the trace exists).
            search_notes: list[str] = []
            searches: list[dict[str, str]] = []
            while True:
                messages = _build_game_messages(
                    SYSTEM_PROMPT_GAME, snapshot_before_path, ctx, question,
                    reflection=last_reflection,
                    search_results="\n\n".join(search_notes) or None,
                )
                over_budget = len(searches) >= cfg.memory_search_max_calls
                raw = model.generate(
                    messages,
                    stop_strings=game_io.MOVE_STOP_STRINGS,
                    # Once the budget is spent the stop pattern is dropped, so
                    # a stray [SEARCH] is inert prose instead of a stall.
                    stop_regex=None if over_budget else SEARCH_TOOL_PATTERN,
                )
                kind, payload, text = classify_move_or_search(raw)
                if kind != "search" or over_budget:
                    break
                results = await mem.search_memory(
                    client, payload, tiers=("semantic", "reasoning"),
                    top_k=cfg.memory_search_top_k, scrub=True,
                )
                search_notes.append(format_search_note(payload, results))
                searches.append({"query": payload, "results": results, "thought": text})
                if len(searches) >= cfg.memory_search_max_calls:
                    search_notes.append(SEARCH_BUDGET_NOTE)
                logger.info("step %d: [SEARCH %s]", steps, payload)
            action = game_io.parse_action(raw) if kind == "move" else None

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
            for s in searches:
                await record_search_tool_call(
                    client, trace, s["thought"], s["query"], s["results"]
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

    # [SEARCH] loop: this privileged mode searches ALL tiers, unscrubbed.
    search_notes: list[str] = []
    n_searches = 0
    while True:
        content: list[dict[str, str]] = [
            {"type": "text", "text": f"Memory context:\n{context_block}"},
        ]
        if search_notes:
            content.append({
                "type": "text",
                "text": (
                    "Results of the memory search(es) you ran while composing "
                    "this reply:\n" + "\n\n".join(search_notes)
                ),
            })
        content.append({"type": "text", "text": f"User: {user_text}"})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DISCUSS}]},
            {"role": "user", "content": content},
        ]
        over_budget = n_searches >= cfg.memory_search_max_calls
        reply = model.generate(
            messages,
            stop_regex=None if over_budget else SEARCH_TOOL_PATTERN,
        )
        query, _ = parse_search_call(reply)
        if query is None or over_budget:
            break
        results = await mem.search_memory(
            client, query, tiers=mem.SEARCH_TIERS,
            top_k=cfg.memory_search_top_k, scrub=False,
        )
        search_notes.append(format_search_note(query, results))
        n_searches += 1
        if n_searches >= cfg.memory_search_max_calls:
            search_notes.append(SEARCH_BUDGET_NOTE)
        logger.info("discuss: [SEARCH %s]", query)

    await client.short_term.add_message(
        session_id=session_id, role="assistant", content=reply,
        metadata={"kind": "discussion"},
    )
    return {"session_id": session_id, "reply": reply, "searches": n_searches}


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
# full access to ground truth (snapshot images + exact Settings JSON), plus
# [SHOW <n>] / [NEXT] / [BACK] tools that move a cursor over the player's
# recorded messages (each fetch = that message's text + the ONE frame the
# player saw + its settings). See :class:`agent.debrief.DebriefSession` for
# the session/loop machinery.

_BLOCK_REVIEWER_STANCE = (
    "The session was played by an earlier instance of YOU; you have NO memory "
    "of playing it. Everything you know about it comes from the recorded "
    "materials in your context and from the tools below. The user may say "
    "'you' meaning the player -- do not get defensive, and NEVER claim you "
    "lack access to the play: retrieve the recorded material with a tool "
    "instead."
)

_BLOCK_PRIVILEGED_VIEW = (
    "The board is the unit square [0,1]x[0,1] framed by four boundary walls; "
    "the agent is the green circle with a red eye showing its facing "
    "direction; the gold is a small yellow circle. During play the player "
    "saw ONLY the screen image. You, the reviewer, additionally see the "
    "exact game Settings for each recorded message (coordinates, angles, "
    "walls). Be precise and quantitative: compute angles and distances from "
    "the settings rather than eyeballing the image. NEVER state settings "
    "values you have not actually seen for that exact message -- fetch the "
    "message first."
)

_BLOCK_GEOMETRY_PRIVILEGED = (
    "GEOMETRY (verified against the renderer -- trust this over intuition):\n"
    "  - Coordinates are in [0, 1]. The x axis points RIGHT and the y axis "
    "points UP on screen: larger y = higher in the image, exactly as in "
    "ordinary graphs.\n"
    "  - 'direction' is theta in radians, kept in [0, 2*pi), measured "
    "CLOCKWISE: increasing theta sweeps the eye the way a clock's hand "
    "sweeps. theta=0 faces right (3 o'clock), pi/2 faces down (6 o'clock), "
    "pi faces left (9 o'clock), 3*pi/2 faces up (12 o'clock).\n"
    "  - ANGLE TO CLOCK DIRECTION: hour = 3 + theta / (pi/6); subtract 12 "
    "if the result exceeds 12. Worked example: theta = 4.55 -> hour = 3 + "
    "8.7 = 11.7 -> between 11 and 12 o'clock, i.e. just shy of straight up, "
    "slightly toward the LEFT (anticlockwise) side. Use this whenever you "
    "translate an angle into words -- do not eyeball it.\n"
    "  - The agent faces a target when theta ~= theta_target = "
    "atan2(y_agent - y_target, x_target - x_agent). NOTE the y-difference "
    "enters NEGATED (agent minus target) while the x-difference is target "
    "minus agent: theta runs clockwise while y points up, so the usual "
    "atan2(dy, dx) would give the wrong sign. The same formula also gives "
    "any point's clock direction from the agent, via the hour rule above.\n"
    "  - ROTATION DIRECTION: to decide which way to rotate, compute diff = "
    "theta_target - theta, then bring it into (-pi, pi] by adding 2*pi if "
    "diff <= -pi, or subtracting 2*pi if diff > pi (one adjustment always "
    "suffices for angles in [0, 2*pi)). Call the result delta. If delta > "
    "0, CLOCKWISE is the shorter rotation; if delta < 0, COUNTER-CLOCKWISE "
    "is. |delta| / (pi/30) is roughly the number of rotation steps needed. "
    "Do NOT use the mod operator here: mod of a negative number is "
    "convention-dependent and can flip your answer -- stick to the "
    "add/subtract-2*pi rule. NEVER compare raw angles without this "
    "normalization: whenever the raw gap |theta_target - theta| exceeds pi, "
    "the OTHER direction is shorter. Worked example: theta = 5.68, "
    "theta_target = 0.75 -> diff = -4.93 -> add 2*pi -> delta = +1.35 -> "
    "clockwise (about 13 steps).\n"
    "  - The [CLOCK] move INCREASES theta by pi/30 (6 degrees) and rotates "
    "clockwise on screen; [ANTICLOCK] decreases theta. One [FORWARD] "
    "advances up to 1/16 of the board along the facing direction."
)

_BLOCK_DEBRIEF_RECORD = (
    "HOW THE RECORD IS ORGANIZED: the player's messages are numbered from 0 "
    "(see the 'Current message' line in your context for the valid range). "
    "Most messages end in a move token ([CLOCK]/[ANTICLOCK]/[FORWARD]); "
    "reflection messages (no move) occur roughly every 30 moves -- to find "
    "one, [SHOW] a multiple of 30 and step with [NEXT]/[BACK] until you hit "
    "it. The user instruction the player was answering is always shown "
    "separately in your context."
)

_BLOCK_DEBRIEF_NAV = (
    "TOOLS: you may inspect any recorded message. To navigate, end your "
    "reply with exactly one of these tokens:\n"
    "  [SHOW n] - jump to recorded message n (e.g. [SHOW 42])\n"
    "  [NEXT]   - advance to the message after the current one\n"
    "  [BACK]   - go to the message before the current one\n"
    "That message's recorded text, the frame the player saw at that moment, "
    "and its exact settings will be placed in your context, and you will be "
    "asked to continue. You may also end your reply with a [SEARCH <query>] "
    "token (described below) instead of a navigation token; its results "
    "include matching recorded messages of THIS session, so searching is the "
    "fast way to find a message worth [SHOW]ing. [SEARCH] matches recorded "
    "message CONTENT semantically; message numbers are not content -- to "
    "open message n, always use [SHOW n], never [SEARCH]. Emit at most one "
    "tool token per reply, at the very end, with nothing after it. When you "
    "have what you need, reply normally without a tool token to finish your "
    "answer."
)

# Shared by every reviewer prompt (scene analyst + debrief): what "analyze
# the player's response" covers. Player replies are multi-part by
# construction (_BLOCK_HOW_TO_PLAY), and reviewers must not fixate on the
# reasoning alone.
_BLOCK_REVIEW_WHOLE_REPLY = (
    "A player reply has multiple parts: the 'OBS:' observation line, the "
    "reasoning prose, and (usually) a move token. When asked to analyze the "
    "player's response, review ALL of these parts, not just the reasoning. "
    "Grade the OBS line against the exact settings separately from the move "
    "choice: perception errors are as important as move errors, and correct "
    "perception with a wrong move, or wrong perception with a lucky move, "
    "should both be called out explicitly."
)

_BLOCK_DEBRIEF_VERDICT = (
    "You may also be asked to produce a final structured verdict on the play "
    "(overall_score 0-10, strengths, weaknesses, per-move notes); do so only "
    "when asked."
)

SYSTEM_PROMPT_DEBRIEF = "\n\n".join([
    "You are reviewing a RECORDED session of a 2D discrete game. "
    + _BLOCK_REVIEWER_STANCE,
    _BLOCK_PRIVILEGED_VIEW,
    _BLOCK_DEBRIEF_RECORD,
    _BLOCK_GEOMETRY_PRIVILEGED,
    _BLOCK_AIM_TOLERANCE_REVIEW,
    _BLOCK_REVIEW_WHOLE_REPLY,
    _BLOCK_DEBRIEF_NAV,
    _search_tool_block(_SEARCH_SCOPE_FULL),
    _BLOCK_TIP_TOOL,
    _BLOCK_DEBRIEF_VERDICT,
])


# ------------------------------------------------- interactive self-eval
#
# Two prompts for the interactive self-eval notebook, both recompositions of
# the blocks above (per .cursor/rules/prompt-composition.mdc): the scene
# player is the play prompt scoped to a single generation over the current
# scene; the scene analyst is the debrief prompt scoped to exactly one
# recorded player message (no [SHOW]/[NEXT]/[BACK] navigation).

_BLOCK_SCENE_SCOPE = (
    "You are responsible for THIS situation only -- the single screen in "
    "front of you, nothing before or after it. For *this situation*, usually "
    "the user will ask for the best move. In that case, reason briefly and "
    "answer with exactly ONE of the available move tokens; your reply ends "
    "there (you will not see the result this turn). The user may instead ask "
    "general questions, like 'are you facing the gold?' or 'is the gold to "
    "your left?' In that case, answer the question in prose with NO move "
    "token."
)

SYSTEM_PROMPT_SCENE_PLAY = "\n\n".join([
    _BLOCK_GAME_INTRO,
    _BLOCK_MOVE_TOKENS,
    _BLOCK_SCENE_SCOPE,
    _BLOCK_HOW_TO_PLAY,
    _BLOCK_AIM_TOLERANCE,
    _BLOCK_CURRENT_SCREEN,
    _search_tool_block(_SEARCH_SCOPE_PLAY),
])

_BLOCK_RATING = (
    "Say explicitly which parts of the player's response were correct and "
    "which were incorrect, and END your final answer with a single overall "
    "rating of the response on the scale -1.0 (completely wrong) to 1.0 "
    "(completely right), on its own line in the form 'RATING: <number>'."
)

_BLOCK_SCENE_ANALYST_SCOPE = (
    "Exactly ONE recorded player message is under review: the player's "
    "latest reply, shown in your context together with the user question it "
    "answered, the frame the player saw, and that frame's exact settings. "
    "There is nothing else to navigate to. If the player's reply ends in a "
    "move token, that move has NOT been applied yet -- you are judging the "
    "decision, not its outcome."
)

SYSTEM_PROMPT_SCENE_ANALYST = "\n\n".join([
    "You are reviewing ONE RECORDED reply from a 2D discrete game. "
    + _BLOCK_REVIEWER_STANCE,
    _BLOCK_PRIVILEGED_VIEW,
    _BLOCK_GEOMETRY_PRIVILEGED,
    _BLOCK_AIM_TOLERANCE_REVIEW,
    _BLOCK_SCENE_ANALYST_SCOPE,
    _BLOCK_REVIEW_WHOLE_REPLY,
    _search_tool_block(_SEARCH_SCOPE_FULL),
    _BLOCK_RATING,
])


def build_scene_analyst_messages(
    player_question: str,
    player_reply: str,
    pending_action: str | None,
    frame_path: str | None,
    settings_json: str | None,
    recent: str,
    question: str,
    search_results: str | None = None,
) -> list[dict]:
    """Assemble one scene-analyst generation's prompt, mirroring
    :func:`build_debrief_messages`' structure (pre-joined text blocks, then
    the ONE frame, then settings + question) but scoped to a single recorded
    player reply instead of a navigable session."""
    scene = [
        "The player was answering this user question: "
        f"\"{player_question}\"",
        "The player's reply under review (this EXACT reply -- the frame the "
        "player saw and its exact settings follow):\n" + player_reply,
    ]
    if pending_action:
        scene.append(
            f"The reply ends in the move token [{pending_action}]. That move "
            "has NOT been applied yet -- judge the decision on the frame "
            "below."
        )
    else:
        bare = game_io.find_bare_move(player_reply)
        if bare:
            scene.append(
                f"FORMAT ERROR (harness-verified): the reply contains the "
                f"bare word '{bare}' WITHOUT brackets. That is NOT a valid "
                f"move token -- only [{bare}] would be -- so NO move will be "
                f"propagated. The player almost certainly intended a move "
                f"and fumbled the format; call this mistake out explicitly "
                f"in your analysis."
            )
        else:
            scene.append("The reply contains no move token (a prose answer).")
    blocks = ["\n\n".join(scene)]
    if recent:
        blocks.append(
            "Conversation so far, including earlier scenes and analyses "
            "(most recent last):\n" + recent
        )
    if search_results:
        blocks.append(
            "Results of the memory search(es) you ran this turn:\n"
            + search_results
        )
    if frame_path:
        blocks.append("The image below is the frame the player saw.")
    else:
        blocks.append("(No frame was saved for this reply.)")

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "\n\n".join(blocks)}
    ]
    tail: list[str] = []
    if frame_path:
        user_content.append({"type": "image", "url": frame_path})
        if settings_json:
            tail.append("Exact settings for this frame:\n" + settings_json)
    tail.append(question)
    user_content.append({"type": "text", "text": "\n\n".join(tail)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SCENE_ANALYST}]},
        {"role": "user", "content": user_content},
    ]

# One lenient matcher shared by the generation-time stop criteria and the
# parser, so stopping and parsing can never disagree. The prompt teaches only
# the canonical forms ([SHOW 42], [NEXT], [BACK], [SEARCH gold tips],
# [WRITE_TIP]), but the model is a small LLM and will occasionally mangle a
# call ([SHOW(42)], [SHOW: 42], [42 SHOW], [ next ], [SEARCH: ...], ...).
# Leniency rules:
#   - SEARCH: a bracket pair that STARTS with the word SEARCH (after \W
#     padding); everything after it up to the closing bracket is the query.
#     Listed FIRST so a query that happens to contain 'show 12' still parses
#     as a search.
#   - SHOW: any SINGLE bracket pair containing SHOW and a number, in either
#     order, with arbitrary junk in between; the first number inside the
#     brackets is the step.
#   - NEXT/BACK/WRITE_TIP: a bracket pair containing the word with only
#     non-alphanumeric padding (\W) allowed -- deliberately TIGHTER than SHOW,
#     because 'next' and 'back' are common English words and bracketed prose
#     like '[the next frame]' must not fire a tool call.
# [^\[\]] / \W confine each match to one bracket pair, so stray brackets or
# prose can neither trigger a stop nor corrupt a parse; an incomplete mangle
# matches nothing and the reply simply ends the turn. Case-insensitivity is
# inline ((?i)) so the SAME pattern string drives both this module's parser
# and the generation-time RegexStopCriteria.

# The SEARCH-call fragment, shared verbatim by the standalone search pattern
# (play / discuss / scene modes) and the full debrief pattern.
_SEARCH_CALL_FRAGMENT = r"\[\W*SEARCH\b[:\s]*(?P<search_query>[^\[\]]+?)\W*\]"

SEARCH_TOOL_PATTERN = r"(?i)" + _SEARCH_CALL_FRAGMENT
SEARCH_TOOL_RE = re.compile(SEARCH_TOOL_PATTERN)

DEBRIEF_TOOL_PATTERN = (
    r"(?i)(?:"
    + _SEARCH_CALL_FRAGMENT +
    r"|\[(?=[^\[\]]*?SHOW)[^\[\]]*?(?P<show_step>\d+)[^\[\]]*?\]"
    r"|\[\W*(?P<nav>NEXT|BACK)\W*\]"
    r"|\[\W*(?P<write_tip>WRITE[\s_\-]*TIP)\W*\]"
    r")"
)
DEBRIEF_TOOL_RE = re.compile(DEBRIEF_TOOL_PATTERN)

# The 'TIP: <one line>' line that must accompany a [WRITE_TIP] call. The LAST
# such line in the reply is the tip (the model may quote earlier proposals).
TIP_LINE_RE = re.compile(r"(?im)^\s*TIP\s*:\s*(?P<tip>.+?)\s*$")

# A reviewer's per-error span: WRONG: "<exact words from the player reply>".
# Quotes optional -- small models drop them -- but the words must still match.
WRONG_SPAN_RE = re.compile(r"(?im)^\s*WRONG\s*:\s*\"?(?P<span>.+?)\"?\s*$")


def parse_wrong_spans(analysis: str, source_text: str) -> dict[str, list[str]]:
    """Extract the reviewer's ``WRONG: "..."`` error spans from ``analysis``
    and verify each against ``source_text`` (the recorded player reply) by
    EXACT substring match.

    Returns ``{"verified": [...], "unverified": [...]}`` (deduplicated, in
    order of first appearance). Unverified spans are returned rather than
    dropped so callers can surface them loudly -- a span the player never
    wrote must never be silently presented as a highlight
    (no-fuzzy-fallbacks)."""
    verified: list[str] = []
    unverified: list[str] = []
    seen: set[str] = set()
    for m in WRONG_SPAN_RE.finditer(analysis):
        span = m.group("span").strip()
        if not span or span in seen:
            continue
        seen.add(span)
        (verified if span in source_text else unverified).append(span)
    return {"verified": verified, "unverified": unverified}


def _clean_search_query(query: str) -> str:
    """Normalize a captured [SEARCH] query: models often quote it (e.g.
    [SEARCH "Message 24"]), and the capture regex's trailing ``\\W*`` eats the
    closing quote but not the opening one -- strip surrounding whitespace and
    quotes symmetrically so the query is never lopsided."""
    return query.strip().strip("\"'").strip()


def parse_search_call(text: str) -> tuple[str | None, str]:
    """Return ``(query, text)`` for the FIRST ``[SEARCH <query>]`` call in
    ``text``, truncating the text right after the call (anything beyond it is
    model-hallucinated tool output -- discard it). ``(None, text)`` if no
    complete call is present."""
    m = SEARCH_TOOL_RE.search(text)
    if not m:
        return None, text
    query = _clean_search_query(m.group("search_query"))
    return (query or None), text[: m.end()]


def classify_move_or_search(raw: str) -> tuple[str, Any, str]:
    """Classify one game-mode generation stopped on either a move token or a
    search token. Returns ``(kind, payload, text)``:

      * ``("search", query, truncated)`` -- the reply's first tool token is a
        [SEARCH]; text is truncated right after it.
      * ``("move", action, raw)``        -- the reply's first tool token is a
        move token.
      * ``("answer", None, raw)``        -- no tool token: a prose answer /
        end of turn.

    Generation stops at the FIRST matching token, so normally only one is
    present; if both somehow are (e.g. the regex stop missed by a token or
    two), the earlier one is the model's actual first decision and wins."""
    search_m = SEARCH_TOOL_RE.search(raw)
    move_m = game_io._MOVE_RE.search(raw)
    if search_m and (move_m is None or search_m.start() < move_m.start()):
        query = _clean_search_query(search_m.group("search_query"))
        if query:
            return "search", query, raw[: search_m.end()]
    if move_m:
        # parse_action takes the LAST move token; consistent here because
        # generation stops at the first one.
        return "move", game_io.parse_action(raw), raw
    return "answer", None, raw


def format_search_note(query: str, results: str) -> str:
    """One formatted entry for the accumulated search-results block that is
    fed back into the prompt after a [SEARCH] call -- the ONE presentation of
    search output, shared by every mode."""
    return f"[SEARCH {query}] returned:\n{results}"


SEARCH_BUDGET_NOTE = (
    "(Your memory-search budget for this reply is exhausted; do NOT search "
    "again. Answer or move now, using the results you already have.)"
)


async def record_search_tool_call(
    client: Any, trace: Any, thought: str, query: str, results: str
) -> None:
    """Record one [SEARCH] call on the turn's reasoning trace: the reply text
    that requested it as the step's thought, the query as the tool arguments."""
    if trace is None:
        return
    step = await client.reasoning.add_step(trace.id, thought=thought)
    await client.reasoning.record_tool_call(
        step.id, "SEARCH", {"query": query}, result={"result_chars": len(results)}
    )


def parse_tip_line(text: str) -> str | None:
    """The tip wording from the last 'TIP: <one line>' line in ``text``, or
    ``None`` if the reply carries no such line."""
    matches = TIP_LINE_RE.findall(text)
    return matches[-1].strip() if matches else None


def parse_debrief_call(text: str) -> tuple[dict[str, Any] | None, str]:
    """Return ``(call, text)`` for the FIRST debrief tool call in ``text``,
    truncating the text right after the call (anything beyond it is
    model-hallucinated tool output -- discard it). ``call`` is one of
    ``{"tool": "SHOW", "step": n}``, ``{"tool": "NEXT"}``,
    ``{"tool": "BACK"}``, ``{"tool": "SEARCH", "query": q}``, or
    ``{"tool": "WRITE_TIP", "tip": t}`` (``t`` is None when the reply lacks
    the required 'TIP:' line); ``(None, text)`` if no complete call is
    present."""
    m = DEBRIEF_TOOL_RE.search(text)
    if not m:
        return None, text
    truncated = text[: m.end()]
    if m.group("search_query") is not None:
        return (
            {"tool": "SEARCH", "query": _clean_search_query(m.group("search_query"))},
            truncated,
        )
    if m.group("show_step") is not None:
        return {"tool": "SHOW", "step": int(m.group("show_step"))}, truncated
    if m.group("write_tip") is not None:
        return {"tool": "WRITE_TIP", "tip": parse_tip_line(truncated)}, truncated
    return {"tool": m.group("nav").upper()}, truncated


def build_debrief_messages(
    trace_block: str,
    recent: str,
    current: dict[str, Any] | None,
    question: str,
    search_results: str | None = None,
) -> list[dict]:
    """Assemble one debrief generation's prompt as at most three content
    parts. Gemma's chat template concatenates adjacent text parts with NO
    separator, so logical blocks are pre-joined with explicit newlines here
    instead of being emitted as separate parts:

      1. text  -- session trace + hanging player instruction + debrief
                  recency window + accumulated search results + the current
                  message's header and text
      2. image -- the ONE frame the player saw at the current message
      3. text  -- that frame's exact settings + the question/continuation

    ``trace_block``: condensed move trace + 'Current message' line.
    ``recent``: recency window of the DEBRIEF conversation (unscrubbed --
    this mode is privileged). ``current``: the message under inspection from
    ``DebriefSession._message_block`` ({header, content, instruction, path,
    settings_json}); None when the session has no player messages.
    ``question``: the user text or continuation nudge. ``search_results``:
    accumulated results of this turn's [SEARCH] calls, if any.
    """
    blocks = ["Recorded play session under analysis:\n" + trace_block]
    if current is not None:
        blocks.append(
            "The player was answering this user instruction: "
            f"\"{current['instruction']}\""
        )
    if recent:
        blocks.append("Debrief conversation so far (most recent last):\n" + recent)
    if search_results:
        blocks.append(
            "Results of the memory search(es) you ran this turn:\n"
            + search_results
        )
    if current is not None:
        cur_text = current["header"] + "\nRecorded message text:\n" + current["content"]
        if current.get("path"):
            cur_text += "\nThe image below is the frame the player saw at this message."
        else:
            cur_text += "\n(No frame was saved for this message.)"
        blocks.append(cur_text)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "\n\n".join(blocks)}
    ]
    tail: list[str] = []
    if current is not None and current.get("path"):
        user_content.append({"type": "image", "url": current["path"]})
        if current.get("settings_json"):
            tail.append(
                "Exact settings for this frame:\n" + current["settings_json"]
            )
    tail.append(question)
    user_content.append({"type": "text", "text": "\n\n".join(tail)})
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
