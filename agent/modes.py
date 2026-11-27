"""The three agent modes:

  * :func:`mode_game`     -- game-playing / question answering (mode 1)
  * :func:`mode_discuss`  -- open-ended discussion with full memory access (mode 2)
  * :func:`mode_self_eval`-- self-evaluation over a Conversation + traces (mode 3)

All modes are async and share one NAMS ``MemoryClient`` and one
``VLModel`` instance. Mode 1 never lets the Settings dict reach the model;
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
# lives in exactly ONE block. Scene-play / scene-analyst / debrief assemble
# those blocks as a short role plus a labeled dump of numbered core-tip
# Preference rows. Discuss still joins blocks into SYSTEM_PROMPT_DISCUSS.
# Fixing a fact in its block fixes every mode that uses it.

# The move-token vocabulary is defined ONCE, in game_io (ACTION_MAP -> ACTIONS
# -> PLAYER_STOP_PATTERN / parse_move). Every prompt block references
# it through these constants -- never as fresh string literals -- so the wire
# format, the parser, and all prompts can only ever change together. The dict
# lookups fail loudly at import if an action name ever changes underneath us.
_TOK = {name: f"[{name}]" for name in game_io.ACTIONS}
_TOK_CLOCK = _TOK["CLOCK"]
_TOK_ANTICLOCK = _TOK["ANTICLOCK"]
_TOK_FORWARD = _TOK["FORWARD"]
# Counted forms are the turn API; the `_TOK` dict lookups keep the loud-fail
# role for the individual canonical tokens.
_TOK_TRIO = "[CLOCK n]/[ANTICLOCK n]/[FORWARD]"
#: Extra player token. NOT in game_io.ACTIONS -- ``apply_action`` never
#: receives it. Self-eval parses it; play / ``mode_game`` treat it as a
#: no-op (``parse_action`` returns None). The unified prompt mentions it;
#: legacy sealed-room sessions grade it but do not end on it.
_TOK_END_GAME = "[END_GAME]"

_SENT_GAME_INTRO = "You are an agent playing a 2D discrete game on a square board."
_SENT_GAME_SCREEN = "You see the current game screen as an image."
_SENT_GAME_WORLD = (
    "The board is the unit square framed by four boundary walls, which may "
    "have an opening; gold pieces (small yellow circles) may be present -- "
    "several, one, or none. You are the green circle with a red eye showing "
    "the direction you are facing."
)

_BLOCK_GAME_INTRO = " ".join([
    _SENT_GAME_INTRO,
    _SENT_GAME_SCREEN,
    _SENT_GAME_WORLD,
    "Your goal depends on what is in the room -- see the variant rules below.",
])

_BLOCK_MOVE_TOKENS = (
    "You make moves by emitting exactly one move token:\n"
    "  [CLOCK n]     - turn clockwise by n steps of pi/30 (6 degrees each), "
    "all in ONE move; e.g. [CLOCK 12] turns 72 degrees clockwise\n"
    "  [ANTICLOCK n] - turn counter-clockwise by n steps, the same way\n"
    f"  {_TOK_FORWARD}     - advance up to 1/16 of the board in the facing "
    "direction\n"
    "The count n is a whole number from 1 to 60. Anchors worth memorizing: "
    "5 steps = one clock hour, 15 = a quarter turn, 30 = a half turn, 60 = "
    "a full circle. Write the token exactly as shown -- one space, digits, "
    "closing bracket. A turn token with no count, like [CLOCK], is read as "
    "1 step -- a handy fine-adjustment nudge when you are almost aimed.\n"
    f"Only {_TOK_FORWARD} moves you; turning only rotates you in place. "
    f"{_TOK_FORWARD} NEVER takes a count: something like [FORWARD 10] is "
    "NOT a move token, does nothing, and is graded as a mistake. To "
    f"advance farther, emit plain {_TOK_FORWARD} again on the next "
    "screen.\n\n"
    "IMPORTANT: ONLY those exact bracketed tokens trigger a move. Talking "
    "about moving in prose (e.g. writing the word 'forward') does NOTHING. "
    "You may reason in plain prose as much as you like; nothing happens "
    "until you emit a bracketed move token, so emit one only when you truly "
    "intend that move."
)

def _block_how_to_play(target: str, obs_object: str, question_example: str) -> str:
    """HOW TO PLAY, parameterized by what the player is aiming at.

    Single-gold prompts pass ``the gold`` (the only objective). The
    multi-gold variant passes ``your current target`` so gold and
    openings are not collapsed into one noun.
    """
    return (
        "HOW TO PLAY -- take these steps in every reply. START your reply with "
        "one structured observation line, read fresh off the CURRENT screen, in "
        "exactly this form:\n"
        "  OBS: I am at <where on the board>; my eye points toward <clock "
        f"direction, e.g. 4 o'clock>; {obs_object}.\n"
        "(Clock directions are as seen on screen: 12 o'clock is up-screen, 3 is "
        "right, 6 is down, 9 is left.) Then REASON, in a sentence or two, about "
        f"where {target} is relative to your red eye: the eye points in the "
        f"direction you face, and {_TOK_FORWARD} sends you straight that way.\n"
        "IF (and only if) the user asked you to play or make a move, choose the "
        "move from that reasoning:\n"
        f"  - If your eye is pointing roughly at {target}, emit {_TOK_FORWARD}.\n"
        f"  - Otherwise, aim your eye at {target} in ONE counted turn: count "
        "the clock hours from your eye direction to it the SHORT way around, "
        "multiply by 5 to get the step count, and emit [CLOCK n] (clockwise, "
        "toward larger clock hours) or [ANTICLOCK n] (counter-clockwise). "
        f"Example: eye toward 12 o'clock, {target} at 4 o'clock -> 4 hours "
        "clockwise -> [CLOCK 20].\n"
        "  - On the next screen, VERIFY before advancing: if your eye now "
        f"points roughly at {target}, emit {_TOK_FORWARD}; if you are still "
        "off, correct with one more counted turn (reverse the direction if "
        "you overshot). When you are nearly aimed, a single-step nudge like "
        "[CLOCK] is perfectly good fine adjustment; what wastes moves is "
        f"creeping toward a FAR-off {target} one step at a time instead of "
        "turning once.\n"
        f"If the user instead asked a question (e.g. '{question_example}'), "
        "answer it in prose after the observation and reasoning and STOP -- "
        "emitting a move token nobody asked for is a format mistake.\n"
        "Always state the observation and reasoning first; looking, then "
        "reasoning, then (only when asked) a single move, is how you decide well."
    )


_BLOCK_HOW_TO_PLAY = _block_how_to_play(
    "your current target",
    "my target, <a remaining gold | an opening/exit>, is at <where on the "
    "board>, toward <clock direction> of me",
    "are you facing your target?",
)

_BLOCK_NOTEPAD = (
    "YOUR NOTEPAD: you have a small notepad that persists for THIS game "
    "session only. To save a note, write a line of this exact form in your "
    "reply, BEFORE your move token:\n"
    "[REMEMBER key: short note]\n"
    "  - 'key' is one short word (letters, digits, underscores); the note is "
    "free text on one line (avoid ']' inside it).\n"
    "  - Every saved note is shown back to you on EVERY future turn in the "
    "'Your notepad' section.\n"
    "  - Saving with a key you already used OVERWRITES that note. That is how "
    "you change your mind on the record.\n"
    "  - Anything written after your move token is LOST -- the move ends your "
    "turn. Save first, then move.\n"
    "Your notepad is the only memory that reliably survives between turns; "
    "older messages scroll away. If a fact matters for future moves, save it. "
    "But write a [REMEMBER] line ONLY when it CHANGES the notepad -- a first "
    "save, or a genuine update. The notepad is shown back to you every turn; "
    "re-saving a note that already says the same thing does nothing, clutters "
    "your reply, and is graded as a mistake. Most turns need NO [REMEMBER] "
    "line at all."
)

def _block_current_screen(target: str) -> str:
    return (
        "DO NOT just copy prior observations from your memories. Make sure you "
        f"evaluate whether you are facing {target} *right now*. Your memories "
        "describe PAST screens; every move changes the screen, so re-derive where "
        f"your red eye points and where {target} is from the CURRENT image before "
        "every single move."
    )


_BLOCK_CURRENT_SCREEN = _block_current_screen("your current target")


def _block_aim_tolerance(target: str) -> str:
    """The one statement of the aim-tolerance rule for a given aim object.
    Shown verbatim to the matching player prompt and quoted verbatim to
    that prompt's reviewer, so player and judge always share one wording.
    """
    return (
        "AIM TOLERANCE: it can be hard to tell your exact facing direction from "
        "the screen, so do not demand pixel-perfect aim. If your best estimate "
        f"is that {target} lies within about 20 degrees (3 turn steps) of your "
        f"facing direction, {_TOK_FORWARD} is a good move -- step forward and "
        "re-assess on the new screen. Inside that band a single fine-tuning "
        "step like [CLOCK] is also fine play. What wastes moves is a long "
        f"chain of tiny turns toward a {target} that is clearly off to one "
        "side or behind you: there, estimate the full count and turn ONCE "
        "with [CLOCK n]/[ANTICLOCK n]."
    )


_BLOCK_AIM_TOLERANCE = _block_aim_tolerance("your current target")


def _block_aim_tolerance_review(tolerance: str, target: str) -> str:
    return (
        "The player's instructions included this aim-tolerance rule, quoted "
        "verbatim:\n\"" + tolerance + "\"\n"
        f"Judge the play against it: a {_TOK_FORWARD} emitted while {target} was "
        "within roughly 20 degrees of the facing direction follows instructions "
        "and must not be penalized as imprecise aim; conversely, a CHAIN of "
        "rotate-only replies inside that tolerance, turn after turn without "
        "ever stepping forward, goes against them (one fine-tuning nudge is "
        "fine). "
        "Measure those 20 degrees from YOUR computed |delta|, never from the "
        "player's 'about 30 degrees', 'close enough', or 'within aim "
        "tolerance' claim. 'Do not demand pixel-perfect aim' is a player "
        "instruction about reading the screen, not a license to bless "
        f"{_TOK_FORWARD} outside the cone."
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


# --------------------------------------------------------- composed prompts

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

#: Pre-filled into the player text box in the notebook; the user may edit it
#: or submit it unchanged.
DEFAULT_PLAYER_QUESTION = "Please make the right move for this position."

# --------------------------------------------------------------- S2 look
#
# The learn-to-look prompt. No [CLOCK] / [FORWARD] / [HOLD].
# A look reply ends by ending the message. A move reply ends with
# [MOVE] on its own last line, which ends the message and wakes S1.
# A sealed empty room ends with [END_GAME], which ends the game and
# does not wake S1. The confirmation lines below are the only copy:
# the trainer samples them from these tuples.

_S2_LOOK_LINES = (
    "Acknowledged, looking there.",
    "Confirmed. Looking at that spot.",
    "Understood, I'll look there.",
    "Yes. Eyes on that spot.",
)
_S2_MOVE_LINES = (
    "Acknowledged, moving there.",
    "Confirmed. Moving to that spot.",
    "Understood, I'll move there.",
    "Yes. Heading there now.",
)

S2_MOVE_TOKEN = "[MOVE]"
S2_MOVE_RE = re.compile(r"\[MOVE\]\s*\Z")
S2_END_TOKEN = _TOK_END_GAME
S2_END_RE = re.compile(r"\[END_GAME\]\s*\Z")
S2_SEALED_NOTE = "none -- room is sealed and empty"
S2_END_REPLY = f"[REMEMBER target: {S2_SEALED_NOTE}]\n{S2_END_TOKEN}"
S2_LOOK_TARGET_USER = "Look at your target."
S2_MOVE_TARGET_USER = "Move to your target."
S2_MOVE_VERDICT_LINES = (
    "The move ate a gold.",
    "The move did not eat a gold.",
)


def s2_move_reply(line: str) -> str:
    """A move confirmation plus the ``[MOVE]`` line that ends it."""
    return f"{line}\n{S2_MOVE_TOKEN}"


def ends_with_move(text: str) -> bool:
    """True when ``text`` ends with ``[MOVE]``, ignoring trailing space."""
    return S2_MOVE_RE.search(text) is not None


def ends_with_end_game(text: str) -> bool:
    """True when ``text`` ends with ``[END_GAME]``, ignoring trailing space."""
    return S2_END_RE.search(text) is not None


_BLOCK_S2_IDENTITY = (
    "You are an agent meant to explore an environment and learn from it. "
    "Besides the words you write, every token also sets two points: s, "
    "where your body is, and v, the place you are attending to. For now, "
    "navigate a game of gold and exits."
)

_BLOCK_S2_WORLD = (
    "The picture is a room on the unit square. You are the green circle. "
    "The red mark on it is your eye and shows which way you face. Gold is a "
    "yellow circle. A gap in a wall is an exit. The top edge of the picture "
    "is y = 1. The right edge is x = 1."
)

_BLOCK_S2_GOAL = (
    "The game: eat every gold, then leave through an exit. Eating a gold "
    "does not end the game. Walking out through an exit ends it."
)

_BLOCK_S2_LOOK_MOVE = (
    "A message may ask you to look at a place, to move to a place, or to "
    "answer a question about the picture or about where you are looking.\n"
    "\n"
    "Look at a place: attend to it and leave your body where it is. Confirm "
    "with one of these lines, and nothing else on that line, then end the "
    "message:\n"
    "\n"
    + "\n".join(f"  {line}" for line in _S2_LOOK_LINES)
    + "\n\n"
    "Move to a place: attend to it, confirm with one of these lines, and "
    "then write [MOVE] on its own last line. [MOVE] ends the message and "
    "sends your body toward v. Do not write anything after [MOVE]:\n"
    "\n"
    + "\n".join(f"  {line}" for line in _S2_MOVE_LINES)
    + "\n"
    "  [MOVE]\n"
    "\n"
    "Your target is the target in your notepad. If the notepad has no "
    "target, or names a gold that is gone, first save a new target on its "
    "own line, then look or move as asked.\n"
    "\n"
    "A question about the picture or about your gaze gets an answer and "
    "then the message ends. Do not write [MOVE] unless you were asked to "
    "move. [END_GAME] has its own rule below."
)

_BLOCK_S2_MOVE_REPORT = (
    "After you move, the next message begins with a Board update that says "
    "whether that move ate a gold and how many golds remain. Open your "
    "reply with exactly one of these lines, then continue:\n"
    "\n"
    f"  {S2_MOVE_VERDICT_LINES[0]}\n"
    "\n"
    "or\n"
    "\n"
    f"  {S2_MOVE_VERDICT_LINES[1]}\n"
    "  [FLAG failure: what you expected, and what happened]\n"
    "\n"
    "Fill in the flag with your notepad target and what the Board update "
    "said. If no gold remains, your next target is an exit. If the room "
    "has no exit either, it is sealed and empty: follow the [END_GAME] rule."
)

_BLOCK_S2_FAILURE = (
    "A move to a gold succeeds only if the next Board update says it ate "
    "a gold. A move to an exit succeeds only if the game ends. Any other "
    "move failed. Flag it for an Analyst version of you and for your user, "
    "on one line that names your notepad target:\n"
    "\n"
    "  [FLAG failure: what you expected, and what happened]\n"
    "\n"
    "After a miss, for example:\n"
    "\n"
    f"  {S2_MOVE_VERDICT_LINES[1]}\n"
    "  [FLAG failure: expected to eat the upper-middle gold; stopped short]\n"
    "\n"
    "Not a [REMEMBER] note."
)

_BLOCK_S2_TIP = (
    "You may save a one-line tip for future games. Write it on its own "
    "line, then go on with the reply:\n"
    "\n"
    "  [TIP: one line]\n"
    "\n"
    "It is saved at once and appears under Saved tips in later games. Save "
    "one when a failure taught you something that will hold on other "
    "boards. Not a [REMEMBER] note."
)

_BLOCK_S2_TARGET = (
    "Keep a short notepad. To save a note, write one line:\n"
    "\n"
    "  [REMEMBER key: short note]\n"
    "\n"
    "The key you keep is target. Name one gold while any gold remains. "
    "Name one exit only when no gold remains. Write it when you have no "
    "target, when that gold has been eaten, or when that exit is gone. "
    "Otherwise do not write it again, and do not switch mid-chase. For "
    "example:\n"
    "\n"
    "  [REMEMBER target: the upper-right gold]\n"
    "  [REMEMBER target: the exit on the left wall]\n"
    "\n"
    "When the room has no gold and no exit, the note is:\n"
    "\n"
    f"  [REMEMBER target: {S2_SEALED_NOTE}]"
)

_BLOCK_S2_END_GAME = (
    "End the game: when the room has no gold and no exit, save the sealed "
    "note above, then write [END_GAME] on its own last line. [END_GAME] "
    "ends the message and the game. Do not write anything after it. An "
    "instruction to pick, look at, or move to your target in a sealed and "
    "empty room gets this reply, never [MOVE]. Also write [END_GAME] when "
    "the message explicitly asks you to end the game. Never write it while "
    "a gold or an exit remains unless asked."
)

# Seed for the S2 system prompt. Numbers stay below 500; 500+ is reserved
# for tips the agent or the user saves. Do not renumber.
CORE_S2_TIPS: list[tuple[str, str]] = [
    ("core_s2_010_identity", _BLOCK_S2_IDENTITY),
    ("core_s2_020_world", _BLOCK_S2_WORLD),
    ("core_s2_030_goal", _BLOCK_S2_GOAL),
    ("core_s2_040_look_move", _BLOCK_S2_LOOK_MOVE),
    ("core_s2_050_move_report", _BLOCK_S2_MOVE_REPORT),
    ("core_s2_060_failure", _BLOCK_S2_FAILURE),
    ("core_s2_070_tip", _BLOCK_S2_TIP),
    ("core_s2_080_target", _BLOCK_S2_TARGET),
    ("core_s2_090_end_game", _BLOCK_S2_END_GAME),
]

S2_SAVED_TIPS_HEADING = "Saved tips, from earlier games:"

def assemble_tip_prompt(
    rows: dict[str, str], prefix: str, heading: str | None,
) -> str:
    """Join one core-tip namespace, seed then any saved tips.

    Seed rows (number below 500) are joined with a blank line, in
    category order. When ``heading`` is set and any 500+ row is present,
    that heading and one indented line per saved tip follow. A category
    that does not belong to ``prefix`` raises. No seed row raises.
    Saved rows with no heading raise rather than being dropped.
    """
    cat_re = re.compile(rf"^{re.escape(prefix)}(\d{{3}})_[a-z0-9_]+$")
    seed: list[tuple[str, str]] = []
    saved: list[tuple[str, str]] = []
    for category, tip in rows.items():
        match = cat_re.match(category)
        if match is None:
            raise ValueError(f"not a {prefix} tip category: {category!r}")
        number = int(match.group(1))
        if number < 500:
            seed.append((category, tip))
        else:
            saved.append((category, tip))
    if not seed:
        raise RuntimeError(f"assemble_tip_prompt: no seed tips for {prefix}")
    seed.sort()
    body = "\n\n".join(tip for _category, tip in seed)
    if not saved:
        return body
    if not heading:
        raise RuntimeError(
            f"assemble_tip_prompt: saved tips under {prefix} but no heading"
        )
    saved.sort()
    lines = "\n".join(f"  {tip}" for _category, tip in saved)
    return body + "\n\n" + heading + "\n" + lines


def assemble_s2_prompt(rows: dict[str, str]) -> str:
    """The S2 system prompt. Thin call to :func:`assemble_tip_prompt`."""
    return assemble_tip_prompt(rows, "core_s2_", S2_SAVED_TIPS_HEADING)


SYSTEM_PROMPT_S2 = assemble_s2_prompt(dict(CORE_S2_TIPS))



async def load_s2_prompt(client: Any) -> str:
    """The S2 system prompt as NAMS holds it. A read or heal failure raises.

    ``ensure_core_tips`` heals the seed rows to this code first, so a
    missing or drifted seed is rewritten before the dump. There is no
    fallback to :data:`SYSTEM_PROMPT_S2`.
    """
    await mem.ensure_core_tips(client)
    return assemble_s2_prompt(await mem.get_core_tips(client, "core_s2_"))


def _run_off_caller_loop(work):
    """Run an async function on a fresh loop in another thread.

    A notebook widget callback is already inside the kernel loop. Python
    3.12 refuses ``run_until_complete`` on that thread. The worker thread
    has no loop, so ``asyncio.run`` is legal there, and it is also legal
    when the caller has no loop at all. ``work`` is called on the worker.
    """
    import asyncio
    import threading

    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(work())
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=_runner, name="nams-sync", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    if "value" not in box:
        raise RuntimeError("NAMS load thread exited without a result")
    return box["value"]


def load_s2_prompt_sync() -> str:
    """:func:`load_s2_prompt` from a notebook cell that already has a loop.

    Connects, loads, and closes the client on a fresh loop in another
    thread. A NAMS failure raises. There is no fallback to the code
    constant.
    """
    async def _go() -> str:
        client = await mem.connect()
        try:
            return await load_s2_prompt(client)
        finally:
            await client.close()

    return _run_off_caller_loop(_go)

_CORE_FA_010_ROLE = "You are the Analyst for an agent that plays a game of gold and exits on the unit square. The agent is the green circle; its red mark is the eye. Gold is yellow. A gap in a wall is an exit. The top edge is y = 1, the right edge x = 1. The agent flags its own failures. Your job is to read those flags and find the common ways it fails. You are not grading one reply. You are looking for patterns across many."

_CORE_FA_020_RECORD = "Each flagged memory has two halves. BEFORE is the reply that acted and failed: the question it was asked, the words it wrote, the picture it was shown, and that picture's exact Settings. AFTER is the reply that noticed and wrote the flag: its question, its words, the picture it saw, and that picture's Settings. The flag text is in the AFTER message. A memory from the first reply of a game has [ERROR, no prior ...] in place of its BEFORE parts.\n\nMemories are numbered from 0, oldest first, across every game. The number is how you come back to one. When you write a note about a memory, write its number."

_CORE_FA_030_EXACT = "Use the Settings, not the picture, for anything numeric. The Settings list every gold as [x, y], the agent as agent_x, agent_y, its facing as direction, and the exits under openings. Distances and bearings come from those numbers.\n\nThe agent also sets two points on every token it writes: s, where it believes its body is, and v, where it is attending. [GAZE n] prints both at the start of the BEFORE message, at the end of each of its lines, and at its final ']'. [GAZE n: a-b] prints every token from a to b. On a move, the controller walks toward the v read at the final ']' token of that message. Compare that v to the gold list. A v that sits on a gold but a move that missed is a different failure from a v that sits in empty space."

_CORE_FA_040_TOOLS = "Tools. End a reply with exactly one token on its own last line, and you will be shown the result and asked to continue. No token ends your turn.\n\n  [LIST]               one line per flagged memory: number, game, flag text\n  [LOAD n]             memory n: both questions, both messages, the BEFORE picture and Settings\n  [LOAD n: parts]      only those parts. Parts: bq bm bi bs aq am ai as gaze\n                       b/a = before/after; q question, m message, i image, s settings\n  [DROP n]  [DROP n: parts]  [DROP ALL]   unload\n  [GAZE n]             s and v for memory n's BEFORE message: at the start, at the end of each line, and at the final ']'\n  [GAZE n: a-b]        s and v for every token from a to b of that message.  [GAZE n: -k] is the last k tokens\n  [NOTE key: text]     write to your notepad.  [FORGET key] removes it\n  [SEARCH words]       flagged messages whose text matches, with their numbers\n\nAt most 4 pictures can be loaded at once. Pick which four. Four BEFORE pictures from four memories is a fair way to look for a shared shape. A loaded memory stays loaded until you drop it."

_CORE_FA_050_METHOD = "Work slowly. Load one memory, say what you see, load another, compare. Keep a note per candidate pattern, and under it the numbers of the memories that fit and the ones that do not. Do not call a pattern until at least three memories fit it and you have said which ones. When you are unsure, ask the user which memory to look at next; the user may know the games. A reply with no tool token hands the conversation back.\n\nWhen the user asks for fixes, propose them then, not before: what the agent should have done in those memories, and what kind of practice would teach it. Keep that part short for now. The analysis comes first."

_CORE_FA_060_VOICE = "Write in plain sentences. Name memories by number. Quote the agent's words when they matter. Do not rate, do not score, do not write a verdict line. Do not write [MOVE], [REMEMBER], or [FLAG]; those are the player's tokens."

CORE_FAILURE_ANALYST_TIPS: list[tuple[str, str]] = [
    ("core_fa_010_role", _CORE_FA_010_ROLE),
    ("core_fa_020_record", _CORE_FA_020_RECORD),
    ("core_fa_030_exact", _CORE_FA_030_EXACT),
    ("core_fa_040_tools", _CORE_FA_040_TOOLS),
    ("core_fa_050_method", _CORE_FA_050_METHOD),
    ("core_fa_060_voice", _CORE_FA_060_VOICE),
]

# Measured 3717 characters once the six blocks are joined with a blank line.
SYSTEM_PROMPT_FAILURE_ANALYST = assemble_tip_prompt(
    dict(CORE_FAILURE_ANALYST_TIPS), "core_fa_", None,
)


async def load_failure_analyst_prompt(client: Any) -> str:
    """The failure-analyst system prompt as NAMS holds it.

    ``ensure_core_tips`` heals the seed rows to this code first. There is
    no fallback to :data:`SYSTEM_PROMPT_FAILURE_ANALYST`.
    """
    await mem.ensure_core_tips(client)
    return assemble_tip_prompt(
        await mem.get_core_tips(client, "core_fa_"), "core_fa_", None,
    )


def load_failure_analyst_prompt_sync() -> str:
    """:func:`load_failure_analyst_prompt` for a notebook cell.

    Connects, loads, and closes the client on a fresh loop in another
    thread. A NAMS failure raises. There is no fallback to the code
    constant.
    """
    async def _go() -> str:
        client = await mem.connect()
        try:
            return await load_failure_analyst_prompt(client)
        finally:
            await client.close()

    return _run_off_caller_loop(_go)


#: User line for a beginning that looks. Not part of the system prompt.
S2_BEGINNING_USER = (
    "For the first message this game, select a target as usual, and look "
    "at it, then end the message."
)

#: User line for a beginning that moves. The reply ends with [MOVE].
S2_BEGINNING_MOVE_USER = (
    "For the first message this game, select a target as usual, and move "
    "to it."
)

#: User line that asks for [END_GAME] even when gold or an exit remains.
S2_END_GAME_USER = "End the game."


#: Pre-filled into the analyst text box in the notebook; the user may edit it
#: or submit it unchanged.
DEFAULT_ANALYST_QUESTION = (
    "Analyze the player's performance, using newly available privileged "
    "information. Go through the response part by part -- the OBS line, the "
    "reasoning, any [REMEMBER] lines, and the move token (or its correct "
    "absence, if the user asked a question) -- and say which parts were "
    "correct and which were incorrect. If the move is a turn, grade its "
    "step count against your computed steps_correct (GRADING TURN "
    "COUNTS). Check each [REMEMBER] line against "
    "the notepad the player saw: a line that merely re-saves what the "
    "notepad already holds is a mistake. Apply your GRADING CALIBRATION "
    "when deciding what counts as mistaken: direction estimates within 2 "
    "clock hours of the truth are NOT mistakes.\n"
    "For EVERY mistaken phrase, add a line of the form:\n"
    "WRONG: \"<the exact words copied from the player's reply>\"\n"
    "(one line per mistake, copied verbatim so the words can be found in "
    "the reply). Put NOTHING else on a WRONG line -- corrections or "
    "suggested alternatives go on their own following line. Then give a "
    "final rating from -1.0 to 1.0."
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
    search_results: str | None = None,
    notepad: str | None = None,
) -> list[dict]:
    user_text = []
    if context:
        user_text.append({"type": "text", "text": f"Memory context:\n{context}"})
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
    if notepad:
        user_text.append({"type": "text", "text": notepad})
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
    That code-side loop is not a prompt multi-move turn: each generation
    still emits at most one move. Otherwise, take a single generation: the
    model either answers the question or emits one move.

    ``[END_GAME]`` is a stop string so generation halts at the token;
    ``game_io.parse_action`` returns None for it, so it is a no-op on the
    board and does not end the session.
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)
    max_steps = max_steps or cfg.max_solve_steps
    play_prompt = (await load_scene_prompts(client))["scene_play"]

    # Legacy training/play board -- sealed one-gold room; eating gold still
    # wins; openings / [END_GAME] exist in the prompt for the multi-gold
    # path but this factory does not spawn them.
    game = game_io.new_bare_game(gameSize=cfg.game_size)
    turns: list[dict[str, Any]] = []

    steps = 0
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
                exclude_analyst=True,
            )
            notes = await mem.get_session_notes(client, session_id)
            notepad = mem.format_notepad(notes)

            # Inner [SEARCH] loop: the model may spend up to
            # cfg.memory_search_max_calls searches before this step's move /
            # answer. Each search's results are fed back and the step is
            # regenerated; searches are recorded on the turn trace below
            # (after the trace exists).
            search_notes: list[str] = []
            searches: list[dict[str, str]] = []
            while True:
                messages = _build_game_messages(
                    play_prompt, snapshot_before_path, ctx, question,
                    search_results="\n\n".join(search_notes) or None,
                    notepad=notepad,
                )
                over_budget = len(searches) >= cfg.memory_search_max_calls
                raw = model.generate(
                    messages,
                    stop_strings=None,
                    # Once the budget is spent the search alternative is
                    # dropped, so a stray [SEARCH] is inert prose instead
                    # of a stall. Counted turn tokens live in the regex.
                    stop_regex=(
                        game_io.PLAYER_STOP_PATTERN if over_budget
                        else game_io.PLAYER_STOP_PATTERN + "|"
                        + SEARCH_TOOL_PATTERN
                    ),
                )
                kind, payload, text = classify_move_or_search(raw)
                if kind != "search" or over_budget:
                    break
                results = await mem.search_memory(
                    client, payload, tiers=("semantic", "reasoning"),
                    top_k=cfg.memory_search_top_k, scrub=True,
                    exclude_analyst=True,
                )
                search_notes.append(format_search_note(payload, results))
                searches.append({"query": payload, "results": results, "thought": text})
                if len(searches) >= cfg.memory_search_max_calls:
                    search_notes.append(SEARCH_BUDGET_NOTE)
                logger.info("step %d: [SEARCH %s]", steps, payload)
            for k, v in game_io.parse_remember_notes(raw):
                await mem.set_session_note(client, session_id, k, v, steps)
            parsed = game_io.parse_move(raw) if kind == "move" else None
            action = parsed[0] if parsed else None
            count = parsed[1] if parsed else 1

            if action:
                gold_collected = game_io.apply_action(game, action, count=count)
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
    """Mode 2: open-ended discussion.

    Always injects the last-K recency window. The ``get_context`` semantic
    dump is added only when ``nams_retrieval`` is on (off on this branch).
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)

    # Recency BEFORE storing the current message: the window should show
    # the PRIOR turns, not the one we are about to answer.
    recent = await mem.get_recent_messages(
        client, session_id, cfg.recent_messages_window,
    )
    context_parts: list[str] = []
    if recent:
        context_parts.append(
            "Recent conversation (most recent last):\n" + recent
        )
    if mem.retrieval_enabled(cfg):
        ctx = await mem.retrieve_context(
            client, query=user_text, session_id=session_id,
        )
        ctx_text = ctx if isinstance(ctx, str) else json.dumps(
            ctx, default=str, indent=2,
        )
        # Recency window is the one channel for recent messages; drop
        # NAMS' built-in "Recent Conversation" (ordered ASC = oldest).
        ctx_text = mem.strip_nams_recent_conversation(ctx_text)
        if ctx_text and ctx_text.strip() not in ("", "{}", "[]"):
            context_parts.append(
                "Relevant memories (semantic search across all tiers):\n"
                + ctx_text
            )
    context_block = (
        "\n\n".join(context_parts) if context_parts else "(no prior context)"
    )

    # Now record the user message (after retrieval, so it isn't self-retrieved).
    await client.short_term.add_message(
        session_id=session_id, role="user", content=user_text,
        metadata={"kind": "discussion"},
    )

    # [SEARCH] loop: this privileged mode searches ALL tiers, unscrubbed.
    search_notes: list[str] = []
    n_searches = 0
    while True:
        content: list[dict[str, str]] = []
        if context_block:
            content.append(
                {"type": "text", "text": f"Memory context:\n{context_block}"}
            )
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
            lines.append(
                f"    settings_json="
                f"{game_io.settings_json_with_openings(s.get('settings_json'))}"
            )
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
    to the model with the evaluator system prompt, then appends the
    verdict to the SAME Conversation as an assistant message
    (metadata.kind='self_evaluation') and records a new reasoning trace
    capturing the evaluation reasoning.
    """
    cfg = cfg or CONFIG
    model = get_model(cfg)
    debrief_prompt = (await load_scene_prompts(client))["debrief"]

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
    sections.append(DEFAULT_ANALYST_QUESTION)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": debrief_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": "\n\n".join(sections)},
        ]},
    ]
    verdict = model.generate(messages, max_new_tokens=cfg.max_new_tokens)

    # Append the verdict to the same Conversation. Tagged so a later player
    # retrieve cannot surface privileged evaluation into play context.
    tagged_verdict = mem.tag_analyst_text(verdict)
    eval_msg = await client.short_term.add_message(
        session_id=session_id, role="assistant", content=tagged_verdict,
        metadata={"kind": "self_evaluation", "evaluated_session": session_id},
    )

    # Record a reasoning trace for the evaluation itself.
    trace = await client.reasoning.start_trace(
        session_id, task=f"self-evaluation of session {session_id}",
        triggered_by_message_id=eval_msg.id,
    )
    step = await client.reasoning.add_step(trace.id, thought=tagged_verdict)
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

# Role statements stay hardcoded -- the one thing not stored as a tip.
ROLE_SCENE_PLAY = _BLOCK_GAME_INTRO

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

# ------------------------------------------------- variant-rule blocks
# Shared by the unified scene-play / scene-analyst / debrief prompts.

_BLOCK_MULTI_GOLD_RULES = (
    "THIS SESSION'S VARIANT RULES: "
    "this room may contain SEVERAL golds (up to 3), or NONE at all. "
    "Eating a gold does NOT end the session -- after you eat one, keep "
    "playing and go for another gold if any remain. The boundary walls "
    "may have an OPENING (a gap leading out of the room); if the room "
    "has no gold left, or never had any, your goal is to find the "
    "opening and walk out through it. Gold and openings are two "
    "different objectives: a gold is a yellow circle you eat; an "
    "opening is a gap in a wall you walk through. Never describe an "
    "opening as gold, and never chase an opening while gold remains."
)

_BLOCK_TARGET_COMMIT = (
    "PICK ONE TARGET AND COMMIT: your notepad's 'target' note is the "
    "thing you are chasing this turn. There are two kinds, and they "
    "are not interchangeable -- name the kind in the note:\n"
    "  - a GOLD (a yellow circle), while any gold remains;\n"
    "  - an OPENING / EXIT (a gap in a wall), only when no gold remains.\n"
    "When more than one gold is visible, choose exactly ONE of them to "
    "eat first -- for example 'the left one' (the leftmost gold on "
    "screen), 'the right one', or 'the near one' (closest to you). Save "
    "your choice in your notepad under the key 'target':\n"
    "[REMEMBER target: the left gold, near the top wall]\n"
    "Your notepad shows your current target every turn. Pursue THAT "
    "target and do NOT switch mid-chase, even if another gold briefly "
    "looks closer.\n"
    "WHEN to write a [REMEMBER target: ...] line -- in exactly these "
    "three situations:\n"
    "  - your notepad has NO 'target' note and a gold or an opening is "
    "visible: pick one and save it. That is your FIRST job that turn.\n"
    "  - the Board update says a gold was EATEN. The eaten gold is "
    "usually your target: check the screen, and if the gold your note "
    "describes is gone, save a NEW target -- another gold if any remain, "
    "otherwise an opening (see WHEN THE ROOM HAS NO GOLD).\n"
    "  - your saved target has disappeared from the screen for any other "
    "reason: same rule -- golds first, then an opening; if the room is "
    "sealed and empty, follow ENDING THE SESSION.\n"
    "WHEN NOT to write one -- every other turn. While you are still "
    "chasing the target your notepad already names, just chase it: no "
    "[REMEMBER] line. Re-saving the target you already have, even in "
    "different words, is a real mistake -- it shows you did not read "
    "your own notepad, and the analyst grades it DOWN. On a normal "
    "chasing turn your reply is observation, reasoning, move -- nothing "
    "else."
)

_BLOCK_NO_GOLD_EXPLORE = (
    "WHEN THE ROOM HAS NO GOLD: your objective has changed. You are "
    "no longer chasing gold -- there is none. Do not wander. Scan the "
    "boundary walls for an opening -- a visible gap in one of the four "
    "walls, also called an exit -- and save it as your target:\n"
    "[REMEMBER target: the opening in the right wall]\n"
    "Name it an opening or an exit in OBS, in reasoning, and in the "
    "notepad. Never call it 'the gold'. Then aim at it the usual way: "
    "ONE counted turn ([CLOCK n]/[ANTICLOCK n]) to put the opening "
    f"roughly dead ahead, then {_TOK_FORWARD} until you walk out "
    "through it."
)

_BLOCK_END_GAME = (
    "ENDING THE SESSION: you have one extra move in this variant: "
    f"{_TOK_END_GAME}. Emit it exactly like a move token. Use it in "
    "exactly two situations:\n"
    "  - the room has NO gold left AND NO opening in any wall: there "
    "is nothing left to do. Save [REMEMBER target: none -- room is "
    f"sealed and empty] and emit {_TOK_END_GAME}.\n"
    "  - the question you were given EXPLICITLY asks you to end the "
    f"game: emit {_TOK_END_GAME} regardless of what is on the board.\n"
    "Do NOT end the session while a gold or an opening remains and "
    "nobody asked you to stop -- finishing the job matters more than "
    "finishing quickly."
)


# Code seed for NAMS Preference rows. Category names carry a zero-padded
# 3-digit number so plain string sort is the dump order. Do not renumber.
# Numbers below 500 are reserved for this seed (FUTURE_GOALS goal 11:
# agent-written tips use 500+).
CORE_PLAYER_TIPS: list[tuple[str, str]] = [
    ("core_player_010_multi_gold_rules", _BLOCK_MULTI_GOLD_RULES),
    ("core_player_020_move_tokens", _BLOCK_MOVE_TOKENS),
    ("core_player_030_scene_scope", _BLOCK_SCENE_SCOPE),
    ("core_player_040_how_to_play", _BLOCK_HOW_TO_PLAY),
    ("core_player_050_notepad", _BLOCK_NOTEPAD),
    ("core_player_060_target_commit", _BLOCK_TARGET_COMMIT),
    ("core_player_070_no_gold_explore", _BLOCK_NO_GOLD_EXPLORE),
    ("core_player_080_end_game", _BLOCK_END_GAME),
    ("core_player_090_aim_tolerance", _BLOCK_AIM_TOLERANCE),
    ("core_player_100_current_screen", _BLOCK_CURRENT_SCREEN),
    ("core_player_110_search_tool", _search_tool_block(_SEARCH_SCOPE_PLAY)),
]

_CORE_TIPS_HEADING = "Long-term tips (loaded from memory):"


def format_core_tips_dump(
    role: str, tips: dict[str, str], exclude: set[str],
) -> str:
    """Assemble a system message: short role, then a labeled dump of
    core-tip Preference rows (category-sort, ``exclude`` dropped)."""
    cats = sorted(c for c in tips if c not in exclude)
    if not cats:
        raise RuntimeError(
            "format_core_tips_dump: no core tips remain after exclude "
            f"{sorted(exclude)!r}"
        )
    entries = ["[" + c + "]\n" + tips[c] for c in cats]
    return role + "\n\n" + _CORE_TIPS_HEADING + "\n\n" + "\n\n".join(entries)


async def load_scene_prompts(client: Any) -> dict[str, str]:
    """Read core-tip Preference rows and assemble the three scene system
    prompts as role + labeled NAMS dump.
    """
    # Reconcile first: session __init__ loads prompts BEFORE the
    # datagen run-start reset, and Preference rows survive that reset.
    # A branch that drops or renames seed categories would otherwise
    # fail get_core_tips' strict set check against the previous
    # checkout's rows. 500+ extras are kept; seed text is healed to
    # the running code (code is source of truth).
    await mem.ensure_core_tips(client)
    player = await mem.get_core_tips(client, "core_player_")
    analyst = await mem.get_core_tips(client, "core_analyst_")
    return {
        "scene_play": format_core_tips_dump(ROLE_SCENE_PLAY, player, set()),
        "scene_analyst": format_core_tips_dump(
            ROLE_SCENE_ANALYST, analyst, SCENE_ANALYST_EXCLUDE,
        ),
        "debrief": format_core_tips_dump(
            ROLE_DEBRIEF, analyst, DEBRIEF_EXCLUDE,
        ),
    }


def build_scene_analyst_messages(
    player_question: str,
    player_reply: str,
    pending_action: str | None,
    frame_path: str | None,
    settings_json: str | None,
    recent: str,
    question: str,
    search_results: str | None = None,
    *,
    system_prompt: str,
    player_notepad: str | None = None,
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
    if player_notepad:
        scene.append(
            "The notepad the player SAW while writing this reply (the harness "
            "injects this block into the player's prompt every turn; [REMEMBER "
            "...] lines inside the reply update it for FUTURE turns only). "
            "Compare any [REMEMBER ...] line in the reply against THIS block "
            "-- a line that merely re-saves what the notepad already holds "
            "is a graded defect:\n"
            + player_notepad
        )
    if pending_action:
        scene.append(
            f"The reply ends in the move token [{pending_action}]. That move "
            "has NOT been applied yet -- judge the decision on the frame "
            "below. Check the user's question first: if it asked for "
            "information rather than a move, this move token was NOT "
            "requested -- that is itself a format mistake to call out."
        )
    invalid = game_io.find_invalid_move_token(player_reply)
    if invalid:
        scene.append(
            f"FORMAT ERROR (harness-verified): the reply contains the "
            f"invalid token '{invalid}'. [FORWARD] never takes a count "
            f"and turn counts must be 1..60; this is NOT a move and was "
            f"ignored. Copy the exact token into a WRONG line and grade "
            f"the wasted turn down."
        )
    elif not pending_action:
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
            scene.append(
                "WARNING: no move token detected. If a move was intended, "
                "this is a Format Error. If the user's question asked for "
                "information rather than a move, prose with no move token "
                "is the CORRECT format -- do not penalize the absence; grade "
                "the answer's content instead."
            )
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
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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
# scoped ((?i:...)) on SEARCH_TOOL_PATTERN so the SAME pattern string drives
# both this module's parser and generation-time RegexStopCriteria, and so
# concatenating it with PLAYER_STOP_PATTERN does not leak IGNORECASE onto
# the uppercase-only move-token alternatives. DEBRIEF_TOOL_PATTERN keeps a
# whole-pattern (?i) -- it is never concatenated with a case-sensitive
# sibling.

# The SEARCH-call fragment, shared verbatim by the standalone search pattern
# (play / discuss / scene modes) and the full debrief pattern.
_SEARCH_CALL_FRAGMENT = r"\[\W*SEARCH\b[:\s]*(?P<search_query>[^\[\]]+?)\W*\]"

SEARCH_TOOL_PATTERN = r"(?i:" + _SEARCH_CALL_FRAGMENT + r")"
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


def _without_analyst_line_tags(text: str) -> str:
    """Strip a leading ``[ANALYST]`` tag from each line, if present.

    The model sometimes emits the persistence tag itself. Line-start
    extractors (RATING / WRONG / TIP / TARGET) must not fail because of it.
    Lines without the tag are unchanged. Delegates to
    :func:`agent.memory.untag_analyst_text` so tagging and parsing share
    one strip rule.
    """
    return mem.untag_analyst_text(text)


# The 'TIP: <one line>' line that must accompany a [WRITE_TIP] call. The LAST
# such line in the reply is the tip (the model may quote earlier proposals).
TIP_LINE_RE = re.compile(r"(?im)^\s*TIP\s*:\s*(?P<tip>.+?)\s*$")

# A reviewer's per-error span: WRONG: "<exact words from the player reply>".
# If the span is double-quoted, ONLY the quoted words count -- models
# (Gemma 4 12B especially) like to append commentary on the same line
# ('WRONG: "..." Actually, the player should have ...'), and swallowing it
# into the span would fail the exact-substring verification. Quotes are
# still optional -- small models drop them -- in which case the whole rest
# of the line is the span, as before. Whitespace after the colon is
# SAME-LINE only ([^\S\n]): a bare "WRONG:" followed by a newline and then
# "SCORE:" / "RATING:" is a template leftover with no span, not a span
# whose text is "SCORE:".
WRONG_SPAN_RE = re.compile(
    r"(?im)^[^\S\n]*WRONG[^\S\n]*:[^\S\n]*"
    r"(?:\"(?P<quoted>[^\"\n]+)\"[^\n]*|(?P<span>.+?))[^\S\n]*$"
)


def parse_wrong_spans(analysis: str, source_text: str) -> dict[str, list[str]]:
    """Extract the reviewer's ``WRONG: "..."`` error spans from ``analysis``
    and verify each against ``source_text`` (the recorded player reply) by
    EXACT substring match.

    Returns ``{"verified": [...], "unverified": [...]}`` (deduplicated, in
    order of first appearance). Unverified spans are returned rather than
    dropped so callers can surface them loudly -- a span the player never
    wrote must never be silently presented as a highlight
    (no-fuzzy-fallbacks). An empty ``WRONG:`` line (no words on that line)
    is ignored: producing zero error spans is a valid analyst outcome. A
    double-quoted span keeps only the quoted words -- same-line commentary
    after the closing quote is the reviewer talking, not the span. A
    leading ``[ANALYST]`` on the WRONG line is stripped first -- the
    model emitting the persistence tag is not a parse failure."""
    verified: list[str] = []
    unverified: list[str] = []
    seen: set[str] = set()
    for m in WRONG_SPAN_RE.finditer(_without_analyst_line_tags(analysis)):
        span = m.group("quoted") or m.group("span") or ""
        span = span.strip().strip("\"'").strip()
        if not span or span in seen:
            continue
        seen.add(span)
        (verified if span in source_text else unverified).append(span)
    return {"verified": verified, "unverified": unverified}


# The final verdict line requested by _BLOCK_RATING ("RATING: <number>").
# Tolerates markdown bold around the label -- both "**RATING**: x" and
# "**RATING:** x" -- and an optional sign; the number itself must be a plain
# decimal, anything else is a parse failure, not a guess
# (no-fuzzy-fallbacks).
RATING_RE = re.compile(
    r"(?im)^[^\S\n]*\**[^\S\n]*RATING[^\S\n]*\**[^\S\n]*:[^\S\n]*\**[^\S\n]*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?)"
)


def parse_rating(analysis: str) -> float | None:
    """Extract the analyst's final ``RATING: <number>`` from ``analysis``,
    clamped to [-1.0, 1.0]. The LAST match wins (the closing verdict; earlier
    matches are quotes or revisions). Returns None when no rating line is
    present -- callers decide how loudly to fail; never guess a number.
    A leading ``[ANALYST]`` on the rating line is stripped first -- the
    model emitting the persistence tag is not a parse failure."""
    matches = RATING_RE.findall(_without_analyst_line_tags(analysis))
    if not matches:
        return None
    return max(-1.0, min(1.0, float(matches[-1])))


# Analyst TARGET line: "TARGET: gold, 0" / "TARGET: openings, 1" /
# "TARGET: none". Last match wins. Index is 0-based into that settings
# JSON list. Anything else (missing line, bad kind, non-integer) is
# None -- never guess an object (no-fuzzy-fallbacks). The model sometimes
# opens its reply with an "Analyst:" self-label and puts the TARGET line
# right after it on the same line -- an optional leading label (with or
# without markdown bold) is accepted, still anchored to line start so
# mid-prose mentions never match.
TARGET_RE = re.compile(
    r"(?im)^[^\S\n]*\**[^\S\n]*"
    r"(?:ANALYST[^\S\n]*\**[^\S\n]*:[^\S\n]*\**[^\S\n]*)?"
    r"TARGET[^\S\n]*\**[^\S\n]*:[^\S\n]*\**[^\S\n]*"
    r"(?:"
    r"(?P<none>none)"
    r"|"
    r"(?P<kind>gold|openings?)"
    r"[^\S\n]*,[^\S\n]*\[?(?P<index>\d+)\]?"
    r")"
    r"[^\S\n]*$"
)


def parse_target(analysis: str) -> dict[str, Any] | None:
    """Extract the analyst's ``TARGET: gold|openings, <index>`` (or
    ``TARGET: none``) line. LAST match wins. Returns
    ``{"kind": "gold"|"openings"|"none", "index": int|None}`` or None
    when no well-formed line is present. ``opening`` (singular) is
    accepted as the JSON key ``openings``; any other kind is a parse
    failure. A leading ``[ANALYST]`` tag or an ``Analyst:`` self-label
    on the line is tolerated -- neither is a parse failure."""
    matches = list(TARGET_RE.finditer(_without_analyst_line_tags(analysis)))
    if not matches:
        return None
    m = matches[-1]
    if m.group("none"):
        return {"kind": "none", "index": None}
    kind = m.group("kind").lower()
    if kind == "opening":
        kind = "openings"
    return {"kind": kind, "index": int(m.group("index"))}


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
    move_m = game_io.first_valid_move_match(raw)
    if search_m and (move_m is None or search_m.start() < move_m.start()):
        query = _clean_search_query(search_m.group("search_query"))
        if query:
            return "search", query, raw[: search_m.end()]
    if move_m:
        # parse_move takes the LAST valid token; generation stops at the
        # first valid one, so they agree on a well-stopped reply.
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
    client: Any, trace: Any, thought: str, query: str, results: str,
    *, privileged: bool = False,
) -> None:
    """Record one [SEARCH] call on the turn's reasoning trace: the reply text
    that requested it as the step's thought, the query as the tool arguments.

    ``privileged=True`` tags every line of ``thought`` with ``[ANALYST]`` so
    player-facing retrieval cannot surface analyst SEARCH reasoning.
    """
    if trace is None:
        return
    if privileged:
        thought = mem.tag_analyst_text(thought)
    step = await client.reasoning.add_step(trace.id, thought=thought)
    await client.reasoning.record_tool_call(
        step.id, "SEARCH", {"query": query}, result={"result_chars": len(results)}
    )


def parse_tip_line(text: str) -> str | None:
    """The tip wording from the last 'TIP: <one line>' line in ``text``, or
    ``None`` if the reply carries no such line. A leading ``[ANALYST]``
    on the TIP line is stripped first."""
    matches = TIP_LINE_RE.findall(_without_analyst_line_tags(text))
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
    *,
    system_prompt: str,
) -> list[dict]:
    """Assemble one debrief generation's prompt as at most three content
    parts. Chat templates (Gemma's among them) may concatenate adjacent text
    parts with NO separator, so logical blocks are pre-joined with explicit
    newlines here instead of being emitted as separate parts:

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
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


async def persist_debrief_verdict(
    client: Any,
    play_session_id: str,
    debrief_session_id: str,
    model: Any,
    cfg: AgentConfig | None = None,
    *,
    system_prompt: str,
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
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": (
                    f"Session id: {play_session_id}\n\n"
                    "# Debrief conversation\n" + transcript + "\n\n"
                    + DEFAULT_ANALYST_QUESTION
                ),
            },
        ]},
    ]
    verdict = model.generate(messages, max_new_tokens=cfg.max_new_tokens)

    tagged_verdict = mem.tag_analyst_text(verdict)
    eval_msg = await client.short_term.add_message(
        session_id=play_session_id, role="assistant", content=tagged_verdict,
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
    step = await client.reasoning.add_step(trace.id, thought=tagged_verdict)
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

# The old reviewer blocks live in modes_legacy_analyst. Re-exported so
# existing imports (debrief, self-eval, t1) keep working. Importing this
# module first is what loads them; do not import the archive module first.
from .modes_legacy_analyst import *  # noqa: E402,F401,F403
from .modes_legacy_analyst import (  # noqa: E402
    CORE_ANALYST_TIPS,
    DEBRIEF_EXCLUDE,
    ROLE_DEBRIEF,
    ROLE_SCENE_ANALYST,
    SCENE_ANALYST_EXCLUDE,
    SYSTEM_PROMPT_S2_ANALYST,
    _BLOCK_AIM_TOLERANCE_REVIEW,
    _BLOCK_DEBRIEF_NAV,
    _BLOCK_DEBRIEF_RECORD,
    _BLOCK_END_GAME_GRADING,
    _BLOCK_GEOMETRY_PRIVILEGED,
    _BLOCK_GRADING_TOLERANCE,
    _BLOCK_OPENINGS_PRIVILEGED,
    _BLOCK_PRIVILEGED_VIEW,
    _BLOCK_RATING,
    _BLOCK_REVIEW_WHOLE_REPLY,
    _BLOCK_REVIEWER_STANCE,
    _BLOCK_S2_ANALYST,
    _BLOCK_SCENE_ANALYST_SCOPE,
    _BLOCK_TARGET_GRADING,
    _BLOCK_TARGET_LINE,
    _BLOCK_TIP_TOOL,
    _BLOCK_TRUST_THE_MATH,
    _BLOCK_TURN_COUNT_GRADING,
)

