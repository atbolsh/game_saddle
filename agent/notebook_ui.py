"""Small ipywidgets UI helpers shared by the notebooks.

- :func:`tame_shift_enter` makes Shift+Enter behave exactly like Enter
  (insert a newline) inside notebook text boxes instead of reaching Jupyter's
  run-cell shortcut -- which would re-run the widget cell and erase whatever
  the user had typed. Buttons remain the only submit mechanism.
- :func:`model_picker` is the architecture dropdown + checkpoint dropdown +
  "save only one set of weights at a time" checkbox + Switch button shown at
  the top of every notebook, wired to the session's ``switch_model``. The
  checkpoint dropdown starts at "[default]" (bare HuggingFace weights) and
  lists every trained adapter found under ``weights/<architecture>/`` (see
  training/TRAINING_OVERVIEW.md), rescanned whenever the architecture
  changes.
- :func:`player_takeover_controls` is the sticky human-player takeover
  used by play and multi-gold self-eval (reply box + Submit +
  pictographic move buttons).
- :func:`room_scenario_bar`, :func:`live_board_row`, and
  :class:`UiBusy` are the shared gold/opening bar, floating
  frame+settings+scratchpad editors, and generating/editing lock.
"""

from __future__ import annotations

import base64
import html as _html
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import ipywidgets as widgets
from IPython.display import HTML, display

from agent.game_io import ACTIONS, opening_axis_center, parse_notepad_edit

#: CSS class marking a Textarea widget as Shift-Enter-tamed.
_TAMED_CLASS = "tame-shift-enter"

# WHY window + capture: JupyterLab / Notebook 7 dispatches keyboard shortcuts
# (including Shift-Enter = run cell) from a keydown listener on `document` in
# the CAPTURE phase. Capture runs top-down (window -> document -> ... ->
# textarea), so a listener on the textarea itself -- capture or bubble --
# always fires AFTER Jupyter's and cannot stop the run-cell command. A capture
# listener on `window` is the only DOM node upstream of `document`, so it
# preempts Jupyter reliably. One global listener (guarded against rebinding on
# cell re-run) handles every tamed textarea; no per-widget binding or polling
# is needed because the check happens per-event on the event's target.
_SCRIPT = """
<script>
(function () {
  if (window.__tameShiftEnterBound) { return; }
  window.__tameShiftEnterBound = true;
  window.addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter" || !ev.shiftKey) { return; }
    var ta = ev.target;
    if (!ta || ta.tagName !== "TEXTAREA" || !ta.closest(".__CLASS__")) { return; }
    ev.preventDefault();
    ev.stopPropagation();
    ev.stopImmediatePropagation();
    var start = ta.selectionStart, end = ta.selectionEnd;
    ta.value = ta.value.slice(0, start) + "\\n" + ta.value.slice(end);
    ta.selectionStart = ta.selectionEnd = start + 1;
    // Bubbling input event so ipywidgets syncs the value to the kernel.
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }, true);
})();
</script>
""".replace("__CLASS__", _TAMED_CLASS)


#: Sentinel label for "no trained checkpoint, bare HF weights".
_DEFAULT_CKPT_LABEL = "[default]"


def model_picker(
    session: Any,
    on_switched: Callable[[dict[str, Any]], None] | None = None,
) -> widgets.VBox:
    """The shared model-switching panel (display it at the TOP of a notebook's
    control cell).

    Two dropdowns. **Architecture** lists every
    ``agent.model.MODEL_REGISTRY`` entry in recommendation order.
    **Checkpoint** starts at "[default]" (bare HuggingFace weights) followed
    by every trained adapter under ``weights/<architecture>/`` (newest
    first); the list is rescanned from disk whenever the architecture
    selection changes and after every switch, so checkpoints saved by a
    training/train.py run appear without re-running the cell. Switching is an
    explicit button press (a dropdown misclick must never start a multi-GB
    download). The checkbox implements "save only one set of weights at a
    time": when checked, a switch first restarts the conversation, then
    deletes every OTHER registry model's cached HF weights before
    downloading the new ones (adapter checkpoints under weights/ are never
    purged); when unchecked, the conversation continues under the new model
    and old weights stay cached.

    ``on_switched(info)`` (if given) fires after a successful switch so the
    notebook can refresh its own view; ``info["restarted"]`` says whether the
    conversation was restarted.
    """
    from .model import MODEL_REGISTRY, list_checkpoints

    current = (
        session.model.spec.key if session.model is not None
        else session.cfg.model_key
    )
    current_ckpt = (
        session.model.checkpoint if session.model is not None
        else session.cfg.model_checkpoint
    )
    arch_dropdown = widgets.Dropdown(
        options=[(spec.label, key) for key, spec in MODEL_REGISTRY.items()],
        value=current if current in MODEL_REGISTRY else None,
        description="Architecture:",
        layout=widgets.Layout(width="460px"),
    )

    def _ckpt_options(arch_key: str | None) -> list[tuple[str, str | None]]:
        opts: list[tuple[str, str | None]] = [(_DEFAULT_CKPT_LABEL, None)]
        if arch_key is not None:
            opts += [(name, name) for name in list_checkpoints(arch_key)]
        return opts

    ckpt_dropdown = widgets.Dropdown(
        options=_ckpt_options(arch_dropdown.value),
        description="Checkpoint:",
        layout=widgets.Layout(width="460px"),
    )
    ckpt_values = [v for _, v in ckpt_dropdown.options]
    ckpt_dropdown.value = current_ckpt if current_ckpt in ckpt_values else None

    def _refresh_ckpts(*_):
        """Rescan weights/<arch>/ from disk; keep the selection if it still
        exists, else fall back to [default]."""
        selected = ckpt_dropdown.value
        ckpt_dropdown.options = _ckpt_options(arch_dropdown.value)
        values = [v for _, v in ckpt_dropdown.options]
        ckpt_dropdown.value = selected if selected in values else None

    arch_dropdown.observe(_refresh_ckpts, names="value")

    one_copy = widgets.Checkbox(
        value=False,
        indent=False,
        description="Save only one set of weights at a time "
                    "(switching restarts the conversation and deletes the "
                    "other cached weights)",
        layout=widgets.Layout(width="640px"),
    )
    switch_btn = widgets.Button(description="Switch model", button_style="warning")
    status = widgets.Output()

    def _on_switch(_):
        key = arch_dropdown.value
        ckpt = ckpt_dropdown.value
        if key is None:
            return
        already = (
            session.model is not None
            and key == session.model.spec.key
            and ckpt == session.model.checkpoint
        )
        if already and not one_copy.value:
            with status:
                status.clear_output()
                print(f"'{key}' + '{ckpt or _DEFAULT_CKPT_LABEL}' is already "
                      "the loaded model.")
            return
        controls = (switch_btn, arch_dropdown, ckpt_dropdown, one_copy)
        for c in controls:
            c.disabled = True
        try:
            with status:
                status.clear_output()
                spec = MODEL_REGISTRY[key]
                if one_copy.value:
                    print("[one-weights mode] restarting the conversation and "
                          "purging other cached weights ...")
                print(f"Switching to {spec.label} ({spec.hf_id})"
                      + (f" + checkpoint '{ckpt}'" if ckpt else "")
                      + "; first use downloads the weights -- this can take "
                        "a while ...")
                info = session.switch_model(
                    key, purge_others=one_copy.value, checkpoint=ckpt
                )
                purge = info.get("purge") or {}
                if purge.get("purged"):
                    print(f"Purged {len(purge['purged'])} cached repo(s), "
                          f"freed {purge['freed_bytes'] / 1e9:.1f} GB: "
                          + ", ".join(purge["purged"]))
                elif one_copy.value:
                    print("No other registry weights were cached; nothing to purge.")
                print(f"Model ready: {info['label']}"
                      + (f" [{ckpt}]" if ckpt else " [default weights]")
                      + ("  [conversation restarted]" if info["restarted"] else ""))
            if on_switched is not None:
                on_switched(info)
        finally:
            for c in controls:
                c.disabled = False
            _refresh_ckpts()

    switch_btn.on_click(_on_switch)
    return widgets.VBox([
        widgets.HBox([arch_dropdown, switch_btn]),
        ckpt_dropdown,
        one_copy,
        status,
    ])


#: Visible faces for the takeover move buttons. The appended token is
#: still ``[FORWARD]`` / ``[CLOCK]`` / ``[ANTICLOCK]``; only the label
#: is a glyph. Order is the play order (step, then the two turns).
_TAKEOVER_MOVE_FACES = (
    ("→", "FORWARD"),
    ("↻", "CLOCK"),
    ("↺", "ANTICLOCK"),
)


def player_takeover_controls(
    on_submit: Callable[[str], None],
    on_mode_change: Callable[[], None] | None = None,
) -> SimpleNamespace:
    """Sticky human-player takeover panel for play and multi-gold self-eval.

    ``on_submit(raw)`` receives the reply-box text, with ``\\n\\n[TOKEN]``
    appended when a move button was pressed. Truncation at the first
    move token happens in ``ask_player``. Takeover stays on until
    **Resume agent**; Restart / New room / Reset do not clear it.

    ``on_mode_change`` (if given) fires after **Takeover** and **Resume
    agent**. The notebooks pass ``_sync_phase`` here: that is the only
    place that re-enables **Ask**, and Resume/Takeover never go through
    it on their own. Without this callback, Ask stays grey after
    Takeover -> move -> Back to player -> Resume agent.

    Returns a namespace with ``takeover_btn``, ``resume_btn``,
    ``reply_box``, ``controls_box`` (reply + submit/move row; hidden
    until takeover), ``toggle_box`` (the two mode buttons), and
    ``sync(player_on)`` to enable/disable with the player phase.
    """
    takeover_on = False
    last_player_on = False
    takeover_btn = widgets.Button(
        description="Takeover", button_style="warning",
        tooltip="Drive the player yourself (no generation)",
    )
    resume_btn = widgets.Button(
        description="Resume agent", button_style="info",
        tooltip="Give the player turn back to the model",
    )
    resume_btn.layout.display = "none"
    reply_box = widgets.Textarea(
        value="",
        placeholder="Write the player's reply (or just press a move button)...",
        description="Player:",
        layout=widgets.Layout(width="600px", height="90px"),
    )
    submit_btn = widgets.Button(description="Submit", button_style="primary")
    move_btns: list[widgets.Button] = []
    for face, action in _TAKEOVER_MOVE_FACES:
        if action not in ACTIONS:
            raise ValueError(f"takeover face {action!r} is not a game action")
        btn = widgets.Button(
            description=face,
            tooltip=action,
            layout=widgets.Layout(width="48px"),
        )
        move_btns.append(btn)

    controls_box = widgets.VBox([
        reply_box,
        widgets.HBox([submit_btn, *move_btns]),
    ])
    controls_box.layout.display = "none"
    toggle_box = widgets.HBox([takeover_btn, resume_btn])

    def _apply_visibility() -> None:
        controls_box.layout.display = None if takeover_on else "none"
        takeover_btn.layout.display = "none" if takeover_on else None
        resume_btn.layout.display = None if takeover_on else "none"

    def sync(player_on: bool) -> None:
        nonlocal last_player_on
        last_player_on = player_on
        takeover_btn.disabled = not player_on
        resume_btn.disabled = False
        live = player_on and takeover_on
        reply_box.disabled = not live
        submit_btn.disabled = not live
        for btn in move_btns:
            btn.disabled = not live

    def _set_takeover(on: bool) -> None:
        nonlocal takeover_on
        takeover_on = on
        _apply_visibility()

    def _notify() -> None:
        # Always refresh this panel first; then let the notebook re-sync
        # Ask / Reset / etc. Resume does not otherwise call _sync_phase.
        sync(last_player_on)
        if on_mode_change is not None:
            on_mode_change()

    def _on_takeover(_) -> None:
        _set_takeover(True)
        _notify()

    def _on_resume(_) -> None:
        _set_takeover(False)
        reply_box.value = ""
        _notify()

    def _fire(extra_token: str | None) -> None:
        raw = reply_box.value
        if extra_token:
            raw = raw + "\n\n" + extra_token
        if not raw.strip():
            return
        on_submit(raw)
        reply_box.value = ""

    takeover_btn.on_click(_on_takeover)
    resume_btn.on_click(_on_resume)
    submit_btn.on_click(lambda _: _fire(None))
    for btn, (_face, action) in zip(move_btns, _TAKEOVER_MOVE_FACES):
        token = f"[{action}]"
        btn.on_click(lambda _, tok=token: _fire(tok))

    return SimpleNamespace(
        takeover_btn=takeover_btn,
        resume_btn=resume_btn,
        reply_box=reply_box,
        submit_btn=submit_btn,
        move_btns=move_btns,
        controls_box=controls_box,
        toggle_box=toggle_box,
        sync=sync,
        is_on=lambda: takeover_on,
    )


def scratchpad_html(text: str, *, height_px: int = 420) -> str:
    """Light-maroon notepad card, game-height and wide; scrolls if taller."""
    escaped = _html.escape(text or "")
    return (
        f"<div style='flex:1 1 640px;min-width:520px;height:{height_px}px;"
        "overflow:auto;background:#f7e6e6;border:1px solid #c69c9c;"
        "border-radius:6px;padding:10px 12px;box-sizing:border-box;"
        "font-family:monospace;white-space:pre-wrap;font-size:13px'>"
        "<div style='font-weight:bold;color:#7a2e2e;margin-bottom:6px'>"
        "Player's scratchpad</div>"
        f"{escaped}</div>"
    )


def display_frame_with_scratchpad(
    path: str | None,
    caption: str,
    notepad: str,
    *,
    width: int = 420,
) -> None:
    """Print ``caption``, then the game image with the notepad to its right."""
    if not path:
        return
    print(caption)
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = Path(path).suffix.lstrip(".").lower() or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    display(HTML(
        "<div style='display:flex;align-items:flex-start;gap:16px;width:100%'>"
        f"<img src='data:image/{suffix};base64,{b64}' width='{width}' "
        f"style='flex:0 0 {width}px;height:auto;display:block'/>"
        f"{scratchpad_html(notepad, height_px=width)}"
        "</div>"
    ))


def tame_shift_enter(*text_widgets) -> None:
    """Make Shift+Enter insert a plain newline in the given Textarea widgets.

    Shift+Enter must behave exactly like Enter inside the box: it does NOT
    submit anything, and it must not reach Jupyter's run-cell shortcut (which
    re-runs the widget cell and erases the input). Each widget is tagged with
    a marker CSS class, and one injected window-level capture-phase keydown
    listener intercepts Shift+Enter on any tagged textarea before Jupyter's
    own document-level shortcut handler can see it.

    Call AFTER creating the widgets, in the same cell that displays them.
    Safe to call repeatedly (cell re-runs): the listener binds once per page.
    """
    for w in text_widgets:
        w.add_class(_TAMED_CLASS)
    display(HTML(_SCRIPT))


class UiBusy:
    """Generating lock shared by play and multi-gold notebooks.

    ``generating`` disables Edit / Ask / Analyze / New room / model switch.
    Scratchpad edit mode or a dirty settings form disables Ask / Analyze.
    Notebooks call :meth:`set_on_change` with their ``_sync_phase``.
    """

    def __init__(self) -> None:
        self.generating = False
        self._on_change: Callable[[], None] | None = None

    def set_on_change(self, fn: Callable[[], None]) -> None:
        self._on_change = fn

    def set_generating(self, on: bool) -> None:
        self.generating = bool(on)
        if self._on_change is not None:
            self._on_change()


def room_scenario_bar() -> SimpleNamespace:
    """Gold-count + opening dropdowns, New room, End game."""
    gold_dd = widgets.Dropdown(
        options=[("random", None), ("0", 0), ("1", 1), ("2", 2), ("3", 3)],
        value=None,
        description="Golds:",
        layout=widgets.Layout(width="220px"),
    )
    opening_dd = widgets.Dropdown(
        options=[("require", "require"), ("forbid", "forbid"),
                 ("random", "any")],
        value="require",
        description="Opening:",
        layout=widgets.Layout(width="220px"),
    )
    new_room_btn = widgets.Button(description="New room", button_style="primary")
    end_btn = widgets.Button(description="End game", button_style="danger")
    return SimpleNamespace(
        gold_dd=gold_dd,
        opening_dd=opening_dd,
        new_room_btn=new_room_btn,
        end_btn=end_btn,
        box=widgets.HBox([gold_dd, opening_dd, new_room_btn, end_btn]),
    )


def _card_html(
    body: str,
    *,
    title: str,
    title_color: str,
    bg: str,
    border: str,
    height_px: int,
    hint: str = "",
) -> str:
    escaped = _html.escape(body or "")
    hint_html = (
        f"<div style='font-size:11px;color:#555;margin-bottom:6px'>"
        f"{_html.escape(hint)}</div>"
        if hint else ""
    )
    return (
        f"<div style='height:{height_px}px;overflow:auto;background:{bg};"
        f"border:1px solid {border};border-radius:6px;padding:10px 12px;"
        "box-sizing:border-box;font-family:monospace;white-space:pre-wrap;"
        "font-size:13px'>"
        f"<div style='font-weight:bold;color:{title_color};margin-bottom:4px'>"
        f"{_html.escape(title)}</div>"
        f"{hint_html}{escaped}</div>"
    )


def _editor_card(
    *,
    title: str,
    title_color: str,
    bg: str,
    border: str,
    height_px: int,
    hint: str,
    edit_label: str,
    render_label: str,
) -> SimpleNamespace:
    """View HTML + labeled Edit/Render buttons on top (always visible).

    Edit does not clobber in-progress text. Render stays enabled whenever
    generation is not running -- it applies the textarea if the user has
    entered edit mode, otherwise the current live JSON.
    """
    edit_btn = widgets.Button(
        description=edit_label,
        tooltip=edit_label,
        layout=widgets.Layout(width="auto", min_width="160px"),
    )
    render_btn = widgets.Button(
        description=render_label,
        button_style="success",
        tooltip=render_label,
        layout=widgets.Layout(width="auto", min_width="180px"),
    )
    view = widgets.HTML()
    textarea = widgets.Textarea(
        value="",
        layout=widgets.Layout(width="100%", height=f"{max(height_px, 160)}px"),
    )
    textarea.layout.display = "none"
    textarea.add_class(_TAMED_CLASS)
    status = widgets.HTML()
    editing = False

    def _set_view(text: str) -> None:
        view.value = _card_html(
            text, title=title, title_color=title_color, bg=bg,
            border=border, height_px=height_px, hint=hint,
        )

    def is_editing() -> bool:
        return editing

    def enter_edit(edit_text: str) -> None:
        nonlocal editing
        if not editing:
            textarea.value = edit_text
        editing = True
        view.layout.display = "none"
        textarea.layout.display = ""
        status.value = ""

    def show_view(view_text: str) -> None:
        nonlocal editing
        editing = False
        textarea.layout.display = "none"
        view.layout.display = ""
        _set_view(view_text)
        status.value = ""

    def set_error(msg: str) -> None:
        status.value = (
            "<div style='color:#a00;font-family:monospace;white-space:pre-wrap'>"
            f"{_html.escape(msg)}</div>"
        )

    def set_disabled(on: bool) -> None:
        edit_btn.disabled = on
        render_btn.disabled = on

    def current_edit_text() -> str:
        return textarea.value

    box = widgets.VBox([
        widgets.HBox([edit_btn, render_btn]),
        status,
        view,
        textarea,
    ], layout=widgets.Layout(
        flex="1 1 380px",
        min_width="340px",
        border=f"2px solid {border}",
        padding="8px",
    ))
    return SimpleNamespace(
        view=view,
        textarea=textarea,
        edit_btn=edit_btn,
        render_btn=render_btn,
        status=status,
        box=box,
        is_editing=is_editing,
        enter_edit=enter_edit,
        show_view=show_view,
        set_error=set_error,
        set_disabled=set_disabled,
        current_edit_text=current_edit_text,
        title=title,
    )


_OPENING_SIDES = ("left", "right", "top", "bottom")
_TWO_PI = 2.0 * math.pi


def _compact_number(
    description: str,
    value: float | int,
    *,
    kind: str = "float",
    width: str = "128px",
    dw: str = "58px",
    on_change: Callable[[Any], None] | None = None,
) -> Any:
    cls = widgets.IntText if kind == "int" else widgets.FloatText
    w = cls(
        value=value,
        description=description,
        step=1 if kind == "int" else 0.01,
        layout=widgets.Layout(width=width),
        style={"description_width": dw},
    )
    if on_change is not None:
        w.observe(on_change, names="value")
    return w


def _settings_card(*, height_px: int, on_form_change: Callable[[], None] | None = None) -> SimpleNamespace:
    """Always-visible teal form for the full settings dict."""
    edit_btn = widgets.Button(
        description="Reload from board",
        tooltip="Discard unrendered form edits and reload from the live board",
        layout=widgets.Layout(width="auto", min_width="160px"),
    )
    render_btn = widgets.Button(
        description="Render game settings",
        button_style="success",
        tooltip="Apply the full settings form to the live board",
        layout=widgets.Layout(width="auto", min_width="180px"),
    )
    hint = widgets.HTML(
        value=(
            "<div style='font-size:11px;color:#0d5c5c;margin:4px 0 8px'>"
            "<b>Game settings</b> — same fields as the old JSON, as widgets. "
            "Direction is 0…2π. Check a side to cut an opening; "
            "<b>center</b> is one number along that edge (y on left/right, "
            "x on top/bottom); <b>width</b> is the gap. from/to are computed "
            "on Render. Boundary-band walls lose to openings; interior walls, "
            "gold, pose, and radii are taken as written."
            "</div>"
        )
    )
    status = widgets.HTML()
    dir_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=_TWO_PI,
        step=0.01,
        description="direction",
        continuous_update=True,
        readout=True,
        readout_format=".3f",
        layout=widgets.Layout(width="98%"),
        style={"description_width": "80px"},
    )
    dir_readout = widgets.HTML()

    sides: dict[str, SimpleNamespace] = {}
    gold_rows: list[SimpleNamespace] = []
    wall_rows: list[SimpleNamespace] = []
    loading = False
    applied: tuple[Any, ...] | None = None

    def _dir_label(rad: float) -> str:
        return (
            "<div style='font-size:11px;color:#555;margin:-4px 0 8px 84px'>"
            f"{rad:.3f} rad  ({rad * 180.0 / math.pi:.1f}°)  ·  0 … 2π"
            "</div>"
        )

    def _sync_side_enabled(side: str) -> None:
        on = bool(sides[side].enabled.value)
        sides[side].center.disabled = not on
        sides[side].width.disabled = not on

    def _form_snapshot() -> tuple[Any, ...]:
        return (
            int(game_size.value),
            float(dir_slider.value),
            float(agent_x.value),
            float(agent_y.value),
            float(agent_r.value),
            float(gold_r.value),
            tuple(
                (float(r.x.value), float(r.y.value)) for r in gold_rows
            ),
            tuple(
                (float(r.x.value), float(r.y.value), float(r.w.value),
                 float(r.h.value), float(r.angle.value))
                for r in wall_rows
            ),
            tuple(
                (
                    bool(sides[s].enabled.value),
                    float(sides[s].center.value),
                    float(sides[s].width.value),
                )
                for s in _OPENING_SIDES
            ),
        )

    def is_editing() -> bool:
        if applied is None:
            return False
        return _form_snapshot() != applied

    def _notify(_change: Any = None) -> None:
        if loading:
            return
        dir_readout.value = _dir_label(float(dir_slider.value))
        if on_form_change is not None:
            on_form_change()

    def _sync_gold_box() -> None:
        gold_box.children = tuple([r.box for r in gold_rows] + [add_gold_btn])

    def _sync_wall_box() -> None:
        wall_box.children = tuple([r.box for r in wall_rows] + [add_wall_btn])

    def _make_gold_row(x: float, y: float) -> SimpleNamespace:
        xw = _compact_number("x", x, width="100px", dw="16px", on_change=_notify)
        yw = _compact_number("y", y, width="100px", dw="16px", on_change=_notify)
        rm = widgets.Button(
            description="×", tooltip="Remove this gold",
            layout=widgets.Layout(width="32px"),
        )
        ns = SimpleNamespace(
            x=xw, y=yw, rm=rm,
            box=widgets.HBox([xw, yw, rm], layout=widgets.Layout(align_items="center")),
        )

        def _remove(_):
            if ns in gold_rows:
                gold_rows.remove(ns)
                _sync_gold_box()
                _notify()

        rm.on_click(_remove)
        return ns

    def _make_wall_row(
        x: float, y: float, w: float, h: float, angle: float,
    ) -> SimpleNamespace:
        xw = _compact_number("x", x, width="88px", dw="14px", on_change=_notify)
        yw = _compact_number("y", y, width="88px", dw="14px", on_change=_notify)
        ww = _compact_number("w", w, width="88px", dw="14px", on_change=_notify)
        hw = _compact_number("h", h, width="88px", dw="14px", on_change=_notify)
        aw = _compact_number("θ", angle, width="88px", dw="14px", on_change=_notify)
        rm = widgets.Button(
            description="×", tooltip="Remove this wall",
            layout=widgets.Layout(width="32px"),
        )
        ns = SimpleNamespace(
            x=xw, y=yw, w=ww, h=hw, angle=aw, rm=rm,
            box=widgets.HBox(
                [xw, yw, ww, hw, aw, rm],
                layout=widgets.Layout(align_items="center"),
            ),
        )

        def _remove(_):
            if ns in wall_rows:
                wall_rows.remove(ns)
                _sync_wall_box()
                _notify()

        rm.on_click(_remove)
        return ns

    def load_form(d: dict[str, Any]) -> None:
        nonlocal loading, applied
        loading = True
        extras: list[str] = []
        try:
            game_size.value = int(d.get("gameSize", 64))
            val = float(d.get("direction", 0.0))
            if val < 0.0 or val > _TWO_PI:
                val = val % _TWO_PI
            dir_slider.value = val
            dir_readout.value = _dir_label(val)
            agent_x.value = float(d.get("agent_x", 0.5))
            agent_y.value = float(d.get("agent_y", 0.5))
            agent_r.value = float(d.get("agent_r", 0.05))
            gold_r.value = float(d.get("gold_r", 0.03))
            gold_rows[:] = []
            for g in d.get("gold") or []:
                if not isinstance(g, (list, tuple)) or len(g) < 2:
                    raise ValueError(f"gold entry {g!r} is not [x, y]")
                gold_rows.append(_make_gold_row(float(g[0]), float(g[1])))
            _sync_gold_box()
            wall_rows[:] = []
            for wall in d.get("walls") or []:
                if not isinstance(wall, (list, tuple)) or len(wall) < 5:
                    raise ValueError(
                        f"wall {wall!r} is not [x, y, w, h, angle]"
                    )
                wall_rows.append(_make_wall_row(
                    float(wall[0]), float(wall[1]), float(wall[2]),
                    float(wall[3]), float(wall[4]),
                ))
            _sync_wall_box()
            used: set[str] = set()
            for op in d.get("openings") or []:
                side = op.get("side")
                if side not in sides:
                    continue
                if side in used:
                    extras.append(side)
                    continue
                used.add(side)
                sides[side].enabled.value = True
                sides[side].center.value = opening_axis_center(op)
                sides[side].width.value = float(op["width"])
            for side in _OPENING_SIDES:
                if side not in used:
                    sides[side].enabled.value = False
                    sides[side].center.value = 0.5
                    sides[side].width.value = 0.2
                _sync_side_enabled(side)
            applied = _form_snapshot()
        finally:
            loading = False
        if extras:
            status.value = (
                "<div style='color:#864;font-family:monospace'>"
                "This board has more than one opening on "
                f"{', '.join(extras)}; the form keeps the first. "
                "Render will replace that side with the form row."
                "</div>"
            )
        else:
            status.value = ""

    def collect(_base: dict[str, Any] | None = None) -> dict[str, Any]:
        openings: list[dict[str, Any]] = []
        for side in _OPENING_SIDES:
            if not sides[side].enabled.value:
                continue
            openings.append({
                "side": side,
                "center": float(sides[side].center.value),
                "width": float(sides[side].width.value),
            })
        return {
            "gameSize": int(game_size.value),
            "direction": float(dir_slider.value),
            "agent_x": float(agent_x.value),
            "agent_y": float(agent_y.value),
            "agent_r": float(agent_r.value),
            "gold_r": float(gold_r.value),
            "gold": [
                [float(r.x.value), float(r.y.value)] for r in gold_rows
            ],
            "walls": [
                [float(r.x.value), float(r.y.value), float(r.w.value),
                 float(r.h.value), float(r.angle.value)]
                for r in wall_rows
            ],
            "openings": openings,
        }

    def set_error(msg: str) -> None:
        status.value = (
            "<div style='color:#a00;font-family:monospace;white-space:pre-wrap'>"
            f"{_html.escape(msg)}</div>"
        )

    def set_disabled(on: bool) -> None:
        edit_btn.disabled = on
        render_btn.disabled = on
        add_gold_btn.disabled = on
        add_wall_btn.disabled = on
        for w in (game_size, dir_slider, agent_x, agent_y, agent_r, gold_r):
            w.disabled = on
        for r in gold_rows:
            r.x.disabled = on
            r.y.disabled = on
            r.rm.disabled = on
        for r in wall_rows:
            r.x.disabled = on
            r.y.disabled = on
            r.w.disabled = on
            r.h.disabled = on
            r.angle.disabled = on
            r.rm.disabled = on
        for side in _OPENING_SIDES:
            sides[side].enabled.disabled = on
            if on:
                sides[side].center.disabled = True
                sides[side].width.disabled = True
            else:
                _sync_side_enabled(side)

    game_size = _compact_number(
        "gameSize", 64, kind="int", width="140px", dw="68px", on_change=_notify,
    )
    agent_x = _compact_number("agent_x", 0.5, on_change=_notify)
    agent_y = _compact_number("agent_y", 0.5, on_change=_notify)
    agent_r = _compact_number("agent_r", 0.05, on_change=_notify)
    gold_r = _compact_number("gold_r", 0.03, on_change=_notify)

    for side in _OPENING_SIDES:
        enabled = widgets.Checkbox(
            value=False,
            description=side,
            indent=False,
            layout=widgets.Layout(width="90px"),
        )
        center = widgets.FloatText(
            value=0.5,
            description="center",
            step=0.01,
            layout=widgets.Layout(width="150px"),
            style={"description_width": "52px"},
        )
        width = widgets.FloatText(
            value=0.2,
            description="width",
            step=0.01,
            layout=widgets.Layout(width="140px"),
            style={"description_width": "46px"},
        )
        enabled.observe(lambda change, s=side: _sync_side_enabled(s), names="value")
        enabled.observe(_notify, names="value")
        center.observe(_notify, names="value")
        width.observe(_notify, names="value")
        sides[side] = SimpleNamespace(
            enabled=enabled, center=center, width=width,
            row=widgets.HBox(
                [enabled, center, width],
                layout=widgets.Layout(align_items="center"),
            ),
        )
        _sync_side_enabled(side)
    dir_slider.observe(_notify, names="value")
    dir_readout.value = _dir_label(0.0)

    add_gold_btn = widgets.Button(
        description="Add gold",
        layout=widgets.Layout(width="auto", min_width="90px"),
    )
    add_wall_btn = widgets.Button(
        description="Add wall",
        layout=widgets.Layout(width="auto", min_width="90px"),
    )

    def _on_add_gold(_):
        gold_rows.append(_make_gold_row(0.5, 0.5))
        _sync_gold_box()
        _notify()

    def _on_add_wall(_):
        wall_rows.append(_make_wall_row(0.4, 0.4, 0.2, 0.2, 0.0))
        _sync_wall_box()
        _notify()

    add_gold_btn.on_click(_on_add_gold)
    add_wall_btn.on_click(_on_add_wall)
    gold_box = widgets.VBox([add_gold_btn])
    wall_box = widgets.VBox([add_wall_btn])

    form = widgets.VBox(
        [
            widgets.HBox([game_size, agent_x, agent_y]),
            widgets.HBox([agent_r, gold_r]),
            dir_slider,
            dir_readout,
            widgets.HTML(
                "<div style='font-size:11px;font-weight:bold;margin:6px 0 2px'>"
                "openings</div>"
            ),
            *[sides[s].row for s in _OPENING_SIDES],
            widgets.HTML(
                "<div style='font-size:11px;font-weight:bold;margin:8px 0 2px'>"
                "gold [x, y]</div>"
            ),
            gold_box,
            widgets.HTML(
                "<div style='font-size:11px;font-weight:bold;margin:8px 0 2px'>"
                "walls [x, y, w, h, angle]</div>"
            ),
            wall_box,
        ],
        layout=widgets.Layout(
            height=f"{max(height_px, 320)}px",
            overflow="auto",
            padding="4px 6px",
            border="1px solid #5aa8a8",
            background="#e6f4f4",
        ),
    )
    box = widgets.VBox([
        widgets.HBox([edit_btn, render_btn]),
        hint,
        status,
        form,
    ], layout=widgets.Layout(
        flex="1 1 420px",
        min_width="400px",
        border="2px solid #5aa8a8",
        padding="8px",
    ))
    return SimpleNamespace(
        box=box,
        edit_btn=edit_btn,
        render_btn=render_btn,
        status=status,
        is_editing=is_editing,
        load_form=load_form,
        collect=collect,
        set_error=set_error,
        set_disabled=set_disabled,
        textarea=None,
    )


def live_board_row(
    session: Any,
    busy: UiBusy,
    *,
    width: int = 420,
    on_changed: Callable[[], None] | None = None,
) -> SimpleNamespace:
    """Sticky teal settings | frame | maroon scratchpad.

    Hidden from the agent until the next generation. Generation cannot
    start while the scratchpad is in edit mode or the settings form is
    dirty; Edit/Reload is disabled while ``busy.generating``.
    """
    frame = widgets.Image(format="png", width=width)
    frame.layout = widgets.Layout(width=f"{width}px", flex=f"0 0 {width}px")
    caption = widgets.HTML()
    body_h = min(width, 280)

    scratch = _editor_card(
        title="Player's scratchpad",
        title_color="#7a2e2e",
        bg="#f7e6e6",
        border="#c69c9c",
        height_px=body_h,
        hint="Edit as JSON {key: value}. Bad format stays in edit mode.",
        edit_label="Edit scratchpad",
        render_label="Render scratchpad",
    )

    def _on_settings_form_change() -> None:
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    settings = _settings_card(
        height_px=body_h, on_form_change=_on_settings_form_change,
    )

    def is_editing() -> bool:
        return scratch.is_editing() or settings.is_editing()

    def refresh(*, force_views: bool = False) -> None:
        path = session.current_frame_path()
        frame.value = Path(path).read_bytes()
        caption.value = (
            "<div style='font-family:monospace;font-size:12px;color:#444'>"
            "current board — live scene; Ask snapshots this, not the "
            "history prints below</div>"
        )
        if force_views or not scratch.is_editing():
            scratch.show_view(session.current_notepad())
        if force_views or not settings.is_editing():
            settings.load_form(session.current_settings_dict())
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    def _sync_buttons() -> None:
        blocked = busy.generating
        scratch.set_disabled(blocked)
        settings.set_disabled(blocked)

    def _on_scratch_edit(_) -> None:
        if busy.generating:
            return
        scratch.enter_edit(session.current_notepad_edit_json())
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    def _on_settings_reload(_) -> None:
        if busy.generating:
            return
        settings.load_form(session.current_settings_dict())
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    def _on_scratch_render(_) -> None:
        if busy.generating:
            return
        raw = scratch.current_edit_text().strip()
        if not raw:
            raw = session.current_notepad_edit_json()
            scratch.enter_edit(raw)
        try:
            notes = parse_notepad_edit(raw)
            session.replace_scratchpad(notes)
        except ValueError as exc:
            scratch.set_error(f"Error, bad format, fix input before saving.\n{exc}")
            return
        scratch.show_view(session.current_notepad())
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    def _on_settings_render(_) -> None:
        if busy.generating:
            return
        try:
            d = settings.collect(session.current_settings_dict())
            session.apply_user_settings(d)
        except ValueError as exc:
            settings.set_error(
                f"Error, fix settings before rendering.\n{exc}"
            )
            return
        frame.value = Path(session.current_frame_path()).read_bytes()
        settings.load_form(session.current_settings_dict())
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    scratch.edit_btn.on_click(_on_scratch_edit)
    scratch.render_btn.on_click(_on_scratch_render)
    settings.edit_btn.on_click(_on_settings_reload)
    settings.render_btn.on_click(_on_settings_render)

    box = widgets.HBox(
        [
            settings.box,
            widgets.VBox([caption, frame], layout=widgets.Layout(
                flex=f"0 0 {width}px",
            )),
            scratch.box,
        ],
        layout=widgets.Layout(width="100%", align_items="flex-start"),
    )
    refresh()
    return SimpleNamespace(
        box=box,
        frame=frame,
        scratch=scratch,
        settings=settings,
        refresh=refresh,
        is_editing=is_editing,
        sync_buttons=_sync_buttons,
    )
