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
  frame+scratchpad+settings editors, and generating/editing lock.
"""

from __future__ import annotations

import base64
import html as _html
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import ipywidgets as widgets
from IPython.display import HTML, display

from agent.game_io import ACTIONS, parse_notepad_edit, parse_settings_edit_json

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
    Either editor in edit mode disables Ask / Analyze. Notebooks call
    :meth:`set_on_change` with their ``_sync_phase``.
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
) -> SimpleNamespace:
    """View HTML + Edit/Render textarea. Edit does not clobber in-progress text."""
    view = widgets.HTML()
    textarea = widgets.Textarea(
        value="",
        layout=widgets.Layout(width="100%", height=f"{max(height_px - 36, 120)}px"),
    )
    textarea.layout.display = "none"
    textarea.add_class(_TAMED_CLASS)
    edit_btn = widgets.Button(description="Edit")
    render_btn = widgets.Button(description="Render", button_style="success")
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
        textarea.layout.display = None
        status.value = ""

    def show_view(view_text: str) -> None:
        nonlocal editing
        editing = False
        textarea.layout.display = "none"
        view.layout.display = None
        _set_view(view_text)
        status.value = ""

    def set_error(msg: str) -> None:
        status.value = (
            "<div style='color:#a00;font-family:monospace;white-space:pre-wrap'>"
            f"{_html.escape(msg)}</div>"
        )

    def set_disabled(on: bool) -> None:
        edit_btn.disabled = on
        render_btn.disabled = on or not editing

    box = widgets.VBox([
        view,
        textarea,
        widgets.HBox([edit_btn, render_btn]),
        status,
    ], layout=widgets.Layout(
        flex="1 1 360px",
        min_width="320px",
        border=f"1px solid {border}",
        padding="6px",
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
        title=title,
    )


def live_board_row(
    session: Any,
    busy: UiBusy,
    *,
    width: int = 420,
    on_changed: Callable[[], None] | None = None,
) -> SimpleNamespace:
    """Sticky frame + maroon scratchpad + teal settings (Edit/Render).

    Hidden from the agent until the next generation. Generation cannot
    start while either card is in edit mode; Edit is disabled while
    ``busy.generating``.
    """
    frame = widgets.Image(format="png", width=width)
    frame.layout = widgets.Layout(width=f"{width}px", flex=f"0 0 {width}px")
    caption = widgets.HTML()

    scratch = _editor_card(
        title="Player's scratchpad",
        title_color="#7a2e2e",
        bg="#f7e6e6",
        border="#c69c9c",
        height_px=width,
        hint="Edit as JSON {key: value}. Bad format stays in edit mode.",
    )
    settings = _editor_card(
        title="Game settings",
        title_color="#0d5c5c",
        bg="#e6f4f4",
        border="#5aa8a8",
        height_px=width,
        hint=(
            "To change the boundary, edit openings. Boundary-band walls "
            "lose if they conflict. Interior walls, gold, and pose are "
            "taken as written. Render strips openings and rebuilds walls."
        ),
    )

    def is_editing() -> bool:
        return scratch.is_editing() or settings.is_editing()

    def refresh(*, force_views: bool = False) -> None:
        path = session.current_frame_path()
        frame.value = Path(path).read_bytes()
        caption.value = (
            "<div style='font-family:monospace;font-size:12px;color:#444'>"
            "current board</div>"
        )
        if force_views or not scratch.is_editing():
            scratch.show_view(session.current_notepad())
        if force_views or not settings.is_editing():
            settings.show_view(
                json.dumps(session.current_settings_dict(), indent=2)
            )
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

    def _on_settings_edit(_) -> None:
        if busy.generating:
            return
        settings.enter_edit(
            json.dumps(session.current_settings_dict(), indent=2)
        )
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    def _on_scratch_render(_) -> None:
        if busy.generating or not scratch.is_editing():
            return
        try:
            notes = parse_notepad_edit(scratch.textarea.value)
            session.replace_scratchpad(notes)
        except ValueError as exc:
            scratch.set_error(f"Error, bad format, fix input before saving.\n{exc}")
            return
        scratch.show_view(session.current_notepad())
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    def _on_settings_render(_) -> None:
        if busy.generating or not settings.is_editing():
            return
        try:
            d = parse_settings_edit_json(settings.textarea.value)
            session.apply_user_settings(d)
        except ValueError as exc:
            settings.set_error(
                f"Error, bad format, fix input before saving.\n{exc}"
            )
            return
        frame.value = Path(session.current_frame_path()).read_bytes()
        settings.show_view(
            json.dumps(session.current_settings_dict(), indent=2)
        )
        _sync_buttons()
        if on_changed is not None:
            on_changed()

    scratch.edit_btn.on_click(_on_scratch_edit)
    scratch.render_btn.on_click(_on_scratch_render)
    settings.edit_btn.on_click(_on_settings_edit)
    settings.render_btn.on_click(_on_settings_render)

    box = widgets.HBox(
        [
            widgets.VBox([caption, frame], layout=widgets.Layout(
                flex=f"0 0 {width}px",
            )),
            widgets.VBox(
                [scratch.box, settings.box],
                layout=widgets.Layout(flex="1 1 520px", min_width="320px"),
            ),
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
