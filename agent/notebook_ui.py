"""Small ipywidgets UI helpers shared by the notebooks.

- :func:`tame_shift_enter` makes Shift+Enter behave exactly like Enter
  (insert a newline) inside notebook text boxes instead of reaching Jupyter's
  run-cell shortcut -- which would re-run the widget cell and erase whatever
  the user had typed. Buttons remain the only submit mechanism.
- :func:`model_picker` is the model dropdown + "save only one set of weights
  at a time" checkbox + Switch button shown at the top of every notebook,
  wired to the session's ``switch_model``.
"""

from __future__ import annotations

from typing import Any, Callable

import ipywidgets as widgets
from IPython.display import HTML, display

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


def model_picker(
    session: Any,
    on_switched: Callable[[dict[str, Any]], None] | None = None,
) -> widgets.VBox:
    """The shared model-switching panel (display it at the TOP of a notebook's
    control cell).

    Dropdown lists every ``agent.model.MODEL_REGISTRY`` entry in
    recommendation order. Switching is an explicit button press (a dropdown
    misclick must never start a multi-GB download). The checkbox implements
    "save only one set of weights at a time": when checked, a switch first
    restarts the conversation, then deletes every OTHER registry model's
    cached weights before downloading the new ones; when unchecked, the
    conversation continues under the new model and old weights stay cached.

    ``on_switched(info)`` (if given) fires after a successful switch so the
    notebook can refresh its own view; ``info["restarted"]`` says whether the
    conversation was restarted.
    """
    from .model import MODEL_REGISTRY

    current = (
        session.model.spec.key if session.model is not None
        else session.cfg.model_key
    )
    dropdown = widgets.Dropdown(
        options=[(spec.label, key) for key, spec in MODEL_REGISTRY.items()],
        value=current if current in MODEL_REGISTRY else None,
        description="Model:",
        layout=widgets.Layout(width="460px"),
    )
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
        key = dropdown.value
        if key is None:
            return
        already = session.model is not None and key == session.model.spec.key
        if already and not one_copy.value:
            with status:
                status.clear_output()
                print(f"'{key}' is already the loaded model.")
            return
        switch_btn.disabled = dropdown.disabled = one_copy.disabled = True
        try:
            with status:
                status.clear_output()
                spec = MODEL_REGISTRY[key]
                if one_copy.value:
                    print("[one-weights mode] restarting the conversation and "
                          "purging other cached weights ...")
                print(f"Switching to {spec.label} ({spec.hf_id}); first use "
                      f"downloads the weights -- this can take a while ...")
                info = session.switch_model(key, purge_others=one_copy.value)
                purge = info.get("purge") or {}
                if purge.get("purged"):
                    print(f"Purged {len(purge['purged'])} cached repo(s), "
                          f"freed {purge['freed_bytes'] / 1e9:.1f} GB: "
                          + ", ".join(purge["purged"]))
                elif one_copy.value:
                    print("No other registry weights were cached; nothing to purge.")
                print(f"Model ready: {info['label']}"
                      + ("  [conversation restarted]" if info["restarted"] else ""))
            if on_switched is not None:
                on_switched(info)
        finally:
            switch_btn.disabled = dropdown.disabled = one_copy.disabled = False

    switch_btn.on_click(_on_switch)
    return widgets.VBox([
        widgets.HBox([dropdown, switch_btn]),
        one_copy,
        status,
    ])


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
