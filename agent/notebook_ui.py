"""Small ipywidgets UI helpers shared by the notebooks.

Currently: :func:`tame_shift_enter`, which makes Shift+Enter behave exactly
like Enter (insert a newline) inside notebook text boxes instead of reaching
Jupyter's run-cell shortcut -- which would re-run the widget cell and erase
whatever the user had typed. Buttons remain the only submit mechanism.
"""

from __future__ import annotations

import uuid

from IPython.display import HTML, display

# One keydown listener per tagged <textarea>, attached in the CAPTURE phase so
# it runs before Jupyter's own Shift-Enter (run cell) shortcut can see the
# event. A data- attribute guards against double-binding when the widget cell
# is re-run. setInterval keeps polling because ipywidgets renders (and can
# re-render) its DOM nodes asynchronously after the script executes.
_SCRIPT_TEMPLATE = """
<script>
(function () {
  function insertNewline(ta) {
    var start = ta.selectionStart, end = ta.selectionEnd;
    ta.value = ta.value.slice(0, start) + "\\n" + ta.value.slice(end);
    ta.selectionStart = ta.selectionEnd = start + 1;
    // Bubbling input event so ipywidgets syncs the value to the kernel.
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }
  function bind(ta) {
    if (ta.dataset.shiftEnterTamed) { return; }
    ta.dataset.shiftEnterTamed = "1";
    ta.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && ev.shiftKey) {
        ev.preventDefault();
        ev.stopPropagation();
        ev.stopImmediatePropagation();
        insertNewline(ta);
      }
    }, true);
  }
  var classes = __CLASSES__;
  var timer = setInterval(function () {
    classes.forEach(function (cls) {
      document.querySelectorAll("." + cls + " textarea").forEach(bind);
    });
  }, 300);
  // Widgets can re-render much later (e.g. scrolling a big notebook), so
  // keep the poller alive but cheap; stop after 10 minutes regardless.
  setTimeout(function () { clearInterval(timer); }, 600000);
})();
</script>
"""


def tame_shift_enter(*text_widgets) -> None:
    """Make Shift+Enter insert a plain newline in the given Textarea widgets.

    Shift+Enter must behave exactly like Enter inside the box: it does NOT
    submit anything, and it must not reach Jupyter's run-cell shortcut (which
    re-runs the widget cell and erases the input). Each widget is tagged with
    a unique CSS class, and one injected script attaches a capture-phase
    keydown handler to the underlying <textarea> of each.

    Call AFTER creating the widgets, in the same cell that displays them.
    """
    classes = []
    for w in text_widgets:
        cls = "tame-shift-enter-" + uuid.uuid4().hex[:8]
        w.add_class(cls)
        classes.append(cls)
    class_list = "[" + ", ".join(f'"{c}"' for c in classes) + "]"
    display(HTML(_SCRIPT_TEMPLATE.replace("__CLASSES__", class_list)))
