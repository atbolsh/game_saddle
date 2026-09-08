"""Process-wide speed-overhaul toggles (env flags).

Defaults match production after the B+C / train 1+3+4 land:

* ``GS_PAD_PARITY`` — off. Logit tripwire only; 1-mod-32 arithmetic stays.
* ``GS_PREFIX_KV`` — on. Resident system-prefix KV at inference.
* ``GS_CHUNKED_KD`` — on. Hidden-then-chunked ``lm_head`` for KD.
* ``GS_TRAIN_PREFIX_KV`` — on. Frozen system-prefix KV at train time.
* ``INFER_BACKEND`` — ``hf`` (default) or ``sglang`` (infer-sglang branch).

Read at call time (not import time) so a bench script can flip them
between modes in one process. Empty / ``0`` / ``false`` / ``no`` / ``off``
are false; anything else is true. Unset uses ``default``.
"""

from __future__ import annotations

import os


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def pad_parity_enabled() -> bool:
    return env_flag("GS_PAD_PARITY", False)


def prefix_kv_enabled() -> bool:
    return env_flag("GS_PREFIX_KV", True)


def chunked_kd_enabled() -> bool:
    return env_flag("GS_CHUNKED_KD", True)


def train_prefix_kv_enabled() -> bool:
    return env_flag("GS_TRAIN_PREFIX_KV", True)


def infer_backend() -> str:
    """``hf`` (B+C) or ``sglang``. Unknown values raise — no silent HF."""
    raw = (os.environ.get("INFER_BACKEND") or "hf").strip().lower()
    if raw not in ("hf", "sglang"):
        raise RuntimeError(
            f"INFER_BACKEND={raw!r} -- want hf or sglang"
        )
    return raw
