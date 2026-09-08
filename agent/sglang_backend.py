"""SGLang generate backend (INFER_BACKEND=sglang).

In-process ``sgl.Engine`` when the pin can load ``gemma-4-12B-it``;
``SGLANG_HTTP_URL`` if this venv's transformers>=5.10 conflicts and a
second venv is serving. Prefix (radix) cache stays on. Per-request
``stop_regex`` matches :class:`agent.model.RegexStopCriteria`.

PEFT checkpoints with ``modules_to_save: embed_vision`` cannot load here.
Merge first (``scripts/export_merged_checkpoint.py``) and set
``SGLANG_MODEL_PATH`` to that folder.

This module never falls back to HF generate. Missing Engine, HTTP, Gemma 4
support, images, or ``stop_regex`` raise with the exact missing piece.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ENGINE: Any = None
_ENGINE_PATH: str | None = None


def _require_model_path(vl: Any) -> str:
    env = (os.environ.get("SGLANG_MODEL_PATH") or "").strip()
    if env:
        return env
    if vl.checkpoint:
        raise RuntimeError(
            f"INFER_BACKEND=sglang cannot load PEFT checkpoint "
            f"{vl.checkpoint!r} (modules_to_save: embed_vision). Merge "
            f"first:\n  python scripts/export_merged_checkpoint.py "
            f"--architecture {vl.spec.key} --checkpoint {vl.checkpoint} "
            f"--out weights/{vl.spec.key}/merged_{vl.checkpoint}\n"
            "then SGLANG_MODEL_PATH=<that folder>. "
            "This backend will not fall back to HF."
        )
    return vl.spec.hf_id


def render_prompt(vl: Any, messages: list[dict]) -> str:
    """``apply_chat_template(..., tokenize=False)`` — lockstep with
    :meth:`VLModel.encode_messages` / ``Collator.build``."""
    if vl.processor is None:
        raise RuntimeError(
            "sglang: VLModel.processor is unset -- load() must still "
            "build the HF processor so tokenization stays lockstep"
        )
    norm = vl.adapter.prepare_messages(messages)
    text = vl.processor.apply_chat_template(
        norm, tokenize=False, add_generation_prompt=True,
    )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            "sglang: apply_chat_template(tokenize=False) returned empty "
            "-- template drift vs Collator.build"
        )
    return text


def _file_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        suffix, "png"
    )
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _pil_to_data_url(image: Any) -> str:
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_refs(messages: list[dict], *, http: bool) -> list[str]:
    """One path / URL / data-URL per image part, in prompt order.

    HTTP mode always uses data URLs for local files (the other venv
    cannot see this box's paths). In-process Engine gets resolved paths.
    A missing file is a hard error — no silent drop.
    """
    refs: list[str] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            img = part.get("image")
            if img is not None and not isinstance(img, str):
                refs.append(_pil_to_data_url(img))
                continue
            src = part.get("url") or part.get("path") or img
            if src is None and part.get("base64"):
                refs.append("data:image/png;base64," + str(part["base64"]))
                continue
            if not isinstance(src, str) or not src:
                raise RuntimeError(
                    f"sglang: image part has no url/path/image/base64 "
                    f"(keys={sorted(part)})"
                )
            if src.startswith("file://"):
                src = src[len("file://"):]
            if src.startswith(("http://", "https://", "data:")):
                refs.append(src)
                continue
            path = Path(src)
            if not path.is_file():
                raise FileNotFoundError(
                    f"sglang: image path is not a file: {src!r}"
                )
            refs.append(_file_to_data_url(path) if http else str(path.resolve()))
    return refs


def sampling_params(
    vl: Any,
    *,
    max_new_tokens: int | None,
    stop_strings: list[str] | None,
    stop_regex: str | None,
) -> dict[str, Any]:
    knobs = vl._sampling_kwargs()
    params: dict[str, Any] = {
        "max_new_tokens": max_new_tokens or vl.cfg.max_new_tokens,
    }
    if not knobs.get("do_sample", True):
        params["temperature"] = 0.0
    else:
        if knobs.get("temperature") is not None:
            params["temperature"] = knobs["temperature"]
        if knobs.get("top_p") is not None:
            params["top_p"] = knobs["top_p"]
        if knobs.get("top_k") is not None:
            params["top_k"] = knobs["top_k"]
    if stop_strings:
        params["stop"] = list(stop_strings)
    if stop_regex:
        params["stop_regex"] = stop_regex
    return params


def _http_url() -> str | None:
    raw = (os.environ.get("SGLANG_HTTP_URL") or "").strip()
    return raw.rstrip("/") or None


def _post_generate(url: str, payload: dict) -> Any:
    try:
        import urllib.request
    except ImportError as exc:
        raise RuntimeError(
            "sglang HTTP client needs urllib.request"
        ) from exc
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"sglang HTTP {url}/generate failed ({type(exc).__name__}: {exc}). "
            "If the server rejected local image paths, this client already "
            "sends data URLs -- check the server log. "
            "This backend will not fall back to HF."
        ) from exc
    return body


def _texts_from_response(resp: Any, n: int) -> list[str]:
    if isinstance(resp, list):
        rows = resp
    elif isinstance(resp, dict) and "text" in resp:
        rows = [resp]
    else:
        raise RuntimeError(
            f"sglang: unexpected generate response type {type(resp)}: "
            f"{str(resp)[:240]}"
        )
    if len(rows) != n:
        raise RuntimeError(
            f"sglang: expected {n} reply(ies), got {len(rows)}"
        )
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or "text" not in row:
            raise RuntimeError(f"sglang: row missing 'text': {row!r}"[:240])
        out.append(str(row["text"]).strip())
    return out


def _get_engine(model_path: str) -> Any:
    global _ENGINE, _ENGINE_PATH
    if _ENGINE is not None:
        if _ENGINE_PATH != model_path:
            raise RuntimeError(
                f"sglang Engine already loaded for {_ENGINE_PATH!r}; "
                f"refusing to also load {model_path!r} in one process"
            )
        return _ENGINE
    try:
        import sglang as sgl
    except ImportError as exc:
        raise RuntimeError(
            "INFER_BACKEND=sglang: sglang is not importable in this venv "
            f"({exc}). Install a recent/nightly pin that loads "
            "gemma-4-12B-it, or run a second venv + "
            "`python -m sglang.launch_server --model-path <merged>` and "
            "set SGLANG_HTTP_URL. This backend will not fall back to HF."
        ) from exc
    if not hasattr(sgl, "Engine"):
        raise RuntimeError(
            "sglang is importable but has no sgl.Engine -- too old, or "
            "the API moved. Pin a nightly that documents Engine + "
            "stop_regex + image_data, or use SGLANG_HTTP_URL. "
            "This backend will not fall back to HF."
        )
    mem = float(os.environ.get("SGLANG_MEM_FRACTION", "0.85"))
    logger.info("sglang: starting Engine model_path=%s", model_path)
    try:
        _ENGINE = sgl.Engine(
            model_path=model_path,
            mem_fraction_static=mem,
        )
    except Exception as exc:
        raise RuntimeError(
            f"sgl.Engine failed to load {model_path!r} "
            f"({type(exc).__name__}: {exc}). Need a pin that knows "
            "gemma-4-12B-it (cookbook + transformers SHA). If this "
            "venv is transformers>=5.10 and conflicts, use a second "
            "venv + HTTP server (SGLANG_HTTP_URL). "
            "This backend will not fall back to HF."
        ) from exc
    _ENGINE_PATH = model_path
    return _ENGINE


def shutdown() -> None:
    global _ENGINE, _ENGINE_PATH
    if _ENGINE is None:
        return
    close = getattr(_ENGINE, "shutdown", None) or getattr(_ENGINE, "close", None)
    if callable(close):
        close()
    _ENGINE = None
    _ENGINE_PATH = None


def generate_one(
    vl: Any,
    messages: list[dict],
    *,
    max_new_tokens: int | None,
    stop_strings: list[str] | None,
    stop_regex: str | None,
) -> str:
    return generate_many(
        vl, [messages],
        max_new_tokens=max_new_tokens,
        stop_strings=stop_strings,
        stop_regex=stop_regex,
    )[0]


def generate_many(
    vl: Any,
    batch: list[list[dict]],
    *,
    max_new_tokens: int | None,
    stop_strings: list[str] | None,
    stop_regex: str | None,
) -> list[str]:
    http = _http_url()
    prompts = [render_prompt(vl, m) for m in batch]
    images = [image_refs(m, http=bool(http)) for m in batch]
    params = sampling_params(
        vl, max_new_tokens=max_new_tokens,
        stop_strings=stop_strings, stop_regex=stop_regex,
    )
    logger.info(
        "sglang batch n=%d stop_regex=%s images=%s",
        len(batch), bool(stop_regex), [len(x) for x in images],
    )
    image_payload: Any
    if all(len(x) == 0 for x in images):
        image_payload = None
    elif all(len(x) == 1 for x in images):
        image_payload = [x[0] for x in images]
    else:
        image_payload = images

    if http:
        parsed = urlparse(http)
        if parsed.scheme not in ("http", "https"):
            raise RuntimeError(f"SGLANG_HTTP_URL is not http(s): {http!r}")
        payload: dict[str, Any] = {
            "text": prompts if len(prompts) > 1 else prompts[0],
            "sampling_params": params,
        }
        if image_payload is not None:
            payload["image_data"] = (
                image_payload if len(prompts) > 1 else image_payload[0]
            )
        resp = _post_generate(http, payload)
        return _texts_from_response(resp, len(prompts))

    engine = _get_engine(_require_model_path(vl))
    kwargs: dict[str, Any] = {
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "sampling_params": params,
    }
    if image_payload is not None:
        kwargs["image_data"] = (
            image_payload if len(prompts) > 1 else image_payload[0]
        )
    try:
        resp = engine.generate(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            f"sgl.Engine.generate rejected kwargs ({exc}). "
            "Need image_data + stop_regex on SamplingParams. "
            "This backend will not fall back to HF."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"sgl.Engine.generate failed ({type(exc).__name__}: {exc}). "
            "If images were rejected, local paths may be unsupported -- "
            "the HTTP path already sends data URLs. "
            "This backend will not fall back to HF."
        ) from exc
    return _texts_from_response(resp, len(prompts))
