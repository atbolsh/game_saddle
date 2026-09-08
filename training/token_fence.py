"""Token-space example length (encode first; chars are not a proxy).

``n_tokens = len(tokenizer(templated_text)) + n_images * image_soft_tokens
           + len(tokenizer(target_text)) + 1``  (terminator)

``templated_text`` is ``apply_chat_template(..., tokenize=False,
add_generation_prompt=True)`` on messages whose image parts are kept as
type=image but not opened (no pixel decode on the 120k-row materialize
pass). The image-soft-token constant is the measured delta between
``tokenize=True`` (one real image) and the text-only template tokenized
-- pinned for the process after the first measurement; a later
disagreement is a hard error.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from agent.model import ADAPTERS

logger = logging.getLogger(__name__)

#: Set by :func:`measure_image_soft_tokens`. None until measured.
_IMAGE_SOFT_TOKENS: int | None = None


def n_images_in(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        for part in (m.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "image":
                n += 1
    return n


def messages_for_template(messages: list[dict]) -> list[dict]:
    """Keep image *slots* for the chat template; do not open files."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                new_content.append({"type": "image"})
            else:
                new_content.append(part)
        out.append({**m, "content": new_content})
    return out


def system_prefix_hash(messages: list[dict]) -> str:
    """Byte hash of the single system dump, or ``no-system``."""
    texts: list[str] = []
    n_sys = 0
    for m in messages:
        if m.get("role") != "system":
            continue
        n_sys += 1
        for part in (m.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text") or "")
    if n_sys != 1 or not texts:
        return "no-system"
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def templated_prompt_text(processor: Any, messages: list[dict]) -> str:
    """``apply_chat_template(..., tokenize=False)`` -- same contract as
    Collator.build / VLModel.encode_messages, without pixels."""
    return processor.apply_chat_template(
        messages_for_template(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def count_example_tokens(
    ex: Any,
    tokenizer: Any,
    processor: Any,
    image_soft_tokens: int,
) -> int:
    """Full trained sequence length in tokens (prompt + target + terminator)."""
    if image_soft_tokens < 0:
        raise ValueError(f"image_soft_tokens {image_soft_tokens}")
    prompt_text = templated_prompt_text(processor, ex.messages)
    prompt_n = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    target_n = len(tokenizer(ex.target_text, add_special_tokens=False)["input_ids"])
    return prompt_n + n_images_in(ex.messages) * image_soft_tokens + target_n + 1


def count_prefix_tokens(
    messages: list[dict], tokenizer: Any, processor: Any
) -> int:
    """System-only token count (no generation prompt). 0 if no system dump."""
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if len(sys_msgs) != 1:
        return 0
    text = processor.apply_chat_template(
        messages_for_template([{"role": "system",
                                "content": sys_msgs[0].get("content")}]),
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def measure_image_soft_tokens(processor: Any, image_path: str) -> int:
    """Delta: tokenize=True with a real image minus text-only template.

    First call pins the process-wide constant. A later call that
    disagrees raises -- the fence is only valid if the constant is
    stable.
    """
    global _IMAGE_SOFT_TOKENS
    tokenizer = getattr(processor, "tokenizer", processor)
    msgs = [{"role": "user", "content": [
        {"type": "image", "url": image_path},
        {"type": "text", "text": "count tokens"},
    ]}]
    adapter = ADAPTERS["gemma"]
    with_img = processor.apply_chat_template(
        adapter.prepare_messages(msgs),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    text = processor.apply_chat_template(
        messages_for_template(msgs),
        tokenize=False,
        add_generation_prompt=True,
    )
    text_n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    full_n = int(with_img["input_ids"].shape[1])
    delta = full_n - text_n
    if delta < 1:
        raise RuntimeError(
            f"image soft-token delta is {delta} (full={full_n}, "
            f"text={text_n}) -- the template did not add image tokens"
        )
    if _IMAGE_SOFT_TOKENS is None:
        _IMAGE_SOFT_TOKENS = delta
        logger.info("token_fence: pinned GEMMA image soft tokens = %d", delta)
        return delta
    if delta != _IMAGE_SOFT_TOKENS:
        raise RuntimeError(
            f"image soft-token delta {delta} != pinned "
            f"{_IMAGE_SOFT_TOKENS} -- fence constant drifted"
        )
    return delta


def image_soft_tokens() -> int:
    if _IMAGE_SOFT_TOKENS is None:
        raise RuntimeError(
            "image soft-token constant is unset -- call "
            "measure_image_soft_tokens once before counting examples"
        )
    return _IMAGE_SOFT_TOKENS


def reset_image_soft_tokens_for_tests() -> None:
    global _IMAGE_SOFT_TOKENS
    _IMAGE_SOFT_TOKENS = None
