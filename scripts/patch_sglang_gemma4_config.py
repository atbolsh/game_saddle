#!/usr/bin/env python3
"""Patch sglang 0.5.19 so Engine() can load gemma-4-12B-it.

Two 0.5.19 holes (still present on main as of 2026-09-08):

1. ``hf_transformers/common.py`` aliases ``gemma4_unified`` onto tower
   ``Gemma4Config`` (sgl-project/sglang#34392 / unmerged #34420). The
   scheduler child then builds ``Gemma4VisionConfig`` and dies on
   ``model_patch_size``.
2. ``models/gemma4_unified.py`` skips parent ``__init__`` (to avoid
   building SigLIP/conformer towers) but never sets ``lm_head_is_tied``.
   Parent ``forward`` reads it during CUDA-graph capture.

Rewrites the installed files. A parent-only monkeypatch would not
survive Engine()'s spawned scheduler. Idempotent. Called from
``scripts/install_sglang.sh``; safe to run alone after a 0.5.19 install.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _sglang_root() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise SystemExit("sglang is not importable in this venv")
    return Path(spec.origin).resolve().parent


def _replace_alias_block(text: str) -> str | None:
    needle = "class _Gemma4UnifiedConfigAlias"
    i = text.find(needle)
    if i < 0:
        return None
    try_start = text.rfind("try:", 0, i)
    if try_start < 0:
        return None
    except_pos = text.find("except ImportError:", i)
    if except_pos < 0:
        return None
    pass_nl = text.find("\n", except_pos)
    if pass_nl < 0:
        return None
    block_end = text.find("\n", pass_nl + 1)
    if block_end < 0:
        block_end = len(text)
    line_start = text.rfind("\n", 0, try_start) + 1
    indent = text[line_start:try_start]
    new_block = (
        f"{indent}try:\n"
        f"{indent}    from transformers import Gemma4UnifiedConfig "
        f"as _HFGemma4UnifiedConfig\n"
        f"\n"
        f"{indent}    _CONFIG_REGISTRY[\"gemma4_unified\"] = "
        f"_HFGemma4UnifiedConfig\n"
        f"{indent}except ImportError:\n"
        f"{indent}    pass\n"
    )
    return text[:try_start] + new_block + text[block_end + 1 :]


def patch_config_alias(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "_Gemma4UnifiedConfigAlias" not in text:
        if "Gemma4UnifiedConfig as _HFGemma4UnifiedConfig" in text:
            return "already"
        raise SystemExit(
            f"{path}: no _Gemma4UnifiedConfigAlias and no official "
            "Gemma4UnifiedConfig registration -- this sglang pin is "
            "not the 0.5.19 layout this patcher knows"
        )
    new = _replace_alias_block(text)
    if new is None or new == text:
        raise SystemExit(f"{path}: found the alias but could not rewrite the try block")
    path.write_text(new, encoding="utf-8")
    return "patched"


def patch_lm_head_is_tied(path: Path) -> str:
    import re

    text = path.read_text(encoding="utf-8")
    if "self.lm_head_is_tied" in text:
        return "already"
    pat = re.compile(
        r"(?P<ind>[ \t]*)text_tie = getattr\(text_config, "
        r"\"tie_word_embeddings\", True\)\n"
        r"(?P=ind)if self\.pp_group\.world_size == 1 and text_tie:\n"
        r"(?P=ind)    self\.lm_head = self\.language_model\.embed_tokens\n"
    )

    def repl(m: re.Match[str]) -> str:
        ind = m.group("ind")
        return (
            f'{ind}text_tie = getattr(text_config, "tie_word_embeddings", True)\n'
            f"{ind}self.lm_head_is_tied = (\n"
            f"{ind}    self.pp_group.world_size == 1 and text_tie\n"
            f"{ind})\n"
            f"{ind}if self.lm_head_is_tied:\n"
            f"{ind}    self.lm_head = self.language_model.embed_tokens\n"
        )

    patched, n = pat.subn(repl, text, count=1)
    if n != 1:
        raise SystemExit(
            f"{path}: expected text_tie / tied-lm_head block not found "
            "(sglang layout changed; this patcher is 0.5.19-shaped)"
        )
    path.write_text(patched, encoding="utf-8")
    return "patched"


def _reload_sglang() -> None:
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]


def verify() -> None:
    from sglang.srt.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedVisionConfig

    cls = _CONFIG_REGISTRY.get("gemma4_unified")
    if cls is not Gemma4UnifiedConfig:
        raise SystemExit(
            f"_CONFIG_REGISTRY['gemma4_unified'] is {cls!r}, "
            "not transformers.Gemma4UnifiedConfig"
        )
    cfg = cls(
        vision_config={
            "model_type": "gemma4_unified_vision",
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "mm_embed_dim": 3840,
            "mm_posemb_size": 1120,
            "output_proj_dims": 3840,
        }
    )
    vis = cfg.vision_config
    if type(vis) is not Gemma4UnifiedVisionConfig:
        raise SystemExit(
            f"vision_config is {type(vis).__name__}, not Gemma4UnifiedVisionConfig"
        )
    if vis.model_patch_size != 48:
        raise SystemExit(f"model_patch_size={vis.model_patch_size!r}, expected 48")
    src = Gemma4UnifiedForConditionalGeneration.__init__.__code__.co_names
    if "lm_head_is_tied" not in src:
        raise SystemExit(
            "Gemma4UnifiedForConditionalGeneration.__init__ still does "
            "not assign lm_head_is_tied"
        )
    print(
        f"[patch-sglang] ok registry=Gemma4UnifiedConfig "
        f"vision={type(vis).__name__} model_patch_size={vis.model_patch_size} "
        f"lm_head_is_tied=yes",
        flush=True,
    )


def main() -> int:
    root = _sglang_root()
    common = root / "srt" / "utils" / "hf_transformers" / "common.py"
    unified = root / "srt" / "models" / "gemma4_unified.py"
    if not common.is_file() or not unified.is_file():
        raise SystemExit(f"sglang layout unexpected under {root}")
    print(f"[patch-sglang] {patch_config_alias(common)}: {common}", flush=True)
    print(f"[patch-sglang] {patch_lm_head_is_tied(unified)}: {unified}", flush=True)
    _reload_sglang()
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
