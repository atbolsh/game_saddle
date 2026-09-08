#!/usr/bin/env python3
"""Patch sglang 0.5.19's gemma4_unified config alias (sgl-project/sglang#34392).

0.5.19 registers ``model_type="gemma4_unified"`` as a subclass of
``Gemma4Config`` (the SigLIP-tower family). ``HfModelConfigParser`` then
reloads the checkpoint through that alias, so ``vision_config`` becomes
``Gemma4VisionConfig`` and ``model_patch_size`` / ``mm_embed_dim`` vanish.
Engine() dies in the scheduler *child* with::

    AttributeError: 'Gemma4VisionConfig' object has no attribute 'model_patch_size'

Upstream fix is sgl-project/sglang#34420 (still open on 2026-09-08). This
rewrites the installed ``hf_transformers/common.py`` so the registry holds
transformers' official ``Gemma4UnifiedConfig``. The child process imports
the patched file; a parent-only monkeypatch would not survive spawn.

Idempotent. Called from ``scripts/install_sglang.sh``; safe to run alone
after a 0.5.19 install.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _common_py() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise SystemExit("sglang is not importable in this venv")
    return (
        Path(spec.origin).resolve().parent
        / "srt"
        / "utils"
        / "hf_transformers"
        / "common.py"
    )


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


def patch(path: Path) -> str:
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


def verify() -> None:
    # Fresh imports after the file rewrite. The child Engine process does
    # the same.
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
    print(
        f"[patch-sglang] ok registry=Gemma4UnifiedConfig "
        f"vision={type(vis).__name__} model_patch_size={vis.model_patch_size}",
        flush=True,
    )


def main() -> int:
    path = _common_py()
    status = patch(path)
    print(f"[patch-sglang] {status}: {path}", flush=True)
    # Drop a stale in-process import so verify sees the rewritten file
    # (Engine's scheduler child always imports fresh).
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
