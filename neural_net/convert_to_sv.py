"""Rewrite a 3-point checkpoint as ``(s, v)`` beside the original.

``python -m neural_net.convert_to_sv --src <label> [--dst <name>]``

``--src`` is a dropdown label under ``weights/``, the same prefix rule
as ``assemble_full``. A localizer directory, a coord-trainer directory,
and a ``weights/full/<name>`` directory are converted. A plain Gemma
adapter has no coordinate file: the command prints that and writes
nothing.

The new directory is ``<source>_sv`` next to the source unless ``--dst``
names one. Nothing is overwritten. ``localizer.pt`` keeps the ``s`` and
``v`` rows of ``q_proj`` and drops ``offset.2``. ``coord_embed.pt``
drops the ``v_bar`` channel. Everything else is copied. The written
blobs must load into a fresh ``Localizer`` and ``CoordEmbedder`` with
``strict=True`` before the command reports success.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from neural_net.assemble_full import _is_gemma, _prefix_hit, _rel
from neural_net.paths import weights_root

_COORD_ORDER = ["s", "v"]
_VBAR_EMBED_KEYS = ("gate_vbar", "proj_vbar.weight", "proj_vbar.bias")


def _convertible(path: Path) -> bool:
    return path.is_dir() and (
        (path / "localizer.pt").is_file() or (path / "coord_embed.pt").is_file()
    )


def _plain_adapter(path: Path) -> bool:
    return (
        path.is_dir()
        and _is_gemma(path)
        and not (path / "localizer.pt").is_file()
        and not (path / "coord_embed.pt").is_file()
    )


def _candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for name in ("localizer.pt", "coord_embed.pt"):
        for marker in root.rglob(name):
            found.add(marker.parent)
    return sorted(found, key=lambda path: path.as_posix())


def resolve_src(root: Path, query: str) -> Path:
    text = query.strip()
    if not text:
        raise SystemExit("--src is empty")
    raw = Path(text)
    bases = [raw] if raw.is_absolute() else [root / raw, raw]
    for path in bases:
        if _convertible(path) or _plain_adapter(path):
            return path
    hits = [
        path for path in _candidates(root)
        if _prefix_hit(_rel(root, path), text)
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        plain = [
            path for path in root.rglob("adapter_config.json")
            if _plain_adapter(path.parent) and _prefix_hit(_rel(root, path.parent), text)
        ]
        plain = sorted({path.parent for path in plain}, key=lambda path: path.as_posix())
        if len(plain) == 1:
            return plain[0]
        if len(plain) > 1:
            shown = "\n".join(_rel(root, path) for path in plain[:40])
            raise SystemExit(f"{text!r} matches {len(plain)} adapters:\n{shown}")
        raise SystemExit(f"no checkpoint matches {text!r} under {root}")
    shown = "\n".join(_rel(root, path) for path in hits[:40])
    raise SystemExit(f"{text!r} matches {len(hits)} checkpoints:\n{shown}")


def convert_localizer_blob(blob: dict) -> dict:
    if list(blob.get("coord_order") or []) == _COORD_ORDER:
        raise RuntimeError("localizer.pt is already (s, v)")
    dim = int(blob["dim"])
    state = dict(blob["state_dict"])
    weight = state["q_proj.weight"]
    bias = state["q_proj.bias"]
    if int(weight.shape[0]) != 3 * dim or int(bias.shape[0]) != 3 * dim:
        raise RuntimeError(
            f"q_proj is {tuple(weight.shape)} / {tuple(bias.shape)}, "
            f"expected 3*{dim}"
        )
    state["q_proj.weight"] = weight[: 2 * dim].contiguous()
    state["q_proj.bias"] = bias[: 2 * dim].contiguous()
    dropped = [key for key in state if key.startswith("offset.2.")]
    if not dropped:
        raise RuntimeError("localizer.pt has no offset.2 parameters")
    for key in dropped:
        del state[key]
    out = dict(blob)
    out["state_dict"] = state
    out["coord_order"] = list(_COORD_ORDER)
    return out


def convert_embed_blob(blob: dict) -> dict:
    if list(blob.get("codes") or []) == _COORD_ORDER:
        raise RuntimeError("coord_embed.pt is already (s, v)")
    state = dict(blob["state_dict"])
    missing = [key for key in _VBAR_EMBED_KEYS if key not in state]
    if missing:
        raise RuntimeError(
            "coord_embed.pt is not a 3-point file; missing " + ", ".join(missing)
        )
    for key in _VBAR_EMBED_KEYS:
        del state[key]
    s_initialized = _fill_missing_s(state)
    out = dict(blob)
    out["state_dict"] = state
    out["codes"] = list(_COORD_ORDER)
    if s_initialized:
        out["s_channel"] = "initialized"
    return out


def _fill_missing_s(state: dict) -> bool:
    """A localizer run saved before the s channel has only v and v_bar.

    That module was never trained. A zero gate adds nothing. The
    projection is a fresh draw, the same init as ``CoordEmbedder``.
    """
    needed = ("gate_s", "proj_s.weight", "proj_s.bias")
    if all(key in state for key in needed):
        return False
    import torch
    from torch import nn

    ref = state["proj_v.weight"]
    if "proj_s.weight" not in state:
        weight = torch.empty(ref.shape, dtype=ref.dtype)
        nn.init.normal_(weight, std=0.02)
        state["proj_s.weight"] = weight
    if "proj_s.bias" not in state:
        state["proj_s.bias"] = torch.zeros(ref.shape[0], dtype=ref.dtype)
    if "gate_s" not in state:
        state["gate_s"] = torch.zeros((), dtype=ref.dtype)
    return True


def _copy_rest(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    for child in src.iterdir():
        if child.name in ("localizer.pt", "coord_embed.pt"):
            continue
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def _patch_meta(dest: Path, src_label: str) -> None:
    path = dest / "train_meta.json"
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["converted_from"] = src_label
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _self_check(dest: Path) -> None:
    import torch

    from neural_net.coord_embed import CoordEmbedder
    from neural_net.localizer import Localizer

    loc_path = dest / "localizer.pt"
    if loc_path.is_file():
        blob = torch.load(loc_path, map_location="cpu", weights_only=True)
        module = Localizer(
            int(blob["hidden_size"]),
            dim=int(blob["dim"]),
            refine_layers=int(blob["refine_layers"]),
            heads=int(blob["heads"]),
            mlp_dim=int(blob["mlp_dim"]),
        )
        module.load_state_dict(blob["state_dict"], strict=True)
    embed_path = dest / "coord_embed.pt"
    if embed_path.is_file():
        blob = torch.load(embed_path, map_location="cpu", weights_only=True)
        module = CoordEmbedder(int(blob["hidden_size"]))
        module.load_state_dict(blob["state_dict"], strict=True)


def convert(src: Path, dest: Path, src_label: str) -> None:
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    import torch

    try:
        _copy_rest(src, dest)
        loc_path = src / "localizer.pt"
        if loc_path.is_file():
            blob = torch.load(loc_path, map_location="cpu", weights_only=True)
            torch.save(convert_localizer_blob(blob), dest / "localizer.pt")
        embed_path = src / "coord_embed.pt"
        if embed_path.is_file():
            blob = torch.load(embed_path, map_location="cpu", weights_only=True)
            converted = convert_embed_blob(blob)
            torch.save(converted, dest / "coord_embed.pt")
            if converted.get("s_channel") == "initialized":
                print(
                    "coord_embed.pt had no s channel; gate_s is 0 and "
                    "proj_s is a fresh draw. This file was not the trained code.",
                    flush=True,
                )
        _patch_meta(dest, src_label)
        _self_check(dest)
    except Exception:
        if dest.exists():
            shutil.rmtree(dest)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a 3-point checkpoint to (s, v).",
    )
    parser.add_argument("--src", required=True, help="Dropdown label under weights/")
    parser.add_argument(
        "--dst",
        help="Directory name to create beside the source (default: <name>_sv)",
    )
    args = parser.parse_args(argv)
    root = weights_root()
    src = resolve_src(root, args.src)
    if _plain_adapter(src):
        print(
            f"{src} is a Gemma adapter with no coordinate file; nothing to convert",
            flush=True,
        )
        return
    if not _convertible(src):
        raise SystemExit(f"{src} has neither localizer.pt nor coord_embed.pt")
    name = args.dst.strip() if args.dst else f"{src.name}_sv"
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise SystemExit("--dst is a directory name, not a path")
    dest = src.parent / name
    label = _rel(root, src)
    print(f"src {src}", flush=True)
    print(f"dst {dest}", flush=True)
    convert(src, dest, label)
    print(f"saved {dest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
