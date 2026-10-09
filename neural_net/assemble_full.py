"""Assemble S1 and S2 into ``weights/full/<output>``.

Two ways in, the same directory out:

* ``--gemma`` and ``--localizer`` and ``--s1`` load the trunk and the
  readout from the two directories the testing_s2 dropdowns list, and
  S1 from the testing_s1 dropdown.
* ``--s2`` and ``--s1`` load a full S2 snapshot (the testing_s2 "Full
  save" dropdown) plus S1.

A name is the dropdown label under ``weights/``. The ``.pt`` on an S1
file is optional. A prefix that matches one checkpoint is that
checkpoint. A prefix that matches several is an error.

The directory is what :meth:`neural_net.loop.S1S2.load` reads: a
``save_all`` Gemma slice, the readout, and ``s1.pt``. The readout and
the coordinate file must already be the ``(s, v)`` format
(``python -m neural_net.convert_to_sv``); the loaders reject a 3-point
file. Run this on the GPU box. It loads Gemma.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_net.paths import full_weights_dir, weights_root

_READOUT = ("localizer.pt", "coord_embed.pt", "train_meta.json")


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_readout(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in _READOUT)


def _is_gemma(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "gemma_meta.json").is_file() or (path / "adapter_config.json").is_file()


def _is_full(path: Path) -> bool:
    """Same rule as the testing_s2 full-save dropdown."""
    if not _is_readout(path):
        return False
    if (path / "gemma" / "gemma_meta.json").is_file():
        return True
    if (path / "adapter_config.json").is_file():
        return True
    try:
        meta = json.loads((path / "train_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(meta.get("gemma_base"))


def _catalog(root: Path, kind: str) -> list[Path]:
    if not root.is_dir():
        return []
    if kind == "s1":
        base = root / "s1"
        if not base.is_dir():
            return []
        files = [path for path in base.rglob("*.pt") if path.is_file()]
        files.sort(key=lambda path: path.as_posix())
        return files
    found: list[Path] = []
    seen: set[Path] = set()
    if kind == "gemma":
        markers = ("adapter_config.json", "gemma_meta.json")
    elif kind in ("localizer", "s2"):
        markers = ("localizer.pt",)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    for name in markers:
        for marker in root.rglob(name):
            parent = marker.parent
            if parent in seen:
                continue
            seen.add(parent)
            if kind == "gemma" and _is_gemma(parent):
                found.append(parent)
            elif kind == "localizer" and _is_readout(parent):
                found.append(parent)
            elif kind == "s2" and _is_full(parent):
                found.append(parent)
    found.sort(key=lambda path: path.as_posix())
    return found


def _prefix_hit(label: str, query: str) -> bool:
    bare = label[:-3] if label.endswith(".pt") else label
    tail = bare.rsplit("/", 1)[-1]
    return (
        label == query
        or bare == query
        or tail == query
        or label.startswith(query)
        or bare.startswith(query)
        or tail.startswith(query)
    )


def _direct(root: Path, query: str, kind: str) -> Path | None:
    raw = Path(query)
    bases = [raw] if raw.is_absolute() else [root / raw, raw]
    options = list(bases)
    if kind == "s1":
        options.extend(
            path.with_suffix(".pt") if path.suffix else Path(str(path) + ".pt")
            for path in bases
        )
    seen: set[Path] = set()
    for path in options:
        if path in seen:
            continue
        seen.add(path)
        if kind == "s1" and path.is_file():
            return path
        if kind == "gemma" and _is_gemma(path):
            return path
        if kind == "localizer" and _is_readout(path):
            return path
        if kind == "s2" and _is_full(path):
            return path
    return None


def resolve_name(root: Path, query: str, kind: str) -> Path:
    """The one checkpoint ``query`` names, as the notebooks would label it."""
    text = query.strip()
    if not text:
        raise SystemExit(f"--{kind} is empty")
    direct = _direct(root, text, kind)
    if direct is not None:
        return direct
    hits = [
        path for path in _catalog(root, kind)
        if _prefix_hit(_rel(root, path), text)
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no {kind} checkpoint matches {text!r} under {root}")
    shown = "\n".join(_rel(root, path) for path in hits[:40])
    extra = "" if len(hits) <= 40 else f"\n... and {len(hits) - 40} more"
    raise SystemExit(
        f"{text!r} matches {len(hits)} {kind} checkpoints:\n{shown}{extra}"
    )


def _output_dir(name: str) -> Path:
    text = name.strip()
    if not text or text in (".", "..") or "/" in text or "\\" in text:
        raise SystemExit(
            "--output is the directory name under weights/full/, not a path"
        )
    dest = full_weights_dir() / text
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    return dest


def _write_sources(dest: Path, sources: dict[str, str]) -> None:
    (dest / "full_meta.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8"
    )


def assemble(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=(
            "Save Gemma + localizer + S1, or a full S2 + S1, "
            "under weights/full/<output>."
        ),
    )
    parser.add_argument(
        "--gemma",
        help="Gemma dropdown label (adapter or gemma_meta directory)",
    )
    parser.add_argument(
        "--localizer",
        help="Localizer dropdown label (localizer.pt directory)",
    )
    parser.add_argument(
        "--s2",
        help="Full-save dropdown label (Gemma and localizer already together)",
    )
    parser.add_argument("--s1", required=True, help="S1 dropdown label; .pt is optional")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory name to create under weights/full/",
    )
    args = parser.parse_args(argv)
    parts = bool(args.gemma or args.localizer)
    whole = bool(args.s2)
    if parts and whole:
        raise SystemExit("pass --s2, or --gemma with --localizer, not both")
    if parts and not (args.gemma and args.localizer):
        raise SystemExit("--gemma and --localizer are both required")
    if not parts and not whole:
        raise SystemExit("pass --gemma and --localizer, or --s2")

    root = weights_root()
    dest = _output_dir(args.output)
    s1_path = resolve_name(root, args.s1, "s1")
    sources = {"s1": _rel(root, s1_path), "output": dest.name}
    if whole:
        s2_path = resolve_name(root, args.s2, "s2")
        sources["s2"] = _rel(root, s2_path)
        print(f"s2 {s2_path}", flush=True)
    else:
        gemma_path = resolve_name(root, args.gemma, "gemma")
        loc_path = resolve_name(root, args.localizer, "localizer")
        sources["gemma"] = _rel(root, gemma_path)
        sources["localizer"] = _rel(root, loc_path)
        print(f"gemma {gemma_path}", flush=True)
        print(f"localizer {loc_path}", flush=True)
    print(f"s1 {s1_path}", flush=True)
    print(f"output {dest}", flush=True)

    from neural_net.loop import S1S2
    from neural_net.s1 import S1
    from neural_net.s2 import GemmaS2

    s1 = S1()
    s1.load_weights(s1_path)
    s2 = GemmaS2()
    if whole:
        s2.load_snapshot(s2_path)
    else:
        s2.load_parts(gemma_path, loc_path)
    net = S1S2(s1=s1, s2=s2)
    net.save(dest)
    _write_sources(dest, sources)
    print(f"saved {dest}", flush=True)
    return dest


def main() -> None:
    try:
        assemble()
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
