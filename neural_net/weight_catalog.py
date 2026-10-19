"""Checkpoint lists for the notebooks that load Gemma, a readout, or both.

Newest first. A directory is one kind or another by the files it holds.
A localizer step whose ``train_meta.json`` only names ``gemma_base`` is a
readout, not a full save. An ``anchor/`` directory and a ``save_gemma``
slice (``gemma_meta.json`` beside ``adapter_config.json``) are not
choices for the parts Gemma dropdown.
"""

from __future__ import annotations

from pathlib import Path

from neural_net.paths import weights_root

_READOUT = ("localizer.pt", "coord_embed.pt", "train_meta.json")


def label(path: Path, root: Path | None = None) -> str:
    """Path relative to ``weights/``, for a dropdown label."""
    base = weights_root() if root is None else root
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _newest(paths: list[Path], stamp) -> list[Path]:
    paths.sort(key=stamp, reverse=True)
    return paths


def is_readout(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in _READOUT)


def is_gemma_parts(path: Path) -> bool:
    """A PEFT adapter ``load_parts`` can take by itself."""
    if not path.is_dir() or path.name == "anchor":
        return False
    if (path / "gemma_meta.json").is_file():
        return False
    return (path / "adapter_config.json").is_file()


def is_s2_full(path: Path) -> bool:
    """Readout plus a Gemma slice in the same directory.

    ``load_snapshot`` can also follow ``train_meta.json``'s ``gemma_base``.
    That pointer alone does not make a localizer step a full save.
    """
    if not is_readout(path):
        return False
    if (path / "gemma" / "gemma_meta.json").is_file():
        return True
    return (path / "adapter_config.json").is_file()


def is_assembled(path: Path) -> bool:
    """``weights/full/<name>``: S1, the readout, and a ``gemma/`` slice."""
    return (
        path.is_dir()
        and (path / "s1.pt").is_file()
        and is_readout(path)
        and (path / "gemma").is_dir()
    )


def is_coord(path: Path) -> bool:
    """LoRA plus ``coord_embed.pt``. No trained Localizer file is required."""
    if not path.is_dir() or path.name == "anchor":
        return False
    return (
        (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
        and (path / "coord_embed.pt").is_file()
    )


def _walk(root: Path, marker: str, keep) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for hit in root.rglob(marker):
        parent = hit.parent
        if parent in seen:
            continue
        seen.add(parent)
        if keep(parent):
            found.append(parent)
    return found


def list_gemma(root: Path | None = None) -> list[Path]:
    base = weights_root() if root is None else root
    return _newest(_walk(base, "adapter_config.json", is_gemma_parts), _mtime)


def list_localizer(root: Path | None = None) -> list[Path]:
    base = weights_root() if root is None else root
    found = _walk(base, "localizer.pt", is_readout)
    return _newest(found, lambda path: _mtime(path / "localizer.pt"))


def list_s2_full(root: Path | None = None) -> list[Path]:
    base = weights_root() if root is None else root
    found = [path for path in _walk(base, "localizer.pt", is_readout) if is_s2_full(path)]
    return _newest(found, lambda path: _mtime(path / "localizer.pt"))


def list_assembled(root: Path | None = None) -> list[Path]:
    base = weights_root() if root is None else root
    full = base / "full"
    if not full.is_dir():
        return []
    found = [path for path in full.iterdir() if is_assembled(path)]
    return _newest(found, lambda path: _mtime(path / "localizer.pt"))


def list_coord(root: Path | None = None) -> list[Path]:
    base = weights_root() if root is None else root
    return _newest(_walk(base, "coord_embed.pt", is_coord), _mtime)


def list_s1(root: Path | None = None) -> list[Path]:
    base = weights_root() if root is None else root
    s1 = base / "s1"
    if not s1.is_dir():
        return []
    files = [path for path in s1.rglob("*.pt") if path.is_file()]
    return _newest(files, _mtime)


def adapter_dir(path: Path) -> Path | None:
    """LoRA directory for ``path``.

    A coordinate checkpoint keeps the adapter beside ``coord_embed.pt``.
    An assembled full save keeps it under ``gemma/``. ``None`` when
    neither is present. An ``anchor/`` directory is not a choice.
    """
    if not path.is_dir() or path.name == "anchor":
        return None
    if (path / "adapter_config.json").is_file() and (
        path / "adapter_model.safetensors"
    ).is_file():
        return path
    nested = path / "gemma"
    if (nested / "adapter_config.json").is_file() and (
        nested / "adapter_model.safetensors"
    ).is_file():
        return nested
    return None


def is_coord_loadable(path: Path) -> bool:
    """``coord_embed.pt`` plus a LoRA, including an assembled full save.

    The localizer and ``s1.pt`` may sit beside the coordinate file.
    A caller that only needs the coordinate code leaves them unread.
    """
    return (
        path.is_dir()
        and (path / "coord_embed.pt").is_file()
        and adapter_dir(path) is not None
    )


def list_coord_loads(root: Path | None = None) -> list[Path]:
    """Coordinate checkpoints and full saves, newest first."""
    base = weights_root() if root is None else root
    return _newest(_walk(base, "coord_embed.pt", is_coord_loadable), _mtime)


def list_s1_loads(root: Path | None = None) -> list[Path]:
    """S1 state dicts: ``weights/s1/**/*.pt`` and each full save's ``s1.pt``.

    A full save's ``localizer.pt`` and ``coord_embed.pt`` are not S1
    weights and are not listed.
    """
    files = list_s1(root)
    seen = {path.resolve() for path in files}
    for directory in list_assembled(root):
        path = directory / "s1.pt"
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        files.append(path)
    return _newest(files, _mtime)
