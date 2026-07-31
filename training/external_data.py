"""External replay datasets: manifest, converters, materialization, sources.

The tracked manifest ``training/datasets.json`` is the single source of
truth (see TRAINING_EXTRA_DATASETS.md for each dataset's role). This module
turns it into two things:

  * **Materialized data** (via training/download_external.py):
    ``data_external/<name>/data.jsonl`` (+ ``images/``) in the canonical
    TrainingExample shape train.py consumes, plus ``meta.json`` (counts,
    mode, date) and fixed probe files under ``data_external/probes/``.
    Download mode comes from ``data_external/settings.json`` written by
    scripts/setup_env.sh: ``"full"`` (default -- the complete dataset lands
    in the HF cache and stays there, no network dependence afterwards) or
    ``"stream"`` (disk-tight boxes: only the consumed shards ever hit disk).
    Both modes produce identical materialized output.

  * **DataSources** (via :func:`sources_from_manifest`):
    :class:`ExternalSource` per enabled entry, with the mixture weight
    computed as ``examples_per_epoch / n_materialized`` -- the per-epoch
    share is the ABSOLUTE count from the manifest and never depends on how
    large the dataset happens to be.

Converters are deliberately tiny per-dataset functions; a malformed row is
a hard error naming the dataset and row number (no-fuzzy-fallbacks: a
training set that silently drops or mangles examples is worse than one
that refuses to build).

Image handling: Cauldron rows carry PIL images; each is saved once (SHA1 of
the encoded bytes) under ``data_external/<name>/images/`` and referenced by
a path RELATIVE to the dataset folder; :class:`ExternalSource` absolutizes
the path at load time, so the data folder can be moved between boxes.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from training.train import VALID_LOSSES, JsonlSource, TrainingExample
from training import synth_navigation

logger = logging.getLogger("train.external_data")

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

#: Root of the materialized data tree (a symlink onto removable storage on
#: the owner's local box; a plain directory made by setup_env.sh elsewhere).
DATA_DIR = Path(os.environ.get("EXTERNAL_DATA_DIR", str(REPO_ROOT / "data_external")))
MANIFEST_PATH = Path(__file__).resolve().parent / "datasets.json"
SETTINGS_PATH = DATA_DIR / "settings.json"

#: Fixed seed for the synthetic generators (probe uses seed+1 internally).
SYNTH_SEED = 2026

VALID_KINDS = ("hf", "synthetic")


# ================================================================ manifest

@dataclass(frozen=True)
class DatasetEntry:
    name: str
    kind: str                      #: "hf" | "synthetic"
    loss: str                      #: "ce" | "kd"
    examples_per_epoch: int
    enabled: bool
    hf_id: str | None = None
    hf_config: str | None = None
    split: str = "train"
    max_rows: int | None = None    #: cap on SOURCE rows read (None = all)
    converter: str | None = None   #: key into CONVERTERS (hf entries)
    generator: str | None = None   #: key into GENERATORS (synthetic entries)
    probe: dict | None = None      #: {"split"?, "n", "kind"}
    #: cap on the training micro-batch for this source's examples (None =
    #: TrainConfig.micro_batch). For long-TARGET KD sources: the KD loss
    #: needs logits over every target position, so batch x ~whole-sequence
    #: x 262k-vocab kept-logit tensors OOM at full micro-batch (t10,
    #: 2026-07-31, openthoughts). Results are unchanged by construction --
    #: weighted_loss normalizes per example (see its docstring).
    micro_batch_cap: int | None = None
    notes: str = ""

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def probe_path(self) -> Path:
        return DATA_DIR / "probes" / f"{self.name}_probe.jsonl"


def load_manifest(path: str | Path | None = None) -> list[DatasetEntry]:
    """Parse + strictly validate the manifest. Unknown converter/generator
    keys, bad loss values, or duplicate names are hard errors -- for enabled
    entries only, so disabled placeholders may name future converters."""
    raw = json.loads(Path(path or MANIFEST_PATH).read_text(encoding="utf-8"))
    entries: list[DatasetEntry] = []
    seen: set[str] = set()
    for obj in raw["datasets"]:
        entry = DatasetEntry(
            name=obj["name"],
            kind=obj["kind"],
            loss=obj["loss"],
            examples_per_epoch=int(obj["examples_per_epoch"]),
            enabled=bool(obj["enabled"]),
            hf_id=obj.get("hf_id"),
            hf_config=obj.get("hf_config"),
            split=obj.get("split") or "train",
            max_rows=obj.get("max_rows"),
            converter=obj.get("converter"),
            generator=obj.get("generator"),
            probe=obj.get("probe"),
            micro_batch_cap=obj.get("micro_batch_cap"),
            notes=obj.get("notes", ""),
        )
        if entry.name in seen:
            raise ValueError(f"manifest: duplicate dataset name {entry.name!r}")
        seen.add(entry.name)
        if entry.kind not in VALID_KINDS:
            raise ValueError(f"manifest: {entry.name}: bad kind {entry.kind!r}")
        if entry.loss not in VALID_LOSSES:
            raise ValueError(f"manifest: {entry.name}: bad loss {entry.loss!r}")
        if entry.enabled:
            if entry.kind == "hf":
                if not entry.hf_id:
                    raise ValueError(f"manifest: {entry.name}: enabled hf entry without hf_id")
                if entry.converter not in CONVERTERS:
                    raise ValueError(
                        f"manifest: {entry.name}: converter {entry.converter!r} "
                        f"not implemented (known: {list(CONVERTERS)})"
                    )
            else:
                if entry.generator not in GENERATORS:
                    raise ValueError(
                        f"manifest: {entry.name}: generator {entry.generator!r} "
                        f"not implemented (known: {list(GENERATORS)})"
                    )
        entries.append(entry)
    return entries


def download_mode() -> str:
    """The global download mode from data_external/settings.json (written by
    scripts/setup_env.sh). Missing file = "full" with a warning, so a bare
    checkout still works."""
    try:
        mode = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))["download_mode"]
    except FileNotFoundError:
        logger.warning(
            "%s not found (run scripts/setup_env.sh); defaulting to "
            "download_mode='full'", SETTINGS_PATH,
        )
        return "full"
    if mode not in ("full", "stream"):
        raise ValueError(f"{SETTINGS_PATH}: bad download_mode {mode!r}")
    return mode


# ============================================================== converters
# Each converter: (row, ctx) -> iterator of canonical records
# {"messages": [...], "target_text": str, "meta": {...}} ("loss" is stamped
# by the materializer from the manifest entry).

class ConvertContext:
    """Per-dataset conversion state: where images go."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.images_dir = out_dir / "images"
        self.n_images = 0

    def save_image(self, img: Any) -> str:
        """Save a PIL image once (SHA1-of-bytes name); return the path
        RELATIVE to the dataset dir."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        name = hashlib.sha1(data).hexdigest()[:20] + ".jpg"
        path = self.images_dir / name
        if not path.exists():
            path.write_bytes(data)
            self.n_images += 1
        return f"images/{name}"


def _text_record(question: str, answer: str, **meta: Any) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ],
        "target_text": answer,
        "meta": meta,
    }


def _conv_gsm8k(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    yield _text_record(row["question"], row["answer"])


def _conv_metamathqa(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    yield _text_record(row["query"], row["response"], type=row.get("type"))


def _conv_orca_math(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    yield _text_record(row["question"], row["answer"])


def _conv_numinamath(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    yield _text_record(row["problem"], row["solution"], source=row.get("source"))


#: ShareGPT-style role names -> chat roles.
_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def _sharegpt_record(system: str | None, conversations: list[dict]) -> dict:
    """First (system?+)user...assistant exchange of a ShareGPT-style
    conversation list; later turns are dropped (one example per row keeps
    the epoch quota meaningful)."""
    messages: list[dict] = []
    if system:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system}]}
        )
    target: str | None = None
    for turn in conversations:
        role = _ROLE_MAP.get(turn.get("from"))
        if role is None:
            raise ValueError(f"unknown ShareGPT role {turn.get('from')!r}")
        if role == "assistant":
            target = turn["value"]
            break
        messages.append(
            {"role": role, "content": [{"type": "text", "text": turn["value"]}]}
        )
    if target is None:
        raise ValueError("conversation has no assistant turn")
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("conversation has no user turn before the assistant")
    return {"messages": messages, "target_text": target, "meta": {}}


def _conv_openthoughts(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    yield _sharegpt_record(row.get("system"), row["conversations"])


def _conv_slimorca(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    yield _sharegpt_record(None, row["conversations"])


def _conv_cauldron(row: dict, ctx: ConvertContext) -> Iterator[dict]:
    """Cauldron rows: {"images": [PIL...], "texts": [{"user", "assistant"}]}
    -- fan out to one example per Q/A pair, all sharing the row's image(s)."""
    image_paths = [ctx.save_image(img) for img in row["images"]]
    if not image_paths:
        raise ValueError("cauldron row without images")
    for qa in row["texts"]:
        content = [{"type": "image", "url": p} for p in image_paths]
        content.append({"type": "text", "text": qa["user"]})
        yield {
            "messages": [{"role": "user", "content": content}],
            "target_text": qa["assistant"],
            "meta": {"source": qa.get("source")},
        }


CONVERTERS: dict[str, Callable[[dict, ConvertContext], Iterator[dict]]] = {
    "gsm8k": _conv_gsm8k,
    "metamathqa": _conv_metamathqa,
    "orca_math": _conv_orca_math,
    "numinamath": _conv_numinamath,
    "openthoughts": _conv_openthoughts,
    "slimorca": _conv_slimorca,
    "cauldron": _conv_cauldron,
}

#: Synthetic generators: name -> (records_fn(n, seed), probe_fn(n, seed)).
GENERATORS: dict[str, tuple[Callable, Callable]] = {
    "navigation": (
        synth_navigation.generate_records,
        synth_navigation.generate_probe,
    ),
}


# ============================================================ probe builders
# Per-dataset gold-answer extraction for HF-based probes. The probe prompt
# appends an explicit answer-format instruction so exact matching is fair.

_ANSWER_SUFFIX = " End your reply with 'ANSWER: <number>'."


def _probe_gsm8k(row: dict) -> dict:
    gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
    if not gold:
        raise ValueError("gsm8k test row without '#### <answer>'")
    return {
        "messages": [
            {"role": "user",
             "content": [{"type": "text", "text": row["question"] + _ANSWER_SUFFIX}]}
        ],
        "answer": gold,
        "meta": {},
    }


PROBE_BUILDERS: dict[str, Callable[[dict], dict]] = {
    "gsm8k": _probe_gsm8k,
}


# ============================================================ materialization

def _load_hf(entry: DatasetEntry, split: str, mode: str) -> Any:
    from datasets import load_dataset

    args = (entry.hf_id, entry.hf_config) if entry.hf_config else (entry.hf_id,)
    logger.info(
        "loading %s (config=%s split=%s mode=%s)",
        entry.hf_id, entry.hf_config, split, mode,
    )
    return load_dataset(
        *args,
        split=split,
        streaming=(mode == "stream"),
        token=os.environ.get("HF_TOKEN") or None,
    )


def _write_jsonl(path: Path, records: Iterator[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def materialize(entry: DatasetEntry, mode: str, force: bool = False) -> dict:
    """Produce ``data_external/<name>/data.jsonl`` (+ images/ + meta.json).
    Idempotent: an existing meta.json skips the work unless ``force``."""
    out_dir = entry.data_dir
    meta_path = out_dir / "meta.json"
    if meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["skipped"] = True
        return meta

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = ConvertContext(out_dir)
    rows_read = 0

    def records() -> Iterator[dict]:
        nonlocal rows_read
        if entry.kind == "synthetic":
            gen_records, _ = GENERATORS[entry.generator]
            for rec in gen_records(entry.max_rows or 10000, SYNTH_SEED):
                rows_read += 1
                rec["loss"] = entry.loss
                yield rec
        else:
            convert = CONVERTERS[entry.converter]
            ds = _load_hf(entry, entry.split, mode)
            it = iter(ds)
            if entry.max_rows:
                it = islice(it, entry.max_rows)
            for i, row in enumerate(it):
                rows_read += 1
                try:
                    for rec in convert(row, ctx):
                        rec["loss"] = entry.loss
                        yield rec
                except Exception as exc:
                    raise ValueError(
                        f"{entry.name}: row {i}: {type(exc).__name__}: {exc}"
                    ) from exc

    n_examples = _write_jsonl(out_dir / "data.jsonl", records())
    if n_examples == 0:
        raise ValueError(f"{entry.name}: materialized zero examples")
    meta = {
        "name": entry.name,
        "hf_id": entry.hf_id,
        "hf_config": entry.hf_config,
        "split": entry.split,
        "mode": mode if entry.kind == "hf" else "synthetic",
        "rows_read": rows_read,
        "examples": n_examples,
        "images": ctx.n_images,
        "loss": entry.loss,
        "bytes": sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "%s: %d examples (%d rows, %d images, %.1f MB)",
        entry.name, n_examples, rows_read, ctx.n_images, meta["bytes"] / 1e6,
    )
    return meta


def materialize_probe(entry: DatasetEntry, mode: str, force: bool = False) -> dict | None:
    """Write the fixed probe slice for entries that declare one. Determinism:
    the FIRST n rows of the probe split (HF) or the seeded generator
    (synthetic) -- the same items every run, forever."""
    if not entry.probe:
        return None
    probe_path = entry.probe_path
    if probe_path.is_file() and not force:
        return {"name": entry.name, "path": str(probe_path), "skipped": True}
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(entry.probe["n"])

    if entry.kind == "synthetic":
        _, gen_probe = GENERATORS[entry.generator]
        records = gen_probe(n, SYNTH_SEED)
    else:
        builder = PROBE_BUILDERS.get(entry.converter)
        if builder is None:
            raise ValueError(
                f"{entry.name}: probe declared but no PROBE_BUILDER for "
                f"converter {entry.converter!r}"
            )
        split = entry.probe.get("split") or entry.split
        ds = _load_hf(entry, split, mode)
        records = (builder(row) for row in islice(iter(ds), n))

    n_written = _write_jsonl(probe_path, records)
    if n_written == 0:
        raise ValueError(f"{entry.name}: probe materialized zero items")
    logger.info("%s: probe with %d items -> %s", entry.name, n_written, probe_path)
    return {"name": entry.name, "path": str(probe_path), "items": n_written}


# ================================================================= sources

class ExternalSource(JsonlSource):
    """DataSource over one materialized external dataset.

    Weight is ``examples_per_epoch / n_materialized`` so the per-epoch share
    is exactly the manifest's absolute count regardless of dataset size.
    Image paths (stored relative to the dataset folder) are absolutized at
    load time."""

    def __init__(self, entry: DatasetEntry):
        meta_path = entry.data_dir / "meta.json"
        if not meta_path.is_file():
            raise RuntimeError(
                f"dataset {entry.name!r} is not materialized under "
                f"{entry.data_dir} -- run: python -m training.download_external"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        super().__init__(
            entry.data_dir / "data.jsonl",
            weight=entry.examples_per_epoch / max(1, int(meta["examples"])),
            default_loss=entry.loss,
        )
        self.name = entry.name
        self.entry = entry

    def examples(self) -> Iterator[TrainingExample]:
        base = str(self.entry.data_dir)
        for ex in super().examples():
            ex.batch_cap = self.entry.micro_batch_cap
            for m in ex.messages:
                content = m.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if (isinstance(part, dict) and part.get("type") == "image"
                            and not os.path.isabs(part["url"])):
                        part["url"] = os.path.join(base, part["url"])
            yield ex


def sources_from_manifest(
    path: str | Path | None = None,
) -> list[ExternalSource]:
    """One ExternalSource per enabled manifest entry with a nonzero epoch
    quota. Fails loudly on any entry that has not been materialized."""
    return [
        ExternalSource(entry)
        for entry in load_manifest(path)
        if entry.enabled and entry.examples_per_epoch > 0
    ]
