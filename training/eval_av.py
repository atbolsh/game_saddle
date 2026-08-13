"""Audio/video comprehension: named checkpoint vs the raw Gemma 4 12B base.

Loads the full eval corpora once (LibriSpeech test-clean transcription +
NExT-QA multiple-choice), draws a fixed-seed sample that should finish in
~20 minutes, and scores the adapter against the frozen base on the SAME
encoded prompts. Input format is the standard HF chat-template path the
Gemma 4 processor already uses for images::

    {"type": "audio", "path": "/abs/clip.wav"}
    {"type": "video", "path": "/abs/clip.mp4"}   # + num_frames / do_sample_frames

A missing tensor key or a processor rejection is a hard error (no-fuzzy-
fallbacks) -- that is a finding, not something to route around.

Usage (remote, after a checkpoint exists)::

    python -m training.eval_av aug12_iter1_step350
    python -m training.eval_av aug12_iter1_step350 --n-audio 5 --n-video 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import string
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# .env (HF_TOKEN) before the first Hub touch -- env-secrets rule.
import agent.config  # noqa: F401
from training.external_data import DATA_DIR, REPO_ROOT

logger = logging.getLogger("train.eval_av")

AUDIO_HF_ID = "openslr/librispeech_asr"
AUDIO_HF_CONFIG = "clean"
AUDIO_HF_SPLIT = "test"
VIDEO_HF_ID = "lmms-lab/NExTQA"
# lmms-eval nextqa_mc_test.yaml: dataset_name MC, test_split test.
# The parquet `video` column is a VidOR id (int64), not bytes; clips live
# in videos.zip and are resolved as NExTVideo/{id}.mp4 (lmms-eval
# nextqa/utils.py get_cache_dir(..., "NExTVideo") + get_video).
VIDEO_HF_CONFIG = "MC"
VIDEO_HF_SPLIT = "test"
VIDEO_ZIP_NAME = "videos.zip"
VIDEO_CLIP_SUBDIR = "NExTVideo"

# Google's canonical transcription prompt asks for digits ("write 3, not
# three"), but LibriSpeech references spell numbers out as words, so that
# instruction would WER-penalize every numeric utterance. The number
# instruction is inverted here to match the reference convention.
AUDIO_PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* Spell numbers out as words, i.e. write 'one point seven' and not "
    "1.7, and write 'three' instead of 3."
)
VIDEO_PROMPT_TAIL = "Answer with the single letter of the correct option."

#: Feature extractor truncates at 480000 samples (~30 s at 16 kHz).
AUDIO_MAX_SECONDS = 30.0


# ===================================================================== scoring

def _normalize_transcript(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).split()


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Standard word-level Levenshtein distance / |reference|. Empty
    reference -> 1.0 unless the hypothesis is also empty (then 0.0)."""
    hyp = _normalize_transcript(hypothesis)
    ref = _normalize_transcript(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def exact_match_normalized(hypothesis: str, reference: str) -> bool:
    return _normalize_transcript(hypothesis) == _normalize_transcript(reference)


_LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def parse_mc_letter(reply: str) -> str | None:
    """First A–E letter in the reply, or None if unparseable."""
    m = _LETTER_RE.search(reply.strip())
    return m.group(1).upper() if m else None


# ================================================================= materialize

def _av_root() -> Path:
    return Path(DATA_DIR) / "av_eval"


def _write_wav(path: Path, array: np.ndarray, sr: int) -> None:
    array = np.asarray(array)
    if array.ndim > 1:
        array = array.mean(axis=-1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype(np.int16)
    else:
        array = array.astype(np.int16)
    max_n = int(AUDIO_MAX_SECONDS * sr)
    if array.shape[0] > max_n:
        array = array[:max_n]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(array.tobytes())


def _array_and_rate_from_samples(samples: Any) -> tuple[np.ndarray, int]:
    """torchcodec AudioSamples -> (mono float array, sample_rate).

    Layout is ``(num_channels, num_samples)`` (HF audio_process docs,
    transformers ``load_audio_torchcodec``). Downmix matches
    ``datasets.features._torchcodec.AudioDecoder.__getitem__("array")``.
    """
    data = samples.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)
    if data.ndim > 1:
        data = np.mean(data, axis=tuple(range(data.ndim - 1)))
    return data, int(samples.sample_rate)


def _audio_from_row(row: dict) -> tuple[np.ndarray, int]:
    audio = row.get("audio")
    if audio is None:
        raise KeyError(
            f"LibriSpeech row has no 'audio' key; keys={sorted(row)}"
        )
    # datasets<4 Audio feature: {"array", "sampling_rate"}.
    if isinstance(audio, dict):
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is None or sr is None:
            raise KeyError(
                f"LibriSpeech audio dict missing array/sampling_rate; "
                f"keys={sorted(audio)}"
            )
        return np.asarray(arr), int(sr)
    # datasets>=4: torchcodec AudioDecoder (HF audio_process:
    #   samples = audio.get_all_samples(); samples.data / samples.sample_rate).
    get_all = getattr(audio, "get_all_samples", None)
    if callable(get_all):
        return _array_and_rate_from_samples(get_all())
    raise TypeError(
        f"LibriSpeech audio is {type(audio).__name__}, expected dict "
        f"with array + sampling_rate or a torchcodec AudioDecoder"
    )


def _librispeech_id(row: dict, index: int) -> str:
    for key in ("id", "file", "chapter_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return f"ls_{index}"


def _librispeech_text(row: dict) -> str:
    for key in ("text", "transcription", "sentence"):
        if key in row and row[key]:
            return str(row[key])
    raise KeyError(
        f"LibriSpeech row has no transcript field; keys={sorted(row)}"
    )


def materialize_audio(n: int, seed: int, hf_id: str, force: bool) -> list[dict]:
    dest = _av_root() / "librispeech"
    meta_path = dest / "meta.json"
    if meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("hf_id") == hf_id and meta.get("seed") == seed
                and meta.get("n") == n):
            items_path = dest / "items.jsonl"
            if items_path.is_file():
                logger.info("reusing %s (n=%d seed=%d)", dest, n, seed)
                return [json.loads(l) for l in items_path.read_text(
                    encoding="utf-8").splitlines() if l.strip()]
    dest.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    logger.info("downloading %s config=%s split=%s (full, no streaming)",
                hf_id, AUDIO_HF_CONFIG, AUDIO_HF_SPLIT)
    ds = load_dataset(hf_id, AUDIO_HF_CONFIG, split=AUDIO_HF_SPLIT)
    if n > len(ds):
        raise ValueError(
            f"--n-audio {n} exceeds LibriSpeech {AUDIO_HF_SPLIT} size "
            f"{len(ds)}"
        )
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    items = []
    for k, idx in enumerate(indices):
        row = ds[idx]
        arr, sr = _audio_from_row(row)
        item_id = _librispeech_id(row, idx)
        wav_path = dest / "clips" / f"{k:04d}.wav"
        _write_wav(wav_path, arr, sr)
        items.append({
            "id": item_id,
            "index": idx,
            "path": str(wav_path.resolve()),
            "gold": _librispeech_text(row),
        })
    (dest / "items.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items),
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps({
        "hf_id": hf_id, "hf_config": AUDIO_HF_CONFIG, "split": AUDIO_HF_SPLIT,
        "seed": seed, "n": n, "item_ids": [it["id"] for it in items],
    }, indent=2), encoding="utf-8")
    logger.info("materialized %d LibriSpeech clips -> %s", n, dest)
    return items


def _nextqa_options(row: dict) -> list[tuple[str, str]]:
    """Return [(letter, text), ...] for a multiple-choice row."""
    if "candidates" in row and row["candidates"] is not None:
        cands = list(row["candidates"])
        if len(cands) < 2:
            raise ValueError(f"NExT-QA candidates too short: {cands!r}")
        return [(chr(ord("A") + i), str(c)) for i, c in enumerate(cands[:5])]
    letters = []
    for i, letter in enumerate("ABCDE"):
        for key in (f"a{i}", f"A{i}", letter, letter.lower(),
                    f"option_{letter}", f"option{letter}"):
            if key in row and row[key] not in (None, ""):
                letters.append((letter, str(row[key])))
                break
    if len(letters) >= 2:
        return letters
    raise KeyError(
        f"NExT-QA row has no options; keys={sorted(row)}"
    )


def _nextqa_answer_letter(row: dict, options: list[tuple[str, str]]) -> str:
    raw = None
    for key in ("answer", "label", "correct", "a"):
        if key in row and row[key] not in (None, ""):
            raw = row[key]
            break
    if raw is None:
        raise KeyError(f"NExT-QA row has no answer; keys={sorted(row)}")
    if isinstance(raw, str) and raw.strip().upper()[:1] in "ABCDE":
        return raw.strip().upper()[:1]
    # MC config: answer is int64 in 0..4 (lmms-eval OPTIONS[doc["answer"]]).
    if isinstance(raw, (int, np.integer)):
        idx = int(raw)
        if 0 <= idx < len(options):
            return options[idx][0]
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
    text = str(raw).strip()
    for letter, opt in options:
        if opt.strip().lower() == text.lower():
            return letter
    raise ValueError(
        f"NExT-QA answer {raw!r} matches no option {options}"
    )


def _download_nextqa_zip(hf_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s/%s (video archive; large, Hub-cached)",
                hf_id, VIDEO_ZIP_NAME)
    path = hf_hub_download(
        repo_id=hf_id, filename=VIDEO_ZIP_NAME, repo_type="dataset",
    )
    return Path(path)


def _extract_nextqa_clips(
    zip_path: Path, video_dir: Path, video_ids: list[str],
) -> None:
    """Pull {id}.mp4 members out of videos.zip (any internal folder).

    lmms-eval looks up ``NExTVideo/{id}.mp4`` after a full unzip; we extract
    only the sampled ids, matching on filename stem so nested zip layouts
    (``NExTVideo/<cat>/<id>.mp4``) still resolve.
    """
    import zipfile

    needed = set(video_ids)
    already = {p.stem for p in video_dir.glob("*.mp4")}
    already |= {p.stem for p in video_dir.glob("*.MP4")}
    still = needed - already
    if not still:
        return
    logger.info("extracting %d/%d clips from %s",
                len(still), len(needed), zip_path.name)
    found: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if not base.lower().endswith(".mp4"):
                continue
            stem = Path(base).stem
            if stem not in still:
                continue
            if stem in found:
                raise RuntimeError(
                    f"{zip_path.name} has multiple members for video id "
                    f"{stem}: {found[stem]!r} and {name!r}"
                )
            found[stem] = name
        missing = still - set(found)
        if missing:
            raise FileNotFoundError(
                f"{zip_path} has no .mp4 whose filename stem matches "
                f"{len(missing)} requested NExT-QA video id(s); "
                f"e.g. {sorted(missing)[:8]}"
            )
        video_dir.mkdir(parents=True, exist_ok=True)
        for stem, member in found.items():
            dest_file = video_dir / f"{stem}.mp4"
            with zf.open(member) as src, dest_file.open("wb") as out:
                shutil.copyfileobj(src, out)


def _nextqa_video_id(row: dict) -> str:
    """MC `video` column is a VidOR id (int64); filename is `{id}.mp4`."""
    if "video" not in row or row["video"] in (None, ""):
        raise KeyError(f"NExT-QA row has no video field; keys={sorted(row)}")
    vid = row["video"]
    name = str(int(vid)) if isinstance(vid, (int, np.integer)) else str(vid)
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return name


def _nextqa_video_file(row: dict, video_dir: Path) -> Path:
    """lmms-eval ``get_video(cache_dir, doc["video"])``: id -> {id}.mp4."""
    name = _nextqa_video_id(row)
    for ext in (".mp4", ".MP4"):
        path = video_dir / f"{name}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"NExT-QA video {name!r} not under {video_dir} "
        f"(expected {name}.mp4; lmms-eval get_video layout)"
    )


def _save_video(src: Any, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(src, (str, Path)):
        src_path = Path(str(src))
        if src_path.is_file():
            shutil.copy2(src_path, dest)
            return
        raise FileNotFoundError(
            f"NExT-QA video path is not a local file: {src!r}"
        )
    if isinstance(src, dict) and "path" in src:
        _save_video(src["path"], dest)
        return
    if isinstance(src, dict) and "bytes" in src:
        dest.write_bytes(src["bytes"])
        return
    raise TypeError(
        f"NExT-QA video is {type(src).__name__}; expected a path or "
        f"bytes dict (keys={sorted(src) if isinstance(src, dict) else 'n/a'})"
    )


def materialize_video(n: int, seed: int, hf_id: str, force: bool) -> list[dict]:
    dest = _av_root() / "nextqa"
    meta_path = dest / "meta.json"
    if meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("hf_id") == hf_id and meta.get("seed") == seed
                and meta.get("n") == n):
            items_path = dest / "items.jsonl"
            if items_path.is_file():
                logger.info("reusing %s (n=%d seed=%d)", dest, n, seed)
                return [json.loads(l) for l in items_path.read_text(
                    encoding="utf-8").splitlines() if l.strip()]
    dest.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    logger.info("downloading %s config=%s split=%s (full, no streaming)",
                hf_id, VIDEO_HF_CONFIG, VIDEO_HF_SPLIT)
    ds = load_dataset(hf_id, VIDEO_HF_CONFIG, split=VIDEO_HF_SPLIT)
    if n > len(ds):
        raise ValueError(
            f"--n-video {n} exceeds {hf_id} {VIDEO_HF_CONFIG}/"
            f"{VIDEO_HF_SPLIT} size {len(ds)}"
        )
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    rows = [ds[idx] for idx in indices]
    video_ids = [_nextqa_video_id(row) for row in rows]
    video_dir = dest / VIDEO_CLIP_SUBDIR
    _extract_nextqa_clips(_download_nextqa_zip(hf_id), video_dir, video_ids)
    items = []
    for k, (idx, row) in enumerate(zip(indices, rows)):
        options = _nextqa_options(row)
        gold = _nextqa_answer_letter(row, options)
        question = str(row.get("question") or row.get("query") or "")
        if not question:
            raise KeyError(f"NExT-QA row has no question; keys={sorted(row)}")
        vid_path = dest / "clips" / f"{k:04d}.mp4"
        _save_video(_nextqa_video_file(row, video_dir), vid_path)
        item_id = str(row.get("video_id") or row.get("qid") or row.get("id")
                      or f"nq_{idx}")
        items.append({
            "id": item_id,
            "index": idx,
            "path": str(vid_path.resolve()),
            "question": question,
            "options": options,
            "gold": gold,
        })
    (dest / "items.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items),
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps({
        "hf_id": hf_id, "hf_config": VIDEO_HF_CONFIG, "split": VIDEO_HF_SPLIT,
        "seed": seed, "n": n, "item_ids": [it["id"] for it in items],
    }, indent=2), encoding="utf-8")
    logger.info("materialized %d NExT-QA clips -> %s", n, dest)
    return items


# ================================================================= generation

def _encode(vl: Any, messages: list[dict],
            processor_kwargs: dict | None = None) -> dict:
    """apply_chat_template through the loaded processor. Video frame
    sampling (num_frames / do_sample_frames) goes in processor_kwargs
    -- transformers 5.x rejects those as top-level **kwargs. A
    processor rejection is a hard error."""
    norm = vl.adapter.prepare_messages(messages)
    inputs = vl.processor.apply_chat_template(
        norm,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs or {},
    )
    if "input_ids" not in inputs:
        raise RuntimeError(
            f"apply_chat_template returned no input_ids; keys={list(inputs)}"
        )
    return vl._move_inputs_to_model(inputs)


def _generate(vl: Any, inputs: dict, max_new_tokens: int) -> str:
    import torch

    prompt_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = vl.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    return vl.processor.decode(
        out[0][prompt_len:], skip_special_tokens=True
    ).strip()


def _prompt_hash(messages: list[dict]) -> str:
    blob = json.dumps(messages, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _audio_messages(path: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": AUDIO_PROMPT},
            {"type": "audio", "path": path},
        ],
    }]


def _video_messages(item: dict) -> list[dict]:
    lines = [item["question"], ""]
    for letter, text in item["options"]:
        lines.append(f"  {letter}. {text}")
    lines.append("")
    lines.append(VIDEO_PROMPT_TAIL)
    return [{
        "role": "user",
        "content": [
            {"type": "video", "path": item["path"]},
            {"type": "text", "text": "\n".join(lines)},
        ],
    }]


def _run_pair(vl: Any, messages: list[dict], max_new_tokens: int,
              processor_kwargs: dict | None = None) -> tuple[str, str]:
    """Encode once; generate with the adapter, then with the adapter
    disabled (the frozen base). Returns (base_reply, ckpt_reply)."""
    inputs = _encode(vl, messages, processor_kwargs=processor_kwargs)
    ckpt_reply = _generate(vl, inputs, max_new_tokens)
    with vl.model.disable_adapter():
        base_reply = _generate(vl, inputs, max_new_tokens)
    return base_reply, ckpt_reply


# ==================================================================== main

def _summarize_audio(rows: list[dict]) -> dict:
    base_wers = [r["base_wer"] for r in rows]
    ckpt_wers = [r["ckpt_wer"] for r in rows]
    return {
        "n": len(rows),
        "base_wer": sum(base_wers) / len(rows),
        "ckpt_wer": sum(ckpt_wers) / len(rows),
        "base_exact": sum(r["base_exact"] for r in rows) / len(rows),
        "ckpt_exact": sum(r["ckpt_exact"] for r in rows) / len(rows),
        "unparseable": 0,
    }


def _summarize_video(rows: list[dict]) -> dict:
    n = len(rows)
    base_ok = sum(r["base_correct"] for r in rows)
    ckpt_ok = sum(r["ckpt_correct"] for r in rows)
    return {
        "n": n,
        "base_acc": base_ok / n,
        "ckpt_acc": ckpt_ok / n,
        "base_unparseable": sum(r["base_letter"] is None for r in rows),
        "ckpt_unparseable": sum(r["ckpt_letter"] is None for r in rows),
    }


def _print_table(audio: dict | None, video: dict | None,
                 audio_s: float, video_s: float) -> None:
    print()
    print("How to read this table")
    print("  delta = ckpt − base. WER: lower is better (positive delta =")
    print("  checkpoint transcribes worse). exact / accuracy: higher is")
    print("  better (positive delta = checkpoint is better). unparse is")
    print("  base/ckpt counts of video replies with no A–E letter; those")
    print("  count as wrong. N is small (~40 audio, ~30 video); a 1–2")
    print("  example swing is noise. This harness asks whether LoRA on")
    print("  the language side degraded audio/video — a near-tie is the")
    print("  healthy outcome.")
    print()
    print(f"{'modality':<10} {'metric':<12} {'base':>8} {'ckpt':>8} "
          f"{'delta':>8} {'N':>5} {'unparse':>8} {'sec':>8}")
    print("-" * 72)
    if audio is not None:
        d = audio["ckpt_wer"] - audio["base_wer"]
        print(f"{'audio':<10} {'WER':<12} {audio['base_wer']:8.3f} "
              f"{audio['ckpt_wer']:8.3f} {d:8.3f} {audio['n']:5d} "
              f"{'—':>8} {audio_s:8.1f}")
        d = audio["ckpt_exact"] - audio["base_exact"]
        print(f"{'audio':<10} {'exact':<12} {audio['base_exact']:8.3f} "
              f"{audio['ckpt_exact']:8.3f} {d:8.3f} {audio['n']:5d} "
              f"{'—':>8} {audio_s:8.1f}")
    if video is not None:
        d = video["ckpt_acc"] - video["base_acc"]
        unp = f"{video['base_unparseable']}/{video['ckpt_unparseable']}"
        print(f"{'video':<10} {'accuracy':<12} {video['base_acc']:8.3f} "
              f"{video['ckpt_acc']:8.3f} {d:8.3f} {video['n']:5d} "
              f"{unp:>8} {video_s:8.1f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint",
                        help="adapter folder name under weights/<arch>/")
    parser.add_argument("--n-audio", type=int, default=40)
    parser.add_argument("--n-video", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=8,
                        help="uniform video frame sample (Gemma 4 processor)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--audio-hf-id", default=AUDIO_HF_ID)
    parser.add_argument("--video-hf-id", default=VIDEO_HF_ID)
    parser.add_argument("--force", action="store_true",
                        help="re-materialize even if a matching sample exists")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    audio_items = [] if args.skip_audio else materialize_audio(
        args.n_audio, args.seed, args.audio_hf_id, args.force)
    video_items = [] if args.skip_video else materialize_video(
        args.n_video, args.seed, args.video_hf_id, args.force)

    from agent.model import VLModel, spec_for

    logger.info("loading gemma-4-12b + checkpoint %s", args.checkpoint)
    vl = VLModel(spec_for("gemma-4-12b"), checkpoint=args.checkpoint).load()
    if not hasattr(vl.model, "disable_adapter"):
        raise RuntimeError(
            f"loaded model is {type(vl.model).__name__}, not a PEFT "
            f"wrapper -- disable_adapter() is required for the base pass"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "logs" / f"av_eval_{args.checkpoint}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    audio_rows: list[dict] = []
    video_rows: list[dict] = []
    audio_s = video_s = 0.0

    with open(results_path, "w", encoding="utf-8") as fout:
        if audio_items:
            t0 = time.time()
            for i, item in enumerate(audio_items, start=1):
                messages = _audio_messages(item["path"])
                base, ckpt = _run_pair(vl, messages, max_new_tokens=192)
                rec = {
                    "id": item["id"], "modality": "audio",
                    "prompt_hash": _prompt_hash(messages),
                    "gold": item["gold"],
                    "base_reply": base, "ckpt_reply": ckpt,
                    "base_wer": word_error_rate(base, item["gold"]),
                    "ckpt_wer": word_error_rate(ckpt, item["gold"]),
                    "base_exact": exact_match_normalized(base, item["gold"]),
                    "ckpt_exact": exact_match_normalized(ckpt, item["gold"]),
                }
                audio_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                logger.info("audio %d/%d  base WER=%.3f  ckpt WER=%.3f",
                            i, len(audio_items), rec["base_wer"], rec["ckpt_wer"])
            audio_s = time.time() - t0

        if video_items:
            t0 = time.time()
            proc_kw = {"num_frames": args.num_frames, "do_sample_frames": True}
            for i, item in enumerate(video_items, start=1):
                messages = _video_messages(item)
                base, ckpt = _run_pair(
                    vl, messages, max_new_tokens=16, processor_kwargs=proc_kw)
                base_letter = parse_mc_letter(base)
                ckpt_letter = parse_mc_letter(ckpt)
                rec = {
                    "id": item["id"], "modality": "video",
                    "prompt_hash": _prompt_hash(messages),
                    "gold": item["gold"],
                    "base_reply": base, "ckpt_reply": ckpt,
                    "base_letter": base_letter, "ckpt_letter": ckpt_letter,
                    "base_correct": base_letter == item["gold"],
                    "ckpt_correct": ckpt_letter == item["gold"],
                }
                video_rows.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                logger.info(
                    "video %d/%d  gold=%s  base=%s  ckpt=%s",
                    i, len(video_items), item["gold"],
                    base_letter or "?", ckpt_letter or "?",
                )
            video_s = time.time() - t0

    audio_sum = _summarize_audio(audio_rows) if audio_rows else None
    video_sum = _summarize_video(video_rows) if video_rows else None
    summary = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "audio": audio_sum, "video": video_sum,
        "audio_seconds": audio_s, "video_seconds": video_s,
        "results": str(results_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    _print_table(audio_sum, video_sum, audio_s, video_s)
    logger.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
