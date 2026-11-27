"""HumanEval pass@1: raw Gemma 4 12B-it against one full-save trunk.

Why this set. A third-party greedy run of ``google/gemma-4-12B-it`` with
reasoning off scored 57.3% on HumanEval and 56.7% on HumanEval+ (42.1%
on MBPP). That is the middle of the range: high enough that a drop is
not noise, low enough that the baseline is not already saturated.
LiveCodeBench v6 on the model card is 72%, and GSM8K in that same
third-party table is 95%, so those would hide a modest regression.
Thinking stays off so the run matches that measurement and stays inside
a couple of hours on a 96 GB RTX 6000. One greedy sample, 164 problems,
``max_new_tokens`` 1024. The two models see the same prompt.

The full save contributes its Gemma trunk only (``weights/full/<name>/gemma``).
The Localizer and S1 are not loaded, and no coordinate code is added.
Text coding has no image tokens for a localizer to read.

    python -m training.humaneval_compare
    python -m training.humaneval_compare <name under weights/full>
    python -m training.humaneval_compare --random 40
    python -m training.humaneval_compare --random 40 --seed 7
    python -m training.humaneval_compare --limit 40

With no name, the save is the first line of a plain ``ls`` of
``weights/full``. Run this on the GPU box. Logs land in ``data_game/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("humaneval_compare")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_game" / "humaneval_compare"
LOG_PATH = REPO_ROOT / "data_game" / "humaneval_compare_orchestrator.log"
HEARTBEAT_PATH = DATA_DIR / "heartbeat.json"
RESULTS_PATH = DATA_DIR / "results.jsonl"
SUMMARY_PATH = DATA_DIR / "summary.json"
CRASH_PATH = DATA_DIR / "crash.txt"

DATASET_ID = "openai/openai_humaneval"
BASE_HF_ID = "google/gemma-4-12B-it"
BASE_KEY = "gemma-4-12b"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_RANDOM_SEED = 0

SYSTEM_PROMPT = (
    "You are a careful Python programmer. "
    "Reply with one python code block and no other text."
)

_FENCE = re.compile(
    r"```(?:python)?[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE,
)


def first_ls_name(directory: Path) -> str:
    """First name a plain ``ls`` of ``directory`` prints."""
    out = subprocess.run(
        ["ls", "-1", str(directory)],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        detail = out.stderr.strip() or f"exit {out.returncode}"
        raise RuntimeError(f"ls {directory} failed: {detail}")
    names = [line for line in out.stdout.splitlines() if line]
    if not names:
        raise RuntimeError(f"{directory} is empty")
    return names[0]


def select_full_save(root: Path, name: str | None) -> Path:
    """One assembled save under ``root``.

    ``name`` None uses :func:`first_ls_name`. The directory must contain
    ``gemma/gemma_meta.json``. A first ``ls`` entry that is not a save
    raises; nothing later in the listing is tried.
    """
    if not root.is_dir():
        raise RuntimeError(f"no full-save directory at {root}")
    if name is None:
        name = first_ls_name(root)
    elif name != Path(name).name or name in (".", ".."):
        raise RuntimeError(
            f"save name must be one directory under {root}, got {name!r}"
        )
    folder = root / name
    if not folder.is_dir():
        raise RuntimeError(f"{folder} is not a directory")
    meta = folder / "gemma" / "gemma_meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"full save {folder} has no {meta}")
    return folder


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Write a self-contained solution. Include the imports, "
                "the function signature, and the docstring.\n\n"
                + prompt
            ),
        },
    ]


def _prompt_header(prompt: str, entry_point: str) -> str:
    """Import lines that sit above ``def entry_point`` in the HumanEval prompt."""
    lines: list[str] = []
    needle = f"def {entry_point}"
    for line in prompt.splitlines():
        if line.startswith(needle):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def extract_program(reply: str, prompt: str, entry_point: str) -> str:
    """The Python the reply offers, stitched to the prompt's imports.

    A fenced block that defines ``entry_point`` wins. A reply with no
    fence is sliced from that ``def``. A body with no ``def`` is appended
    to the original prompt, which already holds the signature.
    """
    text = reply.strip()
    blocks = [m.group(1).strip() for m in _FENCE.finditer(text)]
    chosen = ""
    for block in blocks:
        if f"def {entry_point}" in block:
            chosen = block
            break
    if not chosen and blocks:
        chosen = blocks[0]
    if not chosen:
        idx = text.find(f"def {entry_point}")
        chosen = text[idx:].strip() if idx >= 0 else text
    if f"def {entry_point}" not in chosen:
        return prompt + chosen
    header = _prompt_header(prompt, entry_point)
    if header and header not in chosen:
        return header + "\n\n" + chosen
    return chosen


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_done(path: Path) -> set[tuple[str, str]]:
    """``(model, task_id)`` pairs already scored. A duplicate row raises."""
    done: set[tuple[str, str]] = set()
    if not path.is_file():
        return done
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["model"], row["task_id"])
        if key in done:
            raise RuntimeError(f"duplicate result {key} in {path} line {line_no}")
        done.add(key)
    return done


def _task_index(row: dict[str, Any]) -> int:
    return int(str(row["task_id"]).split("/")[-1])


def load_problems() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="test")
    rows = [dict(row) for row in ds]
    rows.sort(key=_task_index)
    if len(rows) != 164:
        raise RuntimeError(
            f"{DATASET_ID} test split has {len(rows)} rows, expected 164"
        )
    return rows


def choose_problems(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    random_n: int | None,
    seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All problems, the first ``limit``, or a seeded sample of ``random_n``.

    ``--limit`` and ``--random`` together raise. ``--seed`` without
    ``--random`` raises. A sample is drawn with ``random.Random(seed)``
    and then sorted by task number, so both models and a resumed run see
    the same problems in the same order. The default seed is 0.
    """
    n = len(rows)
    if limit is not None and random_n is not None:
        raise ValueError("pass --limit or --random, not both")
    if seed is not None and random_n is None:
        raise ValueError("--seed applies only with --random")
    if limit is None and random_n is None:
        chosen = list(rows)
        mode = "all"
        used_seed = None
    elif limit is not None:
        if limit < 1 or limit > n:
            raise ValueError(f"--limit must be 1..{n}, got {limit}")
        chosen = list(rows[:limit])
        mode = "first"
        used_seed = None
    else:
        assert random_n is not None
        if random_n < 1 or random_n > n:
            raise ValueError(f"--random must be 1..{n}, got {random_n}")
        used_seed = DEFAULT_RANDOM_SEED if seed is None else seed
        picked = random.Random(used_seed).sample(list(rows), random_n)
        picked.sort(key=_task_index)
        chosen = picked
        mode = "random"
    return chosen, {
        "mode": mode,
        "n": len(chosen),
        "seed": used_seed,
        "task_ids": [str(row["task_id"]) for row in chosen],
    }


def _child_env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    path = os.environ.get("PATH")
    if path:
        env["PATH"] = path
    return env


def execute_program(
    program: str, test: str, entry_point: str, timeout_s: float,
) -> tuple[bool, str]:
    """Run one candidate in a fresh ``python -I`` process.

    Returns ``(passed, reason)``. A timeout, a non-zero exit, or a
    program that cannot be written is a failed problem, not a sweep abort.
    """
    source = program + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory(prefix="humaneval_") as tmp:
        script = Path(tmp) / "candidate.py"
        try:
            script.write_text(source, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return False, f"write:{type(exc).__name__}"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                env=_child_env(),
                timeout=timeout_s,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, "pass"
    err = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
    tail = err[-1] if err else f"exit {proc.returncode}"
    return False, tail[:300]


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def encode_prompt(vl: Any, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int]:
    """Tokenize with thinking off. The length is ``input_ids.shape[1]``."""
    encoded = vl.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = vl._move_inputs_to_model(encoded)
    return inputs, int(inputs["input_ids"].shape[1])


def generate_reply(
    vl: Any, inputs: dict[str, Any], n_prompt: int, max_new_tokens: int,
) -> tuple[str, int]:
    """Greedy continuation. The new-token count is the generated tensor length."""
    import torch

    with torch.inference_mode():
        out = vl.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_ids = out[0][n_prompt:]
    text = vl.processor.decode(new_ids, skip_special_tokens=True).strip()
    return text, int(new_ids.shape[0])


def _heartbeat(stage: str, task_id: str, index: int, total: int, n_prompt: int) -> None:
    _write_json_atomic(HEARTBEAT_PATH, {
        "stage": stage,
        "task_id": task_id,
        "index": index,
        "total": total,
        "n_prompt_tokens": n_prompt,
        "t": round(time.time(), 1),
    })


def score_model(
    vl: Any,
    *,
    model_tag: str,
    model_id: str,
    problems: list[dict[str, Any]],
    done: set[tuple[str, str]],
    max_new_tokens: int,
    timeout_s: float,
) -> None:
    total = len(problems)
    for index, problem in enumerate(problems, 1):
        task_id = str(problem["task_id"])
        if (model_tag, task_id) in done:
            logger.info("%s %s already scored (%d/%d)", model_tag, task_id, index, total)
            continue
        prompt = str(problem["prompt"])
        entry = str(problem["entry_point"])
        messages = messages_for(prompt)
        inputs, n_prompt = encode_prompt(vl, messages)
        _heartbeat(model_tag, task_id, index, total, n_prompt)
        _flush_logs()
        started = time.perf_counter()
        reply, n_new = generate_reply(vl, inputs, n_prompt, max_new_tokens)
        if not reply.strip():
            passed, reason = False, "empty"
            program = ""
        else:
            program = extract_program(reply, prompt, entry)
            passed, reason = execute_program(
                program, str(problem["test"]), entry, timeout_s,
            )
        elapsed = round(time.perf_counter() - started, 3)
        row = {
            "model": model_tag,
            "model_id": model_id,
            "task_id": task_id,
            "passed": passed,
            "reason": reason,
            "n_prompt_tokens": n_prompt,
            "n_new_tokens": n_new,
            "seconds": elapsed,
            "reply": reply,
        }
        _append_jsonl(RESULTS_PATH, row)
        done.add((model_tag, task_id))
        logger.info(
            "%s %s %s (%d/%d, %d new tokens, %.1fs) %s",
            model_tag, task_id, "PASS" if passed else "FAIL",
            index, total, n_new, elapsed, reason,
        )
        _flush_logs()


def _load_base():
    from agent.model import VLModel, spec_for

    vl = VLModel(spec_for(BASE_KEY))
    vl.load()
    if vl.model is None:
        raise RuntimeError("baseline load left no model")
    vl.model.eval()
    return vl


def _load_finetune(folder: Path):
    """Base checkpoint, then the save's Gemma slice. No readout, no S1."""
    import torch
    from peft import PeftModel

    from agent.model import VLModel, spec_for

    meta_path = folder / "gemma" / "gemma_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("hf_id") != BASE_HF_ID or meta.get("model_key") != BASE_KEY:
        raise RuntimeError(
            f"{meta_path} must be {BASE_KEY} / {BASE_HF_ID}, got {meta!r}"
        )
    fmt = meta.get("format")
    if fmt not in ("peft", "state_dict"):
        raise RuntimeError(
            f"{meta_path} format must be peft or state_dict, got {fmt!r}"
        )
    vl = VLModel(spec_for(BASE_KEY))
    vl.load()
    if vl.model is None:
        raise RuntimeError("finetune base load left no model")
    gemma_dir = folder / "gemma"
    if fmt == "peft":
        vl.model = PeftModel.from_pretrained(
            vl.model, str(gemma_dir), is_trainable=False,
        )
    else:
        weights = gemma_dir / "gemma.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"missing {weights}")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        vl.model.load_state_dict(state)
    vl.model.eval()
    return vl


def _unload(vl: Any) -> None:
    import gc

    import torch

    if vl is not None:
        vl.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize(
    problems: list[dict[str, Any]],
    save_name: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    task_ids = [str(row["task_id"]) for row in problems]
    rows = []
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    summary: dict[str, Any] = {
        "dataset": DATASET_ID,
        "n": len(task_ids),
        "thinking": False,
        "do_sample": False,
        "save": save_name,
        "selection": selection["mode"],
        "seed": selection["seed"],
        "task_ids": task_ids,
    }
    for tag in ("baseline", "finetune"):
        by_id = {
            row["task_id"]: row for row in rows if row["model"] == tag
        }
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise RuntimeError(
                f"{tag} is missing {len(missing)} tasks, first {missing[0]}"
            )
        passed = sum(1 for task_id in task_ids if by_id[task_id]["passed"])
        summary[tag] = {
            "model_id": by_id[task_ids[0]]["model_id"],
            "passed": passed,
            "n": len(task_ids),
            "pass_at_1": round(passed / len(task_ids), 4),
        }
    base = summary["baseline"]["passed"]
    tuned = summary["finetune"]["passed"]
    summary["delta_passed"] = tuned - base
    _write_json_atomic(SUMMARY_PATH, summary)
    logger.info(
        "baseline %s: %d/%d (%.1f%%)",
        summary["baseline"]["model_id"],
        summary["baseline"]["passed"],
        summary["baseline"]["n"],
        100.0 * summary["baseline"]["pass_at_1"],
    )
    logger.info(
        "finetune %s: %d/%d (%.1f%%)",
        save_name,
        summary["finetune"]["passed"],
        summary["finetune"]["n"],
        100.0 * summary["finetune"]["pass_at_1"],
    )
    logger.info("delta (finetune - baseline): %+d passed", summary["delta_passed"])
    _flush_logs()
    return summary


def _configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(stream)
    root.addHandler(file_handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="directory name under weights/full. "
             "Default: the first name a plain ls of that directory prints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score the first N problems (HumanEval/0 onward). "
             "Default: all 164. Not with --random.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        metavar="N",
        help="score a random N problems instead of the first N. "
             "Both models get that same draw. Not with --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"seed for --random. Default {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="seconds allowed for one candidate's tests.",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        raise SystemExit(f"--max-new-tokens must be >= 1, got {args.max_new_tokens}")
    if args.timeout <= 0:
        raise SystemExit(f"--timeout must be > 0, got {args.timeout}")
    if args.limit is not None and args.random is not None:
        raise SystemExit("pass --limit or --random, not both")
    if args.seed is not None and args.random is None:
        raise SystemExit("--seed applies only with --random")

    _configure_logging()
    # .env before the dataset download and the model load.
    import agent.config  # noqa: F401
    from neural_net.paths import full_weights_dir
    from training.run_weekend import VramMonitor

    folder = select_full_save(full_weights_dir(), args.name)
    problems, selection = choose_problems(
        load_problems(),
        limit=args.limit,
        random_n=args.random,
        seed=args.seed,
    )
    done = load_done(RESULTS_PATH)
    logger.info(
        "HumanEval %d problems (%s, seed=%s), thinking off, greedy, "
        "max_new_tokens=%d, timeout=%ss, save=%s",
        selection["n"], selection["mode"], selection["seed"],
        args.max_new_tokens, args.timeout, folder.name,
    )
    if selection["mode"] != "all":
        logger.info("task ids: %s", " ".join(selection["task_ids"]))
    _flush_logs()
    monitor = VramMonitor("humaneval_compare")
    vl = None
    try:
        monitor.set_stage("baseline")
        vl = _load_base()
        score_model(
            vl,
            model_tag="baseline",
            model_id=BASE_HF_ID,
            problems=problems,
            done=done,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout,
        )
        _unload(vl)
        vl = None
        monitor.set_stage("finetune")
        vl = _load_finetune(folder)
        score_model(
            vl,
            model_tag="finetune",
            model_id=str(folder),
            problems=problems,
            done=done,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout,
        )
        _unload(vl)
        vl = None
        summarize(problems, folder.name, selection)
    except Exception:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CRASH_PATH.write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception("humaneval compare crashed")
        _flush_logs()
        raise
    finally:
        if vl is not None:
            _unload(vl)
        monitor.finish()


if __name__ == "__main__":
    main()
