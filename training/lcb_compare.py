"""LiveCodeBench v6: raw Gemma 4 12B-it against one full-save trunk.

HumanEval is the wrong probe for this harness. With thinking off and
greedy decoding, ``google/gemma-4-12B-it`` scores in the mid-80s to the
90s on it, so a real drop in the finetune has almost no room to show.
The 72% LiveCodeBench v6 figure on the Gemma 4 card is thinking mode.

This run is the newest LiveCodeBench window
(``livecodebench/code_generation_lite``, file ``test6.jsonl``, config
``v6``): 175 problems, 43 easy, 52 medium, 80 hard, from AtCoder and
LeetCode in 2025. Eighty of the 175 are hard, so the headline cannot
sit in the 90s unless the model is actually solving contest problems.
Thinking stays off, sampling stays greedy, and ``max_new_tokens`` stays
1024, which is the setup the model will actually be run under. The
problem count matches HumanEval, and the statements are short, so the
wall clock stays in the same range even though a contest solution is
longer than a HumanEval function.

Each problem is graded on the lite public and private tests. A wrong
answer stops at that test and is recorded as ``wrong answer on test N``.
A crash is the exception line. Passing means every test passed. The
per-test limit defaults to 6 seconds. The full save contributes its
Gemma trunk only (``weights/full/<name>/gemma``). The Localizer and S1
are not loaded, and no coordinate code is added.

Results go to ``data_game/lcb_v6/``.

    python -m training.lcb_compare
    python -m training.lcb_compare <name under weights/full>
    python -m training.lcb_compare --random 40
    python -m training.lcb_compare --random 40 --seed 7
    python -m training.lcb_compare --limit 40

With no name, the save is the first line of a plain ``ls`` of
``weights/full``. Run this on the GPU box.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import pickle
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger("lcb_v6")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_game" / "lcb_v6"
LOG_PATH = REPO_ROOT / "data_game" / "lcb_v6_orchestrator.log"
HEARTBEAT_PATH = DATA_DIR / "heartbeat.json"
RESULTS_PATH = DATA_DIR / "results.jsonl"
SUMMARY_PATH = DATA_DIR / "summary.json"
CRASH_PATH = DATA_DIR / "crash.txt"

DATASET_ID = "livecodebench/code_generation_lite"
DATASET_FILE = "test6.jsonl"
EXPECTED_WINDOW = 175
EXPECTED_DIFFICULTY = {"easy": 43, "medium": 52, "hard": 80}
BASE_HF_ID = "google/gemma-4-12B-it"
BASE_KEY = "gemma-4-12b"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TIMEOUT_S = 6.0
DEFAULT_RANDOM_SEED = 0

# The LiveCodeBench grader prepends this so a solution can call math,
# heapq, List, and so on without importing them. Copied from
# lcb_runner/evaluation/testing_util.py.
PRELUDE = (
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from sys import *\n"
    "from json import *\n"
    "from builtins import *\n"
    "from typing import *\n"
    "import string\n"
    "import re\n"
    "import datetime\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import copy\n"
    "import math\n"
    "import random\n"
    "import statistics\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "import io\n"
    "import sys\n"
    "import json\n"
    "sys.setrecursionlimit(50000)\n"
)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests. Do not use "
    "system calls like exit. The first python code block must be the "
    "whole solution."
)

_FENCE = re.compile(
    r"```(?:python|py)?[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE,
)

# Runs in a fresh interpreter. Compares one functional call against the
# JSON output in case.json. Exit 0 passes, exit 2 is a wrong answer,
# anything else is a crash.
_FUNCTIONAL_RUNNER = r"""
import json
import sys


def main():
    spec = json.loads(open("case.json", encoding="utf-8").read())
    namespace = {}
    source = open("solution.py", encoding="utf-8").read()
    exec(compile(source, "solution.py", "exec"), namespace)
    fn_name = spec["fn_name"]
    if "Solution" in namespace:
        fn = getattr(namespace["Solution"](), fn_name)
    else:
        fn = namespace[fn_name]
    args = [json.loads(line) for line in spec["input"].split("\n")]
    try:
        got = fn(*args)
    except SystemExit:
        print("SystemExit", file=sys.stderr)
        return 1
    if isinstance(got, tuple):
        got = list(got)
    expected = json.loads(spec["output"])
    if got != expected:
        return 2
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except Exception as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(code)
"""


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


def messages_for(problem: dict[str, Any]) -> list[dict[str, str]]:
    """LiveCodeBench's generic chat prompt, stdin or starter code."""
    starter = str(problem.get("starter_code") or "").strip()
    if starter:
        how = (
            "Use the following starter code to write the solution and "
            "enclose your code within delimiters.\n\n"
            f"```python\n{starter}\n```"
        )
    else:
        how = (
            "Read the inputs from stdin, solve the problem, and write the "
            "answer to stdout. Do not directly test on the sample inputs. "
            "Enclose your code within delimiters. When the python program "
            "runs, it must read the inputs, run the algorithm, and write "
            "the output to stdout.\n\n"
            "```python\n# YOUR CODE HERE\n```"
        )
    user = (
        f"### Question:\n{problem['question_content']}\n\n"
        f"### Format: {how}\n\n"
        "### Answer: (use the provided format with backticks)\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_program(reply: str) -> str:
    """The first non-empty fenced block, or the reply itself."""
    text = reply.strip()
    for match in _FENCE.finditer(text):
        block = match.group(1).strip()
        if block:
            return block
    return text


def decode_tests(raw: str, task_id: str, which: str) -> list[dict[str, Any]]:
    """Public tests are JSON. Private tests are JSON, or a JSON string
    stored as base64(zlib(pickle(text))), which is how LiveCodeBench
    ships them. The pickle is only accepted when it unpickles to a string.
    """
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{task_id} {which} tests are missing")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            blob = pickle.loads(
                zlib.decompress(base64.b64decode(raw.encode("utf-8")))
            )
        except Exception as exc:
            raise RuntimeError(
                f"{task_id} {which} tests are neither JSON nor "
                f"the LiveCodeBench compressed form ({type(exc).__name__})"
            ) from exc
        if isinstance(blob, (bytes, bytearray)):
            blob = blob.decode("utf-8")
        if not isinstance(blob, str):
            raise RuntimeError(
                f"{task_id} {which} tests unpickled to "
                f"{type(blob).__name__}, expected a JSON string"
            )
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{task_id} {which} tests unpickled to text that is not JSON"
            ) from exc
    if not isinstance(data, list):
        raise RuntimeError(f"{task_id} {which} tests are not a list")
    for index, test in enumerate(data):
        if not isinstance(test, dict):
            raise RuntimeError(
                f"{task_id} {which} test {index} is not an object"
            )
        for key in ("input", "output", "testtype"):
            if key not in test:
                raise RuntimeError(
                    f"{task_id} {which} test {index} is missing {key}"
                )
        if not isinstance(test["input"], str) or not isinstance(test["output"], str):
            raise RuntimeError(
                f"{task_id} {which} test {index} input/output must be strings"
            )
    return data


def _check_tests(
    task_id: str, tests: list[dict[str, Any]], fn_name: str | None,
) -> None:
    if not tests:
        raise RuntimeError(f"{task_id} has no tests")
    expect = "functional" if fn_name else "stdin"
    for index, test in enumerate(tests):
        if test["testtype"] != expect:
            raise RuntimeError(
                f"{task_id} test {index} is {test['testtype']!r}, "
                f"expected {expect!r}"
            )
        if fn_name is None:
            continue
        try:
            for line in test["input"].split("\n"):
                json.loads(line)
            json.loads(test["output"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{task_id} functional test {index} is not JSON"
            ) from exc


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


def _sort_key(row: dict[str, Any]) -> Any:
    key = row.get("sort_key")
    if key is not None:
        return key
    return str(row["task_id"])


def problems_from_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The v6 window, earliest contest first.

    The window size and the easy/medium/hard counts are checked, so a
    different file cannot be scored as this benchmark.
    """
    if len(raw_rows) != EXPECTED_WINDOW:
        raise RuntimeError(
            f"{DATASET_FILE} has {len(raw_rows)} problems, "
            f"expected {EXPECTED_WINDOW}"
        )
    difficulty_counts = {name: 0 for name in EXPECTED_DIFFICULTY}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        difficulty = str(raw.get("difficulty") or "")
        if difficulty not in EXPECTED_DIFFICULTY:
            raise RuntimeError(f"unexpected difficulty {difficulty!r}")
        difficulty_counts[difficulty] += 1
        task_id = str(raw.get("question_id") or "")
        if not task_id:
            raise RuntimeError("a problem has no question_id")
        if task_id in seen:
            raise RuntimeError(f"duplicate question_id {task_id}")
        seen.add(task_id)
        try:
            meta = json.loads(raw["metadata"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{task_id} metadata is not JSON") from exc
        if not isinstance(meta, dict):
            raise RuntimeError(f"{task_id} metadata is not an object")
        fn_name = meta.get("func_name")
        if fn_name is not None and not isinstance(fn_name, str):
            raise RuntimeError(f"{task_id} func_name is not a string")
        if not fn_name:
            fn_name = None
        for field in ("question_content", "platform", "contest_date"):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise RuntimeError(f"{task_id} is missing {field}")
        public = decode_tests(raw.get("public_test_cases", ""), task_id, "public")
        private = decode_tests(
            raw.get("private_test_cases", ""), task_id, "private",
        )
        tests = public + private
        _check_tests(task_id, tests, fn_name)
        rows.append({
            "task_id": task_id,
            "sort_key": (raw["contest_date"], task_id),
            "difficulty": difficulty,
            "platform": raw["platform"],
            "question_content": raw["question_content"],
            "starter_code": str(raw.get("starter_code") or ""),
            "fn_name": fn_name,
            "tests": tests,
        })
    if difficulty_counts != EXPECTED_DIFFICULTY:
        raise RuntimeError(
            f"{DATASET_FILE} difficulty counts are {difficulty_counts}, "
            f"expected {EXPECTED_DIFFICULTY}"
        )
    rows.sort(key=_sort_key)
    return rows


def load_problems() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    raw_rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{DATASET_FILE} line {line_no} is not JSON"
                ) from exc
    return problems_from_rows(raw_rows)


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
    and then sorted by contest date, so both models and a resumed run see
    the same problems in the same order. The default seed is 0. ``limit``
    takes the prefix of ``rows``; :func:`load_problems` sorts that list
    earliest contest first.
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
        picked.sort(key=_sort_key)
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


def stdout_match(got: str, expected: str) -> bool:
    """Line-stripped equality, then decimal equality, as LiveCodeBench does."""
    got_lines = [line.strip() for line in got.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(got_lines) != len(expected_lines):
        return False
    for got_line, expected_line in zip(got_lines, expected_lines):
        if got_line == expected_line:
            continue
        if _decimals(got_line) is None or _decimals(got_line) != _decimals(expected_line):
            return False
    return True


def _decimals(line: str) -> list[Decimal] | None:
    parts = line.split()
    if not parts:
        return None
    try:
        return [Decimal(part) for part in parts]
    except Exception:
        return None


def _run_limited(
    cmd: list[str],
    cwd: Path,
    timeout_s: float,
    input_bytes: bytes | None = None,
) -> tuple[int | None, str, str]:
    """One child. ``returncode`` None means the time limit killed it."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=_child_env(),
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(input_bytes, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return None, "", "timeout"
    return (
        int(proc.returncode),
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def _reason_from_stderr(code: int, err: str) -> str:
    lines = [line for line in err.strip().splitlines() if line]
    tail = lines[-1] if lines else f"exit {code}"
    return tail[:300]


def grade_solution(
    program: str,
    tests: list[dict[str, Any]],
    fn_name: str | None,
    timeout_s: float,
) -> tuple[bool, str]:
    """Run every test in a fresh ``python -I`` process.

    Stops at the first failure. A timeout, a non-zero exit, or a program
    that cannot be written fails the problem. It does not abort the sweep.
    A wrong answer is reported as such, separately from a crash.
    """
    if not str(program).strip():
        return False, "empty"
    if not tests:
        raise RuntimeError("grade_solution called with no tests")
    with tempfile.TemporaryDirectory(prefix="lcb_") as tmp:
        root = Path(tmp)
        try:
            (root / "solution.py").write_text(
                PRELUDE + "\n" + program, encoding="utf-8",
            )
            if fn_name:
                (root / "runner.py").write_text(
                    _FUNCTIONAL_RUNNER, encoding="utf-8",
                )
        except (OSError, UnicodeError) as exc:
            return False, f"write:{type(exc).__name__}"
        for index, test in enumerate(tests):
            if fn_name:
                (root / "case.json").write_text(
                    json.dumps({
                        "fn_name": fn_name,
                        "input": test["input"],
                        "output": test["output"],
                    }),
                    encoding="utf-8",
                )
                code, _out, err = _run_limited(
                    [sys.executable, "-I", "runner.py"], root, timeout_s,
                )
                if code is None:
                    return False, "timeout"
                if code == 0:
                    continue
                if code == 2:
                    return False, f"wrong answer on test {index}"
                return False, _reason_from_stderr(code, err)
            code, out, err = _run_limited(
                [sys.executable, "-I", "solution.py"],
                root,
                timeout_s,
                input_bytes=test["input"].encode(),
            )
            if code is None:
                return False, "timeout"
            if code != 0:
                return False, _reason_from_stderr(code, err)
            if not stdout_match(out, test["output"]):
                return False, f"wrong answer on test {index}"
    return True, "pass"


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
        messages = messages_for(problem)
        inputs, n_prompt = encode_prompt(vl, messages)
        _heartbeat(model_tag, task_id, index, total, n_prompt)
        _flush_logs()
        started = time.perf_counter()
        reply, n_new = generate_reply(vl, inputs, n_prompt, max_new_tokens)
        program = extract_program(reply) if reply.strip() else ""
        passed, reason = grade_solution(
            program, problem["tests"], problem["fn_name"], timeout_s,
        )
        elapsed = round(time.perf_counter() - started, 3)
        row = {
            "model": model_tag,
            "model_id": model_id,
            "task_id": task_id,
            "difficulty": problem["difficulty"],
            "platform": problem["platform"],
            "passed": passed,
            "reason": reason,
            "n_tests": len(problem["tests"]),
            "n_prompt_tokens": n_prompt,
            "n_new_tokens": n_new,
            "seconds": elapsed,
            "reply": reply,
        }
        _append_jsonl(RESULTS_PATH, row)
        done.add((model_tag, task_id))
        logger.info(
            "%s %s %s %s (%d/%d, %d tests, %d new tokens, %.1fs) %s",
            model_tag, task_id, problem["difficulty"],
            "PASS" if passed else "FAIL",
            index, total, len(problem["tests"]), n_new, elapsed, reason,
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
        "dataset_file": DATASET_FILE,
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
        by_difficulty: dict[str, dict[str, int]] = {}
        by_platform: dict[str, dict[str, int]] = {}
        for task_id in task_ids:
            row = by_id[task_id]
            for field, buckets in (
                ("difficulty", by_difficulty),
                ("platform", by_platform),
            ):
                if field not in row:
                    raise RuntimeError(f"{tag} {task_id} has no {field}")
                bucket = buckets.setdefault(
                    str(row[field]), {"passed": 0, "n": 0},
                )
                bucket["n"] += 1
                if row["passed"]:
                    bucket["passed"] += 1
        summary[tag] = {
            "model_id": by_id[task_ids[0]]["model_id"],
            "passed": passed,
            "n": len(task_ids),
            "pass_at_1": round(passed / len(task_ids), 4),
            "by_difficulty": by_difficulty,
            "by_platform": by_platform,
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
        help="score the first N problems, earliest contest first. "
             f"Default: all {EXPECTED_WINDOW}. Not with --random.",
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
        help="seconds allowed for one test. The problem stops at the first failure.",
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
        "LiveCodeBench v6 %d problems (%s, seed=%s), thinking off, "
        "greedy, max_new_tokens=%d, per-test timeout=%ss, save=%s",
        selection["n"], selection["mode"], selection["seed"],
        args.max_new_tokens, args.timeout, folder.name,
    )
    if selection["mode"] != "all":
        logger.info("task ids: %s", " ".join(selection["task_ids"]))
    _flush_logs()
    monitor = VramMonitor("lcb_v6")
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
        logger.exception("lcb v6 compare crashed")
        _flush_logs()
        raise
    finally:
        if vl is not None:
            _unload(vl)
        monitor.finish()


if __name__ == "__main__":
    main()
