"""Optional prep utility: generation-based self-distillation targets.

The DEFAULT preservation path is the KD loss in train.py (match the
student's logits to the frozen base over the dataset's own target tokens --
no generation, no extra storage). This utility implements the documented
ALTERNATIVE (TRAINING_EXTRA_DATASETS.md): have the UNTRAINED BASE model
answer each materialized prompt itself, then train on those outputs with
plain CE. Use it when KD's logit matching proves too rigid (it pins every
position of someone else's text) or too memory-hungry -- CE on base-generated
text anchors the same behavior with an ordinary forward pass.

Writes ``data_external/<name>/data_selfdistill.jsonl`` beside the original
(same record shape, ``loss`` forced to "ce", ``meta.self_distill`` set), so
a run script can swap it in with a plain ``JsonlSource``. The original
data.jsonl is never touched.

GPU required (loads the base model via agent.model); run on the remote box::

    python -m training.generate_self_distill --dataset slimorca
    python -m training.generate_self_distill --dataset cauldron_vqav2 \
        --limit 2000 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.external_data import load_manifest

logger = logging.getLogger("train.self_distill")


def _absolutized(messages: list[dict], base_dir: Path) -> list[dict]:
    """Copy of ``messages`` with dataset-relative image paths made absolute
    (mirrors ExternalSource.examples; the ORIGINAL record keeps relative
    paths so the output file stays portable)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if (isinstance(part, dict) and part.get("type") == "image"
                        and not Path(part["url"]).is_absolute()):
                    part = {**part, "url": str(base_dir / part["url"])}
                new_content.append(part)
            m = {**m, "content": new_content}
        out.append(m)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True,
                        help="manifest entry name (must be materialized)")
    parser.add_argument("--limit", type=int, default=None,
                        help="regenerate only the first N examples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    entries = {e.name: e for e in load_manifest()}
    if args.dataset not in entries:
        parser.error(f"unknown dataset {args.dataset!r}; "
                     f"manifest has: {sorted(entries)}")
    entry = entries[args.dataset]
    data_path = entry.data_dir / "data.jsonl"
    if not data_path.is_file():
        raise SystemExit(
            f"{data_path} missing -- run: python -m training.download_external "
            f"--only {entry.name}"
        )
    out_path = entry.data_dir / "data_selfdistill.jsonl"

    # Base model, explicitly WITHOUT any adapter checkpoint: self-distillation
    # targets must come from the untrained model, whatever MODEL_CHECKPOINT
    # happens to say in .env.
    from agent.model import get_model, set_default_checkpoint

    set_default_checkpoint(None)
    model = get_model()
    logger.info("base model loaded: %s (checkpoint=None)", model.spec.key)

    n_done = 0
    with open(data_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            if args.limit is not None and n_done >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            reply = model.generate(
                _absolutized(rec["messages"], entry.data_dir),
                max_new_tokens=args.max_new_tokens,
            )
            if not reply.strip():
                raise RuntimeError(
                    f"{data_path}:{lineno}: base model produced an empty "
                    "reply -- refusing to write an empty training target"
                )
            rec["target_text"] = reply
            rec["loss"] = "ce"
            rec.setdefault("meta", {})["self_distill"] = True
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                logger.info("%d examples regenerated ...", n_done)

    logger.info("done: %d examples -> %s", n_done, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
