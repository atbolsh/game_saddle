"""Supervised pretraining for S1.

Each example is one random 0–3 gold board, one target coordinate, and the
oracle's next primitive. Frames go through the same label-safe image noise
as the rest of training. Run on the remote GPU box, from the repo root::

    python -m neural_net.train_s1 --batch-size 384 --workers 48

That command trains until the wall clock hits ``HOURS`` (18). Stop it
whenever you want: Ctrl-C writes ``last.pt`` and a numbered checkpoint,
then exits. ``--steps N`` also stops at N optimizer steps, whichever
limit comes first. ``--hours 0 --steps N`` is a step-only run.

``BATCH_SIZE`` below is the default. ``--batch-size`` overrides it.
A fresh run calls ``S1.init_from_imagenet`` unless ``--no-imagenet`` is
set. ``--resume PATH`` loads a ``torch.save`` state dict and does not
touch ImageNet. Datagen time is logged every step and does not stop
the run.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import multiprocessing as mp
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

# Tunables. Command-line flags override BATCH_SIZE, the worker count, and
# the stopping rule. The 96 GB card can go above the default once a
# step's memory headroom is visible in the log. The run stops at HOURS
# of wall clock or at --steps, whichever comes first. --steps 0 means
# there is no step cap.
BATCH_SIZE = 32
HOURS = 18.0
WORKERS = 8
SAVE_EVERY = 200
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 100

logger = logging.getLogger("s1_pretrain")


def _datagen_one(seed: int) -> dict:
    """One board, one target, one oracle label. Runs in a spawn worker."""
    import random

    import numpy as np
    from PIL import Image

    from agent.game_io import new_multi_gold_game
    from neural_net.oracle import ACTION_INDEX, point_oracle, sample_target
    from neural_net.render import canonical_frame
    from training.image_noise import TRAINING_STRENGTH, noise_image

    random.seed(seed)
    rng = random.Random(seed)
    started = time.perf_counter()
    game = new_multi_gold_game(n_gold=None, opening="any")
    xy = sample_target(game, rng)
    label = ACTION_INDEX[point_oracle(game.settings, xy[0], xy[1])]
    frame = canonical_frame(game)
    noised = noise_image(
        Image.fromarray(frame, mode="RGB"), rng, strength=TRAINING_STRENGTH
    )
    image = np.asarray(noised.convert("RGB"), dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"datagen image shape {image.shape}, expected HxWx3")
    return {
        "image": image,
        "xy": (float(xy[0]), float(xy[1])),
        "label": int(label),
        "worker_s": time.perf_counter() - started,
    }


class _Prefetch:
    def __init__(self, batch_size: int, workers: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        ctx = mp.get_context("spawn")
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        self._pending = self._submit()

    def _submit(self):
        seeds = list(range(self.seed, self.seed + self.batch_size))
        self.seed += self.batch_size
        return [self.pool.submit(_datagen_one, seed) for seed in seeds]

    def take(self) -> tuple[list[dict], float]:
        started = time.perf_counter()
        samples = [future.result() for future in self._pending]
        stall = time.perf_counter() - started
        self._pending = self._submit()
        return samples, stall

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mib() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated() / (1024 * 1024))


def _write_crash(run_dir: Path, exc: BaseException) -> None:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path = run_dir / "crash.txt"
    path.write_text(text, encoding="utf-8")
    _append_jsonl(run_dir / "metrics.jsonl", {
        "kind": "crash",
        "error": f"{type(exc).__name__}: {exc}",
        "t": time.time(),
    })


def _backbone_params(model: torch.nn.Module):
    trunks = []
    heads = []
    for name, param in model.named_parameters():
        if name.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
            trunks.append(param)
        else:
            heads.append(param)
    if not trunks or not heads:
        raise RuntimeError(
            f"S1 param split failed: trunk={len(trunks)} head={len(heads)}"
        )
    return trunks, heads


def _collate(samples: list[dict], device: torch.device):
    shapes = {tuple(sample["image"].shape) for sample in samples}
    if len(shapes) != 1:
        raise RuntimeError(f"datagen produced mixed image shapes {sorted(shapes)}")
    images = np.stack([sample["image"] for sample in samples], axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(images))
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    xy = torch.tensor([sample["xy"] for sample in samples], dtype=torch.float32)
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    non_blocking = device.type == "cuda"
    return (
        tensor.to(device, non_blocking=non_blocking),
        xy.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 oracle SFT")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="stop after this many optimizer steps (0 = no step cap)",
    )
    parser.add_argument(
        "--hours", type=float, default=HOURS,
        help="stop after this many hours of wall clock (0 = no clock). "
             "Ctrl-C saves last.pt and exits.",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.save_every < 1:
        raise SystemExit("batch-size, workers, and save-every must be >= 1")
    if args.steps < 0 or args.hours < 0:
        raise SystemExit("--steps and --hours must be >= 0")
    if args.steps == 0 and args.hours == 0:
        raise SystemExit("set --steps or --hours; both 0 would not stop")
    if not torch.cuda.is_available():
        raise SystemExit("train_s1 requires CUDA; run it on the remote GPU box")

    from neural_net.paths import weights_root
    from neural_net.s1 import S1
    from training.run_weekend import VramMonitor

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = weights_root() / "s1" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir)

    device = torch.device("cuda")
    model = S1().to(device)
    if args.resume is not None:
        model.load_weights(args.resume)
        logger.info("loaded S1 weights from %s", args.resume)
    elif not args.no_imagenet:
        copied = model.init_from_imagenet()
        logger.info("init_from_imagenet copied %d tensors", copied)
    else:
        logger.info("random init (--no-imagenet)")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("S1 parameters: %d (%.2fM)", n_params, n_params / 1e6)

    trunk, head = _backbone_params(model)
    opt = torch.optim.AdamW(
        [
            {"params": trunk, "lr": LR_BACKBONE},
            {"params": head, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    base_lrs = [LR_BACKBONE, LR_HEAD]

    vram = None
    prefetch = None
    metrics = run_dir / "metrics.jsonl"
    step = 0
    t0 = time.perf_counter()

    def _save_stop(reason: str) -> None:
        model.save(run_dir / "last.pt")
        numbered = None
        if step > 0:
            numbered = run_dir / f"step_{step:06d}.pt"
            model.save(numbered)
        elapsed_h = (time.perf_counter() - t0) / 3600.0
        _append_jsonl(metrics, {
            "kind": "stop",
            "reason": reason,
            "step": step,
            "elapsed_h": round(elapsed_h, 4),
            "t": time.time(),
        })
        logger.info(
            "stopped (%s) at step %d after %.2f h; last.pt in %s",
            reason, step, elapsed_h, run_dir,
        )
        if numbered is not None:
            logger.info("saved %s", numbered)

    interrupted_by: dict[str, int | None] = {"signum": None}

    def _on_signal(signum, _frame):
        # SIGINT (Ctrl-C) and SIGTERM both land here. Raising aborts the
        # current step; the handler below writes the checkpoint.
        interrupted_by["signum"] = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    logger.info(
        "budget: hours=%s steps=%s batch=%d workers=%d save_every=%d",
        args.hours or "none", args.steps or "none",
        args.batch_size, args.workers, args.save_every,
    )

    try:
        vram = VramMonitor(f"s1_pretrain_{stamp}")
        vram.set_stage("train")
        prefetch = _Prefetch(args.batch_size, args.workers, args.seed)
        while True:
            if args.steps and step >= args.steps:
                _save_stop("steps")
                break
            if args.hours and (time.perf_counter() - t0) >= args.hours * 3600.0:
                _save_stop("hours")
                break
            scale = 1.0 if step >= WARMUP_STEPS else (step + 1) / WARMUP_STEPS
            for group, base in zip(opt.param_groups, base_lrs):
                group["lr"] = base * scale

            samples, datagen_s = prefetch.take()
            worker_s = float(np.mean([sample["worker_s"] for sample in samples]))

            _atomic_json(run_dir / "heartbeat.json", {
                "step": step,
                "batch": args.batch_size,
                "image": list(samples[0]["image"].shape),
                "gpu_mib_allocated": _gpu_mib(),
                "t": time.time(),
            })

            _sync()
            t_h2d = time.perf_counter()
            images, xy, labels = _collate(samples, device)
            _sync()
            h2d_s = time.perf_counter() - t_h2d

            opt.zero_grad(set_to_none=True)
            _sync()
            t_fwd = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images, xy)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            _sync()
            forward_s = time.perf_counter() - t_fwd
            if not torch.isfinite(loss):
                payload = {"kind": "nonfinite_loss", "step": step, "t": time.time()}
                _append_jsonl(metrics, payload)
                raise RuntimeError(f"non-finite loss at step {step}")

            _sync()
            t_bwd = time.perf_counter()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            _sync()
            backward_s = time.perf_counter() - t_bwd
            if not torch.isfinite(grad_norm):
                _append_jsonl(metrics, {
                    "kind": "nonfinite_grad", "step": step, "t": time.time(),
                })
                raise RuntimeError(f"non-finite grad_norm at step {step}")

            _sync()
            t_opt = time.perf_counter()
            opt.step()
            _sync()
            optim_s = time.perf_counter() - t_opt

            record = {
                "step": step,
                "loss": round(float(loss.detach()), 6),
                "datagen_s": round(datagen_s, 4),
                "worker_s": round(worker_s, 4),
                "h2d_s": round(h2d_s, 4),
                "forward_s": round(forward_s, 4),
                "backward_s": round(backward_s, 4),
                "optim_s": round(optim_s, 4),
                "grad_norm": round(float(grad_norm), 4),
                "lr_backbone": opt.param_groups[0]["lr"],
                "lr_head": opt.param_groups[1]["lr"],
                "gpu_mib_allocated": _gpu_mib(),
            }
            _append_jsonl(metrics, record)
            _atomic_json(run_dir / "last_step.json", record)
            logger.info(
                "step %d loss %.4f datagen %.3fs worker %.3fs fwd %.3fs bwd %.3fs",
                step, record["loss"], datagen_s, worker_s, forward_s, backward_s,
            )

            if (step + 1) % args.save_every == 0:
                ckpt = run_dir / f"step_{step + 1:06d}.pt"
                model.save(ckpt)
                model.save(run_dir / "last.pt")
                logger.info("saved %s", ckpt)
            step += 1
    except KeyboardInterrupt:
        signum = interrupted_by["signum"] or signal.SIGINT
        try:
            _save_stop(signal.Signals(signum).name)
        except Exception:
            logger.exception("could not save last.pt after the interrupt")
        raise SystemExit(128 + signum)
    except Exception as exc:
        _write_crash(run_dir, exc)
        try:
            model.save(run_dir / "last.pt")
        except Exception:
            logger.exception("could not save last.pt after the crash")
        raise
    finally:
        if prefetch is not None:
            prefetch.close()
        if vram is not None:
            vram.finish()


def _configure_logging(run_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream)


if __name__ == "__main__":
    main()
