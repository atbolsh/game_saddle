"""Label-safe image regularization for game frames.

The worry (TRAINING_GAME_TRACES.md, "Visual generalization"): perception
skills learned on this one renderer -- one palette, one sprite, one board
style -- may not generalize. Every frame that reaches the model therefore
gets mild, LABEL-SAFE degradation, sampled per image:

  * slight random crop + rescale back (small enough never to cut off the
    agent or the gold);
  * brightness / contrast / color jitter;
  * 2-6 small DISCOLORED patches -- semi-transparent color tints, each a few
    percent of the image side. Deliberately not dropout: no black boxes, no
    big occlusions, just mild local discoloration;
  * gaussian pixel noise;
  * mild gaussian blur OR a JPEG re-encode (compression artifacts), one of
    the two.

Deliberately EXCLUDED (not label-safe): flips and rotations -- they invert
the clock/bearing semantics that the OBS line and the move token are graded
on.

Used in two places with different strengths:

  * datagen inference (:data:`INFERENCE_STRENGTH`, milder): installed as the
    session's ``image_filter``, so the stored snapshot IS the noised frame
    and player, analyst, NAMS, and the training copy all see identical
    bytes;
  * training (:data:`TRAINING_STRENGTH`): ``GameTraceSource`` re-noises its
    per-run image copies, so every training run sees a fresh degradation on
    top of the stored frame.

Everything is driven by a caller-provided ``random.Random`` -- deterministic
under a fixed seed, varying across a stream of calls.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable

#: Strength presets: datagen inference keeps the frame clearly readable (the
#: player must still be able to play); training pushes a little harder.
INFERENCE_STRENGTH = 0.5
TRAINING_STRENGTH = 1.0

#: Max crop margin per edge at strength 1.0 -- 4% of a side cannot cut off
#: the agent or the gold (both are drawn well inside the board).
_MAX_CROP = 0.04
#: Patch geometry: each side is 2-8% of the image side at strength 1.0.
_PATCH_SIDE = (0.02, 0.08)
#: Patch tint opacity range at strength 1.0 (alpha fraction).
_PATCH_ALPHA = (0.12, 0.30)


def noise_image(img, rng: random.Random, strength: float = TRAINING_STRENGTH):
    """Return a degraded RGB copy of a PIL image. ``strength`` scales every
    magnitude (0 = identity); geometry stays fixed so labels stay true."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    img = img.convert("RGB")
    w, h = img.size
    if strength <= 0:
        return img

    # -- slight crop + rescale back (usually on, never more than _MAX_CROP)
    if rng.random() < 0.7:
        margin = _MAX_CROP * strength
        left = int(w * rng.uniform(0.0, margin))
        top = int(h * rng.uniform(0.0, margin))
        right = w - int(w * rng.uniform(0.0, margin))
        bottom = h - int(h * rng.uniform(0.0, margin))
        img = img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)

    # -- brightness / contrast / color jitter
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast,
                     ImageEnhance.Color):
        factor = 1.0 + rng.uniform(-0.15, 0.15) * strength
        img = enhancer(img).enhance(factor)

    # -- small discolored patches (tints, never opaque, never large)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(rng.randint(2, 6)):
        pw = int(w * rng.uniform(*_PATCH_SIDE))
        ph = int(h * rng.uniform(*_PATCH_SIDE))
        x = rng.randint(0, max(0, w - pw))
        y = rng.randint(0, max(0, h - ph))
        tint = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        alpha = int(255 * rng.uniform(*_PATCH_ALPHA) * strength)
        draw.rectangle([x, y, x + pw, y + ph], fill=(*tint, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # -- gaussian pixel noise
    np_rng = np.random.default_rng(rng.getrandbits(32))
    sigma = rng.uniform(2.0, 6.0) * strength
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np_rng.normal(0.0, sigma, size=arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # -- mild blur OR jpeg artifacts (one of the two)
    if rng.random() < 0.5:
        img = img.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8) * strength)
        )
    else:
        quality = rng.randint(60, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def noise_file(path: str | Path, rng: random.Random,
               strength: float = TRAINING_STRENGTH,
               out_path: str | Path | None = None) -> None:
    """Degrade an image file; in place when ``out_path`` is None. Output
    format follows the destination suffix (PNG for the stored snapshots)."""
    from PIL import Image

    path = Path(path)
    dest = Path(out_path) if out_path is not None else path
    with Image.open(path) as img:
        noise_image(img, rng, strength).save(dest)


def make_image_filter(seed: int,
                      strength: float = INFERENCE_STRENGTH,
                      ) -> Callable[[str], None]:
    """A stateful in-place file noiser for ``session.image_filter``: one
    seeded stream, each successive frame degraded differently but the whole
    run reproducible."""
    rng = random.Random(seed)

    def _filter(path: str) -> None:
        noise_file(path, rng, strength)

    return _filter
