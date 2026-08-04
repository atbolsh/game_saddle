"""Label-safe image regularization for game frames.

The worry (TRAINING_GAME_TRACES.md, "Visual generalization"): perception
skills learned on this one renderer -- one palette, one sprite, one board
style -- may not generalize. Every frame that reaches the model therefore
gets mild, LABEL-SAFE degradation, sampled per image:

  * slight random crop + rescale back (small enough never to cut off the
    agent or the gold);
  * brightness / contrast / color jitter;
  * 2-6 BIG discolored rectangles -- semi-transparent color tints, each
    8-25% of the image side. Placed uniformly at random, so they CAN land
    on the agent or the gold: partial occlusion is deliberate perception
    stress. Still tints, never opaque black boxes -- not dropout;
  * ONE whole-image color drift tint (a full-frame translucent rectangle,
    much weaker than the patches);
  * per-pixel speckle (multiplicative) + gaussian pixel noise (additive);
  * mild gaussian blur OR a JPEG re-encode (compression artifacts), one of
    the two.

Deliberately EXCLUDED (not label-safe): flips and rotations -- they invert
the clock/bearing semantics that the OBS line and the move token are graded
on.

10% of frames (:data:`_SKIP_PROB`) skip ALL of the above and pass through
completely clean: the network must also see uncorrupted boards, or it ends
up miscalibrated on the un-noised frames it meets outside the datagen
harness.

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

Magnitudes are tuned BY EYE in ``notebooks/noise_tuner.ipynb`` (sliders over
a live board frame + a regenerate button); the keyword overrides on
:func:`noise_image` exist for that notebook. When new magnitudes are chosen,
update the module constants here -- the overrides are for experimentation,
production callers pass none.
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
#: the agent or the gold (both are drawn well inside the board). This is the
#: ONLY sprite-protecting guard: the patches below have none, by design.
_MAX_CROP = 0.04
#: Patch geometry: each side is 8-25% of the image side at strength 1.0 --
#: big enough to (sometimes) cover the agent or the gold.
_PATCH_SIDE = (0.08, 0.25)
#: Patch tint opacity range at strength 1.0 (alpha fraction).
_PATCH_ALPHA = (0.12, 0.30)
#: Whole-image color-drift tint opacity at strength 1.0 -- one full-frame
#: translucent rectangle, deliberately weaker than the patches.
_DRIFT_ALPHA = (0.04, 0.10)
#: Speckle: per-pixel MULTIPLICATIVE noise sigma (fraction of the pixel
#: value) at strength 1.0. Tuned by eye in noise_tuner.ipynb (2026-07-31):
#: bumped from the initial (0.01, 0.04) -- frames stay legible to a human
#: even harsher than this, but model perception is more fragile, so the
#: rest of the magnitudes stay at their first-guess values.
_SPECKLE_SIGMA = (0.025, 0.1)
#: Probability that a frame skips ALL degradations and passes through
#: clean (2026-08-04): the network must also see uncorrupted frames --
#: at inference outside datagen there is no noiser at all, and a model
#: that has only ever seen degraded boards is miscalibrated on clean
#: ones. Applies per noise_image call, i.e. independently at datagen
#: (clean stored frame) and at training (no re-noise on top).
_SKIP_PROB = 0.1


def noise_image(img, rng: random.Random, strength: float = TRAINING_STRENGTH,
                *,
                patch_alpha: tuple[float, float] | None = None,
                drift_alpha: tuple[float, float] | None = None,
                speckle_sigma: tuple[float, float] | None = None):
    """Return a degraded RGB copy of a PIL image. ``strength`` scales every
    magnitude (0 = identity); geometry stays fixed so labels stay true.

    The keyword ranges override the module constants (same meaning) -- they
    exist for notebooks/noise_tuner.ipynb. After the up-front _SKIP_PROB
    pass-through draw, every random draw happens UNCONDITIONALLY and
    magnitudes only scale afterwards, so under a fixed seed the same
    patches/tints/noise fields appear at every magnitude -- that is what
    makes the tuner's sliders rescale a frozen scene instead of redrawing
    it (a skipped seed shows the clean frame at every slider setting;
    regenerate to draw a new scene)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    patch_alpha = patch_alpha or _PATCH_ALPHA
    drift_alpha = drift_alpha or _DRIFT_ALPHA
    speckle_sigma = speckle_sigma or _SPECKLE_SIGMA

    img = img.convert("RGB")
    w, h = img.size
    if strength <= 0:
        return img

    # -- 10% pass-through: completely uncorrupted frames (see _SKIP_PROB).
    #    Drawn FIRST so a skipped frame consumes exactly one rng draw and
    #    the seeded stream stays reproducible either way.
    if rng.random() < _SKIP_PROB:
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

    # -- big discolored patches (tints, never opaque; placed uniformly at
    #    random, so covering the agent or the gold is allowed and intended)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(rng.randint(2, 6)):
        pw = int(w * rng.uniform(*_PATCH_SIDE))
        ph = int(h * rng.uniform(*_PATCH_SIDE))
        x = rng.randint(0, max(0, w - pw))
        y = rng.randint(0, max(0, h - ph))
        tint = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        alpha = int(255 * rng.uniform(*patch_alpha) * strength)
        draw.rectangle([x, y, x + pw, y + ph], fill=(*tint, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # -- whole-image color drift: one translucent tint over the entire frame
    #    (the patches' full-frame little sibling)
    tint = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    alpha = int(255 * rng.uniform(*drift_alpha) * strength)
    if alpha > 0:
        drift = Image.new("RGBA", (w, h), (*tint, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), drift).convert("RGB")

    # -- per-pixel noise: multiplicative speckle, then additive gaussian.
    #    Fields are drawn as STANDARD normals and scaled after, so a
    #    magnitude change rescales the same pattern (see docstring).
    np_rng = np.random.default_rng(rng.getrandbits(32))
    arr = np.asarray(img, dtype=np.float32)
    speckle = rng.uniform(*speckle_sigma) * strength
    arr = arr * (1.0 + speckle
                 * np_rng.standard_normal(arr.shape, dtype=np.float32))
    sigma = rng.uniform(2.0, 6.0) * strength
    arr = arr + sigma * np_rng.standard_normal(arr.shape, dtype=np.float32)
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
