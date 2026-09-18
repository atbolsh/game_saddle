"""ViGaL-7B inference helpers (isolated from agent/ / NAMS).

Board observation is an RGB raster only, always resized to 512x512. The
question is either the official Snake / Rotation *instruction* (paper
Appendix A.2, live coordinate dump stripped) or arbitrary custom text.
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

BOARD_SIZE = 512
MODEL_ID = "yunfeixie/ViGaL-7B"
VIGAL_DATA = "yunfeixie/vigal_data"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SAMPLES_DIR = HERE / "samples"

# Appendix A.2 Snake instruction minus the filled state lines
# ({apple_position}, {last_action}, snake/enemy coordinates). Those were a
# second copy of the board; this notebook feeds the board as pixels only.
SNAKE_INSTRUCTION = """Your role is to guide a snake within a Snake game featuring multiple apples.
This game is played on a board of size 10 by 10. The board uses a standard Cartesian coordinate system, where (0,0) represents the bottom-left position and (9,9) is the top-rightmost coordinate.
The current board is the image. Read the snakes, apples, and positions from that image only.
Rules:
1) If you move onto an apple, you grow and gain 1 point.
2) If your head moves to a position where its coordinates (x, y) are outside the board boundaries (meaning x < 0, x > 9, y < 0, or y > 9), or into a space occupied by another snake's body, or into a space occupied by your own body, you die. That's the worst move.
3) The goal is to prioritize snake not die, then efficiently collecting apples. First avoid the worst move, then for each apple, find the nearest apple by calculating Manhattan distances. But only choose best next move to get closer the nearest apple if you can confirm best next move will not run outside the board boundaries, run into the position of another snake, or yourself. Otherwise it will be the worst move.
Decreasing your x coordinate is to the LEFT, increasing your x coordinate is to the RIGHT.
Decreasing your y coordinate is DOWN, increasing your y coordinate is UP.
Read out another snake's position and apple position. Try to predict another snake's next move and avoid colliding with it.
Best answer is one of next move that is the closest to the apple and not lead to your death. Worst answer is all of next moves 1. makes your head's coordinates (x, y) are outside the board boundaries, meaning x < 0, x > 9, y < 0, or y > 9. 2. moves into a position occupied by another snake's body. 3. moves into a position occupied by body of yourself.
Check all the next moves to list out all the worst moves in <worst_answer> tag. If no worst answer, return None for worst answer, e.g., "<worst_answer>None</worst_answer>"
The best answer and the worst answer are mutually exclusive and different.
You need first to give your reasoning process then to choose one of best next move and worst next move from ['UP', 'DOWN', 'LEFT', 'RIGHT'].
The reasoning process and answer are enclosed within <think> </think>, <best_answer> </best_answer> and <worst_answer> </worst_answer> tags, respectively, i.e., "<think> reasoning process here </think> <best_answer> one best move here </best_answer> <worst_answer> all worst moves here </worst_answer>"
"""

# Appendix A.2 Rotation instruction (observation is the four images).
ROTATION_INSTRUCTION = """I'm showing you 4 images. Images 1-2 are an example pair, and Images 3-4 are the test pair. In each pair, the first image shows the initial orientation, and the second shows the object after rotation.
### EXAMPLE OF ROTATION ###
Example: Image 1 shows the initial view and Image 2 shows the object after a 180 degree rotation.
### YOUR TASK ###
Now, considering the transformation from Image 3 (initial) to Image 4 (rotated).
Determine the angle of rotation from Image 3 to Image 4 on the plane
Analyze the rotation carefully using the example pair (Images 1-2) as a reference.
1. Coordinate System Transformation:
- Draw an x-y coordinate system on both original and rotated images with origin at center
- Identify a distinct feature point and note its coordinates in both images
- Apply rotation matrix equations to verify the transformation
Example: A star icon at coordinates (3,1) in the original image appears at (-1,3) in the rotated image. Testing with the 90° clockwise rotation matrix [cos(90°), sin(90°); -sin(90°), cos(90°)] confirms the transformation from (3,1) to (-1,3), verifying a 90° clockwise rotation.
2. Angular Displacement Measurement:
- Mark the image center as the origin in both images
- Draw a straight line from center to a distinctive feature in both images
- Measure the angle between these two lines using counterclockwise as positive
Example: A line from center to a red dot makes a 30° angle with horizontal in the original image. In the rotated image, this line makes a 210° angle with horizontal. The difference (180°) indicates a clockwise 180° rotation.
3. Symmetry Axis Tracking:
- Identify major symmetry axes in the original image
- Locate the same symmetry axes in the rotated image
- Calculate the angular displacement between original and rotated axes
Example: A rectangular logo has vertical and horizontal symmetry axes. After rotation, the vertical axis now points right and horizontal points down. This 90° shift of both axes confirms a clockwise 90° rotation.
4. Triangle Configuration Analysis:
- Select three non-collinear distinct points forming a triangle in both images
- Compare the orientation of this triangle in both images using vector cross products
- Determine rotation angle from the triangle's orientation change
Example: Three points form a right triangle with vertices clockwise arranged. After rotation, the same triangle has its vertices arranged in counterclockwise order while maintaining the same shape. This inversion indicates a clockwise 180° rotation.
5. Polar Coordinate Comparison:
- Convert key points to polar coordinates (r, θ) relative to image center
- Compare θ values of the same features in original and rotated images
- Calculate consistent angular difference across multiple points
Example: A feature at polar angle 45° in the original image appears at 135° in the rotated image. Another feature shifts from 10° to 100°. Both show a +90° shift in polar angle, confirming a clockwise 90° rotation.
Choose the rotation angle from this list: ['counter clockwise 90', '180']
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think> <answer> answer here </answer>"
"""


def load_repo_env() -> Path:
    """Load the repo-root .env before any HuggingFace call."""
    env_path = REPO_ROOT / ".env"
    load_dotenv(env_path)
    return env_path


def load_board_image(src: str | Path | Image.Image | bytes) -> Image.Image:
    """Open anything image-like and return a 512x512 RGB PIL image."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    return img.convert("RGB").resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.LANCZOS)


def image_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    load_board_image(img).save(buf, format="PNG")
    return buf.getvalue()


def default_rotation_paths() -> list[Path]:
    return [SAMPLES_DIR / f"rotation_{i}.png" for i in range(1, 5)]


def extract_best_answer(text: str) -> str | None:
    match = re.search(r"<best_answer>(.*?)</best_answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace("\n", "").replace(".", "").strip()
    return None


def extract_worst_answer(text: str) -> str | None:
    match = re.search(r"<worst_answer>(.*?)</worst_answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace("\n", "").replace(".", "").strip()
    return None


def extract_think(text: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_vigal_reply(text: str) -> dict[str, str | None]:
    return {
        "think": extract_think(text),
        "best_answer": extract_best_answer(text),
        "worst_answer": extract_worst_answer(text),
        "answer": extract_answer(text),
    }


def _model_class() -> type:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq


def load_vigal_model(model_id: str = MODEL_ID) -> tuple[Any, Any]:
    """Load ViGaL-7B + processor. Call once per notebook."""
    import torch
    from transformers import AutoProcessor

    load_repo_env()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_cls = _model_class()
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate_vigal(
    model: Any,
    processor: Any,
    images: list[Image.Image | str | Path],
    question: str,
    *,
    max_new_tokens: int = 2048,
) -> str:
    """One generation: 512x512 RGB image(s) + question text. No board-as-text."""
    from qwen_vl_utils import process_vision_info

    if not images:
        raise ValueError("generate_vigal needs at least one board image")
    if not (question or "").strip():
        raise ValueError("generate_vigal needs a question")
    boards = [load_board_image(im) for im in images]
    with tempfile.TemporaryDirectory(prefix="vigal_") as td:
        paths = []
        for i, board in enumerate(boards):
            dest = Path(td) / f"board_{i}.png"
            board.save(dest, format="PNG")
            paths.append(str(dest))
        messages = [{
            "role": "user",
            "content": (
                [{"type": "image", "image": p} for p in paths]
                + [{"type": "text", "text": question.strip()}]
            ),
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs: dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            kwargs["videos"] = video_inputs
        inputs = processor(**kwargs)
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
    return decoded[0].strip()


def _as_pil(obj: Any) -> Image.Image | None:
    if isinstance(obj, Image.Image):
        return obj
    if obj is None:
        return None
    convert = getattr(obj, "convert", None)
    if callable(convert):
        try:
            return convert("RGB")
        except Exception:
            pass
    if isinstance(obj, dict):
        for key in ("image", "bytes", "path", "pil"):
            if key in obj:
                got = _as_pil(obj[key])
                if got is not None:
                    return got
        return None
    if isinstance(obj, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(obj))
        except Exception:
            return None
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if p.is_file():
            try:
                return Image.open(p)
            except Exception:
                return None
    return None


def collect_images(obj: Any, *, depth: int = 0) -> list[Image.Image]:
    """Walk a HF row and keep only rasters (ignore text / coordinates)."""
    if depth > 10:
        return []
    direct = _as_pil(obj)
    if direct is not None and not isinstance(obj, dict):
        return [direct]
    found: list[Image.Image] = []
    if isinstance(obj, dict):
        pil = _as_pil(obj)
        if pil is not None:
            found.append(pil)
            return found
        for key, val in obj.items():
            if key in ("text", "prompt", "question", "message") and isinstance(val, str):
                continue
            found.extend(collect_images(val, depth=depth + 1))
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            found.extend(collect_images(val, depth=depth + 1))
    return found


def load_hf_sample_images(
    index: int = 0,
    *,
    dataset: str = VIGAL_DATA,
    split: str | None = None,
) -> list[Image.Image]:
    """Load one vigal_data row and return its image(s) only, already 512x512."""
    from datasets import load_dataset

    load_repo_env()
    if split:
        ds = load_dataset(dataset, split=split)
        row = ds[index]
    else:
        bundle = load_dataset(dataset)
        first = next(iter(bundle.values()))
        row = first[index]
    images = [load_board_image(im) for im in collect_images(row)]
    if not images:
        raise ValueError(
            f"{dataset} row {index} had no image files; the board must be pixels"
        )
    return images
