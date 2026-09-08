#!/usr/bin/env bash
#
# install_sglang.sh -- put a Gemma-4-capable SGLang into the current venv.
#
# Not in requirements.txt: the pin wants torch==2.13.0, CUDA-specific
# kernel indexes, and a few pre-release wheels (cuda-tile, flash-attn-4).
# A naive `pip install sglang` on an old uv/pip can silently land 0.5.9,
# which does not know gemma4_unified.
#
# Pin is sglang 0.5.19 (has python/sglang/srt/models/gemma4_unified.py and
# transformers==5.12.1, inside this repo's >=5.10,<5.15 floor). Do NOT
# install the cookbook transformers git SHA -- that would leave the
# verified HF path.
#
# CUDA 13 (this project's 96G box: driver 595 / CUDA 13.2): default
# PyPI extras (flashinfer cu13). CUDA 12: official cu129 force-reinstall
# of torch + sglang-kernel + sgl-deep-gemm after the package lands.
#
# Safe to re-run. Called from setup_env.sh; run alone after a master-era
# setup to make INFER_BACKEND=sglang importable without redoing spaCy /
# GLiNER / download_external.
#
# Usage:
#   bash scripts/install_sglang.sh
#
# Env:
#   SGLANG_PIN      (default sglang==0.5.19)
#   SGLANG_CUDA     (12 or 13; default = nvidia-smi CUDA major)
#   SKIP_SGLANG     if set, no-op (setup_env.sh also honours this)

set -euo pipefail

SGLANG_PIN="${SGLANG_PIN:-sglang==0.5.19}"

log() { printf '[install-sglang] %s\n' "$*"; }
die() { printf '[install-sglang] ERROR: %s\n' "$*" >&2; exit 1; }

if [ -n "${SKIP_SGLANG:-}" ]; then
  log "SKIP_SGLANG set: not installing sglang"
  exit 0
fi

command -v python >/dev/null 2>&1 || die "python not found on PATH"
PYTHON="$(command -v python)"

detect_cuda_major() {
  if [ -n "${SGLANG_CUDA:-}" ]; then
    printf '%s' "${SGLANG_CUDA}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local ver
    ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\).*/\1/p' | head -1 || true)"
    if [ -n "${ver}" ]; then
      printf '%s' "${ver}"
      return 0
    fi
  fi
  python - <<'PY' || true
import sys
try:
    import torch
    v = getattr(torch.version, "cuda", None)
    if v:
        print(v.split(".")[0], end="")
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
}

CUDA_MAJOR="$(detect_cuda_major || true)"
[ -n "${CUDA_MAJOR}" ] || die \
  "cannot detect CUDA major (nvidia-smi / torch.version.cuda). Set SGLANG_CUDA=12 or SGLANG_CUDA=13."

case "${CUDA_MAJOR}" in
  12|13) ;;
  *) die "SGLANG_CUDA/detected major is ${CUDA_MAJOR}; only 12 or 13 are wired" ;;
esac

log "python=${PYTHON}"
log "pin=${SGLANG_PIN}  cuda_major=${CUDA_MAJOR}"
log "this upgrades torch to 2.13.0 and transformers to 5.12.1 (still in-repo range)"

python -m pip install --upgrade pip
python -m pip install --upgrade 'uv>=0.12'
# --prerelease=allow: cuda-tile 1.6.0rc5 and flash-attn-4 betas. --python
# keeps uv inside THIS venv (bare `uv pip install` can miss it).
uv pip install --python "${PYTHON}" --prerelease=allow "${SGLANG_PIN}"

if [ "${CUDA_MAJOR}" = "12" ]; then
  log "CUDA 12: force-reinstall torch + kernels from the cu129 index"
  uv pip install --python "${PYTHON}" --force-reinstall \
    torch==2.13.0 torchaudio==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu129
  uv pip install --python "${PYTHON}" --force-reinstall sglang-kernel \
    --index-url https://docs.sglang.ai/whl/cu129/
  uv pip install --python "${PYTHON}" --force-reinstall sgl-deep-gemm \
    --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
fi

log "verifying import + Engine + gemma4_unified + transformers floor"
python - <<'PY'
from __future__ import annotations

import sys
import traceback
from importlib.metadata import PackageNotFoundError, version as pkg_version

from packaging.version import Version

errors: list[str] = []

try:
    import sglang as sgl
except Exception as exc:
    errors.append(f"import sglang failed: {type(exc).__name__}: {exc}")
    sgl = None

if sgl is not None and not hasattr(sgl, "Engine"):
    errors.append("sglang has no sgl.Engine")

try:
    from sglang.srt.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )
except Exception as exc:
    errors.append(
        "gemma4_unified is not importable "
        f"({type(exc).__name__}: {exc}). This pin cannot load gemma-4-12B-it."
    )
    traceback.print_exc()
    Gemma4UnifiedForConditionalGeneration = None  # noqa: N816

try:
    import transformers
    tv = Version(transformers.__version__.split("+")[0])
except Exception as exc:
    errors.append(f"transformers unreadable: {exc}")
    tv = None

if tv is not None and (tv < Version("5.10") or tv >= Version("5.15")):
    errors.append(
        f"transformers {transformers.__version__} left the repo floor "
        "(need >=5.10,<5.15). Do not install the cookbook git SHA."
    )

ver = "?"
try:
    ver = pkg_version("sglang")
except PackageNotFoundError:
    ver = getattr(sgl, "__version__", "?") if sgl is not None else "?"

if errors:
    print("[install-sglang] VERIFY FAILED:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    print(
        "This script will not fall back to HF. Either fix the pin "
        "(SGLANG_PIN=sglang==0.5.19 or a newer release that still ships "
        "gemma4_unified) or use a second venv:\n"
        "  python -m venv .venv-sglang && . .venv-sglang/bin/activate\n"
        "  bash scripts/install_sglang.sh\n"
        "  python -m sglang.launch_server --model-path "
        "weights/gemma-4-12b/merged_aug27_big_step_iter1_step313 "
        "--host 127.0.0.1 --port 30000\n"
        "  SGLANG_HTTP_URL=http://127.0.0.1:30000 INFER_BACKEND=sglang ...",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"[install-sglang] ok sglang={ver} Engine=yes "
    f"Gemma4UnifiedForConditionalGeneration={Gemma4UnifiedForConditionalGeneration.__name__} "
    f"transformers={transformers.__version__}",
    flush=True,
)
PY

log "done. INFER_BACKEND=sglang should import in this venv."
log "HTTP fallback remains SGLANG_HTTP_URL if Engine() later refuses the merged folder."
