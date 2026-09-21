#!/usr/bin/env bash
#
# vigal_setup.sh -- extras for game_alternative_research/vigal.ipynb
#
# Run AFTER scripts/setup_env.sh. Does NOT install ViGaL's OpenRLHF / vLLM /
# flash-attn train stack. Only:
#   * qwen-vl-utils (Qwen2.5-VL image chat path)
#   * HF cache for yunfeixie/ViGaL-7B
#   * the four Rotation sample PNGs from their repo (docs/resources)
#
# Usage:
#   bash game_alternative_research/vigal_setup.sh
#
# HF credentials: repo-root .env (HF_TOKEN), same as scripts/setup_env.sh.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
MODEL_ID="${VIGAL_MODEL_ID:-yunfeixie/ViGaL-7B}"
SAMPLES="${HERE}/samples"
VIGAL_REPO="https://raw.githubusercontent.com/yunfeixie233/ViGaL/main/docs/resources"

log() { printf '[vigal-setup] %s\n' "$*"; }
die() { printf '[vigal-setup] ERROR: %s\n' "$*" >&2; exit 1; }

command -v python >/dev/null 2>&1 || die "python not found on PATH"

log "installing qwen-vl-utils (not in requirements.txt)"
python -m pip install "qwen-vl-utils>=0.0.10"

log "prefetching ${MODEL_ID} into the HuggingFace cache"
log "  (HF credentials: ${REPO_ROOT}/.env via python-dotenv)"
python - "${MODEL_ID}" "${REPO_ROOT}/.env" <<'PY'
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(sys.argv[2]))

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
print(f"[vigal-setup] snapshot_download({model_id!r}) ...", flush=True)
path = snapshot_download(model_id)
print(f"[vigal-setup] cached at {path}", flush=True)
PY

mkdir -p "${SAMPLES}"
log "fetching Rotation sample frames into ${SAMPLES}"
for i in 1 2 3 4; do
  dest="${SAMPLES}/rotation_${i}.png"
  url="${VIGAL_REPO}/rotation_${i}.png"
  if [ -f "${dest}" ]; then
    log "  already have ${dest}"
    continue
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${dest}" "${url}" || die "failed to download ${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${dest}" "${url}" || die "failed to download ${url}"
  else
    die "need curl or wget to fetch rotation samples"
  fi
  log "  wrote ${dest}"
done

log "done. open game_alternative_research/vigal.ipynb (do not also keep play.ipynb Gemma on the same GPU)"
