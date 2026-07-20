#!/usr/bin/env bash
#
# setup_env.sh -- one-shot Python environment setup for game_saddle.
#
# `pip install -r requirements.txt` is necessary but NOT sufficient: NAMS'
# default entity-extraction pipeline runs spaCy then GLiNER on every message it
# stores, and both need model *weights* that pip cannot install:
#   * spaCy needs a language model downloaded via `python -m spacy download ...`.
#   * GLiNER lazily fetches its weights from HuggingFace on first use; we
#     pre-fetch them here so the first run isn't a surprise network stall and so
#     an offline/gated box fails loudly at setup time instead of mid-game.
#
# Without this step you get, on every stored message:
#   Stage 'SpacyEntityExtractor' failed: spaCy is required for SpacyEntityExtractor.
#   Stage 'GLiNEREntityExtractor' failed: GLiNER is required for GLiNEREntityExtractor.
#
# Safe to re-run (pip + the downloaders are all idempotent).
#
# Usage:
#   bash scripts/setup_env.sh
#
# Env overrides (defaults match NAMS' ExtractionConfig defaults, i.e. what the
# agent actually loads at runtime -- keep them in sync if you change the NAMS
# config):
#   SPACY_MODEL     (default en_core_web_sm)
#   GLINER_MODEL    (default urchade/gliner_medium-v2.1)
#   SKIP_TORCH      (unset)  -- if set, do NOT let requirements.txt pull torch;
#                              install torch yourself first (see the CUDA note in
#                              requirements.txt for driver < 580 / CUDA <= 12.x).

set -euo pipefail

SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"
GLINER_MODEL="${GLINER_MODEL:-urchade/gliner_medium-v2.1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

log() { printf '[setup-env] %s\n' "$*"; }
die() { printf '[setup-env] ERROR: %s\n' "$*" >&2; exit 1; }

command -v python >/dev/null 2>&1 || die "python not found on PATH"

# ------------------------------------------------------------- 1. pip install
REQ_FILE="${REPO_ROOT}/requirements.txt"
if [ -n "${SKIP_TORCH:-}" ]; then
  # Honour a pre-installed CUDA-specific torch: strip torch/torchvision so pip
  # can't pull a wheel that overrides it (install those yourself first -- see
  # the CUDA note at the top of requirements.txt).
  log "SKIP_TORCH set: excluding torch/torchvision from the install"
  REQ_FILE="$(mktemp)"
  grep -viE '^\s*(torch|torchvision)\b' "${REPO_ROOT}/requirements.txt" > "${REQ_FILE}"
fi
log "installing Python dependencies from ${REQ_FILE}"
python -m pip install -r "${REQ_FILE}"

# -------------------------------------------------------- 2. spaCy model dl
# `spacy download` fetches the model wheel matching the installed spaCy version.
log "downloading spaCy model: ${SPACY_MODEL}"
python -m spacy download "${SPACY_MODEL}"

# ------------------------------------------------------- 3. GLiNER weights dl
# Instantiating from_pretrained pulls the weights into the HF cache; nothing
# else is needed at runtime. This is the "python command for loading the model
# weights" that pip can't express.
log "pre-fetching GLiNER weights: ${GLINER_MODEL}"
# Load repo .env into os.environ first so HF_TOKEN (and any other HF auth
# vars) reach huggingface_hub during from_pretrained -- this script never
# imports agent.config, so bare shell env alone would miss a copied-over .env.
python - "${GLINER_MODEL}" "${REPO_ROOT}/.env" <<'PY'
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(sys.argv[2]))

from gliner import GLiNER

model_id = sys.argv[1]
print(f"[setup-env] GLiNER.from_pretrained({model_id!r}) ...", flush=True)
GLiNER.from_pretrained(model_id)
print("[setup-env] GLiNER weights cached.", flush=True)
PY

# --------------------------------------------------------------- 4. verify
log "verifying extractors import cleanly"
python - "${SPACY_MODEL}" <<'PY'
import sys
import spacy

spacy.load(sys.argv[1])
from neo4j_agent_memory.extraction import SpacyEntityExtractor, GLiNEREntityExtractor  # noqa: F401
print("[setup-env] spaCy model loads and NAMS extractors import OK.", flush=True)
PY

log "done. Extraction (spaCy -> GLiNER) is ready; runs will auto-discover entities."
log ""
log "NOTE: this script sets up the PYTHON environment only. Neo4j is a separate"
log "server and is NOT started here. Bring it up next, THEN seed:"
log "  bash scripts/vast_neo4j_launch.sh          # install + start Neo4j (+ APOC, .env)"
log "  python scripts/neo4j_connect_diagnostic.py # verify connectivity (optional)"
log "  python -m agent.runner seed                # seed the semantic model"
