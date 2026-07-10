#!/usr/bin/env bash
#
# reset_semantics.sh -- wipe ALL episodic memory and repopulate the long-term
# semantic model only (Entity + Preference nodes and their relationships).
# Restores the graph to the "semantic seeding only" state -- the status quo
# ante of a fresh box right after seeding. Use it to discard a failed/unwanted
# conversation, e.g. when you cannot run the in-notebook reset cell.
#
# Steps:
#   1. Wipe the neo4j database via `neo4j_db.sh wipe` (stops neo4j, deletes the
#      db files, restarts it empty; auth/password are preserved).
#   2. Run `python -m agent.runner seed` to recreate the Entity + Preference
#      nodes, then `python -m agent.runner link` to (idempotently) add the
#      hardwired entity relationships.
#   3. Unless KEEP_IMAGES=1, clear the on-disk snapshot images (all orphaned
#      once the GameSnapshot nodes are gone).
#
# NOTE: step 1 restarts Neo4j, which drops any live bolt connection -- if a
# notebook is open, re-run its "connect" cell afterwards (that does NOT store a
# conversation; only pressing Ask does).
#
# Usage (run with the project's Python env active):
#   bash scripts/reset_semantics.sh
#   KEEP_IMAGES=1 bash scripts/reset_semantics.sh    # keep memory_images/
#
# Env overrides: NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE (used by
# neo4j_db.sh) and AGENT_IMAGE_DIR (snapshot image directory, default
# memory_images).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() { printf '[reset-semantics] %s\n' "$*"; }

log "1/3 wiping neo4j database"
bash "${SCRIPT_DIR}/neo4j_db.sh" wipe

log "2/3 seeding the semantic model (entities + preferences, then relationships)"
python -m agent.runner seed
python -m agent.runner link

if [[ "${KEEP_IMAGES:-0}" != "1" ]]; then
  IMG_DIR="${AGENT_IMAGE_DIR:-memory_images}"
  if [[ -d "${IMG_DIR}" ]]; then
    log "3/3 clearing snapshot images in ${IMG_DIR} (set KEEP_IMAGES=1 to keep)"
    rm -rf "${IMG_DIR:?}/"*
  else
    log "3/3 no image dir at ${IMG_DIR}; nothing to clear"
  fi
else
  log "3/3 KEEP_IMAGES=1 -- leaving snapshot images in place"
fi

log "done -- graph is back to the semantic-seeding-only state"
