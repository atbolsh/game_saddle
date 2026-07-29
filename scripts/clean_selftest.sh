#!/usr/bin/env bash
# Wipe leftovers from ``python -m training.selftest`` so the suite can be
# re-run cleanly. Does NOT touch data_external/, Neo4j, or real training
# checkpoints (only names matching selftest_*).
#
# Usage (from repo root, or anywhere -- the script cds to the repo root)::
#
#   bash scripts/clean_selftest.sh
#
# After a Ctrl-C during a GPU stage, also check ``nvidia-smi`` for a leftover
# python PID holding VRAM and kill it if present -- this script only cleans
# files on disk.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "Cleaning selftest artifacts under ${REPO_ROOT} ..."

rm -rf data_game/selftest_t5
rm -rf logs/train_selftest_* logs/datagen_stats_selftest_*
# Glob may expand to nothing; nullglob-safe via a loop.
shopt -s nullglob
for d in weights/*/selftest_t4* weights/*/selftest_t4rb* weights/*/selftest_t7*; do
  rm -rf "$d"
done
shopt -u nullglob
rm -rf /tmp/selftest_*

echo "Done. Re-run with: python -m training.selftest all"
echo "(or resume from the failed stage, e.g. python -m training.selftest t3)"
