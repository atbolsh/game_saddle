#!/usr/bin/env bash
# DELETE ME — one-shot handoff written 2026-08-12.
#
# Waits for the in-flight aug11 2-epoch run to finish (including smoke2),
# then launches EXACTLY ONE overnight epoch (prefix aug12) from aug11's
# final checkpoint. Safe to kill; re-running after the overnight has
# already started is a no-op if aug12_state.json already exists with
# done stages (run_weekend skips completed epochs).
#
# Usage (from repo root, after the aug11 nohup is already running):
#   nohup bash scripts/custom_2026-08-12_DELETE_ME_overnight.sh \
#       > overnight_aug12_handoff.log 2>&1 &
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CURRENT_PREFIX="aug11"
CURRENT_LOG="weekend_aug11.log"
CURRENT_STATE="data_game/${CURRENT_PREFIX}_state.json"

NEXT_PREFIX="aug12"
NEXT_LOG="weekend_${NEXT_PREFIX}.log"
NEXT_EPOCHS=1
NEXT_PARALLEL=12   # match the in-flight aug11 run

POLL_SECS=60

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"; }

# Parent orchestrator only (not --train-iter children).
orchestrator_alive() {
  pgrep -f "python -m training.run_weekend --epochs .*--prefix ${CURRENT_PREFIX}" \
    >/dev/null 2>&1
}

# aug11 is finished when train2 is recorded AND smoke2 has a summary
# (smoke is the last stage of each epoch; "done" does not list smoke).
aug11_finished() {
  python3 - "$CURRENT_STATE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
s = json.loads(p.read_text())
done = set(s.get("done") or [])
smoke = s.get("smoke") or {}
sys.exit(0 if ("train2" in done and "2" in smoke) else 1)
PY
}

final_checkpoint() {
  python3 - "$CURRENT_STATE" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
ckpts = s.get("checkpoints") or {}
# Prefer epoch 2; fall back to epoch 1 if train2 failed with no ckpt.
for k in ("2", "1"):
    v = ckpts.get(k)
    if v:
        print(v)
        sys.exit(0)
sys.exit("no checkpoint recorded in " + sys.argv[1])
PY
}

log "DELETE-ME handoff: waiting for ${CURRENT_PREFIX} to finish (poll ${POLL_SECS}s)."
log "Watching state=${CURRENT_STATE} log=${CURRENT_LOG}"

while true; do
  if aug11_finished; then
    log "${CURRENT_PREFIX} state shows train2 + smoke2 complete."
    break
  fi
  if ! orchestrator_alive; then
    # Process died without a clean smoke2 — still hand off if we have
    # any checkpoint; otherwise abort loudly.
    log "WARNING: ${CURRENT_PREFIX} orchestrator process is gone but state is incomplete."
    if [[ -f "$CURRENT_STATE" ]] && final_checkpoint >/dev/null 2>&1; then
      log "Proceeding with whatever checkpoint is recorded (smoke2 may be missing)."
      break
    fi
    log "ERROR: no usable checkpoint in ${CURRENT_STATE}; not launching overnight."
    exit 1
  fi
  log "still waiting (${CURRENT_PREFIX} alive; train2/smoke2 not both done)..."
  sleep "$POLL_SECS"
done

# Let the parent flush its final log line / exit fully.
sleep 5

CKPT="$(final_checkpoint)"
log "Launching ${NEXT_EPOCHS} overnight epoch(s) as prefix=${NEXT_PREFIX} from checkpoint=${CKPT}"

nohup python -m training.run_weekend \
    --epochs "$NEXT_EPOCHS" \
    --prefix "$NEXT_PREFIX" \
    --parallel "$NEXT_PARALLEL" \
    --start-checkpoint "$CKPT" \
    > "$NEXT_LOG" 2>&1 &
NEXT_PID=$!
log "overnight launched: pid=${NEXT_PID} log=${NEXT_LOG}"
log "DELETE ME when you are done reading tomorrow's results."
