#!/usr/bin/env bash
#
# neo4j_db.sh -- save / wipe / load / status for the bare-metal Neo4j database
# used by the game_saddle NAMS memory backend, for hosts that run Neo4j
# directly (no Docker daemon -- e.g. rented Vast.ai containers). This is the
# no-Docker counterpart to the docker-compose stack in docker-compose.yml.
#
# game_saddle keeps all of its memory (Messages, GameSnapshots, reasoning
# Traces, the long-term semantic model) in the single Community `neo4j`
# database, so "save the agent's memories" = "dump the neo4j database".
#
# Subcommands:
#   save <dump_file>      Stop neo4j, dump the neo4j db to <dump_file> via
#                         neo4j-admin, restart neo4j. Non-destructive -- the
#                         graph stays intact. Use this to snapshot the agent's
#                         memory for offline analysis.
#   wipe                  Stop neo4j, delete the neo4j db files, restart (a
#                         fresh empty neo4j db is recreated on start). Auth in
#                         data/dbms is preserved, so the password stays the
#                         same. Use to start a run on a clean graph.
#   load <dump_file>      Stop neo4j, load <dump_file> into the neo4j db via
#                         neo4j-admin --overwrite-destination, restart. Use to
#                         reinstate a previously saved memory graph.
#   status                Print neo4j running state + bolt connectivity + node
#                         count by label.
#
# Save-file format: Neo4j's native `neo4j-admin database dump` output -- a
# binary archive of the store files, only meaningful to `neo4j-admin database
# load` on the same (or newer) Neo4j major version. Not human-readable.
#
# Usage:
#   bash scripts/neo4j_db.sh save logs/memory_run1.dump
#   bash scripts/neo4j_db.sh wipe
#   bash scripts/neo4j_db.sh load logs/memory_run1.dump
#   bash scripts/neo4j_db.sh status
#
# Env overrides:
#   NEO4J_USERNAME  (default neo4j)
#   NEO4J_PASSWORD  (default changeme)
#   NEO4J_DATABASE  (default neo4j -- Community's single db; left as a knob
#                    for a future Enterprise switch)

set -euo pipefail

NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-changeme}"
NEO4J_DB="${NEO4J_DATABASE:-neo4j}"
DATA_DIR="${NEO4J_DATA_DIR:-/var/lib/neo4j/data}"

log() { printf '[neo4j-db] %s\n' "$*"; }
die() { printf '[neo4j-db] ERROR: %s\n' "$*" >&2; exit 1; }

neo4j_is_running() {
  neo4j status 2>/dev/null | grep -qi running
}

wait_for_bolt() {
  local i=0
  while (( i < 90 )); do
    if python3 -c "import socket; socket.socket().connect(('localhost',7687))" 2>/dev/null; then
      log "bolt ready on :7687 after ${i}s"
      return 0
    fi
    sleep 1; i=$((i + 1))
  done
  die "bolt never came back on :7687 -- check /var/log/neo4j/neo4j.log"
}

stop_neo4j() {
  if neo4j_is_running; then
    log "stopping neo4j"
    neo4j stop >/dev/null 2>&1 || true
    # Wait for the process to actually exit so the store files are released.
    local i=0
    while neo4j_is_running && (( i < 60 )); do sleep 1; i=$((i+1)); done
  fi
}

start_neo4j() {
  if neo4j_is_running; then
    log "neo4j already running"
    return 0
  fi
  log "starting neo4j"
  if ! neo4j start 2>/tmp/neo4j_db_start.err; then
    log "neo4j start failed; falling back to console as root"
    head /tmp/neo4j_db_start.err >&2 || true
    mkdir -p /var/log/neo4j
    nohup neo4j console >/var/log/neo4j/console.log 2>&1 &
    echo $! > /var/run/neo4j.pid
  fi
  wait_for_bolt
}

cmd_save() {
  [[ $# -ge 1 ]] || die "save needs <dump_file>"
  local dump_file="$1"
  mkdir -p "$(dirname "$dump_file")"
  stop_neo4j
  log "dumping ${NEO4J_DB} -> ${dump_file}"
  # Neo4j 5+/2026.x CLI: the database is a POSITIONAL arg and the destination
  # is a DIRECTORY (--to-path), inside which it writes "<db>.dump". Dump into a
  # temp dir, then move the produced file to the caller's requested path so
  # callers keep control of the filename.
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  neo4j-admin database dump "${NEO4J_DB}" \
    --to-path="${tmp_dir}" --overwrite-destination
  local produced="${tmp_dir}/${NEO4J_DB}.dump"
  [[ -s "$produced" ]] || { rm -rf "$tmp_dir"; die "dump produced empty/no file (${produced})"; }
  mv -f "$produced" "$dump_file"
  rm -rf "$tmp_dir"
  log "dump ok: ${dump_file} ($(du -h "$dump_file" | cut -f1))"
  start_neo4j
}

cmd_wipe() {
  stop_neo4j
  log "wiping ${NEO4J_DB} database files (auth in ${DATA_DIR}/dbms preserved)"
  rm -rf "${DATA_DIR}/databases/${NEO4J_DB}" "${DATA_DIR}/transactions/${NEO4J_DB}"
  start_neo4j
  log "wiped; ${NEO4J_DB} recreated empty on start"
}

cmd_load() {
  [[ $# -ge 1 ]] || die "load needs <dump_file>"
  local dump_file="$1"
  [[ -f "$dump_file" ]] || die "dump file not found: ${dump_file}"
  stop_neo4j
  log "loading ${dump_file} -> ${NEO4J_DB}"
  # Mirror cmd_save: load takes a POSITIONAL database and a --from-path
  # DIRECTORY that must contain "<db>.dump". Stage the caller's file into a
  # temp dir under that name, then load.
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  cp -f "$dump_file" "${tmp_dir}/${NEO4J_DB}.dump"
  neo4j-admin database load "${NEO4J_DB}" \
    --from-path="${tmp_dir}" --overwrite-destination
  rm -rf "$tmp_dir"
  start_neo4j
  log "load complete"
}

cmd_status() {
  log "neo4j status:"
  neo4j status 2>&1 | sed 's/^/  /' || true
  if python3 -c "import socket; socket.socket().connect(('localhost',7687))" 2>/dev/null; then
    log "bolt :7687 reachable"
  else
    log "bolt :7687 NOT reachable"
  fi
  if neo4j_is_running; then
    log "node counts by label:"
    cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
      "MATCH (n) RETURN labels(n)[0] AS lbl, count(*) AS c ORDER BY c DESC" \
      2>/dev/null | sed 's/^/  /' || true
  fi
}

cmd="${1:-}"; shift || true
case "$cmd" in
  save)   cmd_save "$@" ;;
  wipe)   cmd_wipe ;;
  load)   cmd_load "$@" ;;
  status) cmd_status ;;
  ""|-h|--help|help)
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *) die "unknown subcommand '$cmd' (try: $0 --help)" ;;
esac
