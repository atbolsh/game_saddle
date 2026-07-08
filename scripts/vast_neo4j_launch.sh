#!/usr/bin/env bash
#
# vast_neo4j_launch.sh -- idempotent bare-metal Neo4j setup for the game_saddle
# NAMS memory backend, for rented GPU boxes (Vast.ai / similar) that don't run
# a Docker daemon inside the container. This is the no-Docker replacement for
# `docker compose up -d neo4j`.
#
# What it does (safe to re-run):
#   1. Installs Neo4j Community + OpenJDK 17 via apt if they aren't already
#      installed (many Vast Ubuntu images already have them; this is the
#      fallback for a truly bare box).
#   2. Writes the game_saddle config block into /etc/neo4j/neo4j.conf
#      (idempotent: any previous block is stripped first) -- listen address,
#      bolt/http ports, heap, and the APOC procedure allowlist.
#   3. Writes /etc/neo4j/apoc.conf with the apoc.* settings (Neo4j v5
#      strict_validation rejects apoc.* inside neo4j.conf).
#   4. Downloads the APOC core jar matching the installed Neo4j version into
#      /var/lib/neo4j/plugins (NAMS uses APOC for extraction/dedup).
#   5. Sets the initial password to $NEO4J_PASSWORD (default 'changeme', which
#      matches the .env.example default) -- only on a fresh data dir.
#   6. Starts Neo4j (service-style `neo4j start`; falls back to `neo4j console`
#      backgrounded as root if the neo4j system user is missing).
#   7. Waits for bolt on :7687 and verifies APOC is callable.
#   8. Appends NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD to the repo-root
#      .env so agent/config.py picks them up (idempotent).
#
# The agent then connects with the defaults from agent/config.py:
# NEO4J_URI=bolt://localhost:7687, NEO4J_USERNAME=neo4j,
# NEO4J_PASSWORD=changeme, NEO4J_DATABASE=neo4j.
#
# Tear down / re-provision: to wipe the graph and start over, run
#   bash scripts/neo4j_db.sh wipe
# To save the current graph to a .dump file, run
#   bash scripts/neo4j_db.sh save <dump_file>
#
# Usage:
#   bash scripts/vast_neo4j_launch.sh
#
# Env overrides:
#   NEO4J_PASSWORD  (default changeme)  -- must match the agent's NEO4J_PASSWORD
#   NEO4J_USERNAME  (default neo4j)
#   NEO4J_HEAP_MAX  (default 4G)        -- server.memory.heap.max_size

set -euo pipefail

NEO4J_PASSWORD="${NEO4J_PASSWORD:-changeme}"
NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
NEO4J_HEAP_MAX="${NEO4J_HEAP_MAX:-4G}"

log() { printf '[vast-neo4j] %s\n' "$*"; }
die() { printf '[vast-neo4j] ERROR: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ 1. install
install_neo4j_if_missing() {
  if command -v neo4j >/dev/null 2>&1; then
    log "neo4j already installed: $(neo4j --version)"
    return 0
  fi
  log "neo4j not found; installing via apt (requires network + apt-get)"
  command -v apt-get >/dev/null 2>&1 || die "apt-get not found; this script targets Debian/Ubuntu"
  # Java prerequisite (Neo4j 5.x / 2026.x needs JDK 17+)
  if ! command -v java >/dev/null 2>&1; then
    log "installing openjdk-17-jre-headless"
    apt-get update && apt-get install -y openjdk-17-jre-headless
  fi
  # Neo4j official Debian repo + signing key
  if ! command -v gpg >/dev/null 2>&1; then
    apt-get install -y gnupg
  fi
  if [ ! -f /usr/share/keyrings/neo4j.gpg ]; then
    log "adding Neo4j apt signing key"
    wget -qO - https://debian.neo4j.com/neotechnology.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/neo4j.gpg
  fi
  if [ ! -f /etc/apt/sources.list.d/neo4j.list ]; then
    log "adding Neo4j apt repo"
    echo 'deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest' \
      > /etc/apt/sources.list.d/neo4j.list
  fi
  apt-get update
  apt-get install -y neo4j
  log "installed: $(neo4j --version)"
}

# ----------------------------------------------------------- 2/3. configure
configure_neo4j() {
  local CONF=/etc/neo4j/neo4j.conf
  [ -f "$CONF" ] || die "neo4j.conf not found at $CONF"
  # Strip any previous game_saddle block, then append a fresh one.
  sed -i '/# --- game_saddle NAMS settings ---/,/# --- end game_saddle settings ---/d' "$CONF"
  cat >> "$CONF" <<EOF

# --- game_saddle NAMS settings ---
server.default_listen_address=127.0.0.1
server.bolt.listen_address=:7687
server.http.listen_address=:7474
server.memory.heap.initial_size=512m
server.memory.heap.max_size=${NEO4J_HEAP_MAX}
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
# --- end game_saddle settings ---
EOF
  log "neo4j.conf updated"

  # APOC settings live in apoc.conf in v5 (strict_validation rejects them in
  # neo4j.conf).
  cat > /etc/neo4j/apoc.conf <<'EOF'
apoc.import.file.enabled=true
apoc.export.file.enabled=true
EOF
  chown neo4j:adm /etc/neo4j/apoc.conf 2>/dev/null || true
  log "apoc.conf written"
}

# --------------------------------------------------------- 4. APOC jar
install_apoc_jar() {
  local VERSION
  VERSION="$(neo4j --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  log "detected neo4j ${VERSION}"
  local PLUGINS=/var/lib/neo4j/plugins
  mkdir -p "$PLUGINS"
  local APOC_JAR="$PLUGINS/apoc-${VERSION}-core.jar"
  if [ -f "$APOC_JAR" ]; then
    log "APOC jar already present: $APOC_JAR"
    return 0
  fi
  log "downloading $APOC_JAR"
  curl -L -o "$APOC_JAR" \
    "https://github.com/neo4j/apoc/releases/download/${VERSION}/apoc-${VERSION}-core.jar"
  chown neo4j:adm "$APOC_JAR" 2>/dev/null || true
  log "APOC jar installed"
}

# ------------------------------------------------------- 5. initial password
set_initial_password_if_fresh() {
  if [ -f /var/lib/neo4j/data/dbms/auth.ini ] || [ -d /var/lib/neo4j/data/db ]; then
    log "auth already initialized; skipping set-initial-password"
    log "  (if you don't know the password, stop neo4j and run:"
    log "   neo4j-admin dbms set-default-admin ${NEO4J_USERNAME})"
    return 0
  fi
  neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"
  log "initial password set to '${NEO4J_PASSWORD}'"
}

# ------------------------------------------------------- 6. start
start_neo4j() {
  # If it's already running, leave it.
  if neo4j status 2>/dev/null | grep -qi running; then
    log "neo4j already running"
    return 0
  fi
  if neo4j start 2>/tmp/vast_neo4j_start.err; then
    log "neo4j start ok"
    return 0
  fi
  log "neo4j start failed; falling back to console as root"
  head /tmp/vast_neo4j_start.err >&2 || true
  mkdir -p /var/log/neo4j
  nohup neo4j console >/var/log/neo4j/console.log 2>&1 &
  echo $! > /var/run/neo4j.pid
  log "neo4j console launched, pid $(cat /var/run/neo4j.pid), logs: /var/log/neo4j/console.log"
}

# ------------------------------------------------------- 7. verify
wait_for_bolt() {
  local i=0
  while (( i < 90 )); do
    if python3 -c "import socket; socket.socket().connect(('localhost',7687))" 2>/dev/null; then
      log "bolt ready on :7687 after ${i}s"
      return 0
    fi
    sleep 1; i=$((i + 1))
  done
  die "bolt never came up on :7687 -- check /var/log/neo4j/neo4j.log"
}

verify_apoc() {
  if cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" "RETURN apoc.version() AS v" \
        >/tmp/vast_apoc_check.out 2>/tmp/vast_apoc_check.err; then
    log "APOC callable: $(grep -oE '"[0-9.]+"' /tmp/vast_apoc_check.out | head -1)"
  else
    die "APOC procedure apoc.version() failed -- see /tmp/vast_apoc_check.err"
  fi
}

# Write the Neo4j connection vars into the repo-root .env so agent/config.py
# picks them up explicitly (it loads .env on import via python-dotenv). The
# defaults already match, so this is for explicitness. Idempotent: only appends
# vars that aren't already present.
write_env_file() {
  local env_file
  env_file="$(cd "$(dirname "$0")/.." && pwd)/.env"
  touch "$env_file"
  for kv in "NEO4J_URI=bolt://localhost:7687" \
            "NEO4J_USERNAME=${NEO4J_USERNAME}" \
            "NEO4J_PASSWORD=${NEO4J_PASSWORD}"; do
    local key="${kv%%=*}"
    if ! grep -qE "^${key}=" "$env_file"; then
      printf '%s\n' "$kv" >> "$env_file"
      log "appended ${key}=... to ${env_file}"
    else
      log "${key} already present in ${env_file}; left as-is"
    fi
  done
}

# --------------------------------------------------------------------- main
install_neo4j_if_missing
configure_neo4j
install_apoc_jar
set_initial_password_if_fresh
start_neo4j
wait_for_bolt
verify_apoc
write_env_file

log "Neo4j is ready for game_saddle NAMS:"
log "  bolt:    bolt://localhost:7687"
log "  browser: http://localhost:7474  (user ${NEO4J_USERNAME} / password '${NEO4J_PASSWORD}')"
log "  APOC:    loaded"
log "Next: seed the semantic model with"
log "  python -m agent.runner seed"
