#!/bin/bash
# Usage: ./ssh2scp.sh "ssh -p PORT user@host ..." local_file remote_path
#
# Converts an ssh command (the kind vast.ai gives you) into the equivalent
# scp invocation. Useful for pushing this repo (or a single file) up to a
# rented GPU box without having to remember scp's argument order.
#
# Example (replace PORT, USER, HOST with the values your provider gave you):
#   ./ssh2scp.sh "ssh -p PORT USER@HOST -L 8080:localhost:8080" \
#                .env /root/generative_agents/.env

ssh_cmd="$1"
local_file="$2"
remote_path="$3"

# Extract port (after -p)
port=$(echo "$ssh_cmd" | grep -oP '(?<=-p\s)\d+')

# Extract user@host
userhost=$(echo "$ssh_cmd" | grep -oP '\S+@\S+' | head -1)

echo "scp -P $port $local_file $userhost:$remote_path"
