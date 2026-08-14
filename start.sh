#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and configure CERN credentials."
    exit 1
fi

if command -v podman >/dev/null 2>&1; then
    compose=(podman compose)
elif command -v docker >/dev/null 2>&1; then
    compose=(docker compose)
else
    echo "Neither podman nor docker is installed."
    exit 1
fi

"${compose[@]}" up -d --build

api_port="${LOG_RAG_API_PORT:-8002}"
echo "CMSSDT Log RAG is starting at http://localhost:${api_port}/"
echo "Health check: http://localhost:${api_port}/api/health"
