#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if command -v podman >/dev/null 2>&1; then
    compose=(podman compose)
elif command -v docker >/dev/null 2>&1; then
    compose=(docker compose)
else
    echo "Neither podman nor docker is installed."
    exit 1
fi

"${compose[@]}" down

echo "CMSSDT Log RAG services stopped. Persistent volumes were preserved."
