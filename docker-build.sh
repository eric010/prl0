#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/dist"
docker build --progress=plain --output "type=local,dest=$ROOT/dist" -f "$ROOT/Dockerfile.build" "$ROOT"
echo "Pacote em: $ROOT/dist/prl0-0.1.0.tar.gz"
