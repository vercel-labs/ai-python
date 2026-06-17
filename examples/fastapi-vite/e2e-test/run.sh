#!/usr/bin/env bash
# Launch backend + frontend, run the e2e test, tear everything down.
#
# Requires a configured AI Gateway provider.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

if ! uv run --frozen --with-editable "$ROOT/../.." python - <<'PY'
import sys

import ai

sys.exit(0 if ai.get_provider("vercel").is_configured() else 1)
PY
then
    echo "AI Gateway provider is not configured" >&2
    exit 1
fi

LOGS=$(mktemp -d)
echo "Logs: $LOGS"

BACKEND_PORT=${BACKEND_PORT:-8765}
FRONTEND_PORT=${FRONTEND_PORT:-5765}
export BACKEND_URL="http://localhost:$BACKEND_PORT"
export FRONTEND_URL="http://localhost:$FRONTEND_PORT/"

cleanup() {
    [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
    [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting backend on :$BACKEND_PORT..."
(
    cd "$ROOT/backend"
    uv run --frozen --with-editable "$ROOT/../.." fastapi dev main.py --port "$BACKEND_PORT"
) > "$LOGS/backend.log" 2>&1 &
BACKEND_PID=$!

echo "Starting frontend on :$FRONTEND_PORT..."
(
    cd "$ROOT/frontend"
    pnpm dev --port "$FRONTEND_PORT" --strictPort
) > "$LOGS/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "Waiting for backend..."
until curl -sf "http://127.0.0.1:$BACKEND_PORT/health" > /dev/null 2>&1; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Backend died — see $LOGS/backend.log" >&2
        exit 1
    fi
    sleep 0.5
done

echo "Waiting for frontend..."
until curl -sf "http://127.0.0.1:$FRONTEND_PORT/" > /dev/null 2>&1; do
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "Frontend died — see $LOGS/frontend.log" >&2
        exit 1
    fi
    sleep 0.5
done

echo "Running e2e test..."
cd "$HERE"
[[ -d node_modules ]] || npm install
node e2e-test.mjs
