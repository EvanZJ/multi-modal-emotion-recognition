#!/usr/bin/env bash
set -euo pipefail

# Default environment variables
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-1}
RELOAD=${RELOAD:-false}

# If RELOAD is true, we run uvicorn with --reload (for dev), otherwise regular
if [ "$RELOAD" = "true" ] ; then
  exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
else
  exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi
