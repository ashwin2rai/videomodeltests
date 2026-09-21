#!/usr/bin/env bash
# Container entrypoint. Fetches the fixed stock models (always) and the swappable DiT
# checkpoint (only if DIT_URL is set), then execs into the requested mode.
#
# There is deliberately no persistent volume for MODELS_ROOT (see
# objective/status.md's Docker planning notes) -- both fetches land on the container's
# own ephemeral disk, so this genuinely re-downloads on every fresh container start.
# That's an accepted trade, not a bug: keep it simple rather than half-solving
# persistence with volume-mount logic nobody asked for.
set -euo pipefail

cd /app

echo "=== Fetching fixed stock models (Qwen encoder + video/audio VAEs) ==="
./scripts/fetch_models.sh stock

if [ -n "${DIT_URL:-}" ]; then
  echo "=== Fetching DiT checkpoint from DIT_URL ==="
  ./scripts/fetch_models.sh dit "$DIT_URL" "${DIT_NAME:-}"
else
  echo "WARNING: DIT_URL not set -- no DiT checkpoint will be fetched." >&2
  echo "WARNING: the UI will still start, but will report no model found until" >&2
  echo "WARNING: one is fetched (set DIT_URL and restart, or exec in and run" >&2
  echo "WARNING: './scripts/fetch_models.sh dit <url>' by hand)." >&2
fi

mode="${1:-serve}"

case "$mode" in
  serve)
    echo "=== Starting web UI (real backend) on port ${PORT:-8000} ==="
    exec python web/server.py
    ;;
  serve-mock)
    echo "=== Starting web UI (mock backend) on port ${PORT:-8000} ==="
    export H3_MOCK=1
    exec python web/server.py
    ;;
  check)
    exec python scripts/check_env.py
    ;;
  bash|shell)
    exec bash
    ;;
  *)
    echo "Unknown mode: $mode (expected: serve | serve-mock | check | bash)" >&2
    exit 1
    ;;
esac
