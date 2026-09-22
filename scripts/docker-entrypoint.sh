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

# Pulls the latest commit of the branch/tag this image was built against, so a plain
# `docker run`/pod restart picks up app-code changes without a rebuild. `reset --hard` +
# `clean -fd` (not `-x`) rather than a plain `git pull`, since the image intentionally
# strips tests/objective at build time (see Dockerfile) -- a merge-based pull would
# choke on those as uncommitted local deletions; a hard reset just restores them from
# git, which is harmless. `-fd` without `-x` still respects .gitignore, so inputs/
# outputs/ (this container's actual data) are never touched. Does NOT re-run `uv sync`:
# a pull that only changes pyproject.toml/uv.lock updates the source but not the
# already-baked venv, so that case is detected and warned about explicitly below instead
# of silently running with a stale environment.
if [ "${SKIP_GIT_PULL:-0}" = "1" ]; then
  echo "=== SKIP_GIT_PULL=1 -- using the image's baked-in checkout ($(git rev-parse --short HEAD)) ==="
else
  echo "=== Updating videomodeltests from $(git remote get-url origin) ($(git rev-parse --abbrev-ref HEAD)) ==="
  before_deps=$(git hash-object pyproject.toml uv.lock 2>/dev/null || true)
  if git fetch --depth 1 origin && git reset --hard '@{upstream}' && git clean -fd; then
    echo "Now at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
    if [ "$before_deps" != "$(git hash-object pyproject.toml uv.lock 2>/dev/null || true)" ]; then
      echo "WARNING: pyproject.toml/uv.lock changed -- the image's baked-in venv was NOT updated." >&2
      echo "WARNING: rebuild the image for new/changed dependencies to actually take effect." >&2
    fi
  else
    echo "WARNING: git pull failed -- continuing with the image's baked-in checkout ($(git rev-parse --short HEAD))." >&2
  fi
fi

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
