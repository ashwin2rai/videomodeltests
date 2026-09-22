#!/usr/bin/env bash
# Container entrypoint: updates the checkout, fetches models, then execs into the
# requested mode. No persistent volume (see objective/status.md), so models re-download
# on every fresh container.
set -euo pipefail

cd /app

# Updates /app to the latest commit of the branch it was built from, so a restart picks
# up app-code changes without a rebuild (SKIP_GIT_PULL=1 to pin to what was baked in).
# Uses reset --hard rather than a merge pull: the builder strips tests/objective, which
# a plain pull would see as uncommitted deletions. clean -fd (no -x) still respects
# .gitignore, so inputs/outputs/ are untouched. Does not re-run `uv sync`, so a
# pyproject.toml/uv.lock change is called out below instead of silently going stale.
update_repo() {
  if [ "${SKIP_GIT_PULL:-0}" = "1" ]; then
    echo "=== SKIP_GIT_PULL=1 -- using the image's baked-in checkout ($(git rev-parse --short HEAD)) ==="
    return
  fi
  echo "=== Updating videomodeltests from $(git remote get-url origin) ($(git rev-parse --abbrev-ref HEAD)) ==="
  local before_deps
  before_deps=$(git hash-object pyproject.toml uv.lock 2>/dev/null || true)
  if git fetch --depth 1 origin && git reset --hard '@{upstream}' && git clean -fd; then
    echo "Now at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
    if [ "$before_deps" != "$(git hash-object pyproject.toml uv.lock 2>/dev/null || true)" ]; then
      echo "WARNING: pyproject.toml/uv.lock changed -- rebuild the image, the baked-in venv wasn't updated." >&2
    fi
  else
    echo "WARNING: git pull failed -- continuing with the image's baked-in checkout ($(git rev-parse --short HEAD))." >&2
  fi
}

fetch_models() {
  echo "=== Fetching fixed stock models (Qwen encoder + video/audio VAEs) ==="
  ./scripts/fetch_models.sh stock
  if [ -n "${DIT_URL:-}" ]; then
    echo "=== Fetching DiT checkpoint from DIT_URL ==="
    ./scripts/fetch_models.sh dit "$DIT_URL" "${DIT_NAME:-}"
  else
    echo "WARNING: DIT_URL not set -- no DiT checkpoint fetched. The UI will start but report" >&2
    echo "WARNING: no model found until one is fetched (set DIT_URL and restart, or run" >&2
    echo "WARNING: './scripts/fetch_models.sh dit <url>' by hand)." >&2
  fi
}

update_repo
fetch_models

mode="${1:-serve}"
case "$mode" in
  serve)
    echo "=== Starting web UI (real backend) on port ${PORT:-8000} ==="
    exec python web/server.py
    ;;
  serve-mock)
    echo "=== Starting web UI (mock backend) on port ${PORT:-8000} ==="
    H3_MOCK=1 exec python web/server.py
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
