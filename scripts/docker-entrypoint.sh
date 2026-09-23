#!/usr/bin/env bash
# Container entrypoint: update code from git -> GPU self-check -> fetch models -> serve.
#
# Never exits because something failed. RunPod restarts a container that exits and
# there's no volume, so a crash loop would re-download ~40GB on every restart and the
# error would scroll away. Failures are logged as WARNING and the container stays up.
set -uo pipefail
cd /app

mode="${1:-serve}"
warn() { echo "WARNING: $*" >&2; }

case "$mode" in
  serve | serve-mock) ;;
  check) exec python scripts/check_env.py ;;
  bash | shell) exec bash ;;
  *) echo "Unknown mode: $mode (expected: serve | serve-mock | check | bash)" >&2; exit 1 ;;
esac

# Pick up app-code changes on restart without a rebuild, then re-run the *updated*
# entrypoint. Dependencies are baked into the image, so a lockfile change needs a rebuild.
if [ "${SKIP_GIT_PULL:-0}" != 1 ]; then
  echo "=== Updating /app from $(git remote get-url origin) ==="
  deps_before=$(cat pyproject.toml uv.lock | sha1sum)
  if git fetch --depth 1 origin && git reset --hard '@{upstream}'; then
    echo "Now at $(git log -1 --format='%h %s')"
    [ "$deps_before" = "$(cat pyproject.toml uv.lock | sha1sum)" ] \
      || warn "pyproject.toml/uv.lock changed upstream; rebuild the image to update the venv."
  else
    warn "git update failed; using the checkout baked into the image ($(git rev-parse --short HEAD))."
  fi
  SKIP_GIT_PULL=1 exec bash "$0" "$@"
fi

if [ "$mode" = serve ]; then
  # Seconds to run, and it fails *before* the 40GB download if the image can't do real
  # generation (no GPU/driver too old for this torch build, no C compiler for Triton, ...).
  echo "=== Self-check ==="
  python scripts/check_env.py || warn "self-check failed (see above); real generation will likely fail."

  echo "=== Fetching models ==="
  scripts/fetch_models.sh stock || warn "stock model fetch failed."
  if [ -n "${DIT_URL:-}" ]; then
    scripts/fetch_models.sh dit "$DIT_URL" "${DIT_NAME:-}" || warn "DiT fetch failed."
  else
    warn "DIT_URL not set; no DiT checkpoint fetched."
  fi
else
  export H3_MOCK=1
fi

echo "=== Starting web UI ($mode) on port ${PORT:-8000} ==="
python web/server.py
warn "web server exited with status $?. Keeping the container alive to avoid a restart loop;"
warn "retry with: docker exec / RunPod web terminal -> python web/server.py"
exec sleep infinity
