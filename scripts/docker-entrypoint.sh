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
  # -e .env: a .env written by hand inside a running container is untracked and NOT
  # covered by .gitignore, so a plain `clean -fd` would silently delete the operator's
  # HF_TOKEN on the next restart.
  if git fetch --depth 1 origin && git reset --hard '@{upstream}' && git clean -fd -e .env; then
    echo "Now at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
    if [ "$before_deps" != "$(git hash-object pyproject.toml uv.lock 2>/dev/null || true)" ]; then
      echo "WARNING: pyproject.toml/uv.lock changed -- rebuild the image, the baked-in venv wasn't updated." >&2
    fi
  else
    echo "WARNING: git pull failed -- continuing with the image's baked-in checkout ($(git rev-parse --short HEAD))." >&2
  fi
}

# Returns non-zero if anything the backend needs is missing. Every step reports its own
# status explicitly rather than relying on `set -e`, which is suspended for the whole
# call chain when a function runs inside an `if`.
fetch_models() {
  local ok=0
  echo "=== Fetching fixed stock models (Qwen encoder + video/audio VAEs) ==="
  if ! ./scripts/fetch_models.sh stock; then
    echo "WARNING: stock model fetch failed." >&2
    ok=1
  fi
  if [ -n "${DIT_URL:-}" ]; then
    echo "=== Fetching DiT checkpoint from DIT_URL ==="
    if ! ./scripts/fetch_models.sh dit "$DIT_URL" "${DIT_NAME:-}"; then
      echo "WARNING: DiT checkpoint fetch failed." >&2
      ok=1
    fi
  else
    echo "WARNING: DIT_URL not set -- no DiT checkpoint fetched. The UI will start but report" >&2
    echo "WARNING: no model found until one is fetched (set DIT_URL and restart, or run" >&2
    echo "WARNING: './scripts/fetch_models.sh dit <url>' by hand)." >&2
  fi
  return $ok
}

# Points the server at the checkpoint the fetch just validated. Without this,
# h3.discover_dit_checkpoint() globs diffusion_models/*.safetensors and refuses to guess
# when there's more than one -- which is exactly what happens once a second DIT_URL is
# fetched by hand into a running container.
pin_model_path() {
  local marker="${MODELS_ROOT:-${COMFYUI_ROOT:-/ComfyUI}/models}/diffusion_models/.active-dit"
  if [ -n "${H3_MODEL_PATH:-}" ]; then
    echo "Using H3_MODEL_PATH from the environment: $H3_MODEL_PATH"
    return 0
  fi
  if [ -f "$marker" ]; then
    local candidate
    candidate=$(cat "$marker")
    if [ -f "$candidate" ]; then
      export H3_MODEL_PATH="$candidate"
      echo "Pinned H3_MODEL_PATH=$H3_MODEL_PATH"
    fi
  fi
}

update_repo
if ! fetch_models; then
  # Deliberately does NOT exit. RunPod restarts the container on any non-zero exit, and
  # this image has no persistent volume, so exiting here turns one failed fetch into an
  # endless restart loop that re-downloads tens of GB per iteration and never surfaces a
  # readable error. The server starts regardless: web/server.py's _init_backend()
  # catches the load failure and reports it on /api/status, the pod stays reachable, and
  # the fetch can be retried from a shell without paying for another full download
  # (fetch_models.sh now reuses whatever it already has).
  echo "WARNING: ===================================================================" >&2
  echo "WARNING: Model fetch incomplete -- starting the UI anyway so the pod stays up" >&2
  echo "WARNING: and this log stays readable. The UI will report the backend error." >&2
  echo "WARNING: Retry from a shell (it resumes/reuses, it doesn't start over):" >&2
  echo "WARNING:   ./scripts/fetch_models.sh all \"\$DIT_URL\"" >&2
  echo "WARNING: ===================================================================" >&2
fi
pin_model_path

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
