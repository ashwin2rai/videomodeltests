#!/usr/bin/env bash
# Downloads model weights into MODELS_ROOT (default: h3.MODELS_ROOT).
#
#   stock                    the three fixed files (Qwen encoder + video/audio VAEs)
#                            from Comfy-Org/MiniMax-H3 — never the whole ~480GB repo
#   dit <url> [filename]     a DiT checkpoint into diffusion_models/: huggingface.co URLs
#                            go through `hf` (fast Xet transfers), anything else through curl
#   all <url> [filename]     both
#
# Safe to re-run: files already in place are validated and reused, interrupted
# downloads resume. Every file is checked with h3.validate_checkpoint(), which also
# catches truncated downloads.
#
# Environment (a .env in the repo root fills in anything not already set):
#   HF_TOKEN                 optional, only ever used by `hf` (i.e. only sent to huggingface.co)
#   HF_XET_HIGH_PERFORMANCE  1/0; default: 1 when >=64GB RAM is available to this process
#   FORCE_REFETCH=1          re-download the DiT even if a valid copy is already in place
#   PROGRESS_INTERVAL_SECONDS  how often to log download progress (default 15)
#
# Needs `python3` that can import h3 (the Makefile runs this under `uv run`; the Docker
# image puts its venv on PATH).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STOCK_REPO="Comfy-Org/MiniMax-H3"

if [ -f .env ]; then
  while IFS='=' read -r key value; do
    key="${key#export }"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    value="${value%\"}" value="${value#\"}"
    [ -n "${!key:-}" ] || export "$key=$value"
  done < .env
fi

MODELS_ROOT="${MODELS_ROOT:-$(python3 -c 'import h3; print(h3.MODELS_ROOT)')}"

# Prefer the venv's pinned `hf`; uvx is only for a dev checkout without the gpu group.
if command -v hf >/dev/null 2>&1; then HF=(hf); else HF=(uvx --from 'huggingface_hub[hf_xet]' hf); fi

# hf-xet's high-performance mode buffers aggressively, so only auto-enable it when the
# memory actually available to this container (cgroup limit, not host total) is >=64GB.
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE//[[:space:]]/}"
if [ -z "$HF_XET_HIGH_PERFORMANCE" ]; then
  mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
  if [[ "$limit" =~ ^[0-9]+$ ]] && [ $((limit / 1024)) -lt "$mem_kb" ]; then mem_kb=$((limit / 1024)); fi
  HF_XET_HIGH_PERFORMANCE=0
  [ "$mem_kb" -ge $((64 * 1024 * 1024)) ] && HF_XET_HIGH_PERFORMANCE=1
fi
export HF_XET_HIGH_PERFORMANCE

# Prints OK + a one-line summary, or the validation error; non-zero if unusable.
validate() {
  CHECKPOINT="$1" python3 -c '
import os, sys, h3
try:
    info = h3.validate_checkpoint(os.environ["CHECKPOINT"])
except Exception as e:
    sys.exit(f"INVALID: {e}")
print("OK: %d tensors, dtypes=%s" % (info["tensor_count"], info["dtypes"]))'
}

# Runs "$@" while logging bytes on disk under $1 every few seconds. hf/curl draw
# progress with \r, which never shows up in RunPod's log panel.
with_progress() {
  local target="$1" pid start=$SECONDS
  shift
  "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL_SECONDS:-15}" &
    wait $! || true
    kill -0 "$pid" 2>/dev/null && echo "  ...$((SECONDS - start))s: $(du -sh "$target" 2>/dev/null | cut -f1) on disk"
  done
  wait "$pid"
}

# Up to 3 attempts; both hf and `curl -C -` resume rather than start over.
retry() {
  local n
  for n in 1 2 3; do
    "$@" && return 0
    [ "$n" -lt 3 ] && { echo "WARNING: attempt $n failed, retrying in $((n * 10))s" >&2; sleep $((n * 10)); }
  done
  return 1
}

fetch_stock() {
  local rel bad=0 rels=()
  mapfile -t rels < <(python3 -c '
import h3
for p in (h3.QWEN_ENCODER_PATH, h3.VIDEO_VAE_PATH, h3.AUDIO_VAE_PATH):
    print(p.removeprefix(h3.MODELS_ROOT + "/"))')
  echo "Fetching ${#rels[@]} stock files from $STOCK_REPO..."
  # `hf` skips files whose etag already matches, so this is quick when they're present.
  retry with_progress "$MODELS_ROOT" "${HF[@]}" download "$STOCK_REPO" "${rels[@]}" \
    --local-dir "$MODELS_ROOT" --format quiet >/dev/null
  for rel in "${rels[@]}"; do
    echo -n "$rel: "
    validate "$MODELS_ROOT/$rel" || bad=1
  done
  return "$bad"
}

fetch_dit() {
  local url="$1" name="${2:-}" dir="$MODELS_ROOT/diffusion_models" repo="" rev="" path="" dest src staging
  if [[ "$url" =~ ^https?://huggingface\.co/([^/]+/[^/]+)/(blob|resolve)/([^/]+)/([^?]+) ]]; then
    repo="${BASH_REMATCH[1]}" rev="${BASH_REMATCH[3]}" path="${BASH_REMATCH[4]}"
  fi
  name="${name:-$(basename "${path:-${url%%\?*}}")}"
  # e.g. CivitAI's .../api/download/models/12345 has no extension; the server's glob
  # (and ComfyUI's loader) expect *.safetensors.
  [[ "$name" == *.safetensors ]] || name="$name.safetensors"
  dest="$dir/$name"
  mkdir -p "$dir"

  if [ -f "$dest" ] && [ "${FORCE_REFETCH:-0}" != 1 ]; then
    echo "Found existing $name, validating..."
    if validate "$dest"; then
      echo "$dest" > "$dir/.active-dit"
      return 0
    fi
    rm -f "$dest"
  fi

  # Staged next to the destination (hidden from the server's glob) so the final mv is an
  # atomic same-filesystem rename, and a re-run resumes the partial file.
  staging="$dir/.incoming"
  mkdir -p "$staging"
  if [ -n "$repo" ]; then
    echo "Fetching $path from $repo@$rev via hf..."
    retry with_progress "$staging" "${HF[@]}" download "$repo" "$path" --revision "$rev" \
      --local-dir "$staging" --format quiet >/dev/null
    src="$staging/$path"
  else
    echo "Fetching $name via curl..."
    src="$staging/$name"
    retry with_progress "$staging" curl -fL -sS -C - --retry 3 -o "$src" "$url"
  fi

  if ! validate "$src"; then
    echo "ERROR: downloaded file is not a usable H3 FL2VA checkpoint — check the DiT URL." >&2
    rm -rf "$staging"
    return 1
  fi
  mv -f "$src" "$dest"
  rm -rf "$staging"
  echo "$dest" > "$dir/.active-dit"
  echo "Installed $dest"
}

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

cmd="${1:-}"
case "$cmd" in
  stock) ;;
  dit | all) [ -n "${2:-}" ] || usage ;;
  *) usage ;;
esac

mkdir -p "$MODELS_ROOT"
echo "MODELS_ROOT=$MODELS_ROOT ($(df -h --output=avail "$MODELS_ROOT" | tail -1 | tr -d ' ') free)," \
  "HF_TOKEN $([ -n "${HF_TOKEN:-}" ] && echo set || echo 'not set'), HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"

# One fetch at a time (e.g. the entrypoint's and a manual one from a shell).
exec 9>"$MODELS_ROOT/.fetch.lock"
flock -n 9 || { echo "Waiting for another fetch to finish..."; flock 9; }

case "$cmd" in
  stock) fetch_stock ;;
  dit) fetch_dit "$2" "${3:-}" ;;
  all) fetch_stock && fetch_dit "$2" "${3:-}" ;;
esac
